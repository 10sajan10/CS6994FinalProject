from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

# Import the robust data processing pipeline from your transformer script
from train_discharge_transformer import (
    load_parquet_frames,
    parse_timestamp_columns,
    deduplicate_site_rows,
    infer_columns,
    trailing_one_year_split,
    fit_scalers,
    transform_frame,
    make_persistence_predictions,
    inverse_scale_targets,
    reconstruct_absolute_targets,
    clip_prediction_variation,
    compute_overall_metrics,
    relative_skill_score,
    expand_predictions_by_target_timestamp,
    aggregate_season_metrics,
    aggregate_horizon_season_metrics,
    set_global_seed,
    latest_discharge_column
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_pattern = str(project_root / "streamflow_parquet_v2" / "*precip24avg*.parquet")

    parser = argparse.ArgumentParser(
        description="Train per-site XGBoost baseline models on multihorizon discharge parquet data."
    )
    parser.add_argument(
        "--parquet-pattern",
        default=default_pattern,
        help="Glob pattern for parquet inputs.",
    )
    parser.add_argument(
        "--output-root",
        default=str(project_root),
        help="Project root where xgboost models and metrics will be written.",
    )
    # XGBoost specific hyperparameters
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=-1, help="Number of CPU threads for XGBoost (-1 for all)")
    
    # Post-processing configurations (kept consistent with your transformer)
    parser.add_argument("--variation-clip-pct", type=float, default=0.05)
    parser.add_argument("--variation-clip-history-multiplier", type=float, default=1.5)
    parser.add_argument("--variation-clip-absolute-floor", type=float, default=2.0)
    parser.add_argument("--disable-variation-clip", action="store_true")
    return parser.parse_args()


def train_one_site_xgboost(
    site_df: pd.DataFrame,
    args: argparse.Namespace,
    output_root: Path,
) -> tuple[pd.DataFrame, dict]:
    site_id = int(site_df["site_id"].iloc[0])
    print(f"\n{'=' * 80}")
    print(f"Training XGBoost Baseline for site_id={site_id}")
    print(f"{'=' * 80}")

    lag_cols, target_cols, static_cols = infer_columns(site_df)
    if not lag_cols or not target_cols:
        raise ValueError(f"Site {site_id} is missing lag or target columns.")

    # 1. Data Splitting
    train_df, valid_df, validation_start, validation_end = trailing_one_year_split(site_df)
    print(
        f"Site {site_id}: train rows={len(train_df):,}, validation rows={len(valid_df):,}, "
        f"validation window={validation_start} to {validation_end}"
    )

    # 2. Scaling & Feature Extraction (Reusing the Transformer's logic)
    lag_scaler, static_scaler, target_scaler, static_feature_cols = fit_scalers(
        train_df, lag_cols, static_cols, target_cols,
    )

    train_lags, train_temporal, train_static, train_targets = transform_frame(
        train_df, lag_cols, target_cols, static_cols, static_feature_cols,
        lag_scaler, static_scaler, target_scaler,
    )
    valid_lags, valid_temporal, valid_static, valid_targets = transform_frame(
        valid_df, lag_cols, target_cols, static_cols, static_feature_cols,
        lag_scaler, static_scaler, target_scaler,
    )

    # 3. Flatten 3D sequence features into 2D tabular features for XGBoost
    # train_temporal is shape (Batch, Seq_Len, Temporal_Features)
    X_train = np.concatenate([
        train_lags, 
        train_temporal.reshape(train_temporal.shape[0], -1), 
        train_static
    ], axis=1)

    X_valid = np.concatenate([
        valid_lags, 
        valid_temporal.reshape(valid_temporal.shape[0], -1), 
        valid_static
    ], axis=1)

    site_model_dir = output_root / "models_discharge_xgboost" / str(site_id)
    site_model_dir.mkdir(parents=True, exist_ok=True)

    # 4. Initialize and Train XGBoost
    print(f"Site {site_id}: Fitting XGBoost MultiOutputRegressor...")
    base_xgb = XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        random_state=args.seed,
        n_jobs=args.num_workers
    )
    # MultiOutputRegressor trains one model per horizon step
    model = MultiOutputRegressor(base_xgb)
    model.fit(X_train, train_targets)

    # 5. Predictions & Post-processing
    print(f"Site {site_id}: Generating predictions...")
    scaled_predictions = model.predict(X_valid)

    predicted_deltas = inverse_scale_targets(scaled_predictions, target_scaler)
    y_pred_unclipped = reconstruct_absolute_targets(predicted_deltas, valid_df, lag_cols)
    
    if args.disable_variation_clip:
        y_pred = y_pred_unclipped
    else:
        y_pred = clip_prediction_variation(
            predictions=y_pred_unclipped,
            df=valid_df,
            lag_cols=lag_cols,
            pct_limit=args.variation_clip_pct,
            history_multiplier=args.variation_clip_history_multiplier,
            absolute_floor=args.variation_clip_absolute_floor,
        )
        
    y_true = valid_df[target_cols].to_numpy(dtype=np.float32)
    y_persistence = make_persistence_predictions(valid_df, lag_cols, target_cols)

    # 6. Evaluation
    rmse, mae = compute_overall_metrics(y_true, y_pred)
    persistence_rmse, persistence_mae = compute_overall_metrics(y_true, y_persistence)
    rmse_skill = relative_skill_score(rmse, persistence_rmse)
    mae_skill = relative_skill_score(mae, persistence_mae)
    
    print(f"Site {site_id}: validation RMSE={rmse:.4f}, MAE={mae:.4f}")
    mae_skill_text = f"{mae_skill:.4f}" if mae_skill is not None else "n/a"
    print(
        f"Site {site_id}: persistence RMSE={persistence_rmse:.4f}, MAE={persistence_mae:.4f}, "
        f"model MAE skill={mae_skill_text}"
    )

    # 7. Formatting Outputs
    expanded_predictions = expand_predictions_by_target_timestamp(
        site_id=site_id,
        valid_df=valid_df,
        target_cols=target_cols,
        y_true=y_true,
        y_pred=y_pred,
        y_pred_unclipped=y_pred_unclipped,
        y_persistence=y_persistence,
    )

    run_summary = {
        "site_id": site_id,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "validation_start": str(validation_start),
        "validation_end": str(validation_end),
        "rmse": rmse,
        "mae": mae,
        "variation_clip_enabled": not args.disable_variation_clip,
        "variation_clip_pct": args.variation_clip_pct,
        "variation_clip_history_multiplier": args.variation_clip_history_multiplier,
        "variation_clip_absolute_floor": args.variation_clip_absolute_floor,
        "persistence_rmse": persistence_rmse,
        "persistence_mae": persistence_mae,
        "rmse_skill_vs_persistence": rmse_skill,
        "mae_skill_vs_persistence": mae_skill,
        "model_type": "xgboost_multioutput"
    }
    (site_model_dir / "run_summary_xgb.json").write_text(json.dumps(run_summary, indent=2) + "\n")

    return expanded_predictions, run_summary


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print("Starting discharge XGBoost Baseline pipeline")
    print(f"Parquet pattern: {args.parquet_pattern}")
    print(f"Output root: {output_root}")

    df = load_parquet_frames(args.parquet_pattern)
    df = parse_timestamp_columns(df)
    df = deduplicate_site_rows(df)

    if "site_id" not in df.columns:
        raise ValueError("Expected parquet data to contain a site_id column.")

    all_expanded_predictions = []
    overall_rows = []

    for site_id, site_df in df.groupby("site_id", sort=True):
        site_df = site_df.sort_values("forecast_start_time").reset_index(drop=True)
        expanded_predictions, run_summary = train_one_site_xgboost(site_df, args, output_root)
        all_expanded_predictions.append(expanded_predictions)
        overall_rows.append(run_summary)

    # Aggregate and Save Metrics with `_xgboost_` identifiers
    combined_predictions = pd.concat(all_expanded_predictions, ignore_index=True)
    
    metrics_by_season = aggregate_season_metrics(combined_predictions)
    metrics_path = output_root / "discharge_xgboost_metrics_by_season.csv"
    metrics_by_season.to_csv(metrics_path, index=False)
    print(f"Saved seasonal metrics to {metrics_path}")

    horizon_metrics = aggregate_horizon_season_metrics(combined_predictions)
    horizon_metrics_path = output_root / "discharge_xgboost_metrics_by_season_horizon.csv"
    horizon_metrics.to_csv(horizon_metrics_path, index=False)
    print(f"Saved horizon seasonal metrics to {horizon_metrics_path}")

    overall_metrics_path = output_root / "discharge_xgboost_overall_validation_metrics.csv"
    pd.DataFrame(overall_rows).to_csv(overall_metrics_path, index=False)
    print(f"Saved overall validation metrics to {overall_metrics_path}")

    expanded_predictions_path = output_root / "discharge_xgboost_validation_predictions_long.csv"
    combined_predictions.to_csv(expanded_predictions_path, index=False)
    print(f"Saved validation predictions to {expanded_predictions_path}")


if __name__ == "__main__":
    main()