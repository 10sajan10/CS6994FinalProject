from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import Dataset

try:
    from train_discharge_transformer import (
        TemporalTransformer,
        aggregate_horizon_season_metrics,
        aggregate_season_metrics,
        compute_overall_metrics,
        expand_predictions_by_target_timestamp,
        extract_horizon_number,
        extract_lag_number,
        inverse_scale_targets,
        make_dataloader,
        month_to_season_name,
        predict_batches,
        relative_skill_score,
        set_global_seed,
    )
except ImportError:
    from temporal_transformer.train_discharge_transformer import (
        TemporalTransformer,
        aggregate_horizon_season_metrics,
        aggregate_season_metrics,
        compute_overall_metrics,
        expand_predictions_by_target_timestamp,
        extract_horizon_number,
        extract_lag_number,
        inverse_scale_targets,
        make_dataloader,
        month_to_season_name,
        predict_batches,
        relative_skill_score,
        set_global_seed,
    )


TIME_CONTEXT_COLUMNS = {"year", "month", "day", "hour", "day_of_week", "day_of_year", "season"}
TEMPORAL_SEQUENCE_FEATURE_NAMES = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "day_of_year_frac",
    "season_winter",
    "season_spring",
    "season_summer",
    "season_fall",
]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_pattern = str(
        project_root / "air_temperature_parquet_v2" / "air_temperature_training_site_*_lb168_h24.parquet"
    )

    parser = argparse.ArgumentParser(
        description="Train per-site temporal transformer models on multihorizon air-temperature parquet data."
    )
    parser.add_argument(
        "--parquet-pattern",
        default=default_pattern,
        help="Glob pattern for air-temperature parquet inputs.",
    )
    parser.add_argument(
        "--output-root",
        default=str(project_root),
        help="Project root where air-temperature models and metrics will be written.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--variation-clip-pct", type=float, default=0.05)
    parser.add_argument("--variation-clip-history-multiplier", type=float, default=1.5)
    parser.add_argument("--variation-clip-absolute-floor", type=float, default=2.0)
    parser.add_argument("--disable-variation-clip", action="store_true")
    return parser.parse_args()


def load_parquet_frames(pattern: str) -> pd.DataFrame:
    paths = sorted(Path(p) for p in glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No parquet files matched pattern: {pattern}")

    frames = []
    for path in paths:
        print(f"Loading parquet: {path}")
        df = pd.read_parquet(path)
        df["source_path"] = str(path)
        df["source_mtime"] = float(path.stat().st_mtime)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    print(f"Loaded {len(combined):,} total rows from {len(paths)} parquet file(s)")
    return combined


def parse_timestamp_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in ["history_end_time", "forecast_start_time", "forecast_end_time"]:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce")
    for column in ["history_end_time", "forecast_start_time", "forecast_end_time"]:
        if column not in out.columns:
            raise ValueError(f"Expected timestamp column is missing: {column}")
        bad_rows = int(out[column].isna().sum())
        if bad_rows:
            raise ValueError(f"Found {bad_rows} rows with invalid {column} timestamps.")
    return out


def deduplicate_site_rows(df: pd.DataFrame) -> pd.DataFrame:
    duplicate_keys = ["site_id", "forecast_start_time"]
    duplicate_mask = df.duplicated(subset=duplicate_keys, keep=False)
    if not duplicate_mask.any():
        return df

    ranking_columns: list[str] = []
    version_candidates = ["source_version", "dataset_version", "data_version", "version"]
    for column in version_candidates:
        if column in df.columns and df[column].notna().any():
            ranking_columns.append(column)
            break

    if "source_mtime" in df.columns and df["source_mtime"].notna().any():
        ranking_columns.append("source_mtime")

    if not ranking_columns:
        duplicate_count = int(duplicate_mask.sum())
        print(
            f"Found {duplicate_count:,} duplicate rows by site/timestamp but no reliable "
            "version or file timestamp column was available, so duplicates were kept."
        )
        return df.reset_index(drop=True)

    sort_columns = duplicate_keys + ranking_columns + (["source_path"] if "source_path" in df.columns else [])
    ascending = [True, True] + [True] * len(ranking_columns) + ([True] if "source_path" in df.columns else [])
    deduped = (
        df.sort_values(sort_columns, ascending=ascending)
        .drop_duplicates(subset=duplicate_keys, keep="last")
        .reset_index(drop=True)
    )
    dropped = len(df) - len(deduped)
    if dropped:
        print(
            f"Dropped {dropped:,} duplicate site/timestamp rows, keeping the newest source based on "
            f"{', '.join(ranking_columns)}."
        )
    return deduped


def infer_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    lag_cols = sorted(
        [col for col in df.columns if col.startswith("air_temp_t-")],
        key=extract_lag_number,
        reverse=True,
    )
    target_cols = sorted(
        [col for col in df.columns if col.startswith("target_air_temp_t+")],
        key=extract_horizon_number,
    )

    ignored_columns = {
        "site_id",
        "sample_stride_hours",
        "history_end_time",
        "forecast_start_time",
        "forecast_end_time",
        "source_path",
        "source_mtime",
    }
    unexpected_features = [
        col
        for col in df.columns
        if col not in ignored_columns
        and col not in lag_cols
        and col not in target_cols
        and col not in TIME_CONTEXT_COLUMNS
    ]
    if unexpected_features:
        print(
            "Ignoring non-air-temperature columns in air-temperature-only model: "
            f"{', '.join(unexpected_features)}"
        )

    return lag_cols, target_cols


def build_temporal_sequence_features(df: pd.DataFrame, lag_cols: list[str]) -> np.ndarray:
    if "history_end_time" not in df.columns:
        raise ValueError("Expected history_end_time column for temporal sequence feature construction.")

    history_end_times = pd.to_datetime(df["history_end_time"], errors="coerce")
    if history_end_times.isna().any():
        bad_count = int(history_end_times.isna().sum())
        raise ValueError(f"Found {bad_count} invalid history_end_time values.")

    lag_steps = [extract_lag_number(col) for col in lag_cols]
    num_rows = len(df)
    num_steps = len(lag_cols)
    temporal = np.zeros((num_rows, num_steps, len(TEMPORAL_SEQUENCE_FEATURE_NAMES)), dtype=np.float32)

    for step_idx, lag_step in enumerate(lag_steps):
        timestamps = history_end_times - pd.to_timedelta(lag_step - 1, unit="h")
        season_names = timestamps.dt.month.map(month_to_season_name)
        hours = timestamps.dt.hour.to_numpy(dtype=np.float32)
        day_of_week = timestamps.dt.dayofweek.to_numpy(dtype=np.float32)
        months = timestamps.dt.month.to_numpy(dtype=np.float32)
        day_of_year = timestamps.dt.dayofyear.to_numpy(dtype=np.float32)

        temporal[:, step_idx, 0] = np.sin(2.0 * np.pi * hours / 24.0)
        temporal[:, step_idx, 1] = np.cos(2.0 * np.pi * hours / 24.0)
        temporal[:, step_idx, 2] = np.sin(2.0 * np.pi * day_of_week / 7.0)
        temporal[:, step_idx, 3] = np.cos(2.0 * np.pi * day_of_week / 7.0)
        temporal[:, step_idx, 4] = np.sin(2.0 * np.pi * (months - 1.0) / 12.0)
        temporal[:, step_idx, 5] = np.cos(2.0 * np.pi * (months - 1.0) / 12.0)
        temporal[:, step_idx, 6] = day_of_year / 366.0
        temporal[:, step_idx, 7] = (season_names == "winter").to_numpy(dtype=np.float32)
        temporal[:, step_idx, 8] = (season_names == "spring").to_numpy(dtype=np.float32)
        temporal[:, step_idx, 9] = (season_names == "summer").to_numpy(dtype=np.float32)
        temporal[:, step_idx, 10] = (season_names == "fall").to_numpy(dtype=np.float32)

    return np.nan_to_num(temporal, nan=0.0, posinf=0.0, neginf=0.0)


def trailing_one_year_split(site_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    latest_target_timestamp = pd.Timestamp(site_df["forecast_end_time"].max())
    validation_start = latest_target_timestamp - pd.DateOffset(years=1) + pd.Timedelta(hours=1)

    train_df = site_df.loc[site_df["forecast_end_time"] < validation_start].copy()
    valid_df = site_df.loc[site_df["forecast_start_time"] >= validation_start].copy()
    if train_df.empty or valid_df.empty:
        raise ValueError(
            "Time-based split produced an empty train or validation frame. "
            f"validation_start={validation_start}"
        )

    return train_df, valid_df, validation_start, latest_target_timestamp


def latest_air_temp_column(lag_cols: list[str]) -> str:
    if "air_temp_t-1" not in lag_cols:
        raise ValueError("Expected lag feature air_temp_t-1 to build residual targets.")
    return "air_temp_t-1"


def latest_air_temp_values(df: pd.DataFrame, lag_cols: list[str]) -> np.ndarray:
    latest_col = latest_air_temp_column(lag_cols)
    values = df[latest_col].to_numpy(dtype=np.float32)
    return values.reshape(-1, 1)


def make_delta_targets(df: pd.DataFrame, lag_cols: list[str], target_cols: list[str]) -> np.ndarray:
    target_values = df[target_cols].to_numpy(dtype=np.float32)
    return target_values - latest_air_temp_values(df, lag_cols)


def reconstruct_absolute_targets(delta_values: np.ndarray, df: pd.DataFrame, lag_cols: list[str]) -> np.ndarray:
    return delta_values + latest_air_temp_values(df, lag_cols)


def make_persistence_predictions(df: pd.DataFrame, lag_cols: list[str], target_cols: list[str]) -> np.ndarray:
    latest_values = latest_air_temp_values(df, lag_cols)
    return np.repeat(latest_values, repeats=len(target_cols), axis=1)


def historical_max_hourly_change(df: pd.DataFrame, lag_cols: list[str]) -> np.ndarray:
    lag_values = df[lag_cols].to_numpy(dtype=np.float32)
    hourly_changes = np.abs(np.diff(lag_values, axis=1))
    return np.nan_to_num(hourly_changes.max(axis=1), nan=0.0, posinf=0.0, neginf=0.0)


def clip_prediction_variation(
    predictions: np.ndarray,
    df: pd.DataFrame,
    lag_cols: list[str],
    pct_limit: float,
    history_multiplier: float,
    absolute_floor: float,
) -> np.ndarray:
    clipped = predictions.copy()
    latest_values = latest_air_temp_values(df, lag_cols).reshape(-1)
    history_limit = history_multiplier * historical_max_hourly_change(df, lag_cols)

    previous = latest_values
    for step_idx in range(clipped.shape[1]):
        pct_limit_values = pct_limit * np.abs(previous)
        allowed = np.maximum.reduce(
            [
                np.full_like(previous, absolute_floor, dtype=np.float32),
                pct_limit_values.astype(np.float32),
                history_limit.astype(np.float32),
            ]
        )
        lower = previous - allowed
        upper = previous + allowed
        clipped[:, step_idx] = np.clip(clipped[:, step_idx], lower, upper)
        previous = clipped[:, step_idx]

    return clipped


def fit_scalers(
    train_df: pd.DataFrame,
    lag_cols: list[str],
    target_cols: list[str],
) -> tuple[StandardScaler, StandardScaler]:
    lag_scaler = StandardScaler()
    lag_scaler.fit(train_df[lag_cols].to_numpy(dtype=np.float32).reshape(-1, 1))

    target_deltas = make_delta_targets(train_df, lag_cols, target_cols)
    target_scaler = StandardScaler()
    target_scaler.fit(target_deltas.reshape(-1, 1))

    return lag_scaler, target_scaler


def transform_frame(
    df: pd.DataFrame,
    lag_cols: list[str],
    target_cols: list[str],
    lag_scaler: StandardScaler,
    target_scaler: StandardScaler,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lag_values = df[lag_cols].to_numpy(dtype=np.float32)
    lag_values = lag_scaler.transform(lag_values.reshape(-1, 1)).reshape(lag_values.shape)
    lag_values = np.nan_to_num(lag_values, nan=0.0, posinf=0.0, neginf=0.0)

    temporal_values = build_temporal_sequence_features(df, lag_cols)

    target_values = make_delta_targets(df, lag_cols, target_cols)
    target_values = target_scaler.transform(target_values.reshape(-1, 1)).reshape(target_values.shape)
    target_values = np.nan_to_num(target_values, nan=0.0, posinf=0.0, neginf=0.0)

    return lag_values, temporal_values, target_values


class AirTemperatureSequenceDataset(Dataset):
    def __init__(
        self,
        lag_values: np.ndarray,
        temporal_values: np.ndarray,
        targets: np.ndarray,
    ):
        self.lag_values = lag_values.astype(np.float32)
        self.temporal_values = temporal_values.astype(np.float32)
        self.targets = targets.astype(np.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        lag_sequence = self.lag_values[index][:, None]
        temporal_sequence = self.temporal_values[index]
        if lag_sequence.shape[0] != temporal_sequence.shape[0]:
            raise ValueError("Lag sequence and temporal sequence must have the same number of timesteps.")
        sequence = np.concatenate([lag_sequence, temporal_sequence], axis=1)
        return (
            torch.tensor(sequence, dtype=torch.float32),
            torch.tensor(self.targets[index], dtype=torch.float32),
        )


def train_one_site(
    site_df: pd.DataFrame,
    args: argparse.Namespace,
    output_root: Path,
) -> tuple[pd.DataFrame, dict]:
    site_id = int(site_df["site_id"].iloc[0])
    print(f"\n{'=' * 80}")
    print(f"Training air-temperature transformer for site_id={site_id}")
    print(f"{'=' * 80}")

    lag_cols, target_cols = infer_columns(site_df)
    if not lag_cols or not target_cols:
        raise ValueError(f"Site {site_id} is missing air-temperature lag or target columns.")

    train_df, valid_df, validation_start, validation_end = trailing_one_year_split(site_df)
    print(
        f"Site {site_id}: train rows={len(train_df):,}, validation rows={len(valid_df):,}, "
        f"validation window={validation_start} to {validation_end}"
    )

    lag_scaler, target_scaler = fit_scalers(train_df, lag_cols, target_cols)

    train_lags, train_temporal, train_targets = transform_frame(
        train_df,
        lag_cols,
        target_cols,
        lag_scaler,
        target_scaler,
    )
    valid_lags, valid_temporal, valid_targets = transform_frame(
        valid_df,
        lag_cols,
        target_cols,
        lag_scaler,
        target_scaler,
    )

    train_dataset = AirTemperatureSequenceDataset(train_lags, train_temporal, train_targets)
    valid_dataset = AirTemperatureSequenceDataset(valid_lags, valid_temporal, valid_targets)
    train_loader = make_dataloader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = make_dataloader(valid_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    input_dim = train_dataset[0][0].shape[-1]
    horizon = len(target_cols)
    seq_len = len(lag_cols)
    site_model_dir = output_root / "models_air_temperature" / str(site_id)
    site_model_dir.mkdir(parents=True, exist_ok=True)

    model = TemporalTransformer(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        horizon=horizon,
        learning_rate=args.learning_rate,
        seq_len=seq_len,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=site_model_dir,
        filename="air-temperature-lightning-best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=args.patience,
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        deterministic=True,
        enable_progress_bar=True,
        logger=CSVLogger(save_dir=str(output_root), name="lightning_logs_air_temperature"),
        log_every_n_steps=10,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    best_ckpt_path = checkpoint_callback.best_model_path
    if best_ckpt_path:
        print(f"Site {site_id}: reloading best checkpoint from {best_ckpt_path}")
        best_model = TemporalTransformer.load_from_checkpoint(best_ckpt_path)
    else:
        print(f"Site {site_id}: no checkpoint found, falling back to in-memory model")
        best_model = model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model = best_model.to(device)

    scaled_predictions = predict_batches(best_model, valid_loader, device=device)
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

    best_model_path = site_model_dir / "air_temperature_best_model.pt"
    metadata_path = site_model_dir / "metadata.json"
    residual_base_col = latest_air_temp_column(lag_cols)
    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "site_id": site_id,
            "target_variable": "air_temperature",
            "target_units": "degree Celsius",
            "lag_cols": lag_cols,
            "target_cols": target_cols,
            "target_transform": "delta_from_latest_air_temp",
            "residual_base_col": residual_base_col,
            "variation_clip_enabled": not args.disable_variation_clip,
            "variation_clip_pct": args.variation_clip_pct,
            "variation_clip_history_multiplier": args.variation_clip_history_multiplier,
            "variation_clip_absolute_floor": args.variation_clip_absolute_floor,
            "lag_scaler_mean": lag_scaler.mean_.tolist(),
            "lag_scaler_scale": lag_scaler.scale_.tolist(),
            "temporal_sequence_feature_names": TEMPORAL_SEQUENCE_FEATURE_NAMES,
            "target_scaler_applies_to": "target_air_temp_t+h_minus_air_temp_t-1",
            "target_scaler_mean": target_scaler.mean_.tolist(),
            "target_scaler_scale": target_scaler.scale_.tolist(),
            "model_hparams": best_model.hparams,
            "validation_start": str(validation_start),
            "validation_end": str(validation_end),
        },
        best_model_path,
    )
    print(f"Site {site_id}: saved best model bundle to {best_model_path}")

    metadata = {
        "site_id": site_id,
        "target_variable": "air_temperature",
        "target_units": "degree Celsius",
        "best_model_path": str(best_model_path),
        "best_checkpoint": best_ckpt_path,
        "validation_start": str(validation_start),
        "validation_end": str(validation_end),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "lag_cols": lag_cols,
        "target_cols": target_cols,
        "target_transform": "delta_from_latest_air_temp",
        "residual_base_col": residual_base_col,
        "variation_clip_enabled": not args.disable_variation_clip,
        "variation_clip_pct": args.variation_clip_pct,
        "variation_clip_history_multiplier": args.variation_clip_history_multiplier,
        "variation_clip_absolute_floor": args.variation_clip_absolute_floor,
        "temporal_sequence_feature_names": TEMPORAL_SEQUENCE_FEATURE_NAMES,
        "model_hparams": dict(best_model.hparams),
        "source_files": sorted(site_df["source_path"].dropna().unique().tolist())
        if "source_path" in site_df.columns
        else [],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Site {site_id}: saved metadata to {metadata_path}")

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
        "target_variable": "air_temperature",
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
        "best_checkpoint": best_ckpt_path,
        "best_model_path": str(best_model_path),
    }
    (site_model_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2) + "\n")

    return expanded_predictions, run_summary


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print("Starting air-temperature transformer training pipeline")
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
        expanded_predictions, run_summary = train_one_site(site_df, args, output_root)
        all_expanded_predictions.append(expanded_predictions)
        overall_rows.append(run_summary)

    combined_predictions = pd.concat(all_expanded_predictions, ignore_index=True)
    metrics_by_season = aggregate_season_metrics(combined_predictions)
    metrics_path = output_root / "air_temperature_metrics_by_season.csv"
    metrics_by_season.to_csv(metrics_path, index=False)
    print(f"Saved seasonal metrics to {metrics_path}")

    horizon_metrics = aggregate_horizon_season_metrics(combined_predictions)
    horizon_metrics_path = output_root / "air_temperature_metrics_by_season_horizon.csv"
    horizon_metrics.to_csv(horizon_metrics_path, index=False)
    print(f"Saved horizon seasonal metrics to {horizon_metrics_path}")

    overall_metrics_path = output_root / "air_temperature_overall_validation_metrics.csv"
    pd.DataFrame(overall_rows).to_csv(overall_metrics_path, index=False)
    print(f"Saved overall validation metrics to {overall_metrics_path}")

    expanded_predictions_path = output_root / "air_temperature_validation_predictions_long.csv"
    combined_predictions.to_csv(expanded_predictions_path, index=False)
    print(f"Saved validation predictions to {expanded_predictions_path}")


if __name__ == "__main__":
    main()
