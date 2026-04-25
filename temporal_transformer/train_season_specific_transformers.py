from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import Dataset

try:
    from train_discharge_transformer import (
        TemporalTransformer,
        compute_overall_metrics,
        extract_horizon_number,
        extract_lag_number,
        horizon_bin_label,
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
        compute_overall_metrics,
        extract_horizon_number,
        extract_lag_number,
        horizon_bin_label,
        inverse_scale_targets,
        make_dataloader,
        month_to_season_name,
        predict_batches,
        relative_skill_score,
        set_global_seed,
    )


SEASON_NAMES = {
    0: "winter",
    1: "spring",
    2: "summer",
    3: "fall",
}
SEASON_ORDER = ["winter", "spring", "summer", "fall"]
TIME_CONTEXT_COLUMNS = {"year", "month", "day", "hour", "day_of_week", "day_of_year", "season", "season_name"}
TEMPORAL_SEQUENCE_FEATURE_NAMES = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "day_of_year_frac",
]


@dataclass(frozen=True)
class VariableConfig:
    name: str
    units: str
    lag_prefix: str
    target_prefix: str
    latest_lag_column: str
    model_folder: str
    metrics_prefix: str
    target_transform: str
    target_scaler_applies_to: str
    include_static_features: bool


VARIABLE_CONFIGS = {
    "discharge": VariableConfig(
        name="discharge",
        units="cfs",
        lag_prefix="discharge_t-",
        target_prefix="target_discharge_t+",
        latest_lag_column="discharge_t-1",
        model_folder="models_discharge_season_specific",
        metrics_prefix="discharge",
        target_transform="delta_from_latest_discharge",
        target_scaler_applies_to="target_discharge_t+h_minus_discharge_t-1",
        include_static_features=True,
    ),
    "air_temperature": VariableConfig(
        name="air_temperature",
        units="degree Celsius",
        lag_prefix="air_temp_t-",
        target_prefix="target_air_temp_t+",
        latest_lag_column="air_temp_t-1",
        model_folder="models_air_temperature_season_specific",
        metrics_prefix="air_temperature",
        target_transform="delta_from_latest_air_temp",
        target_scaler_applies_to="target_air_temp_t+h_minus_air_temp_t-1",
        include_static_features=False,
    ),
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train one temporal transformer per variable/site/season."
    )
    parser.add_argument(
        "--output-root",
        default=str(project_root / "season_specific_training"),
        help="Folder where all season-specific models, logs, predictions, and metrics will be written.",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        choices=["discharge", "air_temperature"],
        default=["discharge", "air_temperature"],
        help="Variables to train. Default trains both.",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        choices=SEASON_ORDER,
        default=SEASON_ORDER,
        help="Seasons to train. Default trains all four seasons.",
    )
    parser.add_argument(
        "--discharge-parquet-pattern",
        default=str(project_root / "streamflow_parquet_v2" / "*precip24avg*.parquet"),
    )
    parser.add_argument(
        "--air-temperature-parquet-pattern",
        default=str(project_root / "air_temperature_parquet_v2" / "air_temperature_training_site_*_lb168_h24.parquet"),
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


def parquet_pattern_for_variable(args: argparse.Namespace, variable: str) -> str:
    if variable == "discharge":
        return args.discharge_parquet_pattern
    if variable == "air_temperature":
        return args.air_temperature_parquet_pattern
    raise ValueError(f"Unsupported variable: {variable}")


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


def add_season_name(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "season" in out.columns:
        out["season_name"] = out["season"].map(SEASON_NAMES)
    else:
        out["season_name"] = out["forecast_start_time"].dt.month.map(month_to_season_name)

    if out["season_name"].isna().any():
        out["season_name"] = out["forecast_start_time"].dt.month.map(month_to_season_name)
    return out


def infer_columns(df: pd.DataFrame, config: VariableConfig) -> tuple[list[str], list[str], list[str]]:
    lag_cols = sorted(
        [col for col in df.columns if col.startswith(config.lag_prefix)],
        key=extract_lag_number,
        reverse=True,
    )
    target_cols = sorted(
        [col for col in df.columns if col.startswith(config.target_prefix)],
        key=extract_horizon_number,
    )

    ignored_columns = {
        "site_id",
        "stride",
        "sample_stride_hours",
        "history_end_time",
        "forecast_start_time",
        "forecast_end_time",
        "source_path",
        "source_mtime",
    }
    if not config.include_static_features:
        return lag_cols, target_cols, []

    static_cols = [
        col
        for col in df.columns
        if col not in ignored_columns
        and col not in lag_cols
        and col not in target_cols
        and col not in TIME_CONTEXT_COLUMNS
    ]
    return lag_cols, target_cols, static_cols


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


def latest_values(df: pd.DataFrame, config: VariableConfig) -> np.ndarray:
    if config.latest_lag_column not in df.columns:
        raise ValueError(f"Expected lag feature {config.latest_lag_column} to build residual targets.")
    values = df[config.latest_lag_column].to_numpy(dtype=np.float32)
    return values.reshape(-1, 1)


def make_delta_targets(df: pd.DataFrame, config: VariableConfig, target_cols: list[str]) -> np.ndarray:
    target_values = df[target_cols].to_numpy(dtype=np.float32)
    return target_values - latest_values(df, config)


def reconstruct_absolute_targets(delta_values: np.ndarray, df: pd.DataFrame, config: VariableConfig) -> np.ndarray:
    return delta_values + latest_values(df, config)


def make_persistence_predictions(df: pd.DataFrame, config: VariableConfig, target_cols: list[str]) -> np.ndarray:
    latest = latest_values(df, config)
    return np.repeat(latest, repeats=len(target_cols), axis=1)


def historical_max_hourly_change(df: pd.DataFrame, lag_cols: list[str]) -> np.ndarray:
    lag_values = df[lag_cols].to_numpy(dtype=np.float32)
    hourly_changes = np.abs(np.diff(lag_values, axis=1))
    return np.nan_to_num(hourly_changes.max(axis=1), nan=0.0, posinf=0.0, neginf=0.0)


def clip_prediction_variation(
    predictions: np.ndarray,
    df: pd.DataFrame,
    config: VariableConfig,
    lag_cols: list[str],
    pct_limit: float,
    history_multiplier: float,
    absolute_floor: float,
) -> np.ndarray:
    clipped = predictions.copy()
    latest = latest_values(df, config).reshape(-1)
    history_limit = history_multiplier * historical_max_hourly_change(df, lag_cols)

    previous = latest
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


def build_static_features(df: pd.DataFrame, static_cols: list[str]) -> pd.DataFrame:
    if not static_cols:
        return pd.DataFrame(index=df.index)
    return df[static_cols].copy()


def fit_scalers(
    train_df: pd.DataFrame,
    config: VariableConfig,
    lag_cols: list[str],
    static_cols: list[str],
    target_cols: list[str],
) -> tuple[StandardScaler, StandardScaler | None, StandardScaler, list[str]]:
    lag_scaler = StandardScaler()
    lag_scaler.fit(train_df[lag_cols].to_numpy(dtype=np.float32).reshape(-1, 1))

    static_scaler = None
    static_feature_cols: list[str] = []
    if static_cols:
        static_train = build_static_features(train_df, static_cols)
        static_train = static_train.apply(pd.to_numeric, errors="coerce")
        static_train = static_train.fillna(static_train.median(numeric_only=True))
        static_train = static_train.fillna(0.0)
        static_feature_cols = static_train.columns.tolist()
        static_scaler = StandardScaler()
        static_scaler.fit(static_train.to_numpy(dtype=np.float32))

    target_deltas = make_delta_targets(train_df, config, target_cols)
    target_scaler = StandardScaler()
    target_scaler.fit(target_deltas.reshape(-1, 1))
    return lag_scaler, static_scaler, target_scaler, static_feature_cols


def transform_frame(
    df: pd.DataFrame,
    config: VariableConfig,
    lag_cols: list[str],
    target_cols: list[str],
    static_cols: list[str],
    static_feature_cols: list[str],
    lag_scaler: StandardScaler,
    static_scaler: StandardScaler | None,
    target_scaler: StandardScaler,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lag_values = df[lag_cols].to_numpy(dtype=np.float32)
    lag_values = lag_scaler.transform(lag_values.reshape(-1, 1)).reshape(lag_values.shape)
    lag_values = np.nan_to_num(lag_values, nan=0.0, posinf=0.0, neginf=0.0)

    temporal_values = build_temporal_sequence_features(df, lag_cols)

    if static_feature_cols and static_scaler is not None:
        static_df = build_static_features(df, static_cols)
        static_df = static_df.apply(pd.to_numeric, errors="coerce")
        for column in static_feature_cols:
            if column not in static_df.columns:
                static_df[column] = 0.0
        static_df = static_df[static_feature_cols]
        static_df = static_df.fillna(0.0)
        static_values = static_scaler.transform(static_df.to_numpy(dtype=np.float32))
        static_values = np.nan_to_num(static_values, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        static_values = np.empty((len(df), 0), dtype=np.float32)

    target_values = make_delta_targets(df, config, target_cols)
    target_values = target_scaler.transform(target_values.reshape(-1, 1)).reshape(target_values.shape)
    target_values = np.nan_to_num(target_values, nan=0.0, posinf=0.0, neginf=0.0)
    return lag_values, temporal_values, static_values, target_values


class SeasonSequenceDataset(Dataset):
    def __init__(
        self,
        lag_values: np.ndarray,
        temporal_values: np.ndarray,
        static_values: np.ndarray,
        targets: np.ndarray,
    ):
        self.lag_values = lag_values.astype(np.float32)
        self.temporal_values = temporal_values.astype(np.float32)
        self.static_values = static_values.astype(np.float32)
        self.targets = targets.astype(np.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        lag_sequence = self.lag_values[index][:, None]
        temporal_sequence = self.temporal_values[index]
        if lag_sequence.shape[0] != temporal_sequence.shape[0]:
            raise ValueError("Lag sequence and temporal sequence must have the same number of timesteps.")
        if self.static_values.shape[1] > 0:
            repeated_static = np.repeat(self.static_values[index][None, :], lag_sequence.shape[0], axis=0)
            sequence = np.concatenate([lag_sequence, temporal_sequence, repeated_static], axis=1)
        else:
            sequence = np.concatenate([lag_sequence, temporal_sequence], axis=1)
        return (
            torch.tensor(sequence, dtype=torch.float32),
            torch.tensor(self.targets[index], dtype=torch.float32),
        )


def expand_predictions(
    config: VariableConfig,
    site_id: int,
    season: str,
    valid_df: pd.DataFrame,
    target_cols: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_unclipped: np.ndarray,
    y_persistence: np.ndarray,
) -> pd.DataFrame:
    records: list[dict] = []
    horizon_steps = [extract_horizon_number(col) for col in target_cols]

    for row_idx, (_, row) in enumerate(valid_df.reset_index(drop=True).iterrows()):
        forecast_start = pd.Timestamp(row["forecast_start_time"])
        for step_idx, horizon_step in enumerate(horizon_steps):
            target_timestamp = forecast_start + pd.Timedelta(hours=horizon_step - 1)
            records.append(
                {
                    "target_variable": config.name,
                    "site_id": int(site_id),
                    "season": season,
                    "forecast_start_time": forecast_start,
                    "horizon": int(horizon_step),
                    "horizon_bin": horizon_bin_label(horizon_step),
                    "target_timestamp": target_timestamp,
                    "y_true": float(y_true[row_idx, step_idx]),
                    "y_pred": float(y_pred[row_idx, step_idx]),
                    "y_pred_unclipped": float(y_pred_unclipped[row_idx, step_idx]),
                    "y_persistence": float(y_persistence[row_idx, step_idx]),
                }
            )
    return pd.DataFrame.from_records(records)


def aggregate_metrics(expanded_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in expanded_df.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rmse = float(np.sqrt(mean_squared_error(group["y_true"], group["y_pred"])))
        mae = float(mean_absolute_error(group["y_true"], group["y_pred"]))
        persistence_rmse = float(np.sqrt(mean_squared_error(group["y_true"], group["y_persistence"])))
        persistence_mae = float(mean_absolute_error(group["y_true"], group["y_persistence"]))
        row = {column: value for column, value in zip(group_cols, keys)}
        row.update(
            {
                "n": int(len(group)),
                "rmse": rmse,
                "mae": mae,
                "rmse_skill_vs_persistence": relative_skill_score(rmse, persistence_rmse),
                "mae_skill_vs_persistence": relative_skill_score(mae, persistence_mae),
            }
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if "horizon_bin" in out.columns:
        horizon_order = {"0-6": 0, "6-12": 1, "12-18": 2, "18-24": 3}
        out["horizon_order"] = out["horizon_bin"].map(horizon_order)
    if "season" in out.columns:
        season_order = {season: idx for idx, season in enumerate(SEASON_ORDER)}
        out["season_order"] = out["season"].map(season_order)

    sort_cols = [col for col in ["target_variable", "site_id", "season_order", "horizon_order"] if col in out.columns]
    drop_cols = [col for col in ["season_order", "horizon_order"] if col in out.columns]
    return out.sort_values(sort_cols).drop(columns=drop_cols).reset_index(drop=True)


def train_one_site_season(
    config: VariableConfig,
    site_id: int,
    season: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    lag_cols: list[str],
    target_cols: list[str],
    static_cols: list[str],
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    args: argparse.Namespace,
    output_root: Path,
) -> tuple[pd.DataFrame, dict]:
    print(f"\n{'=' * 80}")
    print(f"Training {config.name} model for site_id={site_id}, season={season}")
    print(f"{'=' * 80}")
    print(
        f"{config.name} site {site_id} {season}: train rows={len(train_df):,}, "
        f"validation rows={len(valid_df):,}, validation window={validation_start} to {validation_end}"
    )

    lag_scaler, static_scaler, target_scaler, static_feature_cols = fit_scalers(
        train_df,
        config,
        lag_cols,
        static_cols,
        target_cols,
    )
    train_lags, train_temporal, train_static, train_targets = transform_frame(
        train_df,
        config,
        lag_cols,
        target_cols,
        static_cols,
        static_feature_cols,
        lag_scaler,
        static_scaler,
        target_scaler,
    )
    valid_lags, valid_temporal, valid_static, valid_targets = transform_frame(
        valid_df,
        config,
        lag_cols,
        target_cols,
        static_cols,
        static_feature_cols,
        lag_scaler,
        static_scaler,
        target_scaler,
    )

    train_dataset = SeasonSequenceDataset(train_lags, train_temporal, train_static, train_targets)
    valid_dataset = SeasonSequenceDataset(valid_lags, valid_temporal, valid_static, valid_targets)
    train_loader = make_dataloader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = make_dataloader(valid_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    input_dim = train_dataset[0][0].shape[-1]
    horizon = len(target_cols)
    seq_len = len(lag_cols)
    site_model_dir = output_root / config.model_folder / f"site_{site_id}" / season
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
        filename=f"{config.name}-site_{site_id}-{season}-best",
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
        logger=CSVLogger(
            save_dir=str(output_root),
            name=f"lightning_logs_{config.name}_season_specific",
            version=f"site_{site_id}_{season}",
        ),
        log_every_n_steps=10,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    best_ckpt_path = checkpoint_callback.best_model_path
    if best_ckpt_path:
        print(f"{config.name} site {site_id} {season}: reloading best checkpoint from {best_ckpt_path}")
        best_model = TemporalTransformer.load_from_checkpoint(best_ckpt_path)
    else:
        print(f"{config.name} site {site_id} {season}: no checkpoint found, falling back to in-memory model")
        best_model = model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model = best_model.to(device)
    scaled_predictions = predict_batches(best_model, valid_loader, device=device)
    predicted_deltas = inverse_scale_targets(scaled_predictions, target_scaler)
    y_pred_unclipped = reconstruct_absolute_targets(predicted_deltas, valid_df, config)
    if args.disable_variation_clip:
        y_pred = y_pred_unclipped
    else:
        y_pred = clip_prediction_variation(
            predictions=y_pred_unclipped,
            df=valid_df,
            config=config,
            lag_cols=lag_cols,
            pct_limit=args.variation_clip_pct,
            history_multiplier=args.variation_clip_history_multiplier,
            absolute_floor=args.variation_clip_absolute_floor,
        )

    y_true = valid_df[target_cols].to_numpy(dtype=np.float32)
    y_persistence = make_persistence_predictions(valid_df, config, target_cols)
    rmse, mae = compute_overall_metrics(y_true, y_pred)
    persistence_rmse, persistence_mae = compute_overall_metrics(y_true, y_persistence)
    rmse_skill = relative_skill_score(rmse, persistence_rmse)
    mae_skill = relative_skill_score(mae, persistence_mae)
    print(f"{config.name} site {site_id} {season}: validation RMSE={rmse:.4f}, MAE={mae:.4f}")

    model_path = site_model_dir / f"{config.name}_site_{site_id}_{season}_best_model.pt"
    metadata_path = site_model_dir / "metadata.json"
    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "target_variable": config.name,
            "target_units": config.units,
            "site_id": site_id,
            "season": season,
            "lag_cols": lag_cols,
            "target_cols": target_cols,
            "target_transform": config.target_transform,
            "residual_base_col": config.latest_lag_column,
            "variation_clip_enabled": not args.disable_variation_clip,
            "variation_clip_pct": args.variation_clip_pct,
            "variation_clip_history_multiplier": args.variation_clip_history_multiplier,
            "variation_clip_absolute_floor": args.variation_clip_absolute_floor,
            "static_feature_cols": static_feature_cols,
            "lag_scaler_mean": lag_scaler.mean_.tolist(),
            "lag_scaler_scale": lag_scaler.scale_.tolist(),
            "temporal_sequence_feature_names": TEMPORAL_SEQUENCE_FEATURE_NAMES,
            "static_scaler_mean": static_scaler.mean_.tolist() if static_scaler is not None else [],
            "static_scaler_scale": static_scaler.scale_.tolist() if static_scaler is not None else [],
            "target_scaler_applies_to": config.target_scaler_applies_to,
            "target_scaler_mean": target_scaler.mean_.tolist(),
            "target_scaler_scale": target_scaler.scale_.tolist(),
            "model_hparams": best_model.hparams,
            "validation_start": str(validation_start),
            "validation_end": str(validation_end),
        },
        model_path,
    )
    metadata = {
        "target_variable": config.name,
        "target_units": config.units,
        "site_id": site_id,
        "season": season,
        "best_model_path": str(model_path),
        "best_checkpoint": best_ckpt_path,
        "validation_start": str(validation_start),
        "validation_end": str(validation_end),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "lag_cols": lag_cols,
        "target_cols": target_cols,
        "target_transform": config.target_transform,
        "residual_base_col": config.latest_lag_column,
        "variation_clip_enabled": not args.disable_variation_clip,
        "variation_clip_pct": args.variation_clip_pct,
        "variation_clip_history_multiplier": args.variation_clip_history_multiplier,
        "variation_clip_absolute_floor": args.variation_clip_absolute_floor,
        "static_feature_cols": static_feature_cols,
        "temporal_sequence_feature_names": TEMPORAL_SEQUENCE_FEATURE_NAMES,
        "model_hparams": dict(best_model.hparams),
        "source_files": sorted(valid_df["source_path"].dropna().unique().tolist())
        if "source_path" in valid_df.columns
        else [],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{config.name} site {site_id} {season}: saved model to {model_path}")

    expanded_predictions = expand_predictions(
        config=config,
        site_id=site_id,
        season=season,
        valid_df=valid_df,
        target_cols=target_cols,
        y_true=y_true,
        y_pred=y_pred,
        y_pred_unclipped=y_pred_unclipped,
        y_persistence=y_persistence,
    )
    run_summary = {
        "target_variable": config.name,
        "site_id": site_id,
        "season": season,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "validation_start": str(validation_start),
        "validation_end": str(validation_end),
        "rmse": rmse,
        "mae": mae,
        "rmse_skill_vs_persistence": rmse_skill,
        "mae_skill_vs_persistence": mae_skill,
        "best_checkpoint": best_ckpt_path,
        "best_model_path": str(model_path),
    }
    (site_model_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2) + "\n")
    return expanded_predictions, run_summary


def train_variable(
    config: VariableConfig,
    pattern: str,
    seasons: list[str],
    args: argparse.Namespace,
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"\n{'#' * 80}")
    print(f"Starting season-specific training for {config.name}")
    print(f"Parquet pattern: {pattern}")
    print(f"{'#' * 80}")

    df = load_parquet_frames(pattern)
    df = parse_timestamp_columns(df)
    df = deduplicate_site_rows(df)
    df = add_season_name(df)
    if "site_id" not in df.columns:
        raise ValueError("Expected parquet data to contain a site_id column.")

    all_predictions = []
    run_summaries = []
    for site_id, site_df in df.groupby("site_id", sort=True):
        site_df = site_df.sort_values("forecast_start_time").reset_index(drop=True)
        lag_cols, target_cols, static_cols = infer_columns(site_df, config)
        if not lag_cols or not target_cols:
            raise ValueError(f"{config.name} site {site_id} is missing lag or target columns.")

        global_train_df, global_valid_df, validation_start, validation_end = trailing_one_year_split(site_df)
        for season in seasons:
            train_df = global_train_df.loc[global_train_df["season_name"] == season].copy()
            valid_df = global_valid_df.loc[global_valid_df["season_name"] == season].copy()
            if train_df.empty or valid_df.empty:
                print(
                    f"Skipping {config.name} site {site_id} {season}: "
                    f"train rows={len(train_df):,}, validation rows={len(valid_df):,}"
                )
                continue
            predictions, summary = train_one_site_season(
                config=config,
                site_id=int(site_id),
                season=season,
                train_df=train_df,
                valid_df=valid_df,
                lag_cols=lag_cols,
                target_cols=target_cols,
                static_cols=static_cols,
                validation_start=validation_start,
                validation_end=validation_end,
                args=args,
                output_root=output_root,
            )
            all_predictions.append(predictions)
            run_summaries.append(summary)

    if not all_predictions:
        raise ValueError(f"No {config.name} season-specific models were trained.")

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    summaries_df = pd.DataFrame(run_summaries)
    return predictions_df, summaries_df


def write_variable_outputs(
    config: VariableConfig,
    predictions_df: pd.DataFrame,
    summaries_df: pd.DataFrame,
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizon_metrics = aggregate_metrics(predictions_df, ["site_id", "season", "horizon_bin"])
    season_metrics = aggregate_metrics(predictions_df, ["site_id", "season"])

    metrics_dir = output_root / "metrics"
    predictions_dir = output_root / "predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    horizon_path = metrics_dir / f"{config.metrics_prefix}_season_specific_metrics_by_season_horizon.csv"
    season_path = metrics_dir / f"{config.metrics_prefix}_season_specific_metrics_by_season.csv"
    overall_path = metrics_dir / f"{config.metrics_prefix}_season_specific_overall_metrics.csv"
    predictions_path = predictions_dir / f"{config.metrics_prefix}_season_specific_validation_predictions_long.csv"

    horizon_metrics.to_csv(horizon_path, index=False)
    season_metrics.to_csv(season_path, index=False)
    summaries_df.to_csv(overall_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    print(f"Saved {config.name} horizon metrics to {horizon_path}")
    print(f"Saved {config.name} seasonal metrics to {season_path}")
    print(f"Saved {config.name} overall metrics to {overall_path}")
    print(f"Saved {config.name} validation predictions to {predictions_path}")
    return horizon_metrics, season_metrics


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    print("Starting season-specific transformer training pipeline")
    print(f"Output root: {output_root}")
    print(f"Variables: {', '.join(args.variables)}")
    print(f"Seasons: {', '.join(args.seasons)}")

    all_predictions = []
    all_summaries = []
    all_horizon_metrics = []
    all_season_metrics = []

    for variable in args.variables:
        config = VARIABLE_CONFIGS[variable]
        pattern = parquet_pattern_for_variable(args, variable)
        predictions_df, summaries_df = train_variable(config, pattern, args.seasons, args, output_root)
        horizon_metrics, season_metrics = write_variable_outputs(config, predictions_df, summaries_df, output_root)

        all_predictions.append(predictions_df)
        all_summaries.append(summaries_df)
        all_horizon_metrics.append(horizon_metrics.assign(target_variable=config.name))
        all_season_metrics.append(season_metrics.assign(target_variable=config.name))

    metrics_dir = output_root / "metrics"
    predictions_dir = output_root / "predictions"
    if all_horizon_metrics:
        combined_horizon = pd.concat(all_horizon_metrics, ignore_index=True)
        combined_cols = [
            "target_variable",
            "site_id",
            "season",
            "horizon_bin",
            "n",
            "rmse",
            "mae",
            "rmse_skill_vs_persistence",
            "mae_skill_vs_persistence",
        ]
        combined_horizon = combined_horizon[combined_cols]
        combined_horizon.to_csv(metrics_dir / "season_specific_metrics_by_variable_horizon.csv", index=False)
    if all_season_metrics:
        pd.concat(all_season_metrics, ignore_index=True).to_csv(
            metrics_dir / "season_specific_metrics_by_variable_season.csv",
            index=False,
        )
    if all_summaries:
        pd.concat(all_summaries, ignore_index=True).to_csv(
            metrics_dir / "season_specific_overall_metrics.csv",
            index=False,
        )
    if all_predictions:
        pd.concat(all_predictions, ignore_index=True).to_csv(
            predictions_dir / "season_specific_validation_predictions_long.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
