from __future__ import annotations

import argparse
import glob
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


SEASON_NAMES = {
    0: "winter",
    1: "spring",
    2: "summer",
    3: "fall",
}

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
    default_pattern = str(project_root / "streamflow_parquet_v2" / "*precip24avg*.parquet")

    parser = argparse.ArgumentParser(
        description="Train per-site temporal transformer models on multihorizon streamflow parquet data."
    )
    parser.add_argument(
        "--parquet-pattern",
        default=default_pattern,
        help="Glob pattern for parquet inputs.",
    )
    parser.add_argument(
        "--output-root",
        default=str(project_root),
        help="Project root where models/ and metrics_by_season.csv will be written.",
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


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def month_to_season_name(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "fall"


def extract_lag_number(column_name: str) -> int:
    match = re.search(r"t-(\d+)$", column_name)
    if not match:
        raise ValueError(f"Could not parse lag number from column: {column_name}")
    return int(match.group(1))


def extract_horizon_number(column_name: str) -> int:
    match = re.search(r"t\+(\d+)$", column_name)
    if not match:
        raise ValueError(f"Could not parse horizon number from column: {column_name}")
    return int(match.group(1))


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


def infer_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    lag_cols = sorted(
        [col for col in df.columns if col.startswith("discharge_t-")],
        key=extract_lag_number,
        reverse=True,
    )
    target_cols = sorted(
        [col for col in df.columns if col.startswith("target_discharge_t+")],
        key=extract_horizon_number,
    )

    dropped_feature_columns = {
        "site_id",
        "stride",
        "sample_stride_hours",
        "history_end_time",
        "forecast_start_time",
        "forecast_end_time",
        "source_path",
    }
    static_cols = [
        col
        for col in df.columns
        if col not in dropped_feature_columns
        and col not in lag_cols
        and col not in target_cols
        and col not in TIME_CONTEXT_COLUMNS
    ]
    return lag_cols, target_cols, static_cols


def add_fixed_season_dummies(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "season" in out.columns:
        season_series = out["season"].map(SEASON_NAMES).fillna("unknown")
        dummies = pd.get_dummies(season_series, prefix="season")
        for name in ["season_winter", "season_spring", "season_summer", "season_fall"]:
            if name not in dummies.columns:
                dummies[name] = 0
        dummies = dummies[["season_winter", "season_spring", "season_summer", "season_fall"]]
        out = pd.concat([out.drop(columns=["season"]), dummies], axis=1)
    return out


def build_static_features(df: pd.DataFrame, static_cols: list[str]) -> pd.DataFrame:
    static_df = df[static_cols].copy()
    static_df = add_fixed_season_dummies(static_df)
    return static_df


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
        # Lag columns are ordered oldest -> newest (for example t-168 ... t-1),
        # and each timestamp is aligned to that same discharge step.
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


def latest_discharge_column(lag_cols: list[str]) -> str:
    if "discharge_t-1" not in lag_cols:
        raise ValueError("Expected lag feature discharge_t-1 to build residual targets.")
    return "discharge_t-1"


def latest_discharge_values(df: pd.DataFrame, lag_cols: list[str]) -> np.ndarray:
    latest_col = latest_discharge_column(lag_cols)
    values = df[latest_col].to_numpy(dtype=np.float32)
    return values.reshape(-1, 1)


def make_delta_targets(df: pd.DataFrame, lag_cols: list[str], target_cols: list[str]) -> np.ndarray:
    target_values = df[target_cols].to_numpy(dtype=np.float32)
    return target_values - latest_discharge_values(df, lag_cols)


def reconstruct_absolute_targets(delta_values: np.ndarray, df: pd.DataFrame, lag_cols: list[str]) -> np.ndarray:
    return delta_values + latest_discharge_values(df, lag_cols)


def make_persistence_predictions(df: pd.DataFrame, lag_cols: list[str], target_cols: list[str]) -> np.ndarray:
    latest_values = latest_discharge_values(df, lag_cols)
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
    latest_values = latest_discharge_values(df, lag_cols).reshape(-1)
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
    static_cols: list[str],
    target_cols: list[str],
) -> tuple[StandardScaler, StandardScaler, StandardScaler, list[str]]:
    static_train = build_static_features(train_df, static_cols)
    static_train = static_train.apply(pd.to_numeric, errors="coerce")
    static_train = static_train.fillna(static_train.median(numeric_only=True))
    static_train = static_train.fillna(0.0)
    static_feature_cols = static_train.columns.tolist()

    lag_scaler = StandardScaler()
    lag_scaler.fit(train_df[lag_cols].to_numpy(dtype=np.float32).reshape(-1, 1))

    static_scaler = StandardScaler()
    static_scaler.fit(static_train.to_numpy(dtype=np.float32))

    target_deltas = make_delta_targets(train_df, lag_cols, target_cols)
    target_scaler = StandardScaler()
    target_scaler.fit(target_deltas.reshape(-1, 1))

    return lag_scaler, static_scaler, target_scaler, static_feature_cols


def transform_frame(
    df: pd.DataFrame,
    lag_cols: list[str],
    target_cols: list[str],
    static_cols: list[str],
    static_feature_cols: list[str],
    lag_scaler: StandardScaler,
    static_scaler: StandardScaler,
    target_scaler: StandardScaler,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lag_values = df[lag_cols].to_numpy(dtype=np.float32)
    lag_values = lag_scaler.transform(lag_values.reshape(-1, 1)).reshape(lag_values.shape)
    lag_values = np.nan_to_num(lag_values, nan=0.0, posinf=0.0, neginf=0.0)

    temporal_values = build_temporal_sequence_features(df, lag_cols)

    static_df = build_static_features(df, static_cols)
    static_df = static_df.apply(pd.to_numeric, errors="coerce")
    for column in static_feature_cols:
        if column not in static_df.columns:
            static_df[column] = 0.0
    static_df = static_df[static_feature_cols]
    static_df = static_df.fillna(0.0)
    static_values = static_scaler.transform(static_df.to_numpy(dtype=np.float32))
    static_values = np.nan_to_num(static_values, nan=0.0, posinf=0.0, neginf=0.0)

    target_values = make_delta_targets(df, lag_cols, target_cols)
    target_values = target_scaler.transform(target_values.reshape(-1, 1)).reshape(target_values.shape)
    target_values = np.nan_to_num(target_values, nan=0.0, posinf=0.0, neginf=0.0)

    return lag_values, temporal_values, static_values, target_values


class StreamflowSequenceDataset(Dataset):
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
        repeated_static = np.repeat(self.static_values[index][None, :], lag_sequence.shape[0], axis=0)
        sequence = np.concatenate([lag_sequence, temporal_sequence, repeated_static], axis=1)

        return (
            torch.tensor(sequence, dtype=torch.float32),
            torch.tensor(self.targets[index], dtype=torch.float32),
        )


class TemporalTransformer(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        horizon: int,
        learning_rate: float,
        seq_len: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(sequence) + self.position_embedding[:, : sequence.shape[1], :]
        encoded = self.encoder(hidden)
        pooled = encoded[:, -1, :]
        return self.head(pooled)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        sequence, target = batch
        prediction = self(sequence)
        loss = self.loss_fn(prediction, target)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        sequence, target = batch
        prediction = self(sequence)
        loss = self.loss_fn(prediction, target)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


def make_dataloader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def inverse_scale_targets(values: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    original_shape = values.shape
    restored = scaler.inverse_transform(values.reshape(-1, 1)).reshape(original_shape)
    return restored


def predict_batches(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for sequences, _ in loader:
            sequences = sequences.to(device)
            predictions = model(sequences).detach().cpu().numpy()
            outputs.append(predictions)
    return np.concatenate(outputs, axis=0)


def compute_overall_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    rmse = math.sqrt(mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1)))
    mae = mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1))
    return float(rmse), float(mae)


def relative_skill_score(model_error: float, baseline_error: float) -> float | None:
    if baseline_error <= 0.0:
        return None
    return float(1.0 - (model_error / baseline_error))


def horizon_bin_label(horizon_step: int) -> str:
    if 1 <= horizon_step <= 6:
        return "0-6"
    if 7 <= horizon_step <= 12:
        return "6-12"
    if 13 <= horizon_step <= 18:
        return "12-18"
    if 19 <= horizon_step <= 24:
        return "18-24"
    raise ValueError(f"Unsupported horizon step: {horizon_step}")


def expand_predictions_by_target_timestamp(
    site_id: int,
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
                    "site_id": int(site_id),
                    "forecast_start_time": forecast_start,
                    "horizon": int(horizon_step),
                    "horizon_bin": horizon_bin_label(horizon_step),
                    "target_timestamp": target_timestamp,
                    "season": month_to_season_name(int(target_timestamp.month)),
                    "y_true": float(y_true[row_idx, step_idx]),
                    "y_pred": float(y_pred[row_idx, step_idx]),
                    "y_pred_unclipped": float(y_pred_unclipped[row_idx, step_idx]),
                    "y_persistence": float(y_persistence[row_idx, step_idx]),
                }
            )
    return pd.DataFrame.from_records(records)


def aggregate_season_metrics(expanded_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (site_id, season), group in expanded_df.groupby(["site_id", "season"], sort=True):
        rmse = math.sqrt(mean_squared_error(group["y_true"], group["y_pred"]))
        mae = mean_absolute_error(group["y_true"], group["y_pred"])
        persistence_rmse = math.sqrt(mean_squared_error(group["y_true"], group["y_persistence"]))
        persistence_mae = mean_absolute_error(group["y_true"], group["y_persistence"])
        rows.append(
            {
                "site_id": int(site_id),
                "season": season,
                "rmse": float(rmse),
                "mae": float(mae),
                "persistence_rmse": float(persistence_rmse),
                "persistence_mae": float(persistence_mae),
                "rmse_skill_vs_persistence": relative_skill_score(float(rmse), float(persistence_rmse)),
                "mae_skill_vs_persistence": relative_skill_score(float(mae), float(persistence_mae)),
            }
        )
    return pd.DataFrame(rows).sort_values(["site_id", "season"]).reset_index(drop=True)


def aggregate_horizon_season_metrics(expanded_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["site_id", "season", "horizon_bin"]
    for (site_id, season, horizon_bin), group in expanded_df.groupby(group_cols, sort=True):
        rmse = math.sqrt(mean_squared_error(group["y_true"], group["y_pred"]))
        mae = mean_absolute_error(group["y_true"], group["y_pred"])
        persistence_rmse = math.sqrt(mean_squared_error(group["y_true"], group["y_persistence"]))
        persistence_mae = mean_absolute_error(group["y_true"], group["y_persistence"])
        rows.append(
            {
                "site_id": int(site_id),
                "season": season,
                "horizon_bin": horizon_bin,
                "n": int(len(group)),
                "rmse": float(rmse),
                "mae": float(mae),
                "persistence_rmse": float(persistence_rmse),
                "persistence_mae": float(persistence_mae),
                "rmse_skill_vs_persistence": relative_skill_score(float(rmse), float(persistence_rmse)),
                "mae_skill_vs_persistence": relative_skill_score(float(mae), float(persistence_mae)),
            }
        )

    horizon_order = {"0-6": 0, "6-12": 1, "12-18": 2, "18-24": 3}
    out = pd.DataFrame(rows)
    out["horizon_order"] = out["horizon_bin"].map(horizon_order)
    return (
        out.sort_values(["site_id", "season", "horizon_order"])
        .drop(columns=["horizon_order"])
        .reset_index(drop=True)
    )


def train_one_site(
    site_df: pd.DataFrame,
    args: argparse.Namespace,
    output_root: Path,
) -> tuple[pd.DataFrame, dict]:
    site_id = int(site_df["site_id"].iloc[0])
    print(f"\n{'=' * 80}")
    print(f"Training transformer for site_id={site_id}")
    print(f"{'=' * 80}")

    lag_cols, target_cols, static_cols = infer_columns(site_df)
    if not lag_cols or not target_cols:
        raise ValueError(f"Site {site_id} is missing lag or target columns.")

    train_df, valid_df, validation_start, validation_end = trailing_one_year_split(site_df)
    print(
        f"Site {site_id}: train rows={len(train_df):,}, validation rows={len(valid_df):,}, "
        f"validation window={validation_start} to {validation_end}"
    )

    lag_scaler, static_scaler, target_scaler, static_feature_cols = fit_scalers(
        train_df,
        lag_cols,
        static_cols,
        target_cols,
    )

    train_lags, train_temporal, train_static, train_targets = transform_frame(
        train_df,
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
        lag_cols,
        target_cols,
        static_cols,
        static_feature_cols,
        lag_scaler,
        static_scaler,
        target_scaler,
    )

    train_dataset = StreamflowSequenceDataset(train_lags, train_temporal, train_static, train_targets)
    valid_dataset = StreamflowSequenceDataset(valid_lags, valid_temporal, valid_static, valid_targets)
    train_loader = make_dataloader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = make_dataloader(valid_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    input_dim = train_dataset[0][0].shape[-1]
    horizon = len(target_cols)
    seq_len = len(lag_cols)
    site_model_dir = output_root / "models" / str(site_id)
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
        filename="lightning-best",
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

    best_model_path = site_model_dir / "best_model.pt"
    metadata_path = site_model_dir / "metadata.json"
    residual_base_col = latest_discharge_column(lag_cols)
    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "site_id": site_id,
            "lag_cols": lag_cols,
            "target_cols": target_cols,
            "target_transform": "delta_from_latest_discharge",
            "residual_base_col": residual_base_col,
            "variation_clip_enabled": not args.disable_variation_clip,
            "variation_clip_pct": args.variation_clip_pct,
            "variation_clip_history_multiplier": args.variation_clip_history_multiplier,
            "variation_clip_absolute_floor": args.variation_clip_absolute_floor,
            "static_feature_cols": static_feature_cols,
            "lag_scaler_mean": lag_scaler.mean_.tolist(),
            "lag_scaler_scale": lag_scaler.scale_.tolist(),
            "temporal_sequence_feature_names": TEMPORAL_SEQUENCE_FEATURE_NAMES,
            "static_scaler_mean": static_scaler.mean_.tolist(),
            "static_scaler_scale": static_scaler.scale_.tolist(),
            "target_scaler_applies_to": "target_discharge_t+h_minus_discharge_t-1",
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
        "best_model_path": str(best_model_path),
        "best_checkpoint": best_ckpt_path,
        "validation_start": str(validation_start),
        "validation_end": str(validation_end),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "lag_cols": lag_cols,
        "target_cols": target_cols,
        "target_transform": "delta_from_latest_discharge",
        "residual_base_col": residual_base_col,
        "variation_clip_enabled": not args.disable_variation_clip,
        "variation_clip_pct": args.variation_clip_pct,
        "variation_clip_history_multiplier": args.variation_clip_history_multiplier,
        "variation_clip_absolute_floor": args.variation_clip_absolute_floor,
        "static_feature_cols": static_feature_cols,
        "temporal_sequence_feature_names": TEMPORAL_SEQUENCE_FEATURE_NAMES,
        "model_hparams": dict(best_model.hparams),
        "source_files": sorted(site_df["source_path"].dropna().unique().tolist()) if "source_path" in site_df.columns else [],
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

    print("Starting temporal transformer training pipeline")
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
    metrics_path = output_root / "metrics_by_season.csv"
    metrics_by_season.to_csv(metrics_path, index=False)
    print(f"Saved seasonal metrics to {metrics_path}")

    horizon_metrics = aggregate_horizon_season_metrics(combined_predictions)
    horizon_metrics_path = output_root / "metrics_by_season_horizon.csv"
    horizon_metrics.to_csv(horizon_metrics_path, index=False)
    print(f"Saved horizon seasonal metrics to {horizon_metrics_path}")

    overall_metrics_path = output_root / "overall_validation_metrics.csv"
    pd.DataFrame(overall_rows).to_csv(overall_metrics_path, index=False)
    print(f"Saved overall validation metrics to {overall_metrics_path}")

    expanded_predictions_path = output_root / "validation_predictions_long.csv"
    combined_predictions.to_csv(expanded_predictions_path, index=False)
    print(f"Saved validation predictions to {expanded_predictions_path}")


if __name__ == "__main__":
    main()
