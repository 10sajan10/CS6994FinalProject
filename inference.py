from __future__ import annotations

import argparse
import csv
import json
import os
import select
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

try:
    import psycopg2
    from psycopg2.extras import DictCursor, Json
except ImportError:  # pragma: no cover - handled at runtime with a clear error.
    psycopg2 = None
    DictCursor = None
    Json = None


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "temporal_transformer") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "temporal_transformer"))

from temporal_transformer.train_discharge_transformer import TemporalTransformer  # noqa: E402


SEASON_ORDER = ["winter", "spring", "summer", "fall"]
SEASON_TO_ID = {name: idx for idx, name in enumerate(SEASON_ORDER)}
TARGET_CHANNEL = "datastream_hourly_insert"


@dataclass(frozen=True)
class TargetConfig:
    name: str
    variable_code: str
    lag_prefix: str
    season_model_folder: str
    allow_negative: bool


@dataclass
class DatastreamEvent:
    datastream_id: int
    site_id: int
    variable_id: int
    variable_code: str
    datetime_utc: datetime
    value: float | None


@dataclass
class LoadedModel:
    family: str
    target_variable: str
    site_id: int
    season: str | None
    path: Path
    bundle: dict[str, Any]
    module: TemporalTransformer


TARGET_CONFIGS = {
    "air_temperature": TargetConfig(
        name="air_temperature",
        variable_code="AirTemp",
        lag_prefix="air_temp_t-",
        season_model_folder="models_air_temperature_season_specific",
        allow_negative=True,
    ),
    "discharge": TargetConfig(
        name="discharge",
        variable_code="Discharge",
        lag_prefix="discharge_t-",
        season_model_folder="models_discharge_season_specific",
        allow_negative=False,
    ),
}
VARIABLE_CODE_TO_TARGET = {
    config.variable_code: name for name, config in TARGET_CONFIGS.items()
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run future inference whenever an exact-hour datastream row is added. "
            "Rows at 15:00:00 are processed; rows at 15:15:00 are ignored."
        )
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--db-host", default=os.environ.get("DB_HOST", "localhost"))
    parser.add_argument("--db-port", type=int, default=int(os.environ.get("DB_PORT", "5433")))
    parser.add_argument("--db-name", default=os.environ.get("DB_NAME", "database"))
    parser.add_argument("--db-user", default=os.environ.get("DB_USER", "admin"))
    parser.add_argument("--db-password", default=os.environ.get("DB_PASSWORD", "password"))
    parser.add_argument("--mode", choices=["poll", "listen"], default="poll")
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--poll-limit", type=int, default=250)
    parser.add_argument("--from-beginning", action="store_true")
    parser.add_argument("--process-datastream-id", type=int)
    parser.add_argument("--install-db-objects", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--verbose-models", action="store_true")
    parser.add_argument(
        "--predictions-csv",
        default=str(PROJECT_ROOT / "inference_predictions.csv"),
        help="Long-format prediction CSV appended after each inference run.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-history-gap-hours", type=int, default=10)
    return parser.parse_args()


def require_psycopg2() -> None:
    if psycopg2 is None:
        raise RuntimeError(
            "psycopg2 is required for database inference. Install project "
            "dependencies with `pip install -r temporal_transformer/requirements.txt`."
        )


def connect_db(args: argparse.Namespace):
    require_psycopg2()
    return psycopg2.connect(
        host=args.db_host,
        port=args.db_port,
        dbname=args.db_name,
        user=args.db_user,
        password=args.db_password,
    )


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def month_to_season_name(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "fall"


def is_exact_hour(timestamp: datetime) -> bool:
    return (
        timestamp.minute == 0
        and timestamp.second == 0
        and timestamp.microsecond == 0
    )


def ceil_hour(timestamp: pd.Timestamp) -> pd.Timestamp:
    if timestamp.minute == 0 and timestamp.second == 0 and timestamp.microsecond == 0:
        return timestamp.replace(nanosecond=0)
    return (timestamp.floor("h") + pd.Timedelta(hours=1)).replace(nanosecond=0)


def extract_step(column_name: str, marker: str) -> int:
    try:
        return int(column_name.rsplit(marker, 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Could not parse step from column {column_name!r}") from exc


def horizon_bin_label(horizon_step: int) -> str:
    if 1 <= horizon_step <= 6:
        return "0-6"
    if 7 <= horizon_step <= 12:
        return "6-12"
    if 13 <= horizon_step <= 18:
        return "12-18"
    if 19 <= horizon_step <= 24:
        return "18-24"
    return f"{horizon_step}"


def has_consecutive_missing_gap(missing: pd.Series, min_gap_hours: int) -> bool:
    run_length = 0
    for value in missing.fillna(False).astype(bool):
        if value:
            run_length += 1
            if run_length >= min_gap_hours:
                return True
        else:
            run_length = 0
    return False


def scaler_transform(values: np.ndarray, mean: list[float], scale: list[float]) -> np.ndarray:
    mean_arr = np.asarray(mean, dtype=np.float32)
    scale_arr = np.asarray(scale, dtype=np.float32)
    scale_arr = np.where(scale_arr == 0.0, 1.0, scale_arr)
    return (values - mean_arr) / scale_arr


def scaler_inverse(values: np.ndarray, mean: list[float], scale: list[float]) -> np.ndarray:
    mean_arr = np.asarray(mean, dtype=np.float32)
    scale_arr = np.asarray(scale, dtype=np.float32)
    return values * scale_arr + mean_arr


def load_model_bundle(path: Path, device: torch.device) -> LoadedModel:
    bundle = torch.load(path, map_location=device, weights_only=False)
    hparams = dict(bundle["model_hparams"])
    model = TemporalTransformer(
        input_dim=int(hparams["input_dim"]),
        hidden_dim=int(hparams["hidden_dim"]),
        num_heads=int(hparams["num_heads"]),
        num_layers=int(hparams["num_layers"]),
        dropout=float(hparams["dropout"]),
        horizon=int(hparams["horizon"]),
        learning_rate=float(hparams.get("learning_rate", 1e-3)),
        seq_len=int(hparams["seq_len"]),
    )
    model.load_state_dict(bundle["model_state_dict"])
    model.to(device)
    model.eval()

    target_variable = bundle.get("target_variable")
    if target_variable is None:
        if "discharge" in path.name or "models_discharge" in str(path):
            target_variable = "discharge"
        elif "air_temperature" in path.name or "models_air_temperature" in str(path):
            target_variable = "air_temperature"
        else:
            raise ValueError(f"Could not infer target variable for {path}")

    family = "season_specific" if "season_specific" in str(path) else "site"
    season = bundle.get("season")
    return LoadedModel(
        family=family,
        target_variable=str(target_variable),
        site_id=int(bundle["site_id"]),
        season=str(season) if season else None,
        path=path,
        bundle=bundle,
        module=model,
    )


def discover_models(project_root: Path, device: torch.device, verbose: bool) -> list[LoadedModel]:
    model_paths: list[Path] = []
    for config in TARGET_CONFIGS.values():
        model_paths.extend(
            sorted(
                (
                    project_root
                    / "season_specific_training"
                    / config.season_model_folder
                ).glob("site_*/*/*_best_model.pt")
            )
        )

    loaded = [load_model_bundle(path, device) for path in model_paths]
    if not loaded:
        raise FileNotFoundError(f"No season-specific model bundles found under {project_root}")

    print(f"Loaded {len(loaded)} season-specific model bundle(s)")
    if verbose:
        for model in loaded:
            print(
                f"  target={model.target_variable}, site_id={model.site_id}, "
                f"season={model.season}, path={model.path}"
            )
    return loaded


def ensure_prediction_tables(conn) -> None:
    sql = """
    CREATE TABLE IF NOT EXISTS inference_state (
        state_key TEXT PRIMARY KEY,
        state_value TEXT NOT NULL,
        updated_at TIMESTAMP NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS model_predictions (
        prediction_id BIGSERIAL PRIMARY KEY,
        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
        source_datastream_id BIGINT,
        model_family TEXT NOT NULL,
        target_variable TEXT NOT NULL,
        site_id INTEGER NOT NULL,
        season TEXT,
        season_key TEXT NOT NULL DEFAULT '',
        history_end_time TIMESTAMP NOT NULL,
        forecast_start_time TIMESTAMP NOT NULL,
        horizon INTEGER NOT NULL,
        horizon_bin TEXT NOT NULL,
        target_timestamp TIMESTAMP NOT NULL,
        prediction DOUBLE PRECISION NOT NULL,
        model_path TEXT NOT NULL,
        feature_row_json JSONB,
        UNIQUE (
            model_family,
            target_variable,
            site_id,
            history_end_time,
            season_key,
            horizon
        )
    );

    ALTER TABLE model_predictions
    ADD COLUMN IF NOT EXISTS prediction DOUBLE PRECISION;
    """
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def install_database_objects(conn) -> None:
    ensure_prediction_tables(conn)
    sql = f"""
    CREATE OR REPLACE FUNCTION notify_hourly_datastream_insert()
    RETURNS trigger AS $$
    DECLARE
        payload TEXT;
        inserted_variable_code TEXT;
    BEGIN
        IF date_trunc('hour', NEW.datetime_utc) = NEW.datetime_utc THEN
            SELECT variable_code
            INTO inserted_variable_code
            FROM variable
            WHERE variable_id = NEW.variable_id;

            payload := json_build_object(
                'datastream_id', NEW.datastream_id,
                'site_id', NEW.site_id,
                'variable_id', NEW.variable_id,
                'variable_code', inserted_variable_code,
                'datetime_utc', NEW.datetime_utc,
                'value', NEW.value
            )::text;

            PERFORM pg_notify('{TARGET_CHANNEL}', payload);
        END IF;
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;

    DROP TRIGGER IF EXISTS datastream_hourly_insert_notify ON datastream;
    CREATE TRIGGER datastream_hourly_insert_notify
    AFTER INSERT ON datastream
    FOR EACH ROW
    EXECUTE FUNCTION notify_hourly_datastream_insert();
    """
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    print("Database inference tables and hourly insert trigger are installed.")


def fetch_event(conn, datastream_id: int) -> DatastreamEvent | None:
    sql = """
        SELECT
            d.datastream_id,
            d.site_id,
            d.variable_id,
            v.variable_code,
            d.datetime_utc,
            d.value
        FROM datastream d
        JOIN variable v ON d.variable_id = v.variable_id
        WHERE d.datastream_id = %s
    """
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(sql, (datastream_id,))
        row = cur.fetchone()
    return row_to_event(row) if row else None


def row_to_event(row: Any) -> DatastreamEvent:
    return DatastreamEvent(
        datastream_id=int(row["datastream_id"]),
        site_id=int(row["site_id"]),
        variable_id=int(row["variable_id"]),
        variable_code=str(row["variable_code"]),
        datetime_utc=row["datetime_utc"],
        value=float(row["value"]) if row["value"] is not None else None,
    )


def fetch_new_events(conn, last_id: int, limit: int) -> list[DatastreamEvent]:
    sql = """
        SELECT
            d.datastream_id,
            d.site_id,
            d.variable_id,
            v.variable_code,
            d.datetime_utc,
            d.value
        FROM datastream d
        JOIN variable v ON d.variable_id = v.variable_id
        WHERE d.datastream_id > %s
        ORDER BY d.datastream_id
        LIMIT %s
    """
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(sql, (last_id, limit))
        return [row_to_event(row) for row in cur.fetchall()]


def fetch_max_datastream_id(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT COALESCE(MAX(datastream_id), 0) FROM datastream")
        return int(cur.fetchone()[0])


def fetch_latest_existing_timestamp(conn, event: DatastreamEvent) -> datetime | None:
    sql = """
        SELECT MAX(datetime_utc)
        FROM datastream
        WHERE site_id = %s
          AND variable_id = %s
          AND datastream_id <> %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (event.site_id, event.variable_id, event.datastream_id))
        row = cur.fetchone()
    return row[0] if row and row[0] is not None else None


def get_state(conn, key: str) -> str | None:
    with conn.cursor() as cur:
        cur.execute("SELECT state_value FROM inference_state WHERE state_key = %s", (key,))
        row = cur.fetchone()
    return str(row[0]) if row else None


def set_state(conn, key: str, value: str) -> None:
    sql = """
        INSERT INTO inference_state (state_key, state_value, updated_at)
        VALUES (%s, %s, NOW())
        ON CONFLICT (state_key)
        DO UPDATE SET state_value = EXCLUDED.state_value, updated_at = NOW()
    """
    with conn.cursor() as cur:
        cur.execute(sql, (key, value))
    conn.commit()


def fetch_raw_observations(
    conn,
    site_id: int,
    variable_code: str,
    start_time: datetime,
    end_time: datetime,
) -> pd.DataFrame:
    sql = """
        SELECT d.datetime_utc, d.value
        FROM datastream d
        JOIN variable v ON d.variable_id = v.variable_id
        WHERE d.site_id = %s
          AND v.variable_code = %s
          AND d.datetime_utc > %s
          AND d.datetime_utc <= %s
        ORDER BY d.datetime_utc
    """
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(sql, (site_id, variable_code, start_time, end_time))
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["datetime_utc", "value"])


def build_hourly_series(
    conn,
    site_id: int,
    variable_code: str,
    start_time: datetime,
    end_time: datetime,
    allow_negative: bool,
    fill_mode: str,
) -> pd.DataFrame:
    index = pd.date_range(start_time, end_time, freq="h")
    hourly = pd.DataFrame({"datetime": index})
    raw_start = start_time - timedelta(hours=1)
    raw = fetch_raw_observations(conn, site_id, variable_code, raw_start, end_time)

    if raw.empty:
        hourly["value"] = np.nan
        hourly["missing_raw"] = True
    else:
        raw["datetime_utc"] = pd.to_datetime(raw["datetime_utc"])
        raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
        if not allow_negative:
            raw.loc[raw["value"] < 0.0, "value"] = np.nan
        raw["datetime"] = raw["datetime_utc"].map(ceil_hour)
        raw = raw.loc[(raw["datetime"] >= index[0]) & (raw["datetime"] <= index[-1])]
        raw = raw.dropna(subset=["value"]).sort_values("datetime_utc")
        bucketed = (
            raw.groupby("datetime", as_index=False)
            .tail(1)[["datetime", "value"]]
            .sort_values("datetime")
        )
        hourly = hourly.merge(bucketed, on="datetime", how="left")
        hourly["missing_raw"] = hourly["value"].isna()

    hourly = hourly.set_index("datetime")
    if fill_mode == "interpolate":
        hourly["value"] = hourly["value"].interpolate(method="time").ffill().bfill()
    elif fill_mode == "ffill":
        hourly["value"] = hourly["value"].ffill().bfill()
    elif fill_mode == "precip":
        hourly["value"] = (
            hourly["value"]
            .interpolate(method="time", limit=3)
            .ffill(limit=3)
            .bfill(limit=3)
        )
    elif fill_mode == "none":
        pass
    else:
        raise ValueError(f"Unsupported fill_mode: {fill_mode}")

    return hourly.reset_index()


def build_temporal_features(history_end_time: datetime, lag_cols: list[str], names: list[str]) -> np.ndarray:
    lag_steps = [extract_step(column, "t-") for column in lag_cols]
    temporal = np.zeros((1, len(lag_cols), len(names)), dtype=np.float32)

    for step_idx, lag_step in enumerate(lag_steps):
        timestamp = pd.Timestamp(history_end_time) - pd.Timedelta(hours=lag_step - 1)
        season_name = month_to_season_name(int(timestamp.month))
        values = {
            "hour_sin": np.sin(2.0 * np.pi * float(timestamp.hour) / 24.0),
            "hour_cos": np.cos(2.0 * np.pi * float(timestamp.hour) / 24.0),
            "dow_sin": np.sin(2.0 * np.pi * float(timestamp.dayofweek) / 7.0),
            "dow_cos": np.cos(2.0 * np.pi * float(timestamp.dayofweek) / 7.0),
            "month_sin": np.sin(2.0 * np.pi * float(timestamp.month - 1) / 12.0),
            "month_cos": np.cos(2.0 * np.pi * float(timestamp.month - 1) / 12.0),
            "day_of_year_frac": float(timestamp.dayofyear) / 366.0,
            "season_winter": float(season_name == "winter"),
            "season_spring": float(season_name == "spring"),
            "season_summer": float(season_name == "summer"),
            "season_fall": float(season_name == "fall"),
        }
        for feature_idx, name in enumerate(names):
            temporal[0, step_idx, feature_idx] = float(values.get(name, 0.0))

    return np.nan_to_num(temporal, nan=0.0, posinf=0.0, neginf=0.0)


def build_static_feature(
    conn,
    column: str,
    history_end_time: datetime,
) -> float:
    if column == "snow_depth_latest":
        start = history_end_time - timedelta(hours=167)
        series = build_hourly_series(
            conn,
            site_id=1,
            variable_code="SnowDepth",
            start_time=start,
            end_time=history_end_time,
            allow_negative=False,
            fill_mode="ffill",
        )
        value = series["value"].iloc[-1]
        if pd.isna(value):
            raise ValueError("Could not build snow_depth_latest from site 1 SnowDepth.")
        return float(value)

    if column == "air_temp_avg_last_24h":
        start = history_end_time - timedelta(hours=23)
        series = build_hourly_series(
            conn,
            site_id=2,
            variable_code="AirTemp",
            start_time=start,
            end_time=history_end_time,
            allow_negative=True,
            fill_mode="interpolate",
        )
        if series["value"].isna().all():
            raise ValueError("Could not build air_temp_avg_last_24h from site 2 AirTemp.")
        return float(series["value"].mean())

    if column == "precip_avg_last_24h":
        start = history_end_time - timedelta(hours=23)
        series = build_hourly_series(
            conn,
            site_id=2,
            variable_code="Precip",
            start_time=start,
            end_time=history_end_time,
            allow_negative=False,
            fill_mode="precip",
        )
        valid_count = int(series["value"].notna().sum())
        if valid_count < 18:
            raise ValueError(
                "Could not build precip_avg_last_24h from site 2 Precip "
                f"because only {valid_count} hourly values were available."
            )
        return float(series["value"].mean())

    if column == "source_mtime":
        return 0.0

    print(f"Warning: unknown static feature {column!r}; using 0.0")
    return 0.0


def build_feature_row(
    conn,
    loaded_model: LoadedModel,
    history_end_time: datetime,
    max_history_gap_hours: int,
) -> dict[str, Any]:
    config = TARGET_CONFIGS[loaded_model.target_variable]
    bundle = loaded_model.bundle
    lag_cols = list(bundle["lag_cols"])
    target_cols = list(bundle["target_cols"])
    history_start = history_end_time - timedelta(hours=len(lag_cols) - 1)

    lag_series = build_hourly_series(
        conn,
        site_id=loaded_model.site_id,
        variable_code=config.variable_code,
        start_time=history_start,
        end_time=history_end_time,
        allow_negative=config.allow_negative,
        fill_mode="interpolate",
    )
    if lag_series["value"].isna().all():
        raise ValueError(
            f"No {config.variable_code} history found for site_id={loaded_model.site_id} "
            f"ending at {history_end_time}."
        )
    if has_consecutive_missing_gap(lag_series["missing_raw"], max_history_gap_hours):
        raise ValueError(
            f"Skipping {config.name} site_id={loaded_model.site_id}: found at least "
            f"{max_history_gap_hours} consecutive missing raw history hours."
        )
    if int(lag_series["value"].isna().sum()):
        raise ValueError(
            f"Could not fill all {config.variable_code} lag values for "
            f"site_id={loaded_model.site_id} ending at {history_end_time}."
        )

    forecast_start = history_end_time + timedelta(hours=1)
    max_horizon = max(extract_step(column, "t+") for column in target_cols)
    forecast_end = forecast_start + timedelta(hours=max_horizon - 1)
    season = month_to_season_name(forecast_start.month)

    row: dict[str, Any] = {
        "site_id": loaded_model.site_id,
        "history_end_time": history_end_time,
        "forecast_start_time": forecast_start,
        "forecast_end_time": forecast_end,
        "year": forecast_start.year,
        "month": forecast_start.month,
        "day": forecast_start.day,
        "hour": forecast_start.hour,
        "day_of_week": forecast_start.weekday(),
        "day_of_year": int(forecast_start.strftime("%j")),
        "season": SEASON_TO_ID[season],
        "sample_stride_hours": 1,
    }

    lag_values = lag_series["value"].to_numpy(dtype=np.float32)
    for column, value in zip(lag_cols, lag_values):
        row[column] = float(value)

    for column in bundle.get("static_feature_cols", []) or []:
        row[column] = build_static_feature(conn, column, history_end_time)

    return row


def build_model_sequence(loaded_model: LoadedModel, feature_row: dict[str, Any]) -> tuple[torch.Tensor, np.ndarray]:
    bundle = loaded_model.bundle
    lag_cols = list(bundle["lag_cols"])
    lag_values = np.array([[feature_row[column] for column in lag_cols]], dtype=np.float32)
    lag_scaled = scaler_transform(
        lag_values.reshape(-1, 1),
        bundle["lag_scaler_mean"],
        bundle["lag_scaler_scale"],
    ).reshape(lag_values.shape)
    lag_scaled = np.nan_to_num(lag_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    temporal_values = build_temporal_features(
        history_end_time=feature_row["history_end_time"],
        lag_cols=lag_cols,
        names=list(bundle.get("temporal_sequence_feature_names", [])),
    )

    static_cols = list(bundle.get("static_feature_cols", []) or [])
    if static_cols:
        static_values = np.array([[feature_row.get(column, 0.0) for column in static_cols]], dtype=np.float32)
        static_values = scaler_transform(
            static_values,
            bundle.get("static_scaler_mean", []),
            bundle.get("static_scaler_scale", []),
        )
        static_values = np.nan_to_num(static_values, nan=0.0, posinf=0.0, neginf=0.0)
        repeated_static = np.repeat(static_values[:, None, :], len(lag_cols), axis=1)
        sequence = np.concatenate([lag_scaled[:, :, None], temporal_values, repeated_static], axis=2)
    else:
        sequence = np.concatenate([lag_scaled[:, :, None], temporal_values], axis=2)

    tensor = torch.tensor(sequence, dtype=torch.float32, device=next(loaded_model.module.parameters()).device)
    return tensor, lag_values


def clip_prediction_variation(
    predictions: np.ndarray,
    lag_values: np.ndarray,
    latest_value: float,
    bundle: dict[str, Any],
) -> np.ndarray:
    if not bool(bundle.get("variation_clip_enabled", False)):
        return predictions

    clipped = predictions.copy()
    hourly_changes = np.abs(np.diff(lag_values.reshape(-1)))
    history_limit = float(np.nanmax(hourly_changes)) if hourly_changes.size else 0.0
    if not np.isfinite(history_limit):
        history_limit = 0.0
    history_limit *= float(bundle.get("variation_clip_history_multiplier", 1.5))
    pct_limit = float(bundle.get("variation_clip_pct", 0.05))
    absolute_floor = float(bundle.get("variation_clip_absolute_floor", 2.0))

    previous = float(latest_value)
    for step_idx in range(clipped.shape[1]):
        allowed = max(absolute_floor, pct_limit * abs(previous), history_limit)
        lower = previous - allowed
        upper = previous + allowed
        clipped[0, step_idx] = float(np.clip(clipped[0, step_idx], lower, upper))
        previous = float(clipped[0, step_idx])
    return clipped


def predict_for_model(
    loaded_model: LoadedModel,
    feature_row: dict[str, Any],
    source_datastream_id: int,
) -> list[dict[str, Any]]:
    sequence, lag_values = build_model_sequence(loaded_model, feature_row)
    bundle = loaded_model.bundle
    target_cols = list(bundle["target_cols"])
    residual_base_col = str(bundle["residual_base_col"])
    latest_value = float(feature_row[residual_base_col])

    with torch.no_grad():
        scaled_predictions = loaded_model.module(sequence).detach().cpu().numpy()

    predicted_deltas = scaler_inverse(
        scaled_predictions,
        bundle["target_scaler_mean"],
        bundle["target_scaler_scale"],
    )
    raw_predictions = predicted_deltas + latest_value
    predictions = clip_prediction_variation(raw_predictions, lag_values, latest_value, bundle)

    rows: list[dict[str, Any]] = []
    forecast_start = feature_row["forecast_start_time"]
    for step_idx, target_col in enumerate(target_cols):
        horizon = extract_step(target_col, "t+")
        target_timestamp = forecast_start + timedelta(hours=horizon - 1)
        rows.append(
            {
                "source_datastream_id": source_datastream_id,
                "model_family": loaded_model.family,
                "target_variable": loaded_model.target_variable,
                "site_id": loaded_model.site_id,
                "season": loaded_model.season,
                "season_key": loaded_model.season or "",
                "history_end_time": feature_row["history_end_time"],
                "forecast_start_time": forecast_start,
                "horizon": horizon,
                "horizon_bin": horizon_bin_label(horizon),
                "target_timestamp": target_timestamp,
                "prediction": float(predictions[0, step_idx]),
                "model_path": str(loaded_model.path),
                "feature_row_json": serialize_feature_row(feature_row),
            }
        )
    return rows


def serialize_feature_row(feature_row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in feature_row.items():
        if isinstance(value, (datetime, pd.Timestamp)):
            out[key] = value.isoformat(sep=" ")
        elif isinstance(value, np.generic):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def matching_models(
    models: list[LoadedModel],
    target_variable: str,
    site_id: int,
    history_end_time: datetime,
) -> list[LoadedModel]:
    forecast_start = history_end_time + timedelta(hours=1)
    season = month_to_season_name(forecast_start.month)
    matches = []
    for model in models:
        if model.target_variable != target_variable or model.site_id != site_id:
            continue
        if model.family != "season_specific":
            continue
        if model.season != season:
            continue
        matches.append(model)
    return matches


def persist_predictions(conn, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    sql = """
        INSERT INTO model_predictions (
            source_datastream_id,
            model_family,
            target_variable,
            site_id,
            season,
            season_key,
            history_end_time,
            forecast_start_time,
            horizon,
            horizon_bin,
            target_timestamp,
            prediction,
            model_path,
            feature_row_json
        )
        VALUES (
            %(source_datastream_id)s,
            %(model_family)s,
            %(target_variable)s,
            %(site_id)s,
            %(season)s,
            %(season_key)s,
            %(history_end_time)s,
            %(forecast_start_time)s,
            %(horizon)s,
            %(horizon_bin)s,
            %(target_timestamp)s,
            %(prediction)s,
            %(model_path)s,
            %(feature_row_json)s
        )
        ON CONFLICT (
            model_family,
            target_variable,
            site_id,
            history_end_time,
            season_key,
            horizon
        )
        DO UPDATE SET
            created_at = NOW(),
            source_datastream_id = EXCLUDED.source_datastream_id,
            forecast_start_time = EXCLUDED.forecast_start_time,
            horizon_bin = EXCLUDED.horizon_bin,
            target_timestamp = EXCLUDED.target_timestamp,
            prediction = EXCLUDED.prediction,
            model_path = EXCLUDED.model_path,
            feature_row_json = EXCLUDED.feature_row_json
    """
    with conn.cursor() as cur:
        for row in rows:
            payload = dict(row)
            payload["feature_row_json"] = Json(row["feature_row_json"])
            cur.execute(sql, payload)
    conn.commit()


def append_predictions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = [
        "source_datastream_id",
        "target_variable",
        "site_id",
        "season",
        "history_end_time",
        "forecast_start_time",
        "horizon",
        "horizon_bin",
        "target_timestamp",
        "prediction",
        "model_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def process_event(
    conn,
    event: DatastreamEvent,
    models: list[LoadedModel],
    args: argparse.Namespace,
) -> int:
    if not is_exact_hour(event.datetime_utc):
        print(
            f"Skip datastream_id={event.datastream_id}: "
            f"{event.datetime_utc} is not an exact hourly timestamp."
        )
        return 0

    target_variable = VARIABLE_CODE_TO_TARGET.get(event.variable_code)
    if target_variable is None:
        print(
            f"Skip datastream_id={event.datastream_id}: variable_code={event.variable_code!r} "
            "does not have a target model."
        )
        return 0

    latest_existing = fetch_latest_existing_timestamp(conn, event)
    if latest_existing is not None and event.datetime_utc <= latest_existing:
        print(
            f"Skip datastream_id={event.datastream_id}: {event.datetime_utc} is not newer "
            f"than the latest existing {event.variable_code} timestamp for site_id={event.site_id} "
            f"({latest_existing})."
        )
        return 0

    selected_models = matching_models(
        models=models,
        target_variable=target_variable,
        site_id=event.site_id,
        history_end_time=event.datetime_utc,
    )
    if not selected_models:
        print(
            f"Skip datastream_id={event.datastream_id}: no loaded {target_variable} "
            f"season-specific model exists for site_id={event.site_id}."
        )
        return 0

    all_rows: list[dict[str, Any]] = []
    for model in selected_models:
        try:
            feature_row = build_feature_row(
                conn,
                loaded_model=model,
                history_end_time=event.datetime_utc,
                max_history_gap_hours=args.max_history_gap_hours,
            )
            all_rows.extend(predict_for_model(model, feature_row, event.datastream_id))
        except Exception as exc:
            print(
                f"Failed inference for datastream_id={event.datastream_id}, "
                f"target={model.target_variable}, site_id={model.site_id}, "
                f"season={model.season}: {exc}"
            )

    if not all_rows:
        return 0

    if args.dry_run:
        print_predictions_preview(all_rows)
    else:
        persist_predictions(conn, all_rows)
        if not args.no_csv:
            append_predictions_csv(Path(args.predictions_csv), all_rows)

    print(
        f"Inference complete for datastream_id={event.datastream_id}: "
        f"{len(all_rows)} prediction row(s)."
    )
    return len(all_rows)


def print_predictions_preview(rows: list[dict[str, Any]], limit: int = 8) -> None:
    print("Dry-run predictions preview:")
    for row in rows[:limit]:
        print(
            f"  {row['target_variable']} site={row['site_id']} "
            f"season={row['season']} horizon={row['horizon']:02d} "
            f"target={row['target_timestamp']} prediction={row['prediction']:.4f}"
        )
    if len(rows) > limit:
        print(f"  ... {len(rows) - limit} more row(s)")


def event_from_notification(payload: str) -> DatastreamEvent:
    data = json.loads(payload)
    return DatastreamEvent(
        datastream_id=int(data["datastream_id"]),
        site_id=int(data["site_id"]),
        variable_id=int(data["variable_id"]),
        variable_code=str(data["variable_code"]),
        datetime_utc=datetime.fromisoformat(str(data["datetime_utc"])),
        value=float(data["value"]) if data.get("value") is not None else None,
    )


def run_listen_loop(conn, models: list[LoadedModel], args: argparse.Namespace) -> None:
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    with conn.cursor() as cur:
        cur.execute(f"LISTEN {TARGET_CHANNEL};")
    print(f"Listening for PostgreSQL notifications on {TARGET_CHANNEL!r}.")

    should_stop = False

    def stop_handler(signum, frame):  # noqa: ARG001
        nonlocal should_stop
        should_stop = True

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    while not should_stop:
        ready, _, _ = select.select([conn], [], [], 5.0)
        if not ready:
            continue
        conn.poll()
        while conn.notifies:
            notify = conn.notifies.pop(0)
            try:
                event = event_from_notification(notify.payload)
                process_event(conn, event, models, args)
            except Exception as exc:
                print(f"Failed to process notification payload={notify.payload!r}: {exc}")

    print("Listener stopped.")


def run_poll_loop(conn, models: list[LoadedModel], args: argparse.Namespace) -> None:
    state_key = "last_processed_datastream_id"
    state = get_state(conn, state_key)
    if args.from_beginning:
        last_id = int(state or 0)
    elif state is None:
        last_id = fetch_max_datastream_id(conn)
        set_state(conn, state_key, str(last_id))
        print(f"Initialized inference state at datastream_id={last_id}; waiting for new rows.")
    else:
        last_id = int(state)
        print(f"Resuming polling from datastream_id={last_id}.")

    should_stop = False

    def stop_handler(signum, frame):  # noqa: ARG001
        nonlocal should_stop
        should_stop = True

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    while not should_stop:
        events = fetch_new_events(conn, last_id, args.poll_limit)
        if not events:
            time.sleep(args.poll_interval)
            continue

        for event in events:
            process_event(conn, event, models, args)
            last_id = max(last_id, event.datastream_id)
            if not args.dry_run:
                set_state(conn, state_key, str(last_id))

    print("Polling stopped.")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    device = choose_device(args.device)
    models = discover_models(project_root, device, verbose=args.verbose_models)

    conn = connect_db(args)
    try:
        if args.install_db_objects:
            install_database_objects(conn)
        elif not (args.dry_run and args.process_datastream_id is not None):
            ensure_prediction_tables(conn)

        if args.process_datastream_id is not None:
            event = fetch_event(conn, args.process_datastream_id)
            if event is None:
                raise ValueError(f"No datastream row found for id={args.process_datastream_id}")
            process_event(conn, event, models, args)
            return

        if args.mode == "listen":
            run_listen_loop(conn, models, args)
        else:
            run_poll_loop(conn, models, args)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
