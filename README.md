# CS6994 Final Project: Logan River Forecasting

This repository contains the database, feature-building, training, and live
inference workflow for Logan River Observatory forecasting. The final forecasting
targets are:

- `discharge` for stream sites `3` and `4`.
- `air_temperature` for climate sites `1` and `2`.

Each model uses a `168` hour lookback window and predicts the next `24` hourly
values. Live inference uses only the season-specific models: one model per
target variable, site, and season.

## End-To-End Workflow

1. **Start PostgreSQL with Docker.**
   The Docker image creates the normalized schema and loads the bundled database
   dump from `init-scripts/`.

2. **Store observations in `datastream`.**
   Environmental observations are normalized into `site`, `variable`, `unit`,
   `method`, `owner`, `qualifier`, `processing_level`, and `datastream`.
   `datastream` is the main time-series table.

3. **Build training parquet files.**
   `build_streamflow_training_parquet_v2_multihorizon.ipynb` converts database
   observations into model-ready rows with lag features and future targets.

4. **Train temporal transformers.**
   The training scripts save model bundles containing the model weights, lag
   columns, target columns, scaler statistics, model hyperparameters, and
   feature metadata needed by inference.

5. **Run live inference.**
   `inference.py` watches new hourly rows in `datastream`, prepares the same
   shape of feature row used during training, selects the correct
   season-specific model, and writes future predictions to `model_predictions`.

6. **Demo the process.**
   `demo.ipynb` inserts hypothetical exact-hour rows, triggers inference, plots
   recent history plus predictions, and deletes the hypothetical data afterward.

## Database Setup

Start the database:

```bash
docker compose up -d --build
```

Connection settings:

| Parameter | Value |
| --- | --- |
| Host | `localhost` |
| Port | `5433` |
| Database | `database` |
| User | `admin` |
| Password | `password` |

The schema is defined in `init-scripts/01-create-schema.sql`.

Main normalized tables:

- `owner`
- `site`
- `unit`
- `method`
- `variable`
- `processing_level`
- `qualifier`
- `datastream`
- `staging`

The ER diagram is saved as:

- `database_er_diagram.png`

## Training Data

Current parquet files:

- `streamflow_parquet_v2/discharge_training_site_3_lb168_air24avg_precip24avg_h24.parquet`
- `streamflow_parquet_v2/discharge_training_site_4_lb168_air24avg_precip24avg_h24.parquet`
- `air_temperature_parquet_v2/air_temperature_training_site_1_lb168_h24.parquet`
- `air_temperature_parquet_v2/air_temperature_training_site_2_lb168_h24.parquet`

Discharge rows include:

- 168 hourly discharge lags for the target stream site.
- Latest snow depth from site `1`.
- 24-hour average air temperature from site `2`.
- 24-hour average precipitation from site `2`.
- Calendar features.
- 24 hourly discharge targets.

Air-temperature rows include:

- 168 hourly air-temperature lags for the target climate site.
- Calendar features.
- 24 hourly air-temperature targets.

See `parquet.txt` for detailed feature definitions and data-quality rules.

## Training

Set up the Python environment:

```bash
bash temporal_transformer/setup_env.sh
```

Run all-season discharge training:

```bash
bash temporal_transformer/run_discharge_training.sh
```

Run all-season air-temperature training:

```bash
bash temporal_transformer/run_air_temperature_training.sh
```

Run season-specific training:

```bash
bash temporal_transformer/run_season_specific_training.sh
```

The season-specific pipeline trains 16 models:

- 2 target variables: `discharge`, `air_temperature`
- 2 sites per target variable
- 4 seasons: `winter`, `spring`, `summer`, `fall`

Season-specific model folders:

- `season_specific_training/models_discharge_season_specific/site_3/<season>/`
- `season_specific_training/models_discharge_season_specific/site_4/<season>/`
- `season_specific_training/models_air_temperature_season_specific/site_1/<season>/`
- `season_specific_training/models_air_temperature_season_specific/site_2/<season>/`

Each saved `*_best_model.pt` bundle contains everything needed for inference:

- model weights
- target variable
- site id
- season
- lag column order
- target horizon column order
- scaler means and scales
- temporal feature names
- static feature names for discharge models
- model hyperparameters

## Live Inference

The live inference entry point is:

```bash
python inference.py
```

Use the project virtual environment:

```bash
.venv/bin/python inference.py
```

`inference.py` is for live prediction generation. It stores prediction rows
only.

### What Triggers Inference

Inference runs only when a new row satisfies all of these conditions:

1. The row is inserted into `datastream`.
2. The timestamp is an exact hourly timestamp, such as `15:00:00`.
3. The timestamp is newer than the latest existing timestamp for the same
   `site_id + variable_id`.
4. The variable has a target model:
   - `AirTemp`
   - `Discharge`
5. A season-specific model exists for the row's target variable, site, and
   forecast-start season.

Rows such as `15:15:00` are ignored. Historical reloads are ignored because
their timestamps are not newer than the latest existing timestamp for that
site/variable.

### Model Selection

For each accepted row:

1. `inference.py` maps the database variable code to a model target:
   - `AirTemp` -> `air_temperature`
   - `Discharge` -> `discharge`
2. The forecast start time is computed as:
   - `history_end_time + 1 hour`
3. The forecast-start month selects the season:
   - December, January, February -> `winter`
   - March, April, May -> `spring`
   - June, July, August -> `summer`
   - September, October, November -> `fall`
4. The matching season-specific model is loaded by:
   - target variable
   - site id
   - season

Example:

```text
Inserted row:
site_id = 4
variable_code = Discharge
datetime_utc = 2026-03-24 19:00:00

Inference model:
target_variable = discharge
site_id = 4
season = spring
```

### Feature Preparation

The inference row is prepared to match training.

For `air_temperature`:

1. Read the previous 168 hourly air-temperature values for that site.
2. Fill short gaps using the same time interpolation style used in training.
3. Add temporal sequence features for each lag timestamp.
4. Apply the saved lag scaler.
5. Pass the sequence to the season-specific model.

For `discharge`:

1. Read the previous 168 hourly discharge values for the stream site.
2. Build static support features:
   - `snow_depth_latest` from site `1`
   - `air_temp_avg_last_24h` from site `2`
   - `precip_avg_last_24h` from site `2`
3. Add temporal sequence features for each lag timestamp.
4. Apply the saved lag and static scalers.
5. Pass the sequence to the season-specific model.

The model predicts scaled deltas from the latest observed value. `inference.py`
inverse-transforms those deltas and reconstructs absolute future predictions.

### Prediction Output

Predictions are stored in the database table:

- `model_predictions`

This table is created by `inference.py` when database objects are installed or
when the script starts in write mode.

Important columns:

- `source_datastream_id`
- `target_variable`
- `site_id`
- `season`
- `history_end_time`
- `forecast_start_time`
- `horizon`
- `horizon_bin`
- `target_timestamp`
- `prediction`
- `model_path`
- `feature_row_json`

There is one row per horizon. A single accepted insert produces 24 prediction
rows.

## Inference Commands

Install the prediction table and Postgres trigger:

```bash
.venv/bin/python inference.py --install-db-objects --mode listen --no-csv --device cpu
```

Run listener mode:

```bash
.venv/bin/python inference.py --mode listen --no-csv --device cpu
```

Run polling mode:

```bash
.venv/bin/python inference.py --mode poll --poll-interval 5 --no-csv --device cpu
```

Dry-run a known row without writing predictions:

```bash
.venv/bin/python inference.py \
  --process-datastream-id 428591 \
  --dry-run \
  --no-csv \
  --device cpu
```

Write predictions for a known row:

```bash
.venv/bin/python inference.py \
  --process-datastream-id 428591 \
  --no-csv \
  --device cpu
```

Check stored predictions:

```bash
docker exec postgres psql -U admin -d database -c "
SELECT
    target_variable,
    site_id,
    season,
    horizon,
    target_timestamp,
    prediction
FROM model_predictions
WHERE source_datastream_id = 428591
ORDER BY horizon;
"
```

## Inference Demo Notebook

The demo notebook is:

- `demo.ipynb`

The notebook demonstrates the full trigger-style flow:

1. Load season-specific models through `inference.py`.
2. Find the latest observation for each modeled target site/variable.
3. Insert one hypothetical exact-hour row per combo using the same latest value.
   Example: latest `03:45`, value `70` -> hypothetical `04:00`, value `70`.
4. Drain the PostgreSQL notification queue.
5. Call `inference.process_event(...)`.
6. Read predictions from `model_predictions`.
7. Plot recent hourly history plus the 24-hour forecast.
8. Delete the hypothetical `datastream` rows and their generated predictions.

## Main Files

- `README.md`
  - Project overview and sequential workflow.
- `Dockerfile`
  - Builds the PostgreSQL image.
- `docker-compose.yml`
  - Runs the local PostgreSQL database on port `5433`.
- `.dockerignore`
  - Keeps the Docker build context limited to the database init files.
- `init-scripts/01-create-schema.sql`
  - Creates the normalized database schema.
- `init-scripts/02-data-dump.sql.gz`
  - Database dump loaded into the Postgres image.
- `ingest_csv.py`
  - CSV-to-staging ingestion helper.
- `normalize_data.sql`
  - Moves staging rows into normalized tables.
- `build_streamflow_training_parquet_v2_multihorizon.ipynb`
  - Builds discharge and air-temperature training parquet files.
- `temporal_transformer/train_season_specific_transformers.py`
  - Trains one model per target variable, site, and season.
- `inference.py`
  - Live inference worker and utility functions.
- `demo.ipynb`
  - Step-by-step inference demonstration notebook.
- `database_er_diagram.png`
  - Rendered database entity relationship diagram.

## Saved Outputs

Model artifacts:

- `models_discharge/`
- `models_air_temperature/`
- `season_specific_training/models_discharge_season_specific/`
- `season_specific_training/models_air_temperature_season_specific/`

Training logs:

- `lightning_logs_discharge/`
- `lightning_logs_air_temperature/`
- `season_specific_training/lightning_logs_discharge_season_specific/`
- `season_specific_training/lightning_logs_air_temperature_season_specific/`

Validation files remain in the repository for offline analysis:

- `season_specific_training/metrics/`
- `season_specific_training/predictions/`
- `discharge_validation_predictions_long.csv`
- `air_temperature_validation_predictions_long.csv`

Live inference output is stored in:

- database table `model_predictions`

## Project Structure

```text
CS6994FinalProject/
+-- README.md
+-- Dockerfile
+-- docker-compose.yml
+-- .dockerignore
+-- init-scripts/
+-- ingest_csv.py
+-- normalize_data.sql
+-- inference.py
+-- demo.ipynb
+-- database_er_diagram.png
+-- build_streamflow_training_parquet_v2_multihorizon.ipynb
+-- temporal_transformer/
+-- streamflow_parquet_v2/
+-- air_temperature_parquet_v2/
+-- models_discharge/
+-- models_air_temperature/
+-- season_specific_training/
```
