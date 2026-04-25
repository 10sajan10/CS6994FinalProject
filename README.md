# CS6994 Final Project: Logan River Forecasting

This repository contains the full data and modeling workflow for multihorizon
forecasting at Logan River Observatory sites. It includes the PostgreSQL data
setup, parquet feature generation notebook, temporal transformer training code,
saved model artifacts, prediction CSVs, and validation metrics.

The final modeling work covers two targets:

- Discharge forecasting for stream sites 3 and 4.
- Air-temperature forecasting for climate sites 1 and 2.

Each model predicts the next 24 hourly values from a 168-hour lookback window.

## Main Files

- `build_streamflow_training_parquet_v2_multihorizon.ipynb`
  - Builds the discharge and air-temperature training parquet files.
- `parquet.txt`
  - Documents the parquet datasets, columns, sampling, and cleaning rules.
- `train.txt`
  - Documents the normal and season-specific training pipelines.
- `metrics.txt`
  - Documents all metric/prediction files and summarizes current results.
- `temporal_transformer/train_discharge_transformer.py`
  - Normal all-season discharge trainer.
- `temporal_transformer/train_air_temperature_transformer.py`
  - Normal all-season air-temperature trainer.
- `temporal_transformer/train_season_specific_transformers.py`
  - Trains one model per variable, site, and season.

## Data Setup

The database can be recreated with Docker:

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

The Docker startup scripts live in `init-scripts/`.

## Parquet Datasets

Current training parquet files:

- `streamflow_parquet_v2/discharge_training_site_3_lb168_air24avg_precip24avg_h24.parquet`
- `streamflow_parquet_v2/discharge_training_site_4_lb168_air24avg_precip24avg_h24.parquet`
- `air_temperature_parquet_v2/air_temperature_training_site_1_lb168_h24.parquet`
- `air_temperature_parquet_v2/air_temperature_training_site_2_lb168_h24.parquet`

Discharge parquet rows contain 168 hours of discharge history, 24-hour mean air
temperature, 24-hour mean precipitation, latest snow depth, calendar features,
and 24 discharge targets.

Air-temperature parquet rows contain only each target site's own 168-hour
air-temperature history, calendar features, and 24 air-temperature targets.

See `parquet.txt` for the complete schema and data-quality rules.

## Training

Set up the Python environment:

```bash
bash temporal_transformer/setup_env.sh
```

Run normal all-season discharge training:

```bash
bash temporal_transformer/run_discharge_training.sh
```

Run normal all-season air-temperature training:

```bash
bash temporal_transformer/run_air_temperature_training.sh
```

Run the full season-specific training pipeline:

```bash
bash temporal_transformer/run_season_specific_training.sh
```

The season-specific pipeline trains 16 models:

- 2 variables: `discharge`, `air_temperature`
- 2 sites per variable
- 4 seasons: `winter`, `spring`, `summer`, `fall`

See `train.txt` for command options, model naming, and output locations.

## Saved Outputs

Normal all-season outputs:

- `models_discharge/`
- `models_air_temperature/`
- `lightning_logs_discharge/`
- `lightning_logs_air_temperature/`
- `discharge_*metrics*.csv`
- `air_temperature_*metrics*.csv`
- `discharge_validation_predictions_long.csv`
- `air_temperature_validation_predictions_long.csv`

Season-specific outputs:

- `season_specific_training/models_discharge_season_specific/`
- `season_specific_training/models_air_temperature_season_specific/`
- `season_specific_training/lightning_logs_discharge_season_specific/`
- `season_specific_training/lightning_logs_air_temperature_season_specific/`
- `season_specific_training/metrics/`
- `season_specific_training/predictions/`

## Current Result Summary

Normal all-season discharge models:

| Site | RMSE | MAE | RMSE skill vs persistence | MAE skill vs persistence |
| --- | ---: | ---: | ---: | ---: |
| 3 | 22.37 | 13.25 | 0.031 | 0.007 |
| 4 | 20.49 | 12.66 | 0.089 | 0.115 |

Normal all-season air-temperature model:

| Site | RMSE | MAE | RMSE skill vs persistence | MAE skill vs persistence |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2.95 | 2.16 | 0.634 | 0.651 |

Season-specific training is most useful for discharge, especially winter, fall,
and summer. Spring discharge is mixed and can hurt the pooled result. For site 1
air temperature, the normal all-season model is better overall than
season-specific models. Site 2 air temperature currently has season-specific
results but no matching normal all-season baseline in the committed metrics.

See `metrics.txt` for the detailed metric file map and interpretation notes.

## Project Structure

```text
CS6994FinalProject/
+-- README.md
+-- parquet.txt
+-- train.txt
+-- metrics.txt
+-- Dockerfile
+-- docker-compose.yml
+-- init-scripts/
+-- normalize_data.sql
+-- ingest_csv.py
+-- build_streamflow_training_parquet_v2_multihorizon.ipynb
+-- streamflow_parquet_v2/
+-- air_temperature_parquet_v2/
+-- temporal_transformer/
+-- models_discharge/
+-- models_air_temperature/
+-- lightning_logs_discharge/
+-- lightning_logs_air_temperature/
+-- season_specific_training/
```
