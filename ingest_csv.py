"""
LRO CSV Ingestion Script
Parses metadata from # header rows and inserts observations into the PostgreSQL staging table.
Usage: python ingest_csv.py
"""

import os
import re
import csv
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime

# -------------------------------------------------------------------
# Database connection settings (match docker-compose.yml)
# -------------------------------------------------------------------
DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "dbname": "database",
    "user": "admin",
    "password": "password",
}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
BATCH_SIZE = 5000  # rows per INSERT batch


def parse_metadata(header_lines: list[str]) -> dict:
    """
    Parse the # comment header lines into a flat metadata dictionary.
    Handles the multi-section LRO CSV format by tracking which section
    we're in, so duplicate keys like 'Name' get mapped correctly.
    """
    meta = {}
    current_section = None

    # Map (section, key) -> staging column name
    key_map = {
        # Workspace
        ("Workspace", "Name"):          "workspace_name",
        ("Workspace", "Owner"):         "owner_name",
        ("Workspace", "Contact Email"): "contact_email",
        # Site Information
        ("Site Information", "Name"):                "site_name",
        ("Site Information", "Description"):         "site_description",
        ("Site Information", "SamplingFeatureType"):  "sampling_feature_type",
        ("Site Information", "SamplingFeatureCode"):  "site_code",
        ("Site Information", "SiteType"):             "site_type",
        # Location Information
        ("Location Information", "Latitude"):        "latitude",
        ("Location Information", "Longitude"):       "longitude",
        ("Location Information", "Elevation_m"):     "elevation_m",
        ("Location Information", "ElevationDatum"):  "elevation_datum",
        ("Location Information", "State"):           "state",
        ("Location Information", "County"):          "county",
        # Datastream Information
        ("Datastream Information", "Name"):                      "datastream_name",
        ("Datastream Information", "Description"):               "datastream_description",
        ("Datastream Information", "ObservationType"):           "observation_type",
        ("Datastream Information", "ResultType"):                "result_type",
        ("Datastream Information", "Status"):                    "status",
        ("Datastream Information", "SampledMedium"):             "sampled_medium",
        ("Datastream Information", "NoDataValue"):               "no_data_value",
        ("Datastream Information", "IntendedTimeSpacing"):       "intended_time_spacing",
        ("Datastream Information", "IntendedTimeSpacingUnit"):   "intended_time_spacing_unit",
        ("Datastream Information", "AggregationStatistic"):      "aggregation_statistic",
        ("Datastream Information", "TimeAggregationInterval"):   "time_aggregation_interval",
        ("Datastream Information", "TimeAggregationIntervalUnit"): "time_aggregation_interval_unit",
        # Method Information
        ("Method Information", "Name"):                    "method_name",
        ("Method Information", "Description"):             "method_description",
        ("Method Information", "MethodCode"):              "method_code",
        ("Method Information", "MethodType"):              "method_type",
        ("Method Information", "MethodLink"):              "method_link",
        ("Method Information", "SensorManufacturerName"):  "sensor_manufacturer_name",
        ("Method Information", "SensorModelName"):         "sensor_model_name",
        ("Method Information", "SensorModelLink"):         "sensor_model_link",
        # Observed Property Information
        ("Observed Property Information", "Name"):         "variable_name",
        ("Observed Property Information", "Definition"):   "variable_definition",
        ("Observed Property Information", "Description"):  "variable_description",
        ("Observed Property Information", "VariableType"): "variable_type",
        ("Observed Property Information", "VariableCode"): "variable_code",
        # Unit Information
        ("Unit Information", "Name"):       "unit_name",
        ("Unit Information", "Symbol"):     "unit_symbol",
        ("Unit Information", "Definition"): "unit_definition",
        ("Unit Information", "UnitType"):   "unit_type",
        # Processing Level Information
        ("Processing Level Information", "Code"):        "processing_level_code",
        ("Processing Level Information", "Definition"):  "processing_level_definition",
        ("Processing Level Information", "Explanation"): "processing_level_explanation",
    }

    for line in header_lines:
        stripped = line.strip()

        # Detect section headers like "# Site Information:"
        section_match = re.match(r"^#\s+(.+?):\s*$", stripped)
        if section_match:
            candidate = section_match.group(1).strip()
            # Only update section if it looks like a real section header
            if candidate in {s for (s, _) in key_map}:
                current_section = candidate
            continue

        # Detect key: value lines like "# Name: Logan River ..."
        kv_match = re.match(r"^#\s+(\w[\w\s]*?):\s*(.+)$", stripped)
        if kv_match and current_section:
            key = kv_match.group(1).strip()
            value = kv_match.group(2).strip()
            col = key_map.get((current_section, key))
            if col:
                # Convert "None" / "N/A" to actual None
                if value in ("None", "N/A", ""):
                    value = None
                meta[col] = value

    return meta


def ingest_csv(filepath: str, conn):
    """Read one CSV file, parse metadata + data rows, and INSERT into staging."""
    filename = os.path.basename(filepath)
    print(f"  Processing: {filename}")

    header_lines = []
    data_lines = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                header_lines.append(line)
            else:
                data_lines.append(line)

    # Parse metadata from header comments
    meta = parse_metadata(header_lines)

    # Parse the CSV data rows (first non-comment line is the column header)
    reader = csv.DictReader(data_lines)

    # Determine no_data_value for filtering
    no_data_val = None
    if meta.get("no_data_value"):
        try:
            no_data_val = float(meta["no_data_value"])
        except ValueError:
            pass

    # Build column list matching the staging table
    columns = [
        "workspace_name", "owner_name", "contact_email",
        "site_code", "site_name", "site_description",
        "sampling_feature_type", "site_type",
        "latitude", "longitude", "elevation_m", "elevation_datum",
        "state", "county",
        "variable_name", "variable_definition", "variable_description",
        "variable_type", "variable_code",
        "unit_name", "unit_symbol", "unit_definition", "unit_type",
        "method_name", "method_description", "method_code",
        "method_type", "method_link",
        "sensor_manufacturer_name", "sensor_model_name", "sensor_model_link",
        "processing_level_code", "processing_level_definition",
        "processing_level_explanation",
        "datastream_name", "datastream_description",
        "observation_type", "result_type", "status", "sampled_medium",
        "no_data_value", "intended_time_spacing", "intended_time_spacing_unit",
        "aggregation_statistic", "time_aggregation_interval",
        "time_aggregation_interval_unit",
        "qualifier_code", "qualifier_description",
        "datetime_utc", "value",
        "source_file",
    ]

    # Convert numeric metadata fields
    def safe_float(val):
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    insert_sql = f"""
        INSERT INTO staging ({', '.join(columns)})
        VALUES %s
    """

    batch = []
    row_count = 0

    with conn.cursor() as cur:
        for row in reader:
            timestamp_str = row.get("ResultTime", "").strip()
            value_str = row.get("Result", "").strip()
            qualifier = row.get("ResultQualifiers", "").strip() or None

            # Parse timestamp
            if not timestamp_str:
                continue
            try:
                ts = datetime.fromisoformat(timestamp_str)
            except ValueError:
                continue

            # Parse value
            value = safe_float(value_str)

            record = (
                meta.get("workspace_name"),
                meta.get("owner_name"),
                meta.get("contact_email"),
                meta.get("site_code"),
                meta.get("site_name"),
                meta.get("site_description"),
                meta.get("sampling_feature_type"),
                meta.get("site_type"),
                safe_float(meta.get("latitude")),
                safe_float(meta.get("longitude")),
                safe_float(meta.get("elevation_m")),
                meta.get("elevation_datum"),
                meta.get("state"),
                meta.get("county"),
                meta.get("variable_name"),
                meta.get("variable_definition"),
                meta.get("variable_description"),
                meta.get("variable_type"),
                meta.get("variable_code"),
                meta.get("unit_name"),
                meta.get("unit_symbol"),
                meta.get("unit_definition"),
                meta.get("unit_type"),
                meta.get("method_name"),
                meta.get("method_description"),
                meta.get("method_code"),
                meta.get("method_type"),
                meta.get("method_link"),
                meta.get("sensor_manufacturer_name"),
                meta.get("sensor_model_name"),
                meta.get("sensor_model_link"),
                meta.get("processing_level_code"),
                meta.get("processing_level_definition"),
                meta.get("processing_level_explanation"),
                meta.get("datastream_name"),
                meta.get("datastream_description"),
                meta.get("observation_type"),
                meta.get("result_type"),
                meta.get("status"),
                meta.get("sampled_medium"),
                safe_float(meta.get("no_data_value")),
                safe_float(meta.get("intended_time_spacing")),
                meta.get("intended_time_spacing_unit"),
                meta.get("aggregation_statistic"),
                safe_float(meta.get("time_aggregation_interval")),
                meta.get("time_aggregation_interval_unit"),
                qualifier,      # qualifier_code (from ResultQualifiers column)
                None,           # qualifier_description
                ts,             # datetime_utc
                value,          # value
                filename,       # source_file
            )

            batch.append(record)
            row_count += 1

            if len(batch) >= BATCH_SIZE:
                execute_values(cur, insert_sql, batch)
                batch = []
                print(f"    Inserted {row_count} rows so far...")

        # Insert remaining rows
        if batch:
            execute_values(cur, insert_sql, batch)

    conn.commit()
    print(f"  Done: {row_count} rows inserted from {filename}")
    return row_count


def main():
    if not os.path.isdir(DATA_DIR):
        print(f"Data directory not found: {DATA_DIR}")
        return

    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not csv_files:
        print("No CSV files found in Data/ directory.")
        return

    print(f"Found {len(csv_files)} CSV file(s) in {DATA_DIR}")
    print(f"Connecting to PostgreSQL at localhost:{DB_CONFIG['port']}...")

    conn = psycopg2.connect(**DB_CONFIG)
    total = 0

    try:
        for csv_file in sorted(csv_files):
            filepath = os.path.join(DATA_DIR, csv_file)
            total += ingest_csv(filepath, conn)
        print(f"\nAll done! Total rows inserted: {total}")
    except Exception as e:
        conn.rollback()
        print(f"Error: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
