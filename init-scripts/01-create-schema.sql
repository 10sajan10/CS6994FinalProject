-- ============================================================
-- LRO Environmental Forecasting Pipeline
-- Normalized Schema (ODM2-inspired)
-- Auto-runs on first container startup.
-- ============================================================

-- 1. Owner
CREATE TABLE IF NOT EXISTS owner (
    owner_id        SERIAL PRIMARY KEY,
    name            VARCHAR(255) NOT NULL,
    owner           VARCHAR(255),
    contact_email   VARCHAR(255)
);

-- 2. Site
CREATE TABLE IF NOT EXISTS site (
    site_id         SERIAL PRIMARY KEY,
    site_code       VARCHAR(50) NOT NULL UNIQUE,
    name            VARCHAR(255) NOT NULL,
    description     VARCHAR(500),
    site_type       VARCHAR(100),
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION,
    elevation_m     DOUBLE PRECISION,
    state           VARCHAR(100),
    county          VARCHAR(100)
);

-- 3. Unit
CREATE TABLE IF NOT EXISTS unit (
    unit_id         SERIAL PRIMARY KEY,
    name            VARCHAR(255) NOT NULL,
    symbol          VARCHAR(50),
    definition      VARCHAR(500),
    unit_type       VARCHAR(100)
);

-- 4. Method
CREATE TABLE IF NOT EXISTS method (
    method_id               SERIAL PRIMARY KEY,
    name                    VARCHAR(255) NOT NULL,
    description             VARCHAR(500),
    method_code             VARCHAR(100),
    method_type             VARCHAR(255),
    method_link             VARCHAR(500),
    sensor_manufacturer_name VARCHAR(255),
    sensor_model_name       VARCHAR(255),
    sensor_model_link       VARCHAR(500)
);

-- 5. Variable (references unit and method)
CREATE TABLE IF NOT EXISTS variable (
    variable_id     SERIAL PRIMARY KEY,
    name            VARCHAR(255) NOT NULL,
    definition      VARCHAR(500),
    description     VARCHAR(500),
    variable_type   VARCHAR(100),
    variable_code   VARCHAR(100),
    unit_id         INTEGER REFERENCES unit(unit_id),
    method_id       INTEGER REFERENCES method(method_id)
);

-- 6. Processing Level
CREATE TABLE IF NOT EXISTS processing_level (
    processing_level_id SERIAL PRIMARY KEY,
    code                VARCHAR(50) NOT NULL,
    definition          VARCHAR(255),
    explanation         VARCHAR(500)
);

-- 7. Qualifier
CREATE TABLE IF NOT EXISTS qualifier (
    qualifier_id    SERIAL PRIMARY KEY,
    qualifier_code  VARCHAR(50) NOT NULL,
    description     VARCHAR(500)
);

-- 8. Datastream (the time-series observations)
CREATE TABLE IF NOT EXISTS datastream (
    datastream_id   BIGSERIAL PRIMARY KEY,
    datetime_utc    TIMESTAMP NOT NULL,
    value           DOUBLE PRECISION,
    site_id         INTEGER NOT NULL REFERENCES site(site_id),
    variable_id     INTEGER NOT NULL REFERENCES variable(variable_id),
    owner_id        INTEGER REFERENCES owner(owner_id),
    qualifier_id    INTEGER REFERENCES qualifier(qualifier_id),
    processing_level_id INTEGER REFERENCES processing_level(processing_level_id)
);

-- ============================================================
-- Staging Table (flat/denormalized - all metadata in one table)
-- Ingest raw CSV data here first, then normalize into tables above.
-- ============================================================
CREATE TABLE IF NOT EXISTS staging (
    staging_id                  BIGSERIAL PRIMARY KEY,

    -- Owner info (from CSV header)
    workspace_name              VARCHAR(255),
    owner_name                  VARCHAR(255),
    contact_email               VARCHAR(255),

    -- Site info (from CSV header)
    site_code                   VARCHAR(50),
    site_name                   VARCHAR(255),
    site_description            VARCHAR(500),
    sampling_feature_type       VARCHAR(100),
    site_type                   VARCHAR(100),
    latitude                    DOUBLE PRECISION,
    longitude                   DOUBLE PRECISION,
    elevation_m                 DOUBLE PRECISION,
    elevation_datum             VARCHAR(50),
    state                       VARCHAR(100),
    county                      VARCHAR(100),

    -- Variable info (from CSV header)
    variable_name               VARCHAR(255),
    variable_definition         VARCHAR(500),
    variable_description        VARCHAR(500),
    variable_type               VARCHAR(100),
    variable_code               VARCHAR(100),

    -- Unit info (from CSV header)
    unit_name                   VARCHAR(255),
    unit_symbol                 VARCHAR(50),
    unit_definition             VARCHAR(500),
    unit_type                   VARCHAR(100),

    -- Method info (from CSV header)
    method_name                 VARCHAR(255),
    method_description          VARCHAR(500),
    method_code                 VARCHAR(100),
    method_type                 VARCHAR(255),
    method_link                 VARCHAR(500),
    sensor_manufacturer_name    VARCHAR(255),
    sensor_model_name           VARCHAR(255),
    sensor_model_link           VARCHAR(500),

    -- Processing level info (from CSV header)
    processing_level_code       VARCHAR(50),
    processing_level_definition VARCHAR(255),
    processing_level_explanation VARCHAR(500),

    -- Datastream metadata (from CSV header)
    datastream_name             VARCHAR(500),
    datastream_description      VARCHAR(500),
    observation_type            VARCHAR(100),
    result_type                 VARCHAR(100),
    status                      VARCHAR(50),
    sampled_medium              VARCHAR(100),
    no_data_value               DOUBLE PRECISION,
    intended_time_spacing       DOUBLE PRECISION,
    intended_time_spacing_unit  VARCHAR(50),
    aggregation_statistic       VARCHAR(100),
    time_aggregation_interval   DOUBLE PRECISION,
    time_aggregation_interval_unit VARCHAR(50),

    -- Qualifier info
    qualifier_code              VARCHAR(50),
    qualifier_description       VARCHAR(500),

    -- Actual observation data (from CSV rows)
    datetime_utc                TIMESTAMP,
    value                       DOUBLE PRECISION,

    -- Housekeeping
    source_file                 VARCHAR(255),
    loaded_at                   TIMESTAMP DEFAULT NOW()
);

-- ============================================================
-- Indexes for common query patterns
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_datastream_site       ON datastream (site_id);
CREATE INDEX IF NOT EXISTS idx_datastream_variable   ON datastream (variable_id);
CREATE INDEX IF NOT EXISTS idx_datastream_datetime   ON datastream (datetime_utc);
CREATE INDEX IF NOT EXISTS idx_datastream_owner      ON datastream (owner_id);
CREATE INDEX IF NOT EXISTS idx_site_code             ON site (site_code);
CREATE INDEX IF NOT EXISTS idx_variable_code         ON variable (variable_code);
CREATE INDEX IF NOT EXISTS idx_staging_datetime      ON staging (datetime_utc);
CREATE INDEX IF NOT EXISTS idx_staging_site_code     ON staging (site_code);
CREATE INDEX IF NOT EXISTS idx_staging_variable_code ON staging (variable_code);
