-- ============================================================
-- LRO Database Normalization Script
-- Moves data from 'staging' to normalized tables.
-- ============================================================

BEGIN;

-- 1. Owner
INSERT INTO owner (name, owner, contact_email)
SELECT DISTINCT owner_name, owner_name, contact_email
FROM staging
WHERE owner_name IS NOT NULL
ON CONFLICT DO NOTHING;

-- 2. Site
INSERT INTO site (site_code, name, description, site_type, latitude, longitude, elevation_m, state, county)
SELECT DISTINCT 
    site_code, 
    site_name, 
    site_description, 
    site_type, 
    latitude, 
    longitude, 
    elevation_m, 
    state, 
    county
FROM staging
WHERE site_code IS NOT NULL
ON CONFLICT (site_code) DO NOTHING;

-- 3. Unit
INSERT INTO unit (name, symbol, definition, unit_type)
SELECT DISTINCT unit_name, unit_symbol, unit_definition, unit_type
FROM staging
WHERE unit_name IS NOT NULL
ON CONFLICT DO NOTHING;

-- 4. Method
INSERT INTO method (name, description, method_code, method_type, method_link, sensor_manufacturer_name, sensor_model_name, sensor_model_link)
SELECT DISTINCT 
    method_name, 
    method_description, 
    method_code, 
    method_type, 
    method_link, 
    sensor_manufacturer_name, 
    sensor_model_name, 
    sensor_model_link
FROM staging
WHERE method_name IS NOT NULL
ON CONFLICT DO NOTHING;

-- 5. Processing Level
INSERT INTO processing_level (code, definition, explanation)
SELECT DISTINCT processing_level_code, processing_level_definition, processing_level_explanation
FROM staging
WHERE processing_level_code IS NOT NULL
ON CONFLICT DO NOTHING;

-- 6. Qualifier
INSERT INTO qualifier (qualifier_code, description)
SELECT DISTINCT qualifier_code, qualifier_description
FROM staging
WHERE qualifier_code IS NOT NULL
ON CONFLICT DO NOTHING;

-- 7. Variable (requires unit and method IDs)
INSERT INTO variable (name, definition, description, variable_type, variable_code, unit_id, method_id)
SELECT DISTINCT 
    s.variable_name, 
    s.variable_definition, 
    s.variable_description, 
    s.variable_type, 
    s.variable_code, 
    u.unit_id, 
    m.method_id
FROM staging s
JOIN unit u ON s.unit_name = u.name
JOIN method m ON s.method_name = m.name
WHERE s.variable_code IS NOT NULL
ON CONFLICT DO NOTHING;

-- 8. Datastream (Observations)
-- This is the table holding the actual 1.16M rows.
INSERT INTO datastream (datetime_utc, value, site_id, variable_id, owner_id, qualifier_id, processing_level_id)
SELECT 
    s.datetime_utc, 
    s.value, 
    st.site_id, 
    v.variable_id, 
    o.owner_id, 
    q.qualifier_id, 
    pl.processing_level_id
FROM staging s
JOIN site st ON s.site_code = st.site_code
JOIN unit u ON s.unit_name = u.name
JOIN method m ON s.method_name = m.name
JOIN variable v ON s.variable_code = v.variable_code AND v.unit_id = u.unit_id AND v.method_id = m.method_id
LEFT JOIN owner o ON s.owner_name = o.name
LEFT JOIN qualifier q ON s.qualifier_code = q.qualifier_code
LEFT JOIN processing_level pl ON s.processing_level_code = pl.code;

COMMIT;

-- Verification counts
SELECT 'owner' as table_name, count(*) FROM owner
UNION ALL
SELECT 'site', count(*) FROM site
UNION ALL
SELECT 'unit', count(*) FROM unit
UNION ALL
SELECT 'method', count(*) FROM method
UNION ALL
SELECT 'variable', count(*) FROM variable
UNION ALL
SELECT 'datastream', count(*) FROM datastream;
