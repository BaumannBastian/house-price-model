-- ------------------------------------
-- cloud/bigquery/marts_views.sql
--
-- RAW -> CORE -> MARTS Views (BigQuery)
-- Placeholders werden per apply_views.py ersetzt:
--   {project}, {raw}, {core}, {marts}
-- ------------------------------------

-- ----------------------
-- CORE: Models (dedupe + run_ts)
-- created_at_utc in RAW ist INTEGER (Epoch ns/us/ms/s)
-- ----------------------
CREATE OR REPLACE VIEW `{project}.{core}.v_models` AS
WITH base AS (
  SELECT
    *,
    CASE
      WHEN created_at_utc IS NULL THEN NULL
      WHEN created_at_utc >= 1000000000000000000 THEN TIMESTAMP_MICROS(DIV(created_at_utc, 1000))  -- ns -> us
      WHEN created_at_utc >= 1000000000000000 THEN TIMESTAMP_MICROS(created_at_utc)                -- us
      WHEN created_at_utc >= 1000000000000 THEN TIMESTAMP_MILLIS(created_at_utc)                   -- ms
      WHEN created_at_utc >= 1000000000 THEN TIMESTAMP_SECONDS(created_at_utc)                     -- s
      ELSE NULL
    END AS created_at_ts
  FROM `{project}.{raw}.models`
),
dedup AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY id
      ORDER BY created_at_ts DESC
    ) AS rn
  FROM base
)
SELECT
  id,
  name,
  version,
  is_champion,
  cv_rmse_mean,
  cv_rmse_std,
  rmse_test,
  r2_test,
  created_at_ts AS created_at_utc,
  file_path,
  SAFE.PARSE_TIMESTAMP('%Y%m%d_%H%M%S', version) AS run_ts,
  DATE(SAFE.PARSE_TIMESTAMP('%Y%m%d_%H%M%S', version)) AS run_date
FROM dedup
WHERE rn = 1;

-- ----------------------
-- CORE: Train CV Predictions (dedupe + run_ts + price_bucket)
-- created_at_utc in RAW ist INTEGER (Epoch ns/us/ms/s)
-- ----------------------
CREATE OR REPLACE VIEW `{project}.{core}.v_train_cv_predictions` AS
WITH base AS (
  SELECT
    *,
    CASE
      WHEN created_at_utc IS NULL THEN NULL
      WHEN created_at_utc >= 1000000000000000000 THEN TIMESTAMP_MICROS(DIV(created_at_utc, 1000))
      WHEN created_at_utc >= 1000000000000000 THEN TIMESTAMP_MICROS(created_at_utc)
      WHEN created_at_utc >= 1000000000000 THEN TIMESTAMP_MILLIS(created_at_utc)
      WHEN created_at_utc >= 1000000000 THEN TIMESTAMP_SECONDS(created_at_utc)
      ELSE NULL
    END AS created_at_ts
  FROM `{project}.{raw}.train_cv_predictions`
),
dedup AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY model_id, kaggle_id, model_version
      ORDER BY created_at_ts DESC
    ) AS rn
  FROM base
)
SELECT
  kaggle_id,
  y_true,
  y_pred_oof,
  abs_error,
  rel_error,
  model_id,
  model_name,
  model_version,
  created_at_ts AS created_at_utc,
  SAFE.PARSE_TIMESTAMP('%Y%m%d_%H%M%S', model_version) AS run_ts,
  DATE(SAFE.PARSE_TIMESTAMP('%Y%m%d_%H%M%S', model_version)) AS run_date,
  CASE
    WHEN y_true < 100000 THEN '0-100k'
    WHEN y_true < 150000 THEN '100k-150k'
    WHEN y_true < 200000 THEN '150k-200k'
    WHEN y_true < 300000 THEN '200k-300k'
    WHEN y_true < 400000 THEN '300k-400k'
    WHEN y_true < 500000 THEN '400k-500k'
    ELSE '500k+'
  END AS price_bucket,
  CASE
    WHEN y_true < 100000 THEN 0
    WHEN y_true < 150000 THEN 1
    WHEN y_true < 200000 THEN 2
    WHEN y_true < 300000 THEN 3
    WHEN y_true < 400000 THEN 4
    WHEN y_true < 500000 THEN 5
    ELSE 6
  END AS price_bucket_order
FROM dedup
WHERE rn = 1;

-- ----------------------
-- CORE: Predictions (created_at_utc ist in RAW bereits TIMESTAMP)
-- ----------------------
CREATE OR REPLACE VIEW `{project}.{core}.v_predictions` AS
WITH base AS (
  SELECT
    *,
    created_at_utc AS created_at_ts
  FROM `{project}.{raw}.predictions`
),
dedup AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY model_id, kaggle_id, model_version
      ORDER BY created_at_ts DESC
    ) AS rn
  FROM base
)
SELECT
  kaggle_id,
  predicted_price,
  model_id,
  model_name,
  model_version,
  created_at_ts AS created_at_utc,
  SAFE.PARSE_TIMESTAMP('%Y%m%d_%H%M%S', model_version) AS run_ts,
  DATE(SAFE.PARSE_TIMESTAMP('%Y%m%d_%H%M%S', model_version)) AS run_date
FROM dedup
WHERE rn = 1;

-- ----------------------
-- MARTS: Model Leaderboard (alle Modelle)
-- ----------------------
CREATE OR REPLACE VIEW `{project}.{marts}.model_leaderboard` AS
SELECT
  run_ts,
  run_date,
  id,
  name,
  version,
  is_champion,
  cv_rmse_mean,
  cv_rmse_std,
  rmse_test,
  r2_test,
  created_at_utc,
  file_path
FROM `{project}.{core}.v_models`;

-- ----------------------
-- MARTS: Champion pro Run (fallback: bestes cv_rmse_mean)
-- ----------------------
CREATE OR REPLACE VIEW `{project}.{marts}.champion_by_run` AS
SELECT
  run_ts,
  run_date,
  version,
  model_id,
  model_name,
  is_champion,
  cv_rmse_mean,
  cv_rmse_std,
  rmse_test,
  r2_test,
  created_at_utc,
  file_path
FROM (
  SELECT
    run_ts,
    run_date,
    version,
    id AS model_id,
    name AS model_name,
    is_champion,
    cv_rmse_mean,
    cv_rmse_std,
    rmse_test,
    r2_test,
    created_at_utc,
    file_path,
    ROW_NUMBER() OVER (
      PARTITION BY version
      ORDER BY is_champion DESC, cv_rmse_mean ASC
    ) AS rn
  FROM `{project}.{core}.v_models`
)
WHERE rn = 1;

-- ----------------------
-- MARTS: CV Error by Bucket (Aggregation)
-- ----------------------
CREATE OR REPLACE VIEW `{project}.{marts}.cv_error_by_bucket` AS
SELECT
  run_ts,
  run_date,
  model_version,
  model_id,
  model_name,
  price_bucket,
  price_bucket_order,
  COUNT(1) AS n,
  SQRT(AVG(POW(y_true - y_pred_oof, 2))) AS rmse,
  AVG(abs_error) AS mean_abs_error,
  AVG(rel_error) AS mean_rel_error,
  MAX(abs_error) AS max_abs_error
FROM `{project}.{core}.v_train_cv_predictions`
GROUP BY
  run_ts, run_date, model_version, model_id, model_name, price_bucket, price_bucket_order;

-- ----------------------
-- MARTS: Top Outliers (Row-level, pro Modell/Run)
-- ----------------------
CREATE OR REPLACE VIEW `{project}.{marts}.top_outliers` AS
SELECT
  run_ts,
  run_date,
  model_version,
  model_id,
  model_name,
  kaggle_id,
  y_true,
  y_pred_oof,
  abs_error,
  rel_error,
  price_bucket,
  price_bucket_order,
  created_at_utc
FROM (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY model_version, model_id
      ORDER BY abs_error DESC
    ) AS rn
  FROM `{project}.{core}.v_train_cv_predictions`
)
WHERE rn <= 100;

-- ----------------------
-- MARTS: Predictions des neuesten Champion-Runs
-- ----------------------
CREATE OR REPLACE VIEW `{project}.{marts}.predictions_latest_champion` AS
WITH latest AS (
  SELECT
    model_id,
    version AS model_version,
    COALESCE(run_ts, created_at_utc) AS sort_ts
  FROM `{project}.{marts}.champion_by_run`
  ORDER BY sort_ts DESC
  LIMIT 1
)
SELECT
  p.kaggle_id,
  p.predicted_price,
  p.model_id,
  p.model_name,
  p.model_version,
  p.created_at_utc,
  p.run_ts,
  p.run_date
FROM `{project}.{core}.v_predictions` p
JOIN latest l
  ON p.model_id = l.model_id
  AND p.model_version = l.model_version;