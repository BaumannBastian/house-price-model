-- BigQuery MARTS views for House Price Model
-- Placeholders: {project}, {raw}, {core}, {marts}

CREATE OR REPLACE VIEW `{project}.{marts}.model_leaderboard` AS
SELECT
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
FROM `{project}.{raw}.models`;


CREATE OR REPLACE VIEW `{project}.{marts}.cv_error_by_bucket` AS
WITH base AS (
  SELECT
    y_true,
    y_pred_oof,
    abs_error,
    rel_error,
    model_id,
    model_name,
    model_version,
    created_at_utc,
    CASE
      WHEN y_true < 100000 THEN "0-100k"
      WHEN y_true < 150000 THEN "100k-150k"
      WHEN y_true < 200000 THEN "150k-200k"
      WHEN y_true < 300000 THEN "200k-300k"
      WHEN y_true < 400000 THEN "300k-400k"
      WHEN y_true < 500000 THEN "400k-500k"
      ELSE "500k+"
    END AS price_bucket
  FROM `{project}.{raw}.train_cv_predictions`
)
SELECT
  price_bucket,
  model_name,
  model_version,
  COUNT(1) AS n,
  SQRT(AVG(POW(y_true - y_pred_oof, 2))) AS rmse,
  AVG(abs_error) AS mean_abs_error,
  AVG(rel_error) AS mean_rel_error,
  MAX(abs_error) AS max_abs_error
FROM base
GROUP BY price_bucket, model_name, model_version;


CREATE OR REPLACE VIEW `{project}.{marts}.top_outliers` AS
SELECT
  kaggle_id,
  y_true,
  y_pred_oof,
  abs_error,
  rel_error,
  model_name,
  model_version,
  created_at_utc
FROM `{project}.{raw}.train_cv_predictions`
ORDER BY abs_error DESC
LIMIT 100;