-- ------------------------------
-- V3__gold_views.sql
--
-- Gold Views (Semantic Layer) fuer Analytics/BI und optionale Exports.
--
-- Design
-- --------------------------------
-- - Views sind "source of truth" fuer KPI-/Bucket-Logik.
-- - PowerBI kann direkt auf diese Views gehen.
-- - scripts/db/export_warehouse_views.py exportiert die Views optional als Parquet/CSV.
-- ------------------------------


-- --------------------------------------------------------
-- 1) Model Leaderboard
-- --------------------------------------------------------
CREATE OR REPLACE VIEW v_gold_model_leaderboard AS
SELECT
    id                  AS model_id,
    name                AS model_name,
    version             AS model_version,
    is_champion         AS is_champion,

    r2_test             AS r2_test,
    rmse_test           AS rmse_test,
    mare_test           AS mare_test,
    mre_test            AS mre_test,

    cv_rmse_mean        AS cv_rmse_mean,
    cv_rmse_std         AS cv_rmse_std,

    max_abs_train_error AS max_abs_train_error,

    created_at          AS created_at
FROM models;


-- --------------------------------------------------------
-- 2) CV Error by Price Bucket
-- --------------------------------------------------------
CREATE OR REPLACE VIEW v_gold_cv_error_by_bucket AS
SELECT
    m.id          AS model_id,
    m.name        AS model_name,
    m.version     AS model_version,
    m.is_champion AS is_champion,

    b.bucket_order AS bucket_order,
    b.bucket_label AS bucket_label,
    b.bucket_lower AS bucket_lower,
    b.bucket_upper AS bucket_upper,

    COUNT(*) AS n_rows,

    AVG(t.abs_error) AS mae,
    SQRT(AVG(POWER(t.predicted_price - t.saleprice_true, 2))) AS rmse,
    AVG(ABS(t.rel_error)) AS mare,

    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY t.abs_error) AS p90_abs_error,
    MAX(t.abs_error) AS max_abs_error

FROM train_cv_predictions t
JOIN models m ON m.id = t.model_id

CROSS JOIN LATERAL (
    SELECT
        CASE
            WHEN t.saleprice_true < 100000 THEN 1
            WHEN t.saleprice_true < 150000 THEN 2
            WHEN t.saleprice_true < 200000 THEN 3
            WHEN t.saleprice_true < 250000 THEN 4
            WHEN t.saleprice_true < 300000 THEN 5
            WHEN t.saleprice_true < 400000 THEN 6
            ELSE 7
        END AS bucket_order,

        CASE
            WHEN t.saleprice_true < 100000 THEN '0-100k'
            WHEN t.saleprice_true < 150000 THEN '100-150k'
            WHEN t.saleprice_true < 200000 THEN '150-200k'
            WHEN t.saleprice_true < 250000 THEN '200-250k'
            WHEN t.saleprice_true < 300000 THEN '250-300k'
            WHEN t.saleprice_true < 400000 THEN '300-400k'
            ELSE '400k+'
        END AS bucket_label,

        CASE
            WHEN t.saleprice_true < 100000 THEN 0
            WHEN t.saleprice_true < 150000 THEN 100000
            WHEN t.saleprice_true < 200000 THEN 150000
            WHEN t.saleprice_true < 250000 THEN 200000
            WHEN t.saleprice_true < 300000 THEN 250000
            WHEN t.saleprice_true < 400000 THEN 300000
            ELSE 400000
        END AS bucket_lower,

        CASE
            WHEN t.saleprice_true < 100000 THEN 100000
            WHEN t.saleprice_true < 150000 THEN 150000
            WHEN t.saleprice_true < 200000 THEN 200000
            WHEN t.saleprice_true < 250000 THEN 250000
            WHEN t.saleprice_true < 300000 THEN 300000
            WHEN t.saleprice_true < 400000 THEN 400000
            ELSE NULL
        END AS bucket_upper
) b

GROUP BY
    m.id, m.name, m.version, m.is_champion,
    b.bucket_order, b.bucket_label, b.bucket_lower, b.bucket_upper;


-- --------------------------------------------------------
-- 3) Top Outliers (per model)
-- --------------------------------------------------------
CREATE OR REPLACE VIEW v_gold_top_outliers AS
SELECT *
FROM (
    SELECT
        m.id          AS model_id,
        m.name        AS model_name,
        m.version     AS model_version,
        m.is_champion AS is_champion,

        t.kaggle_id       AS kaggle_id,
        t.saleprice_true  AS saleprice_true,
        t.predicted_price AS predicted_price,
        t.abs_error       AS abs_error,
        t.rel_error       AS rel_error,
        t.created_at      AS created_at,

        ROW_NUMBER() OVER (PARTITION BY t.model_id ORDER BY t.abs_error DESC) AS error_rank
    FROM train_cv_predictions t
    JOIN models m ON m.id = t.model_id
) ranked
WHERE ranked.error_rank <= 50;