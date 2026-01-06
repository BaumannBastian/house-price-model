-- ------------------------------
-- sql/schema.sql
--
-- House Price Project - Datenbankschema
-- Tabellen:
--   - models
--   - predictions
--   - train_predictions
--   - train_cv_predictions
-- View:
--   - v_predictions_with_model
-- ------------------------------

CREATE TABLE IF NOT EXISTS models (
    id                  SERIAL PRIMARY KEY,
    name                TEXT NOT NULL,
    version             TEXT NOT NULL,
    file_path           TEXT,

    r2_test             DOUBLE PRECISION,
    rmse_test           DOUBLE PRECISION,
    mare_test           DOUBLE PRECISION,
    mre_test            DOUBLE PRECISION,

    cv_rmse_mean        DOUBLE PRECISION,
    cv_rmse_std         DOUBLE PRECISION,

    max_abs_train_error DOUBLE PRECISION,

    hyperparams         JSONB,
    is_champion         BOOLEAN NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_models_name_version
    ON models(name, version);

CREATE INDEX IF NOT EXISTS idx_models_is_champion_created_at
    ON models(is_champion, created_at DESC);

CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL PRIMARY KEY,
    kaggle_id       INTEGER NOT NULL,
    predicted_price DOUBLE PRECISION NOT NULL,
    model_id        INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    FOREIGN KEY (model_id) REFERENCES models(id)
);

CREATE INDEX IF NOT EXISTS idx_predictions_model_id
    ON predictions(model_id);

CREATE INDEX IF NOT EXISTS idx_predictions_kaggle_id
    ON predictions(kaggle_id);

CREATE TABLE IF NOT EXISTS train_predictions (
    id              SERIAL PRIMARY KEY,
    kaggle_id       INTEGER NOT NULL,
    saleprice_true  DOUBLE PRECISION NOT NULL,
    predicted_price DOUBLE PRECISION NOT NULL,
    abs_error       DOUBLE PRECISION NOT NULL,
    rel_error       DOUBLE PRECISION NOT NULL,
    model_id        INTEGER NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    FOREIGN KEY (model_id) REFERENCES models(id)
);

CREATE INDEX IF NOT EXISTS idx_train_predictions_model_id
    ON train_predictions(model_id);

CREATE INDEX IF NOT EXISTS idx_train_predictions_kaggle_id
    ON train_predictions(kaggle_id);

CREATE TABLE IF NOT EXISTS train_cv_predictions (
    id              SERIAL PRIMARY KEY,
    kaggle_id       INTEGER NOT NULL,
    saleprice_true  DOUBLE PRECISION NOT NULL,
    predicted_price DOUBLE PRECISION NOT NULL,
    abs_error       DOUBLE PRECISION NOT NULL,
    rel_error       DOUBLE PRECISION NOT NULL,
    model_id        INTEGER NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    FOREIGN KEY (model_id) REFERENCES models(id)
);

CREATE INDEX IF NOT EXISTS idx_train_cv_predictions_model_id
    ON train_cv_predictions(model_id);

CREATE INDEX IF NOT EXISTS idx_train_cv_predictions_kaggle_id
    ON train_cv_predictions(kaggle_id);

CREATE OR REPLACE VIEW v_predictions_with_model AS
SELECT
    p.id,
    p.kaggle_id,
    p.predicted_price,
    p.model_id,
    m.name       AS model_name,
    m.version    AS model_version,
    p.created_at AS prediction_created_at
FROM predictions p
LEFT JOIN models m
  ON p.model_id = m.id;