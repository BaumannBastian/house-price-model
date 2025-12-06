-- ------------------------------
-- sql/schema.sql
--
-- House Price Project - Datenbankschema
-- Tabellen: models, predictions, Indizes, Foreign Key, View
-- ------------------------------

-- 1. Tabelle 'models' (Modell-Registry)
CREATE TABLE IF NOT EXISTS models (
    id           SERIAL PRIMARY KEY,
    name         TEXT NOT NULL,              -- z.B. 'HistGBR_log'
    version      TEXT NOT NULL,              -- z.B. '20251204-171745'
    file_path    TEXT NOT NULL,              -- Pfad zur .joblib-Datei

    r2_test      DOUBLE PRECISION,
    rmse_test    DOUBLE PRECISION,
    mare_test    DOUBLE PRECISION,
    cv_rmse_mean DOUBLE PRECISION,
    cv_rmse_std  DOUBLE PRECISION,

    hyperparams  JSONB,                      -- Hyperparameter als JSON

    is_champion  BOOLEAN NOT NULL DEFAULT FALSE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2. Tabelle 'predictions' (Prediction-Store)
CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL PRIMARY KEY,
    kaggle_id       INTEGER NOT NULL,        -- Id aus Kaggle-Testdaten
    predicted_price DOUBLE PRECISION NOT NULL,
    model_id        INTEGER,                 -- Referenz auf models.id (Champion o.ä.)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 3. Foreign Key von predictions.model_id -> models.id
--    (idempotent über DROP CONSTRAINT IF EXISTS)
ALTER TABLE predictions
DROP CONSTRAINT IF EXISTS predictions_model_id_fkey;

ALTER TABLE predictions
ADD CONSTRAINT predictions_model_id_fkey
FOREIGN KEY (model_id) REFERENCES models(id);

-- 4. Indizes für typische Abfragen

-- Schnell nach Modell filtern
CREATE INDEX IF NOT EXISTS idx_predictions_model_id
    ON predictions(model_id);

-- Schnell nach Kaggle-Id suchen
CREATE INDEX IF NOT EXISTS idx_predictions_kaggle_id
    ON predictions(kaggle_id);

-- Schnell aktuellen Champion finden
CREATE INDEX IF NOT EXISTS idx_models_is_champion_created_at
    ON models(is_champion, created_at DESC);

-- 5. View für komfortables Joinen von Predictions und Modell-Infos
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