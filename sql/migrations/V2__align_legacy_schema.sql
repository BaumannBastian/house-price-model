-- ------------------------------
-- V2__align_legacy_schema.sql
--
-- Ziel:
-- Diese Migration bringt eine bereits existierende (Legacy) DB auf den aktuellen Stand,
-- ohne dass wir dafür ein Docker-Volume resetten müssen.
--
-- Design:
-- - defensiv / idempotent: ADD COLUMN IF NOT EXISTS, CREATE INDEX IF NOT EXISTS
-- - Foreign Keys werden nur ergänzt, wenn noch kein FK zur Tabelle models existiert
-- - View wird bewusst ohne "Spalten-Umbenennung" definiert, weil Postgres bei
--   CREATE OR REPLACE VIEW keine Umbenennung bestehender View-Spalten erlaubt.
--
-- ------------------------------

-- --------------------------------------------------------
-- 1) Columns
-- --------------------------------------------------------
ALTER TABLE predictions
    ADD COLUMN IF NOT EXISTS model_id INTEGER;


-- --------------------------------------------------------
-- 2) Indexes
-- --------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_models_name_version
    ON models(name, version);

CREATE INDEX IF NOT EXISTS idx_models_is_champion_created_at
    ON models(is_champion, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_predictions_model_id
    ON predictions(model_id);

CREATE INDEX IF NOT EXISTS idx_predictions_kaggle_id
    ON predictions(kaggle_id);

CREATE INDEX IF NOT EXISTS idx_train_predictions_model_id
    ON train_predictions(model_id);

CREATE INDEX IF NOT EXISTS idx_train_predictions_kaggle_id
    ON train_predictions(kaggle_id);

CREATE INDEX IF NOT EXISTS idx_train_cv_predictions_model_id
    ON train_cv_predictions(model_id);

CREATE INDEX IF NOT EXISTS idx_train_cv_predictions_kaggle_id
    ON train_cv_predictions(kaggle_id);


-- --------------------------------------------------------
-- 3) Foreign Keys (nur falls noch nicht vorhanden)
-- --------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint c
        WHERE c.contype = 'f'
          AND c.conrelid = 'predictions'::regclass
          AND c.confrelid = 'models'::regclass
    ) THEN
        ALTER TABLE predictions
            ADD CONSTRAINT fk_predictions_model_id
            FOREIGN KEY (model_id)
            REFERENCES models(id);
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint c
        WHERE c.contype = 'f'
          AND c.conrelid = 'train_predictions'::regclass
          AND c.confrelid = 'models'::regclass
    ) THEN
        ALTER TABLE train_predictions
            ADD CONSTRAINT fk_train_predictions_model_id
            FOREIGN KEY (model_id)
            REFERENCES models(id);
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint c
        WHERE c.contype = 'f'
          AND c.conrelid = 'train_cv_predictions'::regclass
          AND c.confrelid = 'models'::regclass
    ) THEN
        ALTER TABLE train_cv_predictions
            ADD CONSTRAINT fk_train_cv_predictions_model_id
            FOREIGN KEY (model_id)
            REFERENCES models(id);
    END IF;
END $$;


-- --------------------------------------------------------
-- 4) View (keine Umbenennung bestehender Spalten!)
-- --------------------------------------------------------
CREATE OR REPLACE VIEW v_predictions_with_model AS
SELECT
    p.id,
    p.kaggle_id,
    p.predicted_price,
    p.model_id,
    m.name    AS model_name,
    m.version AS model_version,
    p.created_at AS prediction_created_at
FROM predictions p
LEFT JOIN models m
  ON p.model_id = m.id;
