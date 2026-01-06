import os
import json
import logging
from typing import Iterable, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


# --------------------------------------------------------
# Connection / ENV
# --------------------------------------------------------

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "house_prices")
DB_USER = os.getenv("DB_USER", "house")
DB_PASSWORD = os.getenv("DB_PASSWORD", "house")

# Azure: require | Local: disable
DB_SSLMODE = os.getenv("DB_SSLMODE", "require" if ".database.azure.com" in DB_HOST else "disable")


def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        sslmode=DB_SSLMODE,
    )


# --------------------------------------------------------
# Schema Init
# --------------------------------------------------------

def init_predictions_table():
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS predictions (
        id SERIAL PRIMARY KEY,
        kaggle_id INTEGER NOT NULL,
        predicted_price DOUBLE PRECISION NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """

    # Backward compatible: falls table schon ohne model_id existiert
    alter_add_model_id_sql = """
    ALTER TABLE predictions
    ADD COLUMN IF NOT EXISTS model_id INTEGER;
    """

    idx_model_sql = """
    CREATE INDEX IF NOT EXISTS idx_predictions_model_id
    ON predictions (model_id);
    """

    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(create_table_sql)
                cur.execute(alter_add_model_id_sql)
                cur.execute(idx_model_sql)
    finally:
        conn.close()


def init_train_predictions_table():
    create_sql = """
    CREATE TABLE IF NOT EXISTS train_predictions (
        id SERIAL PRIMARY KEY,
        kaggle_id INTEGER NOT NULL,
        saleprice_true DOUBLE PRECISION NOT NULL,
        predicted_price DOUBLE PRECISION NOT NULL,
        abs_error DOUBLE PRECISION NOT NULL,
        rel_error DOUBLE PRECISION NOT NULL,
        model_id INTEGER NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """

    idx_model_sql = """
    CREATE INDEX IF NOT EXISTS idx_train_predictions_model_id
    ON train_predictions (model_id);
    """

    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(create_sql)
                cur.execute(idx_model_sql)
    finally:
        conn.close()


def init_train_cv_predictions_table():
    create_sql = """
    CREATE TABLE IF NOT EXISTS train_cv_predictions (
        id SERIAL PRIMARY KEY,
        kaggle_id INTEGER NOT NULL,
        saleprice_true DOUBLE PRECISION NOT NULL,
        predicted_price DOUBLE PRECISION NOT NULL,
        abs_error DOUBLE PRECISION NOT NULL,
        rel_error DOUBLE PRECISION NOT NULL,
        model_id INTEGER NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """

    idx_model_sql = """
    CREATE INDEX IF NOT EXISTS idx_train_cv_predictions_model_id
    ON train_cv_predictions (model_id);
    """

    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(create_sql)
                cur.execute(idx_model_sql)
    finally:
        conn.close()


def init_models_table():
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS models (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        version TEXT NOT NULL,
        file_path TEXT,

        r2_test DOUBLE PRECISION,
        rmse_test DOUBLE PRECISION,
        mare_test DOUBLE PRECISION,
        mre_test DOUBLE PRECISION,

        cv_rmse_mean DOUBLE PRECISION,
        cv_rmse_std DOUBLE PRECISION,

        max_abs_train_error DOUBLE PRECISION,

        hyperparams JSONB,
        is_champion BOOLEAN NOT NULL DEFAULT FALSE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(create_table_sql)
    finally:
        conn.close()


def init_predictions_view():
    create_view_sql = """
    CREATE OR REPLACE VIEW v_predictions_with_model AS
    SELECT
        p.id AS prediction_id,
        p.kaggle_id,
        p.predicted_price,
        p.model_id,
        p.created_at AS prediction_created_at,

        m.name AS model_name,
        m.version AS model_version,
        m.is_champion AS model_is_champion,
        m.created_at AS model_created_at
    FROM predictions p
    LEFT JOIN models m ON p.model_id = m.id;
    """
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(create_view_sql)
    finally:
        conn.close()


# --------------------------------------------------------
# Existing helper (kept)
# --------------------------------------------------------

def insert_train_predictions(rows: list[tuple]):
    """
    Backward compatible helper:
    rows: [(kaggle_id, y_true, y_pred, abs_error, rel_error, model_id), ...]
    """
    sql = """
    INSERT INTO train_predictions
        (kaggle_id, saleprice_true, predicted_price, abs_error, rel_error, model_id)
    VALUES (%s, %s, %s, %s, %s, %s);
    """
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.executemany(sql, rows)
    finally:
        conn.close()


# --------------------------------------------------------
# MISSING FUNCTIONS (needed by train.py)  ✅
# --------------------------------------------------------

def insert_model(
    *,
    name: str,
    version: str,
    file_path: Optional[str],
    r2_test: Optional[float],
    rmse_test: Optional[float],
    mare_test: Optional[float],
    mre_test: Optional[float],
    cv_rmse_mean: Optional[float],
    cv_rmse_std: Optional[float],
    max_abs_train_error: Optional[float],
    hyperparams: Optional[dict],
    is_champion: bool = False,
) -> int:
    """
    Legt einen Eintrag in models an und gibt model_id zurück.
    """
    sql = """
    INSERT INTO models
        (name, version, file_path,
         r2_test, rmse_test, mare_test, mre_test,
         cv_rmse_mean, cv_rmse_std,
         max_abs_train_error,
         hyperparams,
         is_champion)
    VALUES
        (%s, %s, %s,
         %s, %s, %s, %s,
         %s, %s,
         %s,
         %s,
         %s)
    RETURNING id;
    """
    hyperparams_json = json.dumps(hyperparams) if hyperparams is not None else None

    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        name, version, file_path,
                        r2_test, rmse_test, mare_test, mre_test,
                        cv_rmse_mean, cv_rmse_std,
                        max_abs_train_error,
                        hyperparams_json,
                        is_champion,
                    ),
                )
                return int(cur.fetchone()[0])
    finally:
        conn.close()


def update_model_file_path(model_id: int, file_path: str):
    sql = """
    UPDATE models
    SET file_path = %s
    WHERE id = %s;
    """
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (file_path, model_id))
    finally:
        conn.close()


def set_champion_model(model_id: int):
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE models SET is_champion = FALSE;")
                cur.execute("UPDATE models SET is_champion = TRUE WHERE id = %s;", (model_id,))
    finally:
        conn.close()


def insert_train_cv_predictions(
    *,
    kaggle_ids: Iterable[int],
    y_true: Iterable[float],
    y_pred_oof: Iterable[float],
    model_id: int,
) -> int:
    """
    Out-of-fold (CV) Predictions nach train_cv_predictions.

    rel_error = (pred - true)/true (signiert)
    abs_error separat für Visuals.
    """
    rows = []
    for kid, yt, yp in zip(kaggle_ids, y_true, y_pred_oof):
        yt_f = float(yt)
        yp_f = float(yp)
        abs_err = float(abs(yp_f - yt_f))
        rel_err = float((yp_f - yt_f) / yt_f) if yt_f != 0 else 0.0
        rows.append((int(kid), yt_f, yp_f, abs_err, rel_err, int(model_id)))

    sql = """
    INSERT INTO train_cv_predictions
        (kaggle_id, saleprice_true, predicted_price, abs_error, rel_error, model_id)
    VALUES (%s, %s, %s, %s, %s, %s);
    """

    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.executemany(sql, rows)
        return len(rows)
    finally:
        conn.close()


# --------------------------------------------------------
# DB init wrapper
# --------------------------------------------------------

def init_db():
    init_models_table()
    init_predictions_table()
    init_train_predictions_table()
    init_train_cv_predictions_table()
    try:
        init_predictions_view()
    except Exception:
        logger.warning("Konnte v_predictions_with_model nicht initialisieren (ok).")
