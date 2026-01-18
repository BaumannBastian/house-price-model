# ------------------------------------
# src/db.py
#
# In dieser Datei kapseln wir alle Datenbankzugriffe (PostgreSQL via psycopg2).
#
# Schema-Management
# ------------------------------------
# Das DB-Schema wird NICHT in Python erzeugt.
# Stattdessen werden Tabellen/Indizes/Views versioniert ueber Flyway-Migrationen 
# verwaltet (siehe sql/migrations).
#
# Connection-Konfiguration
# ------------------------------------
# Die Verbindung wird ueber ENV-Variablen konfiguriert:
#   DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DB_SSLMODE
#
# Diese ENV-Variablen werden vor dem Python-Start gesetzt (automatisch durch start_dev.ps1):
#   scripts/set_env_local_db.ps1  (lokal, Docker)
#   scripts/set_env_azure_db.ps1  (cloud, Azure) --Artifact
#
# Fehlen ENV-Variablen, greifen Default-Werte (lokale Docker-DB), damit man das Projekt
# schnell starten kann.
#
# Transaktionen & Ressourcen
# ------------------------------------
# Jede Funktion oeffnet eine Connection, nutzt `with conn:` (commit/rollback) und schliesst
# die Connection am Ende zuverlaessig.
# ------------------------------------

from __future__ import annotations

import json
import logging
import os
import psycopg2

from typing import Iterable, Optional


logger = logging.getLogger(__name__)


# --------------------------------------------------------
# Connection / ENV
# --------------------------------------------------------

# Module-level Defaults: os.getenv nur einmal pro Python-Run
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "house_prices")
DB_USER = os.getenv("DB_USER", "house")
DB_PASSWORD = os.getenv("DB_PASSWORD", "house")

# Azure: require | Local: disable
DB_SSLMODE = os.getenv("DB_SSLMODE", "require" if ".database.azure.com" in DB_HOST else "disable")


def get_connection():
    """Erzeugt eine psycopg2-Connection basierend auf der ENV-Konfiguration."""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        sslmode=DB_SSLMODE,
    )


# --------------------------------------------------------
# Models (Model-Runs)
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
    """Legt einen Eintrag in `models` an und gibt die neue model_id zurueck."""
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
                        name,
                        version,
                        file_path,
                        r2_test,
                        rmse_test,
                        mare_test,
                        mre_test,
                        cv_rmse_mean,
                        cv_rmse_std,
                        max_abs_train_error,
                        hyperparams_json,
                        is_champion,
                    ),
                )
                return int(cur.fetchone()[0])
    finally:
        conn.close()


def update_model_file_path(model_id: int, file_path: str) -> None:
    """Setzt/updated den Artifact-Pfad eines Modell-Runs."""
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


def set_champion_model(model_id: int) -> None:
    """Setzt genau ein Modell als Champion (models.is_champion)."""
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE models SET is_champion = FALSE;")
                cur.execute("UPDATE models SET is_champion = TRUE WHERE id = %s;", (model_id,))
    finally:
        conn.close()


def get_current_champion_id() -> Optional[int]:
    """Gibt die model_id des aktuellen Champions zurueck (oder None)."""
    sql = """
    SELECT id
    FROM models
    WHERE is_champion = TRUE
    ORDER BY created_at DESC
    LIMIT 1;
    """

    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                row = cur.fetchone()
                return int(row[0]) if row else None
    finally:
        conn.close()


def get_latest_model_id_by_name(name: str, version: Optional[str] = None) -> Optional[int]:
    """Gibt die neueste model_id fuer einen Modellnamen zurueck (optional: version fixieren)."""
    if version is None:
        sql = """
        SELECT id
        FROM models
        WHERE name = %s
        ORDER BY created_at DESC
        LIMIT 1;
        """
        params = (name,)
    else:
        sql = """
        SELECT id
        FROM models
        WHERE name = %s AND version = %s
        ORDER BY created_at DESC
        LIMIT 1;
        """
        params = (name, version)

    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
                return int(row[0]) if row else None
    finally:
        conn.close()


def get_model_file_path(model_id: int) -> Optional[str]:
    """Liest file_path fuer eine model_id aus (oder None)."""
    sql = """
    SELECT file_path
    FROM models
    WHERE id = %s;
    """

    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (model_id,))
                row = cur.fetchone()
                return str(row[0]) if row and row[0] is not None else None
    finally:
        conn.close()


# --------------------------------------------------------
# Predictions (Kaggle Test)
# --------------------------------------------------------

def insert_predictions(
    kaggle_ids: Iterable[int],
    y_pred: Iterable[float],
    *,
    model_id: Optional[int] = None,
) -> int:
    """Schreibt Predictions in die Tabelle `predictions` und gibt die Anzahl Inserts zurueck."""
    rows = [(int(kid), float(yp), int(model_id) if model_id is not None else None) for kid, yp in zip(kaggle_ids, y_pred)]

    sql = """
    INSERT INTO predictions (kaggle_id, predicted_price, model_id)
    VALUES (%s, %s, %s);
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
# Train CV Predictions (OOF fuer PowerBI)
# --------------------------------------------------------

def insert_train_cv_predictions(
    *,
    kaggle_ids: Iterable[int],
    y_true: Iterable[float],
    y_pred_oof: Iterable[float],
    model_id: int,
) -> int:
    """Out-of-fold (CV) Predictions nach `train_cv_predictions`."""
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
