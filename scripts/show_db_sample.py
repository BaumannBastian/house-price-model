# ------------------------------
# scripts/show_db_sample.py
#
# Struktur
# ------------------------------
# Zeigt ein paar Beispielzeilen aus zentralen Tabellen / Views, um den aktuellen
# DB-Zustand schnell zu inspizieren (Models, Predictions).
#
# Voraussetzungen
# ------------------------------
# - DB-ENV-Variablen sind gesetzt (local oder azure).
# - Schema ist via Flyway vorhanden (models/predictions/train_* + View).
#
# Usage
# ------------------------------
# python -m scripts.show_db_sample
# python -m scripts.show_db_sample --limit 10
# ------------------------------

from __future__ import annotations

import argparse
import sys

from src.db import get_connection


def _configure_stdout_utf8() -> None:
    """Macht stdout auf Windows robuster (falls reconfigure verfuegbar ist)."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def show_models(limit: int) -> None:
    sql = """
    SELECT id, name, version, is_champion, created_at
    FROM models
    ORDER BY created_at DESC
    LIMIT %s;
    """

    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (limit,))
                rows = cur.fetchall()

        print(f"\nLetzte {len(rows)} Modelle in 'models':")
        for mid, name, version, is_champion, created_at in rows:
            flag = "*" if is_champion else " "
            print(f"[{mid:3d}] {flag} {name:18s} | {version:20s} | {created_at}")

    finally:
        conn.close()


def show_predictions_with_model(limit: int) -> None:
    sql = """
    SELECT
        id,
        kaggle_id,
        predicted_price,
        model_name,
        model_version,
        prediction_created_at
    FROM v_predictions_with_model
    ORDER BY prediction_created_at DESC
    LIMIT %s;
    """

    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (limit,))
                rows = cur.fetchall()

        print(f"\nLetzte {len(rows)} Eintraege aus 'v_predictions_with_model':")
        for pid, kaggle_id, predicted_price, model_name, model_version, created_at in rows:
            print(
                f"[{pid:4d}] KaggleId={kaggle_id:4d} | "
                f"pred={predicted_price:10.2f} | "
                f"model={model_name:12s} ({model_version}) | "
                f"{created_at}"
            )

    finally:
        conn.close()


def main() -> None:
    _configure_stdout_utf8()

    parser = argparse.ArgumentParser(description="Zeigt Beispielzeilen aus der DB.")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    show_models(limit=args.limit)
    show_predictions_with_model(limit=args.limit)


if __name__ == "__main__":
    main()
