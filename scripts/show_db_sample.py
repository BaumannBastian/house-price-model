# ------------------------------
# scripts/show_db_sample.py
#
# In dieser Python-Datei werden einige Beispielzeilen aus den
# Tabellen bzw. der View der Azure-PostgreSQL-Datenbank gelesen,
# um den aktuellen Zustand (Modelle, Predictions) zu inspizieren.
#
# Voraussetzung:
# - DB-ENV-Variablen sind gesetzt (z.B. via scripts/set_env_azure_db.ps1).
# ------------------------------

from src.db import get_connection


def show_models(limit: int = 5) -> None:
    """
    Gibt die letzten ``limit`` Einträge aus der Tabelle ``models`` aus.
    """
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
        for row in rows:
            mid, name, version, is_champion, created_at = row
            champ_flag = "🏆" if is_champion else " "
            print(f"[{mid:3d}] {champ_flag} {name:15s} | {version:20s} | {created_at}")
    finally:
        conn.close()


def show_predictions_with_model(limit: int = 5) -> None:
    """
    Gibt einige Zeilen aus der View ``v_predictions_with_model`` aus.
    """
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
        print(f"\nLetzte {len(rows)} Einträge aus 'v_predictions_with_model':")
        for row in rows:
            (
                pid,
                kaggle_id,
                predicted_price,
                model_name,
                model_version,
                created_at,
            ) = row
            print(
                f"[{pid:4d}] KaggleId={kaggle_id:4d} | "
                f"pred={predicted_price:10.2f} | "
                f"model={model_name:12s} ({model_version}) | "
                f"{created_at}"
            )
    finally:
        conn.close()


def main() -> None:
    show_models(limit=5)
    show_predictions_with_model(limit=5)


if __name__ == "__main__":
    main()