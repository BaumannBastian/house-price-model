# ------------------------------
# scripts/init_schema.py
#
# In dieser Python-Datei wird das vollständige Datenbankschema aus
# sql/schema.sql gegen die aktuell konfigurierte PostgreSQL-Datenbank
# angewendet (lokal oder Azure).
#
# Voraussetzung:
# - Die DB-Verbindung ist über Umgebungsvariablen (DB_HOST, DB_PORT,
#   DB_NAME, DB_USER, DB_PASSWORD, DB_SSLMODE) korrekt gesetzt.
# - Die Datei sql/schema.sql existiert im Projektroot.
# ------------------------------

from pathlib import Path

from src.db import get_connection


SCHEMA_PATH = Path("sql/schema.sql")


def main() -> None:
    """
    Liest die Datei ``sql/schema.sql`` ein und führt den kompletten
    SQL-Inhalt auf der aktuellen Datenbank aus.

    Dadurch werden u.a. Tabellen, Indizes und Views angelegt. Das
    Skript ist idempotent, sofern in der Schema-Datei IF NOT EXISTS
    bzw. CREATE OR REPLACE verwendet wird.
    """
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema-Datei nicht gefunden: {SCHEMA_PATH}")

    sql = SCHEMA_PATH.read_text(encoding="utf-8")

    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql)
        print(f"Schema aus {SCHEMA_PATH} erfolgreich angewendet ✅")
    finally:
        conn.close()


if __name__ == "__main__":
    main()