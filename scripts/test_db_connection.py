# ------------------------------
# scripts/test_db_connection.py
#
# In dieser Python-Datei wird die Verbindung zur aktuell konfigurierten
# PostgreSQL-Datenbank getestet (lokal oder Azure), basierend auf den
# Umgebungsvariablen DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
# und DB_SSLMODE.
# ------------------------------


from src.db import get_connection


def main() -> None:
    """
    Testet die DB-Verbindung und gibt grundlegende Infos aus.

    Es wird ein einfacher SELECT auf ``version()``, ``current_database()``
    und ``current_user`` ausgeführt. Schlägt die Verbindung fehl, wird
    eine Exception geworfen.
    """
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version(), current_database(), current_user;")
                version, dbname, user = cur.fetchone()
        print("Verbindung OK ✅")
        print(f"version      : {version}")
        print(f"database name: {dbname}")
        print(f"current user : {user}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
