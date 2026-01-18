# ------------------------------
# scripts/test_db_connection.py
#
# Struktur
# ------------------------------
# Minimaler, robuster Healthcheck fuer die aktuell konfigurierte DB-Verbindung.
#
# Basis
# ------------------------------
# - nutzt src.db.get_connection() als Single Source of Truth fuer Connection-Details
# - ExitCode 0 = OK, ExitCode 1 = Fehler
#
# Usage
# ------------------------------
# python -m scripts.test_db_connection
# python -m scripts.test_db_connection --quiet
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


def main() -> int:
    _configure_stdout_utf8()

    parser = argparse.ArgumentParser(description="Testet die DB-Verbindung (ExitCode 0/1).")
    parser.add_argument("--quiet", action="store_true", help="Keine Ausgabe, nur ExitCode.")
    args = parser.parse_args()

    conn = None
    try:
        conn = get_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version(), current_database(), current_user;")
                version, dbname, user = cur.fetchone()

        if not args.quiet:
            print("Verbindung OK")
            print(f"version      : {version}")
            print(f"database name: {dbname}")
            print(f"current user : {user}")

        return 0

    except Exception as e:
        if not args.quiet:
            print("DB-Verbindung fehlgeschlagen:", str(e))
        return 1

    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
