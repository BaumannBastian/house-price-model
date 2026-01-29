# ------------------------------------
# scripts/bigquery/apply_views.py
#
# Erstellt/aktualisiert CORE und MARTS Views in BigQuery.
#
# Usage
# ------------------------------------
#   $env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\service_account.json"
#   $env:BQ_PROJECT_ID="house-price-model"
#
#   python -m scripts.bigquery.apply_views
# ------------------------------------

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from google.cloud import bigquery


def setup_logging(debug: bool) -> logging.Logger:
    logger = logging.getLogger("bq-views")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG if debug else logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply BigQuery views (CORE/MARTS).")
    p.add_argument("--project", type=str, default=None, help="GCP project id (env: BQ_PROJECT_ID)")
    p.add_argument("--raw", type=str, default="house_prices_raw")
    p.add_argument("--core", type=str, default="house_prices_core")
    p.add_argument("--marts", type=str, default="house_prices_marts")
    p.add_argument("--location", type=str, default="EU")
    p.add_argument("--sql", type=str, default="cloud/bigquery/marts_views.sql")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def _ensure_dataset(client: bigquery.Client, dataset_id: str, location: str, logger: logging.Logger) -> None:
    try:
        client.get_dataset(dataset_id)
        return
    except Exception:
        pass

    ds = bigquery.Dataset(dataset_id)
    if location:
        ds.location = location
    client.create_dataset(ds)
    logger.info("Dataset erstellt: %s", dataset_id)


def main() -> int:
    args = parse_args()
    logger = setup_logging(args.debug)

    project = args.project
    if project is None:
        import os

        project = os.environ.get("BQ_PROJECT_ID")
    if not project:
        raise ValueError("BQ project fehlt. Nutze --project oder env:BQ_PROJECT_ID.")

    client = bigquery.Client(project=project)

    _ensure_dataset(client, f"{project}.{args.core}", args.location, logger)
    _ensure_dataset(client, f"{project}.{args.marts}", args.location, logger)

    sql_path = Path(args.sql)
    sql = sql_path.read_text(encoding="utf-8")
    sql = sql.format(project=project, raw=args.raw, core=args.core, marts=args.marts)

    job = client.query(sql)
    job.result()

    logger.info("Views aktualisiert via: %s", str(sql_path.as_posix()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
