# ------------------------------------
# scripts/bigquery/load_raw_tables.py
#
# Lädt reproduzierbare RAW-Exports aus data/warehouse/raw nach BigQuery.
#
# Erwartete Dateien (Parquet):
# - data/warehouse/raw/models.parquet
# - data/warehouse/raw/train_cv_predictions.parquet
# - data/warehouse/raw/predictions.parquet (optional)
#
# Usage
# ------------------------------------
# PowerShell (Windows):
#   $env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\service_account.json"
#   $env:BQ_PROJECT_ID="house-price-model"
#   python -m scripts.bigquery.load_raw_tables --dataset house_prices_raw
#
# ------------------------------------

from __future__ import annotations

import argparse

import logging
from pathlib import Path
from typing import Optional

from google.cloud import bigquery


def setup_logging(debug: bool) -> logging.Logger:
    logger = logging.getLogger("bq-load")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG if debug else logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load RAW parquet exports into BigQuery.")
    p.add_argument("--project", type=str, default=None, help="GCP project id (env: BQ_PROJECT_ID)")
    p.add_argument("--dataset", type=str, default="house_prices_raw")
    p.add_argument("--input-dir", type=str, default="data/warehouse/raw")
    p.add_argument("--write-disposition", choices=["truncate", "append"], default="truncate")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def _write_disposition(arg: str) -> str:
    return bigquery.WriteDisposition.WRITE_TRUNCATE if arg == "truncate" else bigquery.WriteDisposition.WRITE_APPEND


def _ensure_dataset(client: bigquery.Client, dataset_id: str, location: Optional[str], logger: logging.Logger) -> None:
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


def _load_parquet(
    client: bigquery.Client,
    dataset: str,
    table: str,
    parquet_path: Path,
    write_disp: str,
    logger: logging.Logger,
) -> None:
    table_id = f"{client.project}.{dataset}.{table}"

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET,
        write_disposition=write_disp,
    )

    with parquet_path.open("rb") as f:
        job = client.load_table_from_file(f, table_id, job_config=job_config)

    job.result()
    logger.info("Loaded %s -> %s", str(parquet_path.as_posix()), table_id)


def main() -> int:
    args = parse_args()
    logger = setup_logging(args.debug)

    project = args.project
    if project is None:
        import os

        project = os.environ.get("BQ_PROJECT_ID")
    if not project:
        raise ValueError("BQ project fehlt. Nutze --project oder env:BQ_PROJECT_ID.")

    input_dir = Path(args.input_dir)
    write_disp = _write_disposition(args.write_disposition)

    client = bigquery.Client(project=project)

    dataset_id = f"{project}.{args.dataset}"
    _ensure_dataset(client, dataset_id, location=None, logger=logger)

    mapping = {
        "models": input_dir / "models.parquet",
        "train_cv_predictions": input_dir / "train_cv_predictions.parquet",
        "predictions": input_dir / "predictions.parquet",
    }

    for table, path in mapping.items():
        if not path.exists():
            logger.info("Skip %s (Datei fehlt): %s", table, str(path.as_posix()))
            continue
        _load_parquet(client, args.dataset, table, path, write_disp, logger)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())