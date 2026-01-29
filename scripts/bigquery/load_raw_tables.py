# ------------------------------------
# scripts/bigquery/load_raw_tables.py
#
# Laedt reproduzierbare RAW-Exports aus data/warehouse/raw nach BigQuery.
#
# Erwartete Dateien (Parquet):
# - data/warehouse/raw/models.parquet
# - data/warehouse/raw/train_cv_predictions.parquet
# - data/warehouse/raw/predictions.parquet (optional)
#
# Verhalten
# ------------------------------------
# - Default: append
# - Skip-Logik (default aktiv): Wenn ein Run (version/model_version) in der Zieltabelle
#   bereits existiert, wird der Load fuer diese Tabelle uebersprungen (verhindert Duplikate).
# - Schema-Update: erlaubt neue Spalten (Field Addition).
#
# Usage
# ------------------------------------
# PowerShell (Windows):
#   $env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\service_account.json"
#   $env:BQ_PROJECT_ID="house-price-model"
#
#   # append (default) + skip if run already exists
#   python -m scripts.bigquery.load_raw_tables --dataset house_prices_raw
#
#   # append without skip (force duplicates possible)
#   python -m scripts.bigquery.load_raw_tables --dataset house_prices_raw --no-skip-existing
#
#   # truncate (replaces tables)
#   python -m scripts.bigquery.load_raw_tables --dataset house_prices_raw --write-disposition truncate
# ------------------------------------

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
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
    p.add_argument("--location", type=str, default="EU")
    p.add_argument("--input-dir", type=str, default="data/warehouse/raw")
    p.add_argument("--write-disposition", choices=["truncate", "append"], default="append")
    p.add_argument("--no-skip-existing", action="store_true", help="Disable skip logic and always load.")
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


def _extract_versions_from_parquet(parquet_path: Path, version_col: str) -> Sequence[str]:
    if not parquet_path.exists():
        return []
    df = pd.read_parquet(parquet_path, columns=[version_col])
    if version_col not in df.columns:
        return []
    vals = df[version_col].dropna().astype(str).unique().tolist()
    return vals


def _table_has_any_version(
    client: bigquery.Client,
    dataset: str,
    table: str,
    version_col: str,
    versions: Sequence[str],
) -> bool:
    if not versions:
        return False

    table_id = f"{client.project}.{dataset}.{table}"
    sql = f"""
    SELECT 1
    FROM `{table_id}`
    WHERE {version_col} IN UNNEST(@versions)
    LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("versions", "STRING", list(versions))]
    )
    it = client.query(sql, job_config=job_config).result()
    for _ in it:
        return True
    return False


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
        schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
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
    _ensure_dataset(client, dataset_id, location=args.location, logger=logger)

    # Table mapping: local parquet + column name that identifies the run
    tables = [
        ("models", input_dir / "models.parquet", "version"),
        ("train_cv_predictions", input_dir / "train_cv_predictions.parquet", "model_version"),
        ("predictions", input_dir / "predictions.parquet", "model_version"),
    ]

    skip_existing = (not args.no_skip_existing) and (write_disp == bigquery.WriteDisposition.WRITE_APPEND)

    for table_name, parquet_path, version_col in tables:
        if not parquet_path.exists():
            logger.info("Skip %s (Datei fehlt): %s", table_name, str(parquet_path.as_posix()))
            continue

        if skip_existing:
            versions = _extract_versions_from_parquet(parquet_path, version_col)
            try:
                exists = _table_has_any_version(client, args.dataset, table_name, version_col, versions)
            except Exception:
                # Wenn die Tabelle noch nicht existiert oder Schema noch nicht passt -> einfach laden
                exists = False

            if exists:
                logger.info("Skip %s (Run existiert bereits): %s", table_name, ", ".join(versions[:5]))
                continue

        _load_parquet(client, args.dataset, table_name, parquet_path, write_disp, logger)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())