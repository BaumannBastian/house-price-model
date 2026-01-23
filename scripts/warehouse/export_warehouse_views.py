# ------------------------------------
# scripts/warehouse/export_warehouse_views.py
#
# Exportiert Warehouse-Views als Dateisnapshots (optional).
# ------------------------------------

from __future__ import annotations

import argparse
import logging
import sys
import pandas as pd

from datetime import datetime
from pathlib import Path
from typing import Literal

from src.db import get_connection


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _ensure_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
    except Exception as e:
        raise RuntimeError("Parquet-Support fehlt (pyarrow). Nutze --format csv oder installiere pyarrow.") from e


def _write_table(df: pd.DataFrame, out_path: Path, fmt: Literal["parquet", "csv"]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        df.to_csv(out_path, index=False)
        return

    _ensure_pyarrow()
    df.to_parquet(out_path, index=False, engine="pyarrow")


def _read_view(view_name: str) -> pd.DataFrame:
    sql = f"SELECT * FROM {view_name};"
    conn = get_connection()
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export warehouse analytics views to parquet/csv.")
    p.add_argument("--out-dir", type=str, default="data/exports/warehouse_gold", help="Export base directory")
    p.add_argument("--format", type=str, choices=["parquet", "csv"], default="parquet", help="Output format")
    p.add_argument("--keep-history", action="store_true", help="Also write a versioned copy per run")
    p.add_argument("--run-id", type=str, default="", help="Optional run identifier (default: timestamp)")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    fmt: Literal["parquet", "csv"] = args.format  # type: ignore[assignment]

    run_id = args.run_id.strip() or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    suffix = "parquet" if fmt == "parquet" else "csv"

    datasets = {
        "model_leaderboard": "v_gold_model_leaderboard",
        "cv_error_by_bucket": "v_gold_cv_error_by_bucket",
        "top_outliers": "v_gold_top_outliers",
    }

    for dataset_name, view_name in datasets.items():
        logger.info("Export %s (%s)", dataset_name, view_name)
        df = _read_view(view_name)

        dataset_dir = out_dir / dataset_name
        _write_table(df, dataset_dir / f"latest.{suffix}", fmt)

        if args.keep_history:
            _write_table(df, dataset_dir / f"run_{run_id}.{suffix}", fmt)

    logger.info("Export fertig: %s", out_dir.as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main())