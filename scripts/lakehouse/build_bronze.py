# ------------------------------------
# scripts/lakehouse/build_bronze.py
#
# Bronze Layer: raw CSV -> Parquet/CSV Snapshot
# ------------------------------------

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import pandas as pd

from datetime import datetime
from pathlib import Path
from typing import Literal


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _ensure_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
    except Exception as e:
        raise RuntimeError("Parquet-Support fehlt (pyarrow). Nutze --format csv oder installiere pyarrow.") from e


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_table(df: pd.DataFrame, out_path: Path, fmt: Literal["parquet", "csv"]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        df.to_csv(out_path, index=False)
        return

    _ensure_pyarrow()
    df.to_parquet(out_path, index=False, engine="pyarrow")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Bronze layer (raw ingest).")
    p.add_argument("--raw-dir", type=str, default="data/raw", help="Input directory containing train.csv/test.csv")
    p.add_argument("--out-dir", type=str, default="data/lakehouse/bronze", help="Bronze output directory")
    p.add_argument("--format", type=str, choices=["parquet", "csv"], default="parquet", help="Output format")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    raw_dir = (PROJECT_ROOT / args.raw_dir).resolve()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    fmt: Literal["parquet", "csv"] = args.format  # type: ignore[assignment]
    suffix = "parquet" if fmt == "parquet" else "csv"

    train_csv = raw_dir / "train.csv"
    test_csv = raw_dir / "test.csv"

    if not train_csv.exists() or not test_csv.exists():
        missing = [p.name for p in [train_csv, test_csv] if not p.exists()]
        raise FileNotFoundError(f"Fehlende Input-Dateien: {', '.join(missing)}")

    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    _write_table(df_train, out_dir / "train_raw" / f"latest.{suffix}", fmt)
    _write_table(df_test, out_dir / "test_raw" / f"latest.{suffix}", fmt)

    meta = {
        "built_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "format": fmt,
        "inputs": {
            "train_csv": {"path": str(train_csv.as_posix()), "sha256": _sha256(train_csv), "rows": int(df_train.shape[0])},
            "test_csv": {"path": str(test_csv.as_posix()), "sha256": _sha256(test_csv), "rows": int(df_test.shape[0])},
        },
    }

    meta_dir = out_dir / "_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "raw_hashes.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    logger.info("Bronze build fertig: %s", out_dir.as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main())