# ------------------------------------
# scripts/lakehouse/build_gold.py
#
# Gold Layer: silver -> feature tables
# - zeilenweises Feature Engineering
# - ordinal Mapping (deterministisch)
# ------------------------------------

from __future__ import annotations

import argparse
import logging
import sys
import pandas as pd

from pathlib import Path
from typing import Literal

from src.features import new_feature_engineering, ordinal_mapping


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _ensure_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
    except Exception as e:
        raise RuntimeError("Parquet-Support fehlt (pyarrow). Nutze --format csv oder installiere pyarrow.") from e


def _read_table(path: Path, fmt: Literal["parquet", "csv"]) -> pd.DataFrame:
    if fmt == "csv":
        return pd.read_csv(path)

    _ensure_pyarrow()
    return pd.read_parquet(path, engine="pyarrow")


def _write_table(df: pd.DataFrame, out_path: Path, fmt: Literal["parquet", "csv"]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        df.to_csv(out_path, index=False)
        return

    _ensure_pyarrow()
    df.to_parquet(out_path, index=False, engine="pyarrow")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Gold layer (feature tables).")
    p.add_argument("--silver-dir", type=str, default="data/lakehouse/silver", help="Silver base directory")
    p.add_argument("--out-dir", type=str, default="data/lakehouse/gold", help="Gold output directory")
    p.add_argument("--format", type=str, choices=["parquet", "csv"], default="parquet", help="Input/output format")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    silver_dir = (PROJECT_ROOT / args.silver_dir).resolve()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    fmt: Literal["parquet", "csv"] = args.format  # type: ignore[assignment]
    suffix = "parquet" if fmt == "parquet" else "csv"

    train_in = silver_dir / "train_clean" / f"latest.{suffix}"
    test_in = silver_dir / "test_clean" / f"latest.{suffix}"

    if not train_in.exists() or not test_in.exists():
        raise FileNotFoundError("Silver Inputs fehlen (train_clean/test_clean latest.*).")

    df_train = _read_table(train_in, fmt)
    df_test = _read_table(test_in, fmt)

    train_id = df_train["Id"]
    train_y = df_train["SalePrice"]
    train_X = df_train.drop(columns=["Id", "SalePrice"])

    test_id = df_test["Id"]
    test_X = df_test.drop(columns=["Id"])

    train_X = new_feature_engineering(train_X)
    train_X = ordinal_mapping(train_X)

    test_X = new_feature_engineering(test_X)
    test_X = ordinal_mapping(test_X)

    df_train_out = pd.concat([train_id, train_X, train_y], axis=1)
    df_test_out = pd.concat([test_id, test_X], axis=1)

    _write_table(df_train_out, out_dir / "train_features" / f"latest.{suffix}", fmt)
    _write_table(df_test_out, out_dir / "test_features" / f"latest.{suffix}", fmt)

    logger.info("Gold build fertig: %s", out_dir.as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main())