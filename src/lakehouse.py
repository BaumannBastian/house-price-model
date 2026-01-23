# ------------------------------------
# src/lakehouse.py
#
# Loader fuer Lakehouse Layers (lokal, file-basiert).
# ------------------------------------

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAKEHOUSE_DIR = PROJECT_ROOT / "data" / "lakehouse"


def _ensure_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
    except Exception as e:
        raise RuntimeError("Parquet-Support fehlt (pyarrow).") from e


def _read_parquet(path: Path) -> pd.DataFrame:
    _ensure_pyarrow()
    return pd.read_parquet(path, engine="pyarrow")


def load_gold_train_features(path: str | Path = LAKEHOUSE_DIR / "gold" / "train_features" / "latest.parquet") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Gold train_features nicht gefunden: {path.as_posix()}")
    return _read_parquet(path)


def load_gold_test_features(path: str | Path = LAKEHOUSE_DIR / "gold" / "test_features" / "latest.parquet") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Gold test_features nicht gefunden: {path.as_posix()}")
    return _read_parquet(path)