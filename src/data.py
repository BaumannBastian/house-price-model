# ------------------------------------
# src/data.py
#
# In dieser Python-Datei werden Hilfsfunktionen bereitgestellt, um
# den Kaggle-House-Prices-Train-/Test-Datensatz (CSV) sowie optional
# bereits vorbereitete Gold-Features (Parquet) als Pandas-DataFrames zu laden.
# ------------------------------------

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_train_data(path: str | Path = "data/train.csv") -> pd.DataFrame:
    """Lädt den Kaggle-Train-Datensatz (train.csv) als DataFrame."""
    return pd.read_csv(Path(path))


def load_test_data(path: str | Path = "data/test.csv") -> pd.DataFrame:
    """Lädt den Kaggle-Test-Datensatz (test.csv) als DataFrame."""
    return pd.read_csv(Path(path))


def load_gold_train_data(
    path: str | Path = "data/feature_store/train_gold.parquet",
) -> pd.DataFrame:
    """Lädt Gold-Train-Features (inkl. SalePrice) als DataFrame.

    Erwartete Spalten:
      - Id
      - SalePrice
      - Feature-Spalten (numerisch)
    """
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    return pd.read_parquet(p)


def load_gold_test_data(
    path: str | Path = "data/feature_store/test_gold.parquet",
) -> pd.DataFrame:
    """Lädt Gold-Test-Features als DataFrame.

    Erwartete Spalten:
      - Id
      - Feature-Spalten (numerisch)
    """
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    return pd.read_parquet(p)