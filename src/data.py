# src/data.py

import pandas as pd
from pathlib import Path


def load_train_data(path: str | Path = "data/raw/train.csv") -> pd.DataFrame:
    """
    Lädt den Kaggle-Train-Datensatz als DataFrame.
    """
    path = Path(path)
    return pd.read_csv(path)