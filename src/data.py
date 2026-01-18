# ------------------------------------
# src/data.py
#
# In dieser Python-Datei werden Hilfsfunktionen bereitgestellt, um
# den Kaggle-House-Prices-Train-Datensatz (und ggf. weitere CSV-Dateien)
# als Pandas-DataFrames zu laden.
# ------------------------------------

import pandas as pd

from pathlib import Path


def load_train_data(path: str | Path = "data/raw/train.csv") -> pd.DataFrame:
    """
    Lädt den Kaggle-Train-Datensatz als Pandas-DataFrame.

    Diese Funktion kapselt das Einlesen der Rohdaten aus der
    House-Prices-Kaggle-Challenge und sorgt dafür, dass im gesamten
    Projekt ein einheitlicher Standardpfad verwendet wird.

    Parameter
    ----------
    path : str | pathlib.Path, optional
        Pfad zur CSV-Datei mit den Trainingsdaten. Standard ist
        ``"data/raw/train.csv"`` relativ zum Projekt-Root.

    Returns
    -------
    pandas.DataFrame
        DataFrame mit allen Spalten des Trainingsdatensatzes genau so,
        wie sie in der CSV-Datei vorliegen (noch ohne Feature-Engineering,
        Encoding oder Skalierung).
    """
    path = Path(path)
    return pd.read_csv(path)
