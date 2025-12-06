# ------------------------------
# src/preprocessing.py
#
# In dieser Python-Datei wird ein ColumnTransformer erstellt, der
# kategoriale Features per One-Hot-Encoding transformiert und
# numerische Features unverändert durchreicht.
# ------------------------------

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Erzeugt einen ColumnTransformer für kategoriale und numerische Features.

    Es werden automatisch:
    - alle Spalten mit Datentyp ``object`` als kategorial interpretiert und
      mit einem ``OneHotEncoder`` (dichtes Array, ``sparse_output=False``)
      kodiert,
    - alle übrigen Spalten als numerisch behandelt und im ``remainder='passthrough'``
      unverändert durchgereicht.

    Voraussetzung ist, dass:
    - ordinale Merkmale bereits vorab auf Integer gemappt wurden,
    - nur „echte“ kategoriale Merkmale noch den Typ ``object`` besitzen.

    Parameter
    ----------
    X : pandas.DataFrame
        Beispiel-DataFrame (z.B. Trainingsdaten) mit allen Spalten, aus
        denen der Preprocessor die Spaltentypen ableiten soll.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        ColumnTransformer, der fertige Feature-Matrizen für sklearn-Modelle
        erzeugt und sich in Pipelines integrieren lässt.
    """
    # kategoriale Spalten bestimmen
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    # restliche Spalten sind numerisch
    numeric_features = [col for col in X.columns if col not in categorical_features]

    # ColumnTransformer erstellen
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            )
        ],
        remainder="passthrough",  # numerische Features einfach durchreichen
    )

    return preprocessor