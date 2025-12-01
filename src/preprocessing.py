# src/preprocessing.py

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Erzeugt einen ColumnTransformer mit:
    - OneHotEncoder für alle object-Spalten
    - Passthrough für alle numerischen Spalten

    Erwartet ein DataFrame, in dem:
    - alle ordinalen Features schon per Mapping auf integer gebracht wurden
    - nur noch echte kategoriale Features den Typ 'object' haben
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
