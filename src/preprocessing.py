# ------------------------------------
# src/preprocessing.py
#
# In dieser Python-Datei werden Preprocessing-Pipelines gebaut.
#
# Motivation / Design
# ------------------------------------
# Wir bauen zwei Preprocessing-Varianten:
#
# 1) kind="ohe_dense"
#    - kategoriale Features: Imputation (konstant) + One-Hot-Encoding (dicht)
#    - numerische Features : Imputation (Median) + optionales Scaling
#    -> geeignet für Linear/RandomForest und insbesondere TorchMLP.
#
# 2) kind="hgb_native"
#    - kategoriale Features: werden als pandas 'category' durchgereicht
#    - numerische Features : Imputation (Median)
#    -> geeignet für HistGradientBoostingRegressor mit
#       categorical_features="from_dtype" (native Categorical Support).
#
# Wichtig:
# - Statistik-basierte Imputation (Median/Modus) passiert hier im Pipeline-Fit
#   und damit sauber Fold-spezifisch in CV (kein Leakage).
# - Custom Transformer (CategoryCaster) muss sklearn-clone-kompatibel sein:
#   __init__ darf Parameter NICHT kopieren/modifizieren, sonst RuntimeError.
#
# Log-Target Wrapping:
# ------------------------------------
# Wir halten hier eine kleine Hilfsfunktion `wrap_log_target`, die ein Modell optional
# in einen TransformedTargetRegressor mit log1p/expm1-Transformation wrapppt.
# So nutzen sklearn-Modelle und TorchMLP exakt dieselbe Log-Target-Logik.
# ------------------------------------

from __future__ import annotations

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------------
# Feature Preprocessing Pipelines
# --------------------------------------------------------

# Custom Transformer: CategoryCaster
class CategoryCaster(BaseEstimator, TransformerMixin):
    """Konvertiert ausgewählte Spalten auf pandas 'category' dtype.

    WICHTIG (sklearn clone):
    - __init__ darf den Parameter 'columns' nicht kopieren oder verändern.
    - Deshalb speichern wir 'columns' exakt so, wie er übergeben wurde.
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.columns:
            if c in X.columns:
                X[c] = X[c].astype("category")
        return X


# Build Preprocessor
def build_preprocessor(
    X: pd.DataFrame,
    *,
    kind: str = "ohe_dense",
    min_frequency: int | float = 10,
    scale_numeric: bool = False,
):
    """Erzeugt eine Preprocessing-Pipeline basierend auf den Spaltentypen.

    Parameter
    ----------
    X : pandas.DataFrame
        Beispiel-DataFrame, aus dem Spaltentypen abgeleitet werden.
    kind : str
        "ohe_dense" oder "hgb_native" (siehe Kopfkommentar).
    min_frequency : int | float
        OneHotEncoder: seltene Kategorien werden gruppiert (reduziert Dimensionalität).
        - int   => Kategorien mit Häufigkeit < int werden zusammengefasst
        - float => Kategorien mit relativer Häufigkeit < float werden zusammengefasst
    scale_numeric : bool
        Wenn True: numerische Features werden (nach Imputation) standardisiert.
        Sinnvoll für TorchMLP.

    Returns
    -------
    sklearn Transformer
        Ein Transformer, der in Pipelines genutzt werden kann.
    """
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    if kind == "hgb_native":
        transformers = []
        if numeric_cols:
            transformers.append(
                (
                    "num",
                    Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                    numeric_cols,
                )
            )

        ct = ColumnTransformer(
            transformers=transformers,
            remainder="passthrough",
        ).set_output(transform="pandas")

        return Pipeline(
            steps=[
                ("cast_cat", CategoryCaster(categorical_cols)),
                ("prep", ct),
            ]
        )

    if kind == "ohe_dense":
        transformers = []

        if numeric_cols:
            num_steps = [("imputer", SimpleImputer(strategy="median"))]
            if scale_numeric:
                num_steps.append(("scaler", StandardScaler()))
            transformers.append(("num", Pipeline(steps=num_steps), numeric_cols))

        if categorical_cols:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                            (
                                "ohe",
                                OneHotEncoder(
                                    handle_unknown="ignore",
                                    sparse_output=False,
                                    min_frequency=min_frequency,
                                ),
                            ),
                        ]
                    ),
                    categorical_cols,
                )
            )

        if not transformers:
            return ColumnTransformer(transformers=[], remainder="passthrough")

        return ColumnTransformer(transformers=transformers, remainder="drop")

    raise ValueError(f"Unknown kind='{kind}'")


# --------------------------------------------------------
# Hilfsfunktion: Log-Target Wrapping
# --------------------------------------------------------

def wrap_log_target(model: BaseEstimator, use_log_target: bool) -> TransformedTargetRegressor:
    """Wrappt ein Modell optional in eine log1p/expm1 Target-Transformation."""
    if use_log_target:
        func = np.log1p
        inverse_func = np.expm1
    else:
        func = None
        inverse_func = None

    return TransformedTargetRegressor(
        regressor=model,
        func=func,
        inverse_func=inverse_func,
    )