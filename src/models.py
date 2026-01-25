# ------------------------------------
# src/models.py
#
# In dieser Python-Datei werden sklearn-Modelle als Pipelines bereitgestellt,
# jeweils wahlweise mit oder ohne Log-Transformation des Targets über
# TransformedTargetRegressor von sklearn.
#
# Feature Engineering
# ------------------------------------
# HistGradientBoostingRegressor wird so gebaut, dass er native kategoriale
# Features verarbeiten kann (categorical_features="from_dtype").
# Dazu muss der Preprocessor für dieses Modell pandas DataFrames mit
# category-dtypes durchreichen (siehe src/preprocessing.py, kind="hgb_native").
# ------------------------------------

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.preprocessing import wrap_log_target


# --------------------------------------------------------
# Build sklearn Regressor Pipelines: Linear, RF, HGB
# --------------------------------------------------------

def build_linear_regression_model(
    preprocessor: BaseEstimator,
    use_log_target: bool = False,
) -> TransformedTargetRegressor:
    """Erzeugt eine Pipeline aus Preprocessing und LinearRegression."""
    base = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )
    return wrap_log_target(base, use_log_target)


def build_random_forest_model(
    preprocessor: BaseEstimator,
    use_log_target: bool = False,
) -> TransformedTargetRegressor:
    """Erzeugt eine Pipeline aus Preprocessing und RandomForestRegressor."""
    base = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return wrap_log_target(base, use_log_target)


def build_histogram_based_model(
    preprocessor: BaseEstimator,
    use_log_target: bool = False,
) -> TransformedTargetRegressor:
    """Erzeugt eine Pipeline aus Preprocessing und HistGradientBoostingRegressor.

    Hinweis:
    - Für native Categorical Support muss der Preprocessor pandas DataFrames
      mit category-dtypes liefern (src/preprocessing.py, kind="hgb_native").
    """
    base = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    max_depth=6,
                    learning_rate=0.1,
                    max_iter=300,
                    random_state=42,
                    categorical_features="from_dtype",
                ),
            ),
        ]
    )
    return wrap_log_target(base, use_log_target)