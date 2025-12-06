# ------------------------------
# src/models.py
#
# In dieser Python-Datei werden sklearn-Modelle (Linear Regression,
# Random Forest und Histogram-Based Gradient Boosting) als Pipelines
# bereitgestellt – jeweils wahlweise mit oder ohne Log-Transformation
# des Targets über TransformedTargetRegressor.
# ------------------------------

import numpy as np

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


def build_linear_regression_model(
    preprocessor: ColumnTransformer,
    use_log_target: bool = False,
) -> TransformedTargetRegressor:
    """
    Erzeugt eine Pipeline aus Preprocessing und LinearRegression.

    Aufbau:
    - ``preprocessor``: ColumnTransformer für Feature-Engineering/Encoding.
    - ``regressor``: klassische lineare Regression ohne Regularisierung.
    Optional wird das Target über ``log1p``/``expm1`` transformiert.

    Parameter
    ----------
    preprocessor : sklearn.compose.ColumnTransformer
        Vorverarbeitung, die rohe House-Prices-Daten in numerische
        Feature-Matrizen überführt.
    use_log_target : bool, optional
        Wenn ``True``, wird ``SalePrice`` intern mit ``np.log1p`` transformiert
        und die Vorhersagen mit ``np.expm1`` zurücktransformiert.
        Wenn ``False``, wird direkt auf dem Original-Target trainiert.

    Returns
    -------
    sklearn.compose.TransformedTargetRegressor
        Regressor, der die vollständige Pipeline inklusive optionaler
        Target-Transformation kapselt und sich wie ein sklearn-Modell
        (``fit``, ``predict``, ``score``) verhält.
    """
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    if use_log_target:
        func = np.log1p
        inverse_func = np.expm1
    else:
        # Identität
        func = None
        inverse_func = None

    return TransformedTargetRegressor(
        regressor=model,
        func=func,
        inverse_func=inverse_func,
    )


def build_random_forest_model(
    preprocessor: ColumnTransformer,
    use_log_target: bool = False,
) -> TransformedTargetRegressor:
    """
    Erzeugt eine Pipeline aus Preprocessing und RandomForestRegressor.

    Aufbau:
    - ``preprocess``: ColumnTransformer für Encoding & Feature-Auswahl.
    - ``regressor``: Random Forest mit fester Konfiguration
      (z.B. 300 Bäume, fester ``random_state``).
    Optional kann das Target per ``log1p``/``expm1`` transformiert werden.

    Parameter
    ----------
    preprocessor : sklearn.compose.ColumnTransformer
        Vorverarbeitung der Eingabedaten (z.B. aus ``build_preprocessor``).
    use_log_target : bool, optional
        Wenn ``True``, wird das Target im Trainingsprozess geloggt und die
        Vorhersagen werden zurücktransformiert.

    Returns
    -------
    sklearn.compose.TransformedTargetRegressor
        Pipeline-Regressor mit Random Forest als Kernmodell und optionaler
        Target-Transformation.
    """
    model = Pipeline(
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


def build_histogram_based_model(
    preprocessor: ColumnTransformer,
    use_log_target: bool = False,
) -> TransformedTargetRegressor:
    """
    Erzeugt eine Pipeline aus Preprocessing und HistGradientBoostingRegressor.

    Das Histogram-Gradient-Boosting-Modell ist typischerweise dein
    „Champion-Modell“ und wird hier mit einer festen, gut funktionierenden
    Hyperparameter-Konfiguration aufgebaut. Auch hier kann optional auf
    log-transformiertem Target trainiert werden.

    Parameter
    ----------
    preprocessor : sklearn.compose.ColumnTransformer
        Vorverarbeitungs-Pipeline, die die rohen Eingabedaten in eine
        numerische Feature-Matrix umwandelt.
    use_log_target : bool, optional
        Wenn ``True``, wird das Target mit ``np.log1p`` transformiert und
        Vorhersagen via ``np.expm1`` zurücktransformiert.

    Returns
    -------
    sklearn.compose.TransformedTargetRegressor
        Pipeline-Regressor mit Histogram-Based Gradient Boosting als
        Kernmodell und optionaler log-Target-Transformation.
    """
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    max_depth=6,
                    learning_rate=0.1,
                    max_iter=300,
                    random_state=42,
                ),
            ),
        ]
    )

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