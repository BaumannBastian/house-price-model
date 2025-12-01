# src/models.py

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor 


# Linear Regression Model
def build_linear_regression_model(
    preprocessor: ColumnTransformer,
    use_log_target: bool = False,
) -> TransformedTargetRegressor:
    """
    Preprocessing + LinearRegression auf SalePrice.
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

# Random Forest Model
def build_random_forest_model(
    preprocessor: ColumnTransformer,
    use_log_target: bool = False,
) -> TransformedTargetRegressor:
    """
    Preprocessing + RandomForestRegressor auf SalePrice.
    """
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
            )),
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

# Histogram-Based Gradient Boosting Model
def build_histogram_based_model(
    preprocessor: ColumnTransformer,
    use_log_target: bool = False,
) -> TransformedTargetRegressor:
    """
    Preprocessing + HistGradientBoostingRegressor auf SalePrice.
    """
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", HistGradientBoostingRegressor(
                max_depth=6,
                learning_rate=0.1,
                max_iter=300,
                random_state=42,
            )),
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
