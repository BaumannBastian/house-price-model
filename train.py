# ------------------------------------
# train.py
#
# Dieses Skript trainiert mehrere Modelle für das Kaggle House Prices Projekt
# und speichert die Modelle als .joblib Artefakte.
#
# Struktur
# ------------------------------------
# - train-only
#   Trainiert alle Modelle auf den gesamten Trainingsdaten (Full-Data),
#   speichert Artefakte und schreibt einen Model-Run in die DB.
#
# - analysis
#   Führt eine KFold-CV (gemeinsame Splits für alle Modelle) auf dem Trainingsset durch,
#   schreibt OOF-Predictions in train_cv_predictions (für PowerBI),
#   loggt CV-Metriken in models und setzt den Champion nach CV-RMSE.
#
# Usage
# ------------------------------------
# - python train.py startet "train-only" Mode
# - python train.py --mode analysis startet "analysis" Mode
# - DB-Schema wird via Flyway gemanaged (siehe start_dev.ps1 / docker-compose.yml)
# ------------------------------------

from __future__ import annotations

import argparse
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline

from src.data import load_train_data
from src.features import missing_value_treatment, new_feature_engineering, ordinal_mapping
from src.models import (
    build_linear_regression_model,
    build_random_forest_model,
    build_histogram_based_model,
)
from src.nn_models import build_torch_mlp_model
from src.preprocessing import build_preprocessor

from src.db import (
    insert_model,
    insert_train_cv_predictions,
    set_champion_model,
    update_model_file_path,
)


# --------------------------------------------------------
# Logging
# --------------------------------------------------------


def setup_logging(debug: bool) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Handler nur einmal hinzufügen
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG if debug else logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger


# --------------------------------------------------------
# CLI
# --------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train-only", "analysis"], default="train-only")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-splits", type=int, default=5)
    return parser.parse_args()


# --------------------------------------------------------
# Metrics
# --------------------------------------------------------


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mare(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), 1e-9)
    return float(np.mean(np.abs(y_pred - y_true) / denom))


def mre(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), 1e-9)
    return float(np.mean((y_pred - y_true) / denom))


# --------------------------------------------------------
# CV + Eval
# --------------------------------------------------------


def eval_on_holdout(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> Tuple[float, float, float, float]:
    y_pred = model.predict(X_test)
    return (
        float(r2_score(y_test, y_pred)),
        rmse(y_test, y_pred),
        float(mean_absolute_error(y_test, y_pred)),
        mare(y_test, y_pred),
    )


def _extract_hyperparams(model: Pipeline) -> Optional[dict]:
    """
    Versucht, Config/Parameter aus dem letzten Step herauszuziehen.
    (Best effort, weil scikit-learn Pipelines + TargetTransformer etc.)
    """
    try:
        if isinstance(model, TransformedTargetRegressor):
            base = model.regressor_
        else:
            base = model

        if hasattr(base, "named_steps"):
            est = base.named_steps.get("model", None)
        else:
            est = base

        if est is None:
            return None

        # TorchMLP / sklearn Modelle haben meist get_params
        if hasattr(est, "get_params"):
            params = est.get_params(deep=False)
            # Nicht zu groß machen: nur simple types
            simple = {}
            for k, v in params.items():
                if isinstance(v, (int, float, str, bool, type(None))):
                    simple[k] = v
            return simple

        return None
    except Exception:
        return None


# --------------------------------------------------------
# Main
# --------------------------------------------------------


def main() -> None:
    """Führt train.py im gewählten Modus aus."""
    warnings.filterwarnings("ignore", category=UserWarning)

    args = parse_args()
    logger = setup_logging(args.debug)

    run_version = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Rohdaten laden
    df = load_train_data()
    kaggle_ids = df["Id"].to_numpy()
    Y = df["SalePrice"].to_numpy()

    X_raw = df.drop(columns=["Id", "SalePrice"])

    # deterministische, zeilenweise Schritte (kein Leakage)
    X = missing_value_treatment(X_raw)
    X = new_feature_engineering(X)
    X = ordinal_mapping(X)

    # Preprocessor-Builder (modell-spezifisch)
    def prep_ohe(df_like):
        return build_preprocessor(df_like, kind="ohe_dense", min_frequency=10, scale_numeric=False)

    def prep_nn(df_like):
        return build_preprocessor(df_like, kind="ohe_dense", min_frequency=10, scale_numeric=True)

    def prep_hgb(df_like):
        return build_preprocessor(df_like, kind="hgb_native")

    configs = [
        ("LinearRegression", build_linear_regression_model, False, prep_ohe),
        ("LinearRegression_log", build_linear_regression_model, True, prep_ohe),
        ("RandomForest", build_random_forest_model, False, prep_ohe),
        ("RandomForest_log", build_random_forest_model, True, prep_ohe),
        ("HistGBR", build_histogram_based_model, False, prep_hgb),
        ("HistGBR_log", build_histogram_based_model, True, prep_hgb),
        ("TorchMLP", build_torch_mlp_model, False, prep_nn),
        ("TorchMLP_log", build_torch_mlp_model, True, prep_nn),
    ]

    # ---------------------------------------------------------------------
    # MODE: train-only
    # ---------------------------------------------------------------------
    if args.mode == "train-only":
        logger.info("Mode: train-only")

        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        for name, builder, use_log, prep_builder in configs:
            logger.info("=== Train-only: %s (use_log_target=%s) ===", name, use_log)

            preprocessor = prep_builder(X)

            model = builder(preprocessor, use_log_target=use_log)

            model.fit(X, Y)

            out_path = models_dir / f"{name}_{run_version}.joblib"
            joblib.dump(model, out_path)

            hyperparams = _extract_hyperparams(model)

            model_id = insert_model(
                name=name,
                version=run_version,
                file_path=str(out_path),
                r2_test=None,
                rmse_test=None,
                mare_test=None,
                mre_test=None,
                cv_rmse_mean=None,
                cv_rmse_std=None,
                max_abs_train_error=None,
                hyperparams=hyperparams,
                is_champion=False,
            )

            logger.info("Gespeichert: %s (model_id=%s) -> %s", name, model_id, out_path)

        logger.info("Train-only fertig. Alle Modelle gespeichert.")
        return

    # ---------------------------------------------------------------------
    # MODE: analysis
    # ---------------------------------------------------------------------
    logger.info("Mode: analysis")
    # Erwartet: DB-Schema ist bereits via Flyway vorhanden

    (
        X_train,
        X_test,
        Y_train,
        Y_test,
        ids_train,
        _ids_test,
    ) = train_test_split(
        X,
        Y,
        kaggle_ids,
        test_size=0.2,
        random_state=args.seed,
    )

    logger.info(
        "Erzeuge CV-Splits (KFold=%d, seed=%d) – werden für alle Modelle wiederverwendet.",
        args.cv_splits,
        args.seed,
    )
    kf = KFold(n_splits=args.cv_splits, shuffle=True, random_state=args.seed)
    splits = list(kf.split(X_train))

    results = []
    best_cv_rmse = float("inf")
    best_config = None
    best_model_id = None

    for name, builder, use_log, prep_builder in configs:
        logger.info("=== Analysis: %s (use_log_target=%s) ===", name, use_log)

        preprocessor = prep_builder(X_train)

        fold_rmses = []
        oof_pred = np.zeros(len(X_train), dtype=float)

        for fold_idx, (tr_idx, va_idx) in enumerate(splits, start=1):
            logger.info("[%s] CV fold %d/%d: fit", name, fold_idx, args.cv_splits)

            X_tr = X_train.iloc[tr_idx]
            y_tr = Y_train[tr_idx]

            X_va = X_train.iloc[va_idx]
            y_va = Y_train[va_idx]

            model = builder(preprocessor, use_log_target=use_log)
            model.fit(X_tr, y_tr)

            pred_va = model.predict(X_va)
            oof_pred[va_idx] = pred_va

            fold_rmse = rmse(y_va, pred_va)
            fold_rmses.append(fold_rmse)

            logger.info("[%s] CV fold %d/%d: RMSE = %.2f", name, fold_idx, args.cv_splits, fold_rmse)

        cv_rmse_mean = float(np.mean(fold_rmses))
        cv_rmse_std = float(np.std(fold_rmses))

        # Fit final on X_train for test metrics (holdout)
        final_model = builder(preprocessor, use_log_target=use_log)
        final_model.fit(X_train, Y_train)
        r2_t, rmse_t, mae_t, mare_t = eval_on_holdout(final_model, X_test, Y_test)

        mre_t = mre(Y_test, final_model.predict(X_test))

        # best effort max train error
        abs_err = np.abs(oof_pred - Y_train)
        max_abs_train_error = float(np.max(abs_err))

        model_id = insert_model(
            name=name,
            version=run_version,
            file_path=None,
            r2_test=r2_t,
            rmse_test=rmse_t,
            mare_test=mae_t,
            mre_test=mre_t,
            cv_rmse_mean=cv_rmse_mean,
            cv_rmse_std=cv_rmse_std,
            max_abs_train_error=max_abs_train_error,
            hyperparams=_extract_hyperparams(final_model),
            is_champion=False,
        )

        # write OOF rows for PowerBI
        inserted = insert_train_cv_predictions(
            kaggle_ids=ids_train,
            y_true=Y_train,
            y_pred_oof=oof_pred,
            model_id=model_id,
        )

        logger.info(
            "[%s] CV-RMSE = %.2f ± %.2f | Test RMSE = %.2f | Test R² = %.4f | OOF rows = %d",
            name,
            cv_rmse_mean,
            cv_rmse_std,
            rmse_t,
            r2_t,
            inserted,
        )

        results.append((name, cv_rmse_mean, cv_rmse_std, rmse_t, r2_t, model_id))

        if cv_rmse_mean < best_cv_rmse:
            best_cv_rmse = cv_rmse_mean
            best_config = (name, builder, use_log, prep_builder)
            best_model_id = model_id

    # Summary
    results.sort(key=lambda x: x[1])
    logger.info("=== Summary (sorted by CV-RMSE) ===")
    for name, mean_, std_, rmse_t, r2_t, _mid in results:
        logger.info("%-16s | CV-RMSE = %9.2f ± %8.2f | Test RMSE = %9.2f | Test R² = %.4f", name, mean_, std_, rmse_t, r2_t)

    assert best_config is not None and best_model_id is not None
    best_name, best_builder, best_use_log, best_prep_builder = best_config
    logger.info("Champion: %s (model_id=%s, CV-RMSE=%.2f)", best_name, best_model_id, best_cv_rmse)

    # Fit Champion on full data + save artifact
    logger.info("Fit Champion auf Full-Data und speichere Artifact.")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    preprocessor_full = best_prep_builder(X)
    champion_model = best_builder(preprocessor_full, use_log_target=best_use_log)
    champion_model.fit(X, Y)

    out_path = models_dir / f"{best_name}_{run_version}.joblib"
    joblib.dump(champion_model, out_path)

    set_champion_model(best_model_id)
    update_model_file_path(best_model_id, str(out_path))

    logger.info("Champion gespeichert: %s", out_path)


if __name__ == "__main__":
    main()
