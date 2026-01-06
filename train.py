# ------------------------------
# train.py
#
# In dieser Python-Datei werden die Trainingsdaten geladen, Features
# erzeugt und Modelle trainiert.
#
# Modi
# ----
# 1) train-only:
#    - jedes Modell wird einmal auf dem kompletten train.csv trainiert
#    - jedes Modell wird als .joblib gespeichert
#    - pro Modell wird eine Zeile in 'models' geschrieben (file_path gesetzt)
#
# 2) analysis:
#    - gemeinsamer Train/Test Split (Holdout) für Test-Metriken
#    - gemeinsamer KFold Split (gleiche Folds für alle Modelle)
#    - pro Modell: OOF/CV Predictions werden in 'train_cv_predictions' gespeichert
#    - Champion wird per cv_rmse_mean bestimmt
#    - Champion wird auf Full-Data neu gefittet und gespeichert (file_path)
# ------------------------------

from src.data import load_train_data
from src.features import (
    missing_value_treatment,
    new_feature_engineering,
    ordinal_mapping,
)
from src.preprocessing import build_preprocessor
from src.models import (
    build_linear_regression_model,
    build_random_forest_model,
    build_histogram_based_model,
)
from src.nn_models import build_torch_mlp_model
from src.db import (
    init_db,
    init_predictions_view,
    init_models_table,
    init_train_cv_predictions_table,
    insert_model,
    insert_train_cv_predictions,
    set_champion_model,
    update_model_file_path,
)

import argparse
import logging
import warnings
import joblib
import numpy as np

from datetime import datetime
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# Optional: nur falls du es im Projekt hast (typischerweise in src/data.py)
try:
    from src.data import load_test_data  # type: ignore
except Exception:
    load_test_data = None


def parse_args() -> argparse.Namespace:
    """
    Parst Kommandozeilenargumente für das Trainingsskript.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["analysis", "train-only"],
        default="train-only",
        help="analysis = CV + OOF speichern, train-only = nur Full-Train + Modelle speichern",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-splits", type=int, default=5)

    parser.add_argument(
        "--nn-verbose",
        action="store_true",
        help="TorchMLP während des Trainings Loss ausgeben",
    )
    parser.add_argument(
        "--nn-plot-loss",
        action="store_true",
        help="TorchMLP Losskurve nach Training als PNG speichern (nur Full-Fit)",
    )

    return parser.parse_args()


def setup_logging(debug: bool = False) -> logging.Logger:
    """
    Richtet das Logging für das Trainingsskript ein.

    Wichtig:
    - force=True verhindert doppelte Handler / doppelte Ausgabe.
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
    return logging.getLogger("train")


def _extract_hyperparams(model) -> dict:
    """
    Extrahiert Hyperparameter des inneren Regressors (Pipeline-Step "regressor").
    """
    try:
        base = model.regressor
        inner_reg = base.named_steps["regressor"]
        return inner_reg.get_params(deep=False)
    except Exception:
        return {}


def _maybe_set_n_jobs_max(model) -> None:
    """
    Setzt n_jobs=-1 für Modelle, die das unterstützen (beschleunigt CV auf lokalen Maschinen).
    """
    try:
        inner = model.regressor.named_steps["regressor"]
        params = inner.get_params(deep=False)
        if "n_jobs" in params:
            inner.set_params(n_jobs=-1)
    except Exception:
        pass


def _oof_predictions_from_splits(
    logger: logging.Logger,
    model_name: str,
    builder,
    preprocessor,
    use_log_target: bool,
    X_train,
    Y_train,
    splits,
    nn_verbose: bool = False,
) -> tuple[np.ndarray, float, float]:
    """
    Berechnet Out-of-Fold (OOF) Predictions auf Basis vorgegebener CV-Splits.

    Wichtig:
    - splits werden außerhalb einmal erstellt und für alle Modelle wiederverwendet,
      damit jedes Modell exakt die gleichen Folds sieht.
    """
    oof_pred = np.empty(len(X_train), dtype=float)
    oof_pred[:] = np.nan

    fold_rmses = []
    n_folds = len(splits)

    for fold_idx, (tr_idx, val_idx) in enumerate(splits, start=1):
        logger.info("[%s] CV fold %d/%d: fit", model_name, fold_idx, n_folds)

        model_fold = builder(preprocessor, use_log_target=use_log_target)
        _maybe_set_n_jobs_max(model_fold)

        if nn_verbose:
            try:
                inner_mlp = model_fold.regressor.named_steps["regressor"]
                inner_mlp.verbose = True
            except Exception:
                pass

        model_fold.fit(X_train.iloc[tr_idx], Y_train[tr_idx])
        pred_val = model_fold.predict(X_train.iloc[val_idx])

        oof_pred[val_idx] = pred_val

        rmse_fold = float(np.sqrt(mean_squared_error(Y_train[val_idx], pred_val)))
        fold_rmses.append(rmse_fold)

        logger.info("[%s] CV fold %d/%d: RMSE = %.2f", model_name, fold_idx, n_folds, rmse_fold)

    if np.isnan(oof_pred).any():
        raise RuntimeError("OOF-Predictions enthalten NaNs – CV Splits haben nicht alle Indizes befüllt.")

    return oof_pred, float(np.mean(fold_rmses)), float(np.std(fold_rmses))


def main() -> None:
    """
    Führt train.py im gewählten Modus aus.
    """
    warnings.filterwarnings("ignore", category=UserWarning)

    args = parse_args()
    logger = setup_logging(args.debug)

    # DB Grundsetup (idempotent)
    init_models_table()
    init_db()
    try:
        init_predictions_view()
    except Exception:
        logger.warning("Konnte v_predictions_with_model nicht initialisieren (ok).")

    run_version = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Rohdaten laden
    df = load_train_data()

    kaggle_ids = df["Id"].to_numpy()
    Y = df["SalePrice"].to_numpy()

    X_raw = df.drop(columns=["Id", "SalePrice"])

    X = missing_value_treatment(X_raw)
    X = new_feature_engineering(X)
    X = ordinal_mapping(X)

    configs = [
        ("LinearRegression", build_linear_regression_model, False),
        ("LinearRegression_log", build_linear_regression_model, True),
        ("RandomForest", build_random_forest_model, False),
        ("RandomForest_log", build_random_forest_model, True),
        ("HistGBR", build_histogram_based_model, False),
        ("HistGBR_log", build_histogram_based_model, True),
        ("TorchMLP", build_torch_mlp_model, False),
        ("TorchMLP_log", build_torch_mlp_model, True),
    ]

    # ---------------------------------------------------------------------
    # MODE: train-only
    # ---------------------------------------------------------------------
    if args.mode == "train-only":
        logger.info("Mode: train-only")
        logger.info("Baue Preprocessor auf Full-Data.")
        preprocessor_full = build_preprocessor(X)

        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        for name, builder, use_log in configs:
            logger.info("=== Train-only: %s (use_log_target=%s) ===", name, use_log)

            model = builder(preprocessor_full, use_log_target=use_log)
            _maybe_set_n_jobs_max(model)

            if args.nn_verbose and name.startswith("TorchMLP"):
                try:
                    inner_mlp = model.regressor.named_steps["regressor"]
                    inner_mlp.verbose = True
                except Exception:
                    pass

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
                is_champion=False,  # train-only setzt keinen Champion um
            )

            logger.info("Gespeichert: %s (model_id=%s) -> %s", name, model_id, out_path)

        logger.info("Train-only fertig. Alle Modelle gespeichert.")
        return

    # ---------------------------------------------------------------------
    # MODE: analysis
    # ---------------------------------------------------------------------
    logger.info("Mode: analysis")
    init_train_cv_predictions_table()

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

    logger.info("Baue Preprocessor auf X_train.")
    preprocessor = build_preprocessor(X_train)

    logger.info("Erzeuge CV-Splits (KFold=%d, seed=%d) – werden für alle Modelle wiederverwendet.", args.cv_splits, args.seed)
    kf = KFold(n_splits=args.cv_splits, shuffle=True, random_state=args.seed)
    splits = list(kf.split(X_train))

    results = []
    best_cv_rmse = float("inf")
    best_config = None
    best_model_id = None

    for name, builder, use_log in configs:
        logger.info("=== Analysis: %s (use_log_target=%s) ===", name, use_log)

        # 1) OOF/CV Predictions (gleiche Folds für alle Modelle)
        oof_pred, cv_rmse_mean, cv_rmse_std = _oof_predictions_from_splits(
            logger=logger,
            model_name=name,
            builder=builder,
            preprocessor=preprocessor,
            use_log_target=use_log,
            X_train=X_train,
            Y_train=Y_train,
            splits=splits,
            nn_verbose=args.nn_verbose and name.startswith("TorchMLP"),
        )

        # 2) Fit auf vollem X_train für Holdout-Test-Metriken
        model = builder(preprocessor, use_log_target=use_log)
        _maybe_set_n_jobs_max(model)

        if args.nn_verbose and name.startswith("TorchMLP"):
            try:
                inner_mlp = model.regressor.named_steps["regressor"]
                inner_mlp.verbose = True
            except Exception:
                pass

        model.fit(X_train, Y_train)

        y_pred_test = model.predict(X_test)

        rmse_test = float(np.sqrt(mean_squared_error(Y_test, y_pred_test)))
        r2_test = float(r2_score(Y_test, y_pred_test))

        rel_errors_test = (y_pred_test - Y_test) / Y_test
        mre_test = float(rel_errors_test.mean())
        mare_test = float(np.abs(rel_errors_test).mean())

        # Für "Worst error" ist OOF sinnvoller als Train-Fit (generalization-like)
        oof_abs_errors = np.abs(oof_pred - Y_train)
        max_abs_oof_error = float(oof_abs_errors.max())

        # 3) DB Write: models + train_cv_predictions
        hyperparams = _extract_hyperparams(model)

        model_id = insert_model(
            name=name,
            version=run_version,
            file_path=None,  # nur Champion bekommt file_path (unten)
            r2_test=r2_test,
            rmse_test=rmse_test,
            mare_test=mare_test,
            mre_test=mre_test,
            cv_rmse_mean=cv_rmse_mean,
            cv_rmse_std=cv_rmse_std,
            max_abs_train_error=max_abs_oof_error,
            hyperparams=hyperparams,
            is_champion=False,
        )

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
            rmse_test,
            r2_test,
            inserted,
        )

        # Champion Tracking
        if cv_rmse_mean < best_cv_rmse:
            best_cv_rmse = cv_rmse_mean
            best_config = (name, builder, use_log)
            best_model_id = model_id

        results.append((name, cv_rmse_mean, cv_rmse_std, rmse_test, r2_test))

    if best_config is None or best_model_id is None:
        raise RuntimeError("Kein Champion-Modell gefunden – Config-Liste leer?")

    # Champion markieren
    champion_name, champion_builder, champion_use_log = best_config
    set_champion_model(best_model_id)

    logger.info("=== Summary (sorted by CV-RMSE) ===")
    for name, cv_mean, cv_std, rmse_t, r2_t in sorted(results, key=lambda x: x[1]):
        logger.info("%-18s | CV-RMSE = %10.2f ± %8.2f | Test RMSE = %10.2f | Test R² = %.4f", name, cv_mean, cv_std, rmse_t, r2_t)

    logger.info("Champion: %s (model_id=%s, CV-RMSE=%.2f)", champion_name, best_model_id, best_cv_rmse)

    # Champion auf Full-Data neu fitten und speichern (für predict.py)
    logger.info("Fit Champion auf Full-Data und speichere Artifact.")
    preprocessor_full = build_preprocessor(X)
    champion_model = champion_builder(preprocessor_full, use_log_target=champion_use_log)
    _maybe_set_n_jobs_max(champion_model)

    champion_model.fit(X, Y)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    out_path = models_dir / f"{champion_name}_{run_version}.joblib"
    joblib.dump(champion_model, out_path)

    update_model_file_path(best_model_id, str(out_path))
    logger.info("Champion gespeichert: %s", out_path)


if __name__ == "__main__":
    main()