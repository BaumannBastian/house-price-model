# ------------------------------------
# train.py
#
# Dieses Skript trainiert mehrere Modelle für das Kaggle House Prices Projekt.
# Die Modelle werden als .joblib Artefakte gespeichert. Zusätzlich werden
# die wichtigsten Laufzeitdaten als Parquet-Dateien exportiert, sodass sie
# in BigQuery (raw layer) geladen und später in Marts/PowerBI genutzt werden können.
#
# Struktur
# ------------------------------------
# - train-only
#   Trainiert alle Modelle auf den gesamten Trainingsdaten (Full-Data)
#   und speichert das Champion-Modell als Artefakt.
#
# - analysis
#   Führt eine KFold-CV (gemeinsame Splits für alle Modelle) durch,
#   exportiert OOF-Predictions und Model-Metriken als Parquet
#   und setzt den Champion nach CV-RMSE.
#
# Usage
# ------------------------------------
# - python train.py
# - python train.py --mode analysis
# - python train.py --mode analysis --data-source gold
# ------------------------------------

from __future__ import annotations

import argparse
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from scripts.databricks.sync_feature_store import sync_feature_store
from src.data import load_gold_train_data, load_train_data
from src.features import missing_value_treatment, new_feature_engineering, ordinal_mapping
from src.models import (
    build_histogram_based_model,
    build_linear_regression_model,
    build_random_forest_model,
)
from src.nn_models import build_torch_mlp_model
from src.preprocessing import build_preprocessor


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


@dataclass(frozen=True)
class ModelConfig:
    name: str
    version: str
    model_factory: Callable[[], object]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _export_parquet(df: pd.DataFrame, path: Path) -> None:
    _ensure_dir(path.parent)
    df.to_parquet(path, index=False)


def _cv_oof_predictions(
    model_factory: Callable[[], object],
    X: pd.DataFrame,
    y: np.ndarray,
    kf: KFold,
) -> Tuple[float, float, np.ndarray]:
    """KFold-CV mit OOF-Predictions.

    Returns:
        (rmse_mean, rmse_std, oof_pred)
    """
    rmses: List[float] = []
    oof_pred = np.full(shape=(len(X),), fill_value=np.nan, dtype=float)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X), start=1):
        model = model_factory()
        model.fit(X.iloc[tr_idx], y[tr_idx])
        pred = model.predict(X.iloc[val_idx])

        rmse = float(np.sqrt(mean_squared_error(y[val_idx], pred)))
        rmses.append(rmse)
        oof_pred[val_idx] = pred

        logging.info("Fold %d RMSE: %.4f", fold, rmse)

    rmse_mean = float(np.mean(rmses))
    rmse_std = float(np.std(rmses))
    return rmse_mean, rmse_std, oof_pred


def _evaluate_holdout_metrics(
    model_factory: Callable[[], object],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> Tuple[float, float]:
    """Trainiert auf Train-Split und evaluiert auf Holdout-Split."""
    model = model_factory()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    rmse_test = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2_test = float(r2_score(y_test, pred))
    return rmse_test, r2_test


def _prepare_xy_from_csv(train_csv_path: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = load_train_data(train_csv_path)

    kaggle_ids = df["Id"].to_numpy()
    y = df["SalePrice"].to_numpy()

    X = df.drop(columns=["Id", "SalePrice"])
    X = missing_value_treatment(X)
    X = new_feature_engineering(X)
    X = ordinal_mapping(X)

    return X, y, kaggle_ids


def _prepare_xy_from_gold(gold_train_path: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = load_gold_train_data(gold_train_path)

    if "Id" not in df.columns or "SalePrice" not in df.columns:
        raise ValueError("Gold-Train-Daten müssen mindestens die Spalten 'Id' und 'SalePrice' enthalten.")

    kaggle_ids = df["Id"].to_numpy()
    y = df["SalePrice"].to_numpy()

    X = df.drop(columns=["Id", "SalePrice"])
    return X, y, kaggle_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train-only", "analysis"], default="train-only")

    parser.add_argument("--data-source", choices=["csv", "gold"], default="csv")
    parser.add_argument("--input", default="data/raw/train.csv", help="Pfad zu train.csv (nur bei --data-source csv)")
    parser.add_argument(
        "--gold-train-path",
        default="data/feature_store/train_gold.parquet",
        help="Pfad zu train_gold.parquet (nur bei --data-source gold)",
    )

    parser.add_argument("--kfolds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="models")
    parser.add_argument("--warehouse-raw-dir", default="data/warehouse/raw")

    args = parser.parse_args()

    if args.data_source == "gold":
        sync_feature_store()

    run_version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    created_at_utc = datetime.now(timezone.utc)

    if args.data_source == "csv":
        X, y, kaggle_ids = _prepare_xy_from_csv(args.input)
    else:
        X, y, kaggle_ids = _prepare_xy_from_gold(args.gold_train_path)

    # ab hier: IDENTISCH in beiden Modi
    prep_dense = build_preprocessor(X, kind="ohe_dense", scale_numeric=False)
    prep_tree  = prep_dense
    prep_nn    = build_preprocessor(X, kind="ohe_dense", scale_numeric=True)
    prep_hgb   = build_preprocessor(X, kind="hgb_native")


    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X,
        y,
        kaggle_ids,
        test_size=0.2,
        random_state=args.seed,
    )

    kf = KFold(n_splits=args.kfolds, shuffle=True, random_state=args.seed)

    model_configs: List[ModelConfig] = [
        ModelConfig(
            name="LinearRegression",
            version=run_version,
            model_factory=lambda: build_linear_regression_model(prep_dense, use_log_target=False),
        ),
        ModelConfig(
            name="LinearRegression_log",
            version=run_version,
            model_factory=lambda: build_linear_regression_model(prep_dense, use_log_target=True),
        ),
        ModelConfig(
            name="RandomForest",
            version=run_version,
            model_factory=lambda: build_random_forest_model(prep_tree, use_log_target=False),
        ),
        ModelConfig(
            name="RandomForest_log",
            version=run_version,
            model_factory=lambda: build_random_forest_model(prep_tree, use_log_target=True),
        ),
        ModelConfig(
            name="HistGBR",
            version=run_version,
            model_factory=lambda: build_histogram_based_model(prep_hgb, use_log_target=False),
        ),
        ModelConfig(
            name="HistGBR_log",
            version=run_version,
            model_factory=lambda: build_histogram_based_model(prep_hgb, use_log_target=True),
        ),
        ModelConfig(
            name="TorchMLP",
            version=run_version,
            model_factory=lambda: build_torch_mlp_model(prep_nn, use_log_target=False),
        ),
        ModelConfig(
            name="TorchMLP_log",
            version=run_version,
            model_factory=lambda: build_torch_mlp_model(prep_nn, use_log_target=True),
        ),
    ]

    output_dir = Path(args.output)
    _ensure_dir(output_dir)

    warehouse_raw_dir = Path(args.warehouse_raw_dir)
    _ensure_dir(warehouse_raw_dir)

    if args.mode == "train-only":
        best_name = None
        best_rmse = float("inf")
        best_factory = None

        for cfg in model_configs:
            rmse_mean, rmse_std, _ = _cv_oof_predictions(cfg.model_factory, X, y, kf)
            logging.info("%s | cv_rmse_mean=%.4f (std=%.4f)", cfg.name, rmse_mean, rmse_std)

            if rmse_mean < best_rmse:
                best_rmse = rmse_mean
                best_name = cfg.name
                best_factory = cfg.model_factory

        assert best_factory is not None and best_name is not None

        champion_model = best_factory()
        champion_model.fit(X, y)

        model_id = str(uuid.uuid4())
        model_path = output_dir / f"{model_id}.joblib"
        joblib.dump(champion_model, model_path)

        logging.info("Champion: %s | cv_rmse_mean=%.4f", best_name, best_rmse)
        logging.info("Model saved: %s", model_path)

        models_df = pd.DataFrame(
            [
                {
                    "id": model_id,
                    "name": best_name,
                    "version": run_version,
                    "is_champion": True,
                    "cv_rmse_mean": best_rmse,
                    "cv_rmse_std": None,
                    "rmse_test": None,
                    "r2_test": None,
                    "created_at_utc": created_at_utc,
                    "file_path": str(model_path.as_posix()),
                }
            ]
        )
        _export_parquet(models_df, warehouse_raw_dir / "models.parquet")
        return

    model_rows: List[Dict[str, object]] = []
    oof_rows: List[Dict[str, object]] = []

    best_cfg: ModelConfig | None = None
    best_rmse_mean = float("inf")

    for cfg in model_configs:
        model_id = str(uuid.uuid4())

        rmse_mean, rmse_std, oof_pred = _cv_oof_predictions(cfg.model_factory, X_train, y_train, kf)
        rmse_test, r2_test = _evaluate_holdout_metrics(cfg.model_factory, X_train, y_train, X_test, y_test)

        logging.info(
            "%s | cv_rmse_mean=%.4f (std=%.4f) | rmse_test=%.4f | r2_test=%.4f",
            cfg.name,
            rmse_mean,
            rmse_std,
            rmse_test,
            r2_test,
        )

        model_rows.append(
            {
                "id": model_id,
                "name": cfg.name,
                "version": cfg.version,
                "is_champion": False,
                "cv_rmse_mean": rmse_mean,
                "cv_rmse_std": rmse_std,
                "rmse_test": rmse_test,
                "r2_test": r2_test,
                "created_at_utc": created_at_utc,
                "file_path": None,
            }
        )

        valid_mask = ~np.isnan(oof_pred)
        ids_used = ids_train[valid_mask]
        y_true_used = y_train[valid_mask]
        y_pred_used = oof_pred[valid_mask]

        abs_err = np.abs(y_true_used - y_pred_used)
        rel_err = abs_err / np.maximum(y_true_used, 1.0)

        for kid, yt, yp, ae, re_ in zip(ids_used, y_true_used, y_pred_used, abs_err, rel_err):
            oof_rows.append(
                {
                    "kaggle_id": int(kid),
                    "y_true": float(yt),
                    "y_pred_oof": float(yp),
                    "abs_error": float(ae),
                    "rel_error": float(re_),
                    "model_id": model_id,
                    "model_name": cfg.name,
                    "model_version": cfg.version,
                    "created_at_utc": created_at_utc,
                }
            )

        if rmse_mean < best_rmse_mean:
            best_rmse_mean = rmse_mean
            best_cfg = cfg

    assert best_cfg is not None

    champion_id = None
    for row in model_rows:
        if row["name"] == best_cfg.name:
            row["is_champion"] = True
            champion_id = row["id"]
            break

    assert champion_id is not None

    champion_model = best_cfg.model_factory()
    champion_model.fit(X, y)

    model_path = output_dir / f"{champion_id}.joblib"
    joblib.dump(champion_model, model_path)

    for row in model_rows:
        if row["id"] == champion_id:
            row["file_path"] = str(model_path.as_posix())
            break

    logging.info("Champion: %s | cv_rmse_mean=%.4f", best_cfg.name, best_rmse_mean)
    logging.info("Model saved: %s", model_path)

    models_df = pd.DataFrame(model_rows)
    oof_df = pd.DataFrame(oof_rows)

    _export_parquet(models_df, warehouse_raw_dir / "models.parquet")
    _export_parquet(oof_df, warehouse_raw_dir / "train_cv_predictions.parquet")

    logging.info("Export fertig: %s", warehouse_raw_dir)


if __name__ == "__main__":
    main()