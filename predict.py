# ------------------------------------
# predict.py
#
# Dieses Skript lädt ein gespeichertes Modell (.joblib) und erzeugt Kaggle-Predictions.
# Zusätzlich werden Predictions als Parquet exportiert, sodass sie in BigQuery (raw layer)
# geladen und in Marts/PowerBI verwendet werden können.
# ------------------------------------

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import joblib
import pandas as pd

from scripts.databricks.sync_feature_store import sync_feature_store
from src.data import load_gold_test_data, load_test_data
from src.features import missing_value_treatment, new_feature_engineering, ordinal_mapping


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _resolve_model(
    models_parquet: Path,
    model_id: Optional[str],
    model_name: Optional[str],
) -> Tuple[str, str, str, Path]:
    """Ermittelt (model_id, model_name, model_version, file_path) aus models.parquet."""
    if not models_parquet.exists():
        raise FileNotFoundError(
            f"models.parquet nicht gefunden: {models_parquet}. Bitte zuerst 'python train.py --mode analysis' ausführen."
        )

    df = pd.read_parquet(models_parquet)
    if df.empty:
        raise ValueError(f"models.parquet ist leer: {models_parquet}")

    if model_id is not None:
        sel = df[df["id"].astype(str) == str(model_id)]
        if sel.empty:
            raise ValueError(f"Model-ID nicht gefunden in models.parquet: {model_id}")
        row = sel.sort_values("created_at_utc").iloc[-1]
    elif model_name is not None:
        sel = df[df["name"] == model_name]
        if sel.empty:
            raise ValueError(f"Model-Name nicht gefunden in models.parquet: {model_name}")
        row = sel.sort_values("created_at_utc").iloc[-1]
    else:
        if "is_champion" in df.columns:
            sel = df[df["is_champion"] == True]
        else:
            sel = df.iloc[0:0]

        if sel.empty:
            row = df.sort_values("created_at_utc").iloc[-1]
        else:
            row = sel.sort_values("created_at_utc").iloc[-1]

    mid = str(row["id"])
    mname = str(row["name"])
    mver = str(row["version"])
    fpath = row.get("file_path", None)

    if fpath is None or (isinstance(fpath, float) and pd.isna(fpath)):
        raise ValueError(f"Für das ausgewählte Modell ist kein file_path gesetzt (id={mid}, name={mname}).")

    return mid, mname, mver, Path(str(fpath))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-source", choices=["csv", "gold"], default="csv")

    parser.add_argument("--test-path", default="data/raw/test.csv", help="Pfad zu test.csv (nur bei --data-source csv)")
    parser.add_argument(
        "--gold-test-path",
        default="data/feature_store/test_gold.parquet",
        help="Pfad zu test_gold.parquet (nur bei --data-source gold)",
    )

    parser.add_argument("--models-parquet", default="data/warehouse/raw/models.parquet")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--model-name", default=None)

    parser.add_argument("--output-csv", default="data/submission.csv")
    parser.add_argument("--warehouse-raw-dir", default="data/warehouse/raw")

    args = parser.parse_args()

    if args.data_source == "gold":
        sync_feature_store()

    models_parquet = Path(args.models_parquet)
    model_id, model_name, model_version, model_path = _resolve_model(
        models_parquet=models_parquet,
        model_id=args.model_id,
        model_name=args.model_name,
    )

    logging.info("Model: %s (%s) | version=%s | path=%s", model_name, model_id, model_version, model_path)

    model = joblib.load(model_path)

    if args.data_source == "csv":
        df_test = load_test_data(args.test_path)
        kaggle_ids = df_test["Id"].copy()
        X = df_test.drop(columns=["Id"])
        X = missing_value_treatment(X)
        X = new_feature_engineering(X)
        X = ordinal_mapping(X)
    else:
        df_test = load_gold_test_data(args.gold_test_path)
        if "Id" not in df_test.columns:
            raise ValueError("Gold-Test-Daten müssen mindestens die Spalte 'Id' enthalten.")
        kaggle_ids = df_test["Id"].copy()
        X = df_test.drop(columns=["Id"])

    pred = model.predict(X)

    out_csv = Path(args.output_csv)
    _ensure_dir(out_csv.parent)

    submission = pd.DataFrame({"Id": kaggle_ids, "SalePrice": pred})
    submission.to_csv(out_csv, index=False)
    logging.info("Submission saved: %s", out_csv)

    created_at_utc = datetime.now(timezone.utc)
    raw_dir = Path(args.warehouse_raw_dir)
    _ensure_dir(raw_dir)

    pred_df = pd.DataFrame(
        {
            "kaggle_id": kaggle_ids.astype(int),
            "predicted_price": pred.astype(float),
            "model_id": model_id,
            "model_name": model_name,
            "model_version": model_version,
            "created_at_utc": created_at_utc,
        }
    )
    pred_df.to_parquet(raw_dir / "predictions.parquet", index=False)
    logging.info("Export fertig: %s", raw_dir)


if __name__ == "__main__":
    main()