# ------------------------------------
# predict.py
#
# In dieser Python-Datei wird ein Modell geladen (standardmäßig der Champion),
# auf den Kaggle-House-Prices-Testdatensatz angewendet, die Vorhersagen werden
# als CSV gespeichert und zusätzlich in die PostgreSQL-Datenbank geschrieben.
#
# Usage
# ------------------------------------
# - python predict.py : aktueller Champion aus der DB (models.is_champion = TRUE)
# - python predict.py --model-name oder --model-id : spezifisches Modell laden
# - python predict.py --skip-db : keine Predictions in die DB schreiben
# ------------------------------------

from __future__ import annotations

import argparse
import joblib
import pandas as pd

from pathlib import Path

from src.db import get_current_champion_id, get_latest_model_id_by_name, get_model_file_path, insert_predictions
from src.features import missing_value_treatment, new_feature_engineering, ordinal_mapping
from src.lakehouse import load_gold_test_features


DEFAULT_INPUT_PATH = Path("data/raw/test.csv")
DEFAULT_OUTPUT_PATH = Path("predictions/predictions.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-source", choices=["raw", "lakehouse"], default="raw")

    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH))

    parser.add_argument("--model-id", type=int, default=None)
    parser.add_argument("--model-name", type=str, default=None)

    parser.add_argument("--skip-db", action="store_true")

    return parser.parse_args()


def resolve_model_path(args: argparse.Namespace) -> tuple[int, Path]:
    if args.model_id is not None:
        model_id = int(args.model_id)
    elif args.model_name is not None:
        model_id = get_latest_model_id_by_name(args.model_name)
        if model_id is None:
            raise ValueError(f"Kein Model gefunden mit name={args.model_name}")
    else:
        model_id = get_current_champion_id()
        if model_id is None:
            raise ValueError("Kein Champion in der DB gesetzt (models.is_champion).")

    file_path = get_model_file_path(model_id)
    if file_path is None:
        raise ValueError(f"Model file_path ist NULL fuer model_id={model_id}")

    return model_id, Path(file_path)


def main() -> None:
    args = parse_args()

    model_id, model_path = resolve_model_path(args)

    if args.data_source == "lakehouse":
        df = load_gold_test_features()
        kaggle_ids = df["Id"].values
        X = df.drop(columns=["Id"])
    else:
        df = pd.read_csv(args.input)
        kaggle_ids = df["Id"].values

        X_raw = df.drop(columns=["Id"])
        X = missing_value_treatment(X_raw)
        X = new_feature_engineering(X)
        X = ordinal_mapping(X)

    model = joblib.load(model_path)
    y_pred = model.predict(X)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame({"Id": kaggle_ids, "SalePrice": y_pred})
    out_df.to_csv(output_path, index=False)

    if not args.skip_db:
        n_inserted = insert_predictions(kaggle_ids, y_pred, model_id=model_id)
        print(f"{n_inserted} Predictions in die Datenbank geschrieben (model_id={model_id}).")

    print(f"Predictions gespeichert unter: {output_path.resolve()}")
    print(f"Verwendetes Modell: model_id={model_id}, file_path={model_path}")


if __name__ == "__main__":
    main()