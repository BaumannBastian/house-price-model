# ------------------------------
# predict.py
#
# In dieser Python-Datei wird ein Modell geladen (standardmäßig der Champion),
# auf den Kaggle-House-Prices-Testdatensatz angewendet, die Vorhersagen werden
# als CSV gespeichert und zusätzlich in die PostgreSQL-Datenbank geschrieben.
#
# Auswahl des Modells:
# - default: aktueller Champion aus der DB (models.is_champion = TRUE)
# - optional: --model-name oder --model-id
# ------------------------------

from src.features import missing_value_treatment, new_feature_engineering, ordinal_mapping
from src.db import (
    init_db,
    insert_predictions,
    get_current_champion_id,
    get_model_file_path,
    get_latest_model_id_by_name,
)

import argparse
import pandas as pd
import joblib

from pathlib import Path


DEFAULT_INPUT_PATH = Path("data/raw/test.csv")
DEFAULT_OUTPUT_PATH = Path("predictions/predictions.csv")


def parse_args() -> argparse.Namespace:
    """
    Parst Kommandozeilenargumente für predict.py.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH))

    parser.add_argument("--model-id", type=int, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--version", type=str, default=None)

    parser.add_argument("--skip-db", action="store_true", help="Keine DB Inserts durchführen (nur CSV schreiben).")

    return parser.parse_args()


def _resolve_model_id(args: argparse.Namespace) -> int:
    """
    Bestimmt, welches Modell verwendet werden soll und gibt model_id zurück.
    """
    if args.model_id is not None and args.model_name is not None:
        raise ValueError("Bitte entweder --model-id oder --model-name verwenden, nicht beides.")

    if args.model_id is not None:
        return int(args.model_id)

    if args.model_name is not None:
        model_id = get_latest_model_id_by_name(args.model_name, version=args.version)
        if model_id is None:
            raise RuntimeError(f"Kein Modell gefunden für name='{args.model_name}' (version={args.version}).")
        return int(model_id)

    champion_id = get_current_champion_id()
    if champion_id is None:
        raise RuntimeError("Kein Champion in der DB gefunden. Bitte zuerst `python train.py --mode analysis` ausführen.")
    return int(champion_id)


def main() -> None:
    """
    Führt Inferenz aus und schreibt CSV + optional DB.
    """
    args = parse_args()

    model_id = _resolve_model_id(args)
    file_path = get_model_file_path(model_id)

    if file_path is None:
        raise FileNotFoundError(
            f"Kein file_path in models für model_id={model_id}. "
            "Bitte das Modell zuerst speichern (train-only oder analysis-Champion)."
        )

    model_path = Path(file_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact nicht gefunden unter: {model_path}. "
            "Pfad aus DB stimmt nicht oder Datei wurde gelöscht."
        )

    # Modell laden
    model = joblib.load(model_path)

    # Eingabedaten laden
    input_path = Path(args.input)
    df = pd.read_csv(input_path)

    # Gleiche Feature-Pipeline wie im Training
    X_raw = df.drop(columns=["Id"])
    X = missing_value_treatment(X_raw)
    X = new_feature_engineering(X)
    X = ordinal_mapping(X)

    # Predictions machen
    y_pred = model.predict(X)

    # Output-CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)

    out_df = pd.DataFrame(
        {
            "Id": df["Id"],
            "SalePrice": y_pred,
        }
    )
    out_df.to_csv(output_path, index=False)

    # Optional: DB Insert
    if not args.skip_db:
        init_db()
        n_inserted = insert_predictions(df["Id"].values, y_pred, model_id=model_id)
        print(f"{n_inserted} Predictions in die Datenbank geschrieben (model_id={model_id}).")

    print(f"Predictions gespeichert unter: {output_path.resolve()}")
    print(f"Verwendetes Modell: model_id={model_id}, file_path={model_path}")


if __name__ == "__main__":
    main()