# ------------------------------
# predict.py
#
# In dieser Python-Datei wird das Champion-Modell geladen, auf den
# Kaggle-House-Prices-Testdatensatz angewendet, die Vorhersagen werden
# als CSV gespeichert und zusätzlich in die PostgreSQL-Datenbank geschrieben.
# ------------------------------

from src.features import missing_value_treatment, new_feature_engineering, ordinal_mapping
from src.db import init_db, insert_predictions, get_current_champion_id

import pandas as pd
import joblib

from pathlib import Path


MODEL_PATH = Path("models/HistGBR_log.joblib")
INPUT_PATH = Path("data/raw/test.csv")
OUTPUT_PATH = Path("predictions/predictions.csv")


def main() -> None:
    """
    Führt das Inferenz-Skript für das Champion-Modell aus.

    Ablauf
    ------
    1. Lädt das gespeicherte Champion-Modell aus ``MODEL_PATH``.
    2. Lädt den Kaggle-Testdatensatz aus ``INPUT_PATH``.
    3. Wendet dieselbe Feature-Pipeline wie im Training an:
       Missing-Value-Treatment, Feature-Engineering und Ordinal-Encoding.
    4. Berechnet Vorhersagen für ``SalePrice``.
    5. Speichert die Predictions als CSV unter ``OUTPUT_PATH``.
    6. Schreibt dieselben Predictions zusätzlich in die Datenbanktabelle
       ``predictions`` inkl. optionaler ``model_id`` des Champion-Modells.

    Parameter
    ----------
    None

    Returns
    -------
    None
        Die Funktion hat keinen Rückgabewert. Ergebnisse werden auf
        der Festplatte (CSV) und in der Datenbank persistiert.
    """
    # Modell laden
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modell nicht gefunden unter {MODEL_PATH}. "
            "Bitte zuerst `python train.py` ausführen."
        )
    model = joblib.load(MODEL_PATH)

    # Eingabedaten laden
    df = pd.read_csv(INPUT_PATH)

    # Gleiche Feature-Pipeline wie im Training
    X = missing_value_treatment(df.copy())
    X = new_feature_engineering(X)
    X = ordinal_mapping(X)

    # Predictions machen
    y_pred = model.predict(X)

    # Output-DataFrame erzeugen und speichern
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    out_df = pd.DataFrame(
        {
            "Id": df["Id"],
            "SalePrice": y_pred,
        }
    )
    out_df.to_csv(OUTPUT_PATH, index=False)

    # Predictions zusätzlich in die Datenbank schreiben
    init_db()  # stellt sicher, dass 'predictions' existiert und 'model_id' hat

    champion_id = get_current_champion_id()
    if champion_id is None:
        print(
            "Warnung: Kein Champion-Modell in 'models' gefunden. "
            "Speichere Predictions ohne model_id."
        )
    else:
        print(f"Verwende Champion model_id={champion_id} für Predictions.")

    n_inserted = insert_predictions(df["Id"].values, y_pred, model_id=champion_id)
    print(f"{n_inserted} Predictions in die Datenbank geschrieben.")
    print(f"Predictions gespeichert unter: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()