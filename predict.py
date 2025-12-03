# predict.py

from src.features import missing_value_treatment, new_feature_engineering, ordinal_mapping
import joblib
import pandas as pd
from pathlib import Path


MODEL_PATH = Path("models/HistGBR_log.joblib")
INPUT_PATH = Path("data/raw/test.csv")
OUTPUT_PATH = Path("predictions/predictions.csv")

def main() -> None:
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
    print(f"Predictions gespeichert unter: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
