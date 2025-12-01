# train.py

from src.data import load_train_data
from src.features import missing_value_treatment, new_feature_engineering, ordinal_mapping
from src.preprocessing import build_preprocessor
from src.models import build_linear_regression_model, build_random_forest_model, build_histogram_based_model
import numpy as np
import argparse
import logging
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="enable debug logging",
    )
    return parser.parse_args()

def main() -> None:
    # Debug-Logging aktivieren
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s:%(name)s: %(message)s",
    )

    logger = logging.getLogger(__name__)

    # Rohdaten laden
    df = load_train_data("data/raw/train.csv")

    target_col = "SalePrice"
    Y = df[target_col].copy()
    X_raw = df.drop(columns=[target_col])

    # Feature-Pipeline: Missing Values → neue Features → Ordinal Mapping
    X = missing_value_treatment(X_raw)
    X = new_feature_engineering(X)
    X = ordinal_mapping(X)

    # Train/Test-Split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
    )

    logger.debug("X_train:", X_train.shape, "X_test:", X_test.shape)
    logger.debug("Y_train:", Y_train.shape, "Y_test:", Y_test.shape)

    # Preprocessor aus den Trainingsdaten bauen
    preprocessor = build_preprocessor(X_train)

    logger.info("Preprocessor gebaut.")
    logger.info("Starte Training.")

    # Modelle bauen, trainieren und vergleichen
    configs = [
        ("LinearRegression",      build_linear_regression_model,    False),
        ("LinearRegression_log",  build_linear_regression_model,    True),
        ("RandomForest",          build_random_forest_model,        False),
        ("RandomForest_log",      build_random_forest_model,        True),
        ("HistGBR",               build_histogram_based_model,      False),
        ("HistGBR_log",           build_histogram_based_model,      True),
    ]

    results = []

    best_cv_rmse = float("inf")
    best_config = None

    for name, builder, use_log in configs:
        print(f"\n=== Train {name} (use_log_target={use_log}) ===")
        model = builder(preprocessor, use_log_target=use_log)

        # 5a. Cross-Validation auf dem Trainingsset
        cv_scores = cross_val_score(
            model,
            X_train,
            Y_train,
            cv=5,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        rmse_cv = -cv_scores  # Vorzeichen umdrehen
        mean_cv_rmse = rmse_cv.mean()

        if mean_cv_rmse < best_cv_rmse:
            best_cv_rmse = mean_cv_rmse
            best_config = (name, builder, use_log)


        print(f"{name}: CV-RMSE = {rmse_cv.mean():.2f} ± {rmse_cv.std():.2f}")

        # 5b. Training auf dem ganzen Trainingsset
        model.fit(X_train, Y_train)

        # 5c. Evaluation auf dem Testset
        Y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
        r2 = r2_score(Y_test, Y_pred)

        print(f"{name}: Test R² = {r2:.4f}, Test RMSE = {rmse:.2f}")
        results.append((name, r2, rmse, rmse_cv.mean(), rmse_cv.std()))

    print("\n=== Zusammenfassung ===")
    for name, r2, rmse, rmse_cv_mean, rmse_cv_std in results:
        print(
            f"{name:20s} | "
            f"Test R² = {r2:.4f} | "
            f"Test RMSE = {rmse:.2f} | "
            f"CV-RMSE = {rmse_cv_mean:.2f} ± {rmse_cv_std:.2f}"
        )

    logger.debug("X shape: %s", X.shape)
    logger.debug("Y shape: %s", Y.shape)

    champion_name, champion_builder, champion_use_log = best_config
    print(f"\nChampion (nach CV): {champion_name} (use_log_target={champion_use_log})")

    # Preprocessor neu auf allen Daten fitten
    preprocessor_full = build_preprocessor(X)

    champion_model = champion_builder(preprocessor_full, use_log_target=champion_use_log)
    champion_model.fit(X, Y)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    out_path = models_dir / f"{champion_name}.joblib"

    joblib.dump(champion_model, out_path)
    print(f"Champion-Modell gespeichert unter: {out_path}")


if __name__ == "__main__":
    main()
