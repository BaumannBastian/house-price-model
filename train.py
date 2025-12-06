# ------------------------------
# train.py
#
# In dieser Python-Datei werden die Trainingsdaten geladen, Features
# erzeugt, mehrere sklearn- und PyTorch-Modelle trainiert und verglichen.
# Das beste Modell (Champion) wird gespeichert und seine Metadaten in der
# PostgreSQL-Datenbank registriert.
# ------------------------------

from src.data import load_train_data
from src.features import missing_value_treatment, new_feature_engineering, ordinal_mapping
from src.preprocessing import build_preprocessor
from src.models import (
    build_linear_regression_model,
    build_random_forest_model,
    build_histogram_based_model,
)
from src.nn_models import build_torch_mlp_model
from src.db import init_models_table, insert_model

import argparse
import logging
import warnings
import joblib
import numpy as np

from datetime import datetime
from pathlib import Path
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def parse_args() -> argparse.Namespace:
    """
    Parst Kommandozeilenargumente für das Trainingsskript.

    Unterstützte Flags
    ------------------
    --debug :
        Aktiviert detaillierteres Logging (DEBUG-Level).
    --nn-verbose :
        Gibt während des TorchMLP-Trainings den Loss auf der Konsole aus.
    --nn-plot-loss :
        Speichert nach dem TorchMLP-Training eine Loss-Kurve als PNG
        im Ordner ``plots``.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.Namespace
        Namespace mit den Attributen ``debug``, ``nn_verbose`` und
        ``nn_plot_loss``.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--nn-verbose",
        action="store_true",
        help="TorchMLP während des Trainings Loss ausgeben",
    )
    parser.add_argument(
        "--nn-plot-loss",
        action="store_true",
        help="Loss-Kurve des TorchMLP nach dem Training als PNG speichern",
    )
    return parser.parse_args()


def setup_logging(debug: bool) -> logging.Logger:
    """
    Richtet konsistentes Logging für Konsole und Logdatei ein.

    Es wird ein ``logs/``-Verzeichnis erstellt, ein Root-Logger
    konfiguriert und sowohl ein Konsolen- als auch ein Datei-Handler
    (``logs/train.log``) registriert. Zusätzlich werden Python-Warnings
    in das Logging-System umgeleitet.

    Im Debug-Modus (``debug=True``) werden alle Meldungen auf DEBUG-Level
    ausgegeben, ansonsten auf INFO-Level. Bestimmte sklearn-
    ``FutureWarning``-Meldungen werden im Normalmodus unterdrückt.

    Parameters
    ----------
    debug : bool
        Ob das Skript im Debug-Modus läuft. Beeinflusst Log-Level und
        Behandlung von Warnings.

    Returns
    -------
    logging.Logger
        Modul-Logger für dieses Skript, der an den konfigurierten
        Root-Logger angebunden ist.
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Root-Logger holen & Level setzen
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if debug else logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Alte Handler entfernen, falls mehrfach aufgerufen
    if root.handlers:
        root.handlers.clear()

    # Konsole
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_handler.setFormatter(fmt)

    # Datei
    file_handler = logging.FileHandler(
        log_dir / "train.log",
        mode="w",
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)  # immer alles ins File
    file_handler.setFormatter(fmt)

    root.addHandler(console_handler)
    root.addHandler(file_handler)

    # Warnings in Logging umleiten
    logging.captureWarnings(True)

    if debug:
        # im Debug-Mode: Warnings normal loggen
        warnings.filterwarnings("default")
    else:
        # im normalen Mode: die sklearn-Pipeline-FutureWarnings unterdrücken
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message="This Pipeline instance is not fitted yet.*",
        )

    # Modul-Logger zurückgeben
    return logging.getLogger(__name__)


def main() -> None:
    """
    Hauptfunktion des Trainingsskripts.

    Ablauf
    ------
    1. Parst Kommandozeilenargumente und richtet das Logging ein.
    2. Lädt den Kaggle-Train-Datensatz und trennt Target (``SalePrice``)
       von den Features.
    3. Führt Missing-Value-Treatment, Feature-Engineering und Ordinal-Encoding
       für die Input-Features durch.
    4. Splittet die Daten in Trainings- und Testset.
    5. Baut verschiedene Modelle (Linear Regression, Random Forest,
       Histogram-Gradient-Boosting, TorchMLP) jeweils mit/ohne Log-Target.
    6. Führt Cross-Validation und Test-Set-Evaluation durch und bestimmt
       ein Champion-Modell basierend auf dem CV-RMSE.
    7. Trainiert den Champion auf allen Daten, speichert ihn als ``.joblib``
       und legt einen Eintrag in der Tabelle ``models`` in PostgreSQL an.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Die Funktion hat keinen Rückgabewert. Ergebnisse werden über
        Logging, Konsolenausgabe, gespeicherte Modell-Dateien und
        Datenbankeinträge sichtbar.
    """
    # Kommandozeilenargumente parsen und Logging einrichten
    args = parse_args()
    logger = setup_logging(args.debug)

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

    logger.debug("X_train shape: %s, X_test shape: %s", X_train.shape, X_test.shape)
    logger.debug("Y_train shape: %s, Y_test shape: %s", Y_train.shape, Y_test.shape)

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
        ("TorchMLP",              build_torch_mlp_model,            False),
        # ("TorchMLP_log",          build_torch_mlp_model,            True),
    ]

    results: list[tuple[str, float, float, float, float, float, float, float]] = []

    best_cv_rmse = float("inf")
    best_config: tuple[str, callable, bool] | None = None

    for name, builder, use_log in configs:
        logger.info("=== Train %s (use_log_target=%s) ===", name, use_log)

        model = builder(preprocessor, use_log_target=use_log)

        # Falls TorchMLP und --nn-verbose: Verbose aktivieren
        if args.nn_verbose and name.startswith("TorchMLP"):
            # TTR → Pipeline → TorchMLPRegressor
            inner_mlp = model.regressor.named_steps["regressor"]
            inner_mlp.verbose = True

        print(f"\n=== Train {name} (use_log_target={use_log}) ===")

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

        rel_errors = (Y_pred - Y_test) / Y_test

        mre = rel_errors.mean()                     # signed, mittlerer relativer Fehler
        mare = np.abs(rel_errors).mean()            # mittlerer absoluter relativer Fehler
        rrmse = np.sqrt((rel_errors**2).mean())     # relative RMSE

        if args.nn_plot_loss and name.startswith("TorchMLP"):
            import matplotlib.pyplot as plt  # lokal, um harte Abhängigkeit zu vermeiden

            Path("plots").mkdir(exist_ok=True)

            inner_mlp = model.regressor_.named_steps["regressor"]
            losses = getattr(inner_mlp, "train_losses_", None)

            if losses is not None and len(losses) > 0:
                plt.figure()
                plt.plot(losses)
                plt.xlabel("Epoch")
                plt.ylabel("MSE loss")
                plt.yscale("log")
                out_path = Path("plots") / f"{name}_loss_curve.png"
                plt.savefig(out_path, bbox_inches="tight")
                plt.close()
                logger.info(
                    "Losskurve für %s gespeichert unter: %s",
                    name,
                    out_path.resolve(),
                )
            else:
                logger.warning(
                    "Keine train_losses_ für %s gefunden – wurde fit() aufgerufen?",
                    name,
                )

        print(f"{name}: Test R² = {r2:.4f}, Test RMSE = {rmse:.2f}")
        results.append(
            (name, r2, rmse, mre, mare, rrmse, rmse_cv.mean(), rmse_cv.std())
        )

    if best_config is None:
        raise RuntimeError("Kein Champion-Modell gefunden – Config-Liste leer?")

    init_models_table()

    print("\n=== Zusammenfassung ===")
    for name, r2, rmse, mre, mare, rrmse, rmse_cv_mean, rmse_cv_std in results:
        print(
            f"{name:20s} | "
            f"Test R² = {r2:.4f} | "
            f"Test RMSE = {rmse:.2f} | "
            f"MRE = {mre*100:6.2f}% | "
            f"MARE = {mare*100:5.1f}% | "
            f"RRMSE = {rrmse*100:5.1f}% | "
            f"CV-RMSE = {rmse_cv_mean:.2f} ± {rmse_cv_std:.2f}"
        )

    logger.debug("X shape: %s", X.shape)
    logger.debug("Y shape: %s", Y.shape)

    champion_name, champion_builder, champion_use_log = best_config
    print(f"\nChampion (nach CV): {champion_name} (use_log_target={champion_use_log})")

    # Metrics des Champions aus der results-Liste holen
    champion_row = next(
        (row for row in results if row[0] == champion_name),
        None,
    )
    if champion_row is None:
        raise RuntimeError(
            f"Champion-Metriken für {champion_name} nicht gefunden."
        )

    (
        _,
        champion_r2,
        champion_rmse,
        champion_mre,
        champion_mare,
        champion_rrmse,
        champion_cv_mean,
        champion_cv_std,
    ) = champion_row

    # Preprocessor neu auf allen Daten fitten
    preprocessor_full = build_preprocessor(X)

    champion_model = champion_builder(
        preprocessor_full,
        use_log_target=champion_use_log,
    )
    champion_model.fit(X, Y)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    out_path = models_dir / f"{champion_name}.joblib"

    joblib.dump(champion_model, out_path)

    # Hyperparameter des inneren Regressors holen
    # (gilt für Sklearn-Modelle UND deinen TorchMLP, weil BaseEstimator)
    base = champion_model.regressor      # TransformedTargetRegressor.regressor
    inner_reg = base.named_steps["regressor"]  # eigentlicher Estimator

    try:
        hyperparams = inner_reg.get_params(deep=False)
    except Exception:
        hyperparams = {}

    # einfache Versionskennung – z.B. Datum + Modellname
    version = datetime.now().strftime("%Y%m%d-%H%M%S")

    insert_model(
        name=champion_name,
        version=version,
        file_path=str(out_path),
        r2_test=champion_r2,
        rmse_test=champion_rmse,
        mare_test=champion_mare,
        cv_rmse_mean=champion_cv_mean,
        cv_rmse_std=champion_cv_std,
        is_champion=True,
        hyperparams=hyperparams,
    )

    print(f"Champion-Modell gespeichert unter: {out_path}")


if __name__ == "__main__":
    main()