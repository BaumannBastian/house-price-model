# House Price Model

End-to-End Machine-Learning-Projekt zur Vorhersage von Hauspreisen  
(basierend auf dem Kaggle-Datensatz *House Prices – Advanced Regression Techniques*).

Ziel des Projekts:

- saubere ML-Pipeline mit `scikit-learn` und `PyTorch`
- reproduzierbares Training inkl. Cross-Validation
- Model Registry in PostgreSQL
- Prediction-Store in PostgreSQL
- Docker-Setup, das sich später in eine Cloud-Umgebung (AWS) heben lässt

---

## Projektstruktur

    .
    ├─ src/
    │  ├─ data.py            # Laden der Rohdaten
    │  ├─ features.py        # Missing-Value-Treatment & Feature Engineering
    │  ├─ preprocessing.py   # ColumnTransformer & Feature-Listen
    │  ├─ models.py          # klassische ML-Modelle (Linear, RF, HistGBR)
    │  ├─ nn_models.py       # TorchMLPRegressor + Pipeline
    │  ├─ db.py              # PostgreSQL-Anbindung (models & predictions)
    │  └─ __init__.py
    ├─ train.py              # Trainiert alle Modelle, wählt Champion, loggt in DB
    ├─ predict.py            # Lädt Champion, erzeugt Predictions & schreibt in DB
    ├─ sql/
    │  └─ schema.sql         # vollständiges DB-Schema (models, predictions, Indizes, View)
    ├─ docs/
    │  ├─ architecture.md    # Architektur-Übersicht (Datenfluss, Komponenten)
    │  └─ experiments.md     # Experiment-Log (Modelle & Ergebnisse)
    ├─ models/               # gespeicherte Champion-Modelle (.joblib)
    ├─ predictions/          # Predictions als CSV
    ├─ logs/                 # Logging (z.B. train.log)
    ├─ plots/                # Trainingsplots (z.B. TorchMLP-Losskurven)
    ├─ requirements.txt
    ├─ Dockerfile
    ├─ .dockerignore
    └─ .gitignore

---

## Daten

### Dateibasiert

- Basis: Kaggle „House Prices: Advanced Regression Techniques“
- Train-Datei: `data/raw/train.csv` (lokal, nicht versioniert)
- Test-Datei (für Kaggle-Submission): `data/raw/test.csv` (lokal, nicht versioniert)
- Predictions: `predictions/predictions.csv` (wird von `predict.py` erzeugt)

> Hinweis: `data/`, `models/`, `logs/`, `plots/` und `predictions` sind in
> `.gitignore` und `.dockerignore` eingetragen, da es sich um Artefakte bzw.
> lokale Daten handelt.

### Datenbank (PostgreSQL)

- Läuft in einem Docker-Container (`postgres:16`)
- Datenbank: `house_prices`
- Tabellen:
  - `models`  
    → Model Registry (Name, Version, Pfad zur `.joblib`, Metriken, Hyperparameter, Champion-Flag)
  - `predictions`  
    → Prediction-Store (`kaggle_id`, `predicted_price`, `model_id`, `created_at`)
- View:
  - `v_predictions_with_model`  
    → Join von `predictions` und `models` (z.B. für BI/Monitoring)
- Vollständiges Schema: `sql/schema.sql`

---

## Pipeline (kurz)

1. **Laden & Preprocessing**  
   - Missing Values: None/Median/Modus-Strategien (siehe `features.py`)
   - Feature Engineering: `TotalSF`, `TotalBath`, `HouseAge`, etc.
   - Ordinale Features: Mapping auf `int` (z.B. Ex/Gd/TA/Fa/Po → 5…1)
   - `ColumnTransformer` + `OneHotEncoder` für kategoriale Features

2. **Modelle**
   - `LinearRegression` (+ optional log-Target via `TransformedTargetRegressor`)
   - `RandomForestRegressor`
   - `HistGradientBoostingRegressor`
   - `TorchMLPRegressor` (PyTorch-Multi-Layer-Perceptron, sklearn-kompatibel)

3. **Evaluation**
   - Train/Test-Split (80/20)
   - Cross-Validation (5-fold)
   - Metriken:
     - R²
     - RMSE (in Euro)
     - MARE / RRMSE (relative Fehler in %)

4. **Persistenz**
   - Champion-Modell wird als `.joblib` unter `models/` gespeichert
   - Eintrag in `models` (Postgres) inkl. Metriken und Hyperparametern
   - `predict.py` schreibt Vorhersagen zusätzlich in `predictions` (Postgres)

---

## Aktueller Stand (kurze Summary)

- Bester CV-Performer: **HistGBR mit log-Target** (`HistGBR_log`)
- Typischer Fehler:
  - ca. 9–10 % relativer Fehler (MARE),
  - RMSE ~ 29–30k € (je nach Modell)
- TorchMLP:
  - R² ~ 0.87,
  - aber schlechtere relative Fehler und höhere CV-RMSE als HistGBR_log.

Details zu Experimenten und Metriken stehen in `docs/experiments.md`.

---

## Setup

### Python-Umgebung

    python -m venv .venv
    # Windows:
    .venv\Scripts\activate
    # Linux/Mac:
    source .venv/bin/activate

    pip install -r requirements.txt

### PostgreSQL in Docker

    docker run --name house-price-postgres ^
      -e POSTGRES_USER=house ^
      -e POSTGRES_PASSWORD=house ^
      -e POSTGRES_DB=house_prices ^
      -p 5432:5432 ^
      -d postgres:16

(unter Linux/Mac `^` durch `\` ersetzen)

Die Verbindungseinstellungen sind in `src/db.py` hinterlegt:

- Host: `localhost`
- Port: `5432`
- DB: `house_prices`
- User: `house`
- Passwort: `house`

---

## Docker-Image für das Training (optional)

Das Projekt bringt ein einfaches Dockerfile für das Training mit.  
Voraussetzung: Die PostgreSQL-Instanz ist vom Container aus erreichbar.

    # Image bauen
    docker build -t house-price-train .

    # Training im Container ausführen
    docker run --rm house-price-train

(Standard-CMD im Container ist `python train.py`.)

---

## Wie ausführen

### Training

    python train.py

- trainiert alle konfigurierten Modelle,
- berechnet Test- und CV-Metriken,
- wählt einen Champion (nach CV-RMSE),
- trainiert den Champion auf allen Daten,
- speichert ihn als `.joblib` unter `models/`,
- trägt ihn in der Tabelle `models` (PostgreSQL) ein (inkl. Hyperparametern).

### Predictions

    python predict.py

- lädt das aktuelle Champion-Modell aus `models/`,
- erzeugt Vorhersagen für den Testdatensatz,
- schreibt `predictions/predictions.csv`,
- fügt alle Vorhersagen in die Tabelle `predictions` ein
  (inkl. Verweis auf `models.id` des Champions).

---

## Roadmap / weitere Planung

Geplante Erweiterungen:

- **Cloud (AWS)**  
  - RDS-Postgres  
  - Modellablage in S3  
  - Container-Deployment (z.B. ECS/Fargate) für Training und/oder Prediction-API

- **API-Layer**  
  - z.B. FastAPI-Service, der das Champion-Modell als REST-Endpunkt bereitstellt

- **Terraform**  
  - IaC-Definition der Cloud-Ressourcen (DB, Compute, S3)

- **BI / Visualisierung**  
  - Tableau an PostgreSQL anbinden  
  - Dashboards auf Basis von `models`, `predictions` und `v_predictions_with_model`

- **Erweiterte Experimente**  
  - systematisches Hyperparameter-Tuning  
  - zusätzliche Feature-Engineering-Schritte  
  - weitere Modellklassen (XGBoost/LightGBM)