# Architektur – House Price Model

Dieses Dokument beschreibt die grobe Architektur des Projekts:
- Komponenten
- Datenfluss
- Datenbank-Schema
- Container-Setup
- geplante Cloud-Überführung

---

## 1. Komponentenübersicht

### 1.1 Code-Module

- `src/data.py`  
  Laden der Rohdaten (CSV) und ggf. Auftrennen in Features/Target.

- `src/features.py`  
  Kapselt alle Feature-bezogenen Transformationen:
  - `missing_value_treatment(df)`  
    → behandelt fehlende Werte (None/Median/Modus)
  - `new_feature_engineering(df)`  
    → erzeugt abgeleitete Features (z.B. `TotalSF`, `TotalBath`, `HouseAge`, `RemodAge`, `TotalPorchSF`)
  - `ordinal_mapping(df)`  
    → mappt ordinale Kategorien (z.B. Ex/Gd/TA/Fa/Po) auf Integer-Skalen

- `src/preprocessing.py`  
  - hält Listen für `numeric_features` und `categorical_features`
  - baut einen `ColumnTransformer` mit:
    - `OneHotEncoder(handle_unknown="ignore")` für kategoriale Spalten
    - `remainder="passthrough"` für numerische/ordinale Features
  - zentrale Funktion (z.B.) `build_preprocessor(X)`.

- `src/models.py`  
  Enthält Funktionen zum Bauen klassischer ML-Modelle als Pipelines:
  - `build_linear_regression_model(preprocessor, use_log_target=False)`
  - `build_random_forest_model(preprocessor, use_log_target=False)`
  - `build_hist_gradient_boosting_model(preprocessor, use_log_target=False)`
  Jedes Modell wird in einen `TransformedTargetRegressor` verpackt, um optional mit log-Target zu arbeiten.

- `src/nn_models.py`  
  - `TorchMLPRegressor`  
    sklearn-kompatibler Wrapper um ein PyTorch-MLP (Feedforward-Netz) mit:
    - ReLU-Aktivierungen
    - Adam-Optimizer
    - optionalem Learning-Rate-Scheduler
    - Early Stopping mit Validierungsset
    - Loss-Logging (`train_losses_`, `val_losses_`)
  - `build_torch_mlp_model(preprocessor, use_log_target=False)`  
    Pipeline: `preprocess → StandardScaler → TorchMLPRegressor`, verpackt in `TransformedTargetRegressor`, falls `use_log_target=True`.

- `src/db.py`  
  Enthält alle DB-bezogenen Funktionen:
  - `get_connection()`  
    → stellt Verbindung zu Postgres (`house_prices`) her
  - `init_db()` / `init_models_table()` / ggf. `init_predictions_table()`  
    → legt Tabellen an, falls nicht vorhanden
  - `insert_model(...)`  
    → schreibt Modell-Eintrag in `models`
  - `insert_predictions(kaggle_ids, predictions, model_id)`  
    → schreibt Vorhersagen in `predictions`
  - `get_current_champion_id()`  
    → liefert `id` des aktuellen Champion-Modells
  - `fetch_last_predictions(limit)`  
    → holt letzte Predictions (z.B. für Debugging/Analyse)

- `train.py`  
  Orchestriert das Training:
  1. Daten laden (`data.py`)
  2. Preprocessing-Pipeline bauen (`preprocessing.py`)
  3. Modell-Builder definieren (`models.py`, `nn_models.py`)
  4. Alle Modelle trainieren, Metriken berechnen
  5. Champion anhand des CV-RMSE auswählen
  6. Champion auf allen Daten neu trainieren
  7. Champion als `.joblib` speichern und in `models` eintragen
  8. Logging (Konsole + optional File-Logger)

- `predict.py`  
  Orchestriert die Prediction:
  1. Champion-Modell aus `models/` laden
  2. Testdaten laden und preprocessen
  3. Vorhersagen erzeugen
  4. `predictions/predictions.csv` schreiben
  5. Vorhersagen in die DB-Tabelle `predictions` eintragen (inkl. `model_id` des Champions)

---

## 2. Datenfluss

### 2.1 Training

1. **Input**  
   - `data/raw/train.csv` (Kaggle-Train-Datei; nicht versioniert im Repo)

2. **Preprocessing & Feature Engineering**
   - Missing-Value-Treatment:
     - kategorische Qualitäts-Features: `"None"` statt NaN
     - numerische Spalten: Median
     - einige Kategorien: Modus
   - Feature Engineering:
     - `HouseAge` = `YrSold - YearBuilt`
     - `RemodAge` = `YrSold - YearRemodAdd`
     - `TotalSF` = `TotalBsmtSF + 1stFlrSF + 2ndFlrSF`
     - `TotalBath` = `FullBath + 0.5 * HalfBath + BsmtFullBath + 0.5 * BsmtHalfBath`
     - `TotalPorchSF` = Summe verschiedener Porch-Flächen
   - Ordinal-Encodings:
     - mehrere Mapping-Dicts für Qualitätsstufen (Ex/Gd/TA/Fa/Po) usw.
   - `ColumnTransformer`:
     - One-Hot-Encoding der `object`-Spalten
     - numerische/ordinale Features werden durchgereicht (`passthrough`)

3. **Modelltraining**
   - Modelle werden jeweils als Pipeline `preprocessor → regressor` gebaut:
     - LinearRegression
     - RandomForestRegressor
     - HistGradientBoostingRegressor
     - TorchMLPRegressor (inkl. StandardScaler in der Pipeline)
   - Für jedes Modell:
     - 5-fold Cross-Validation (RMSE, in Euro, auf Originalskala)
     - Train/Test-Split (z.B. 80/20) zur finalen Bewertung

4. **Modellauswahl (Champion)**
   - es werden pro Modell u.a. berechnet:
     - R² (Test)
     - RMSE (Test)
     - MRE, MARE, RRMSE (relative Fehler)
     - CV-RMSE (Mittelwert + Standardabweichung)
   - Champion-Kriterium:
     - derzeit: minimaler CV-RMSE
   - Champion wird auf **allen** verfügbaren Trainingsdaten neu trainiert.

5. **Persistenz**
   - Champion-Pipeline wird mit `joblib.dump` unter `models/<NAME>.joblib` gespeichert
   - `insert_model(...)` schreibt einen Eintrag in die `models`-Tabelle:
     - Name, Version, Pfad
     - Metriken (R², RMSE, MARE, CV-RMSE)
     - Hyperparameter (JSONB)
     - `is_champion = TRUE` (ggf. vorher andere Champions auf FALSE setzen)

### 2.2 Prediction

1. **Input**
   - Testdaten aus `data/raw/test.csv` (Struktur wie Kaggle-Test)

2. **Ablauf**
   - aktueller Champion wird mittels `get_current_champion_id()` / Dateipfad ermittelt
   - Modell wird aus der `.joblib` geladen
   - es werden dieselben Preprocessing-Stufen wie im Training ausgeführt (dank Pipeline)
   - Vorhersagen für alle Test-Samples werden erzeugt

3. **Persistenz**
   - `predictions/predictions.csv` wird geschrieben:
     - Spalten: `Id`, `SalePrice`
   - `insert_predictions(...)` schreibt Vorhersagen in die Tabelle `predictions`:
     - `kaggle_id` (aus Testdaten)
     - `predicted_price`
     - `model_id` (FK auf `models.id`)
     - `created_at` (Timestamp)

4. **Auswertung**
   - über DBeaver, SQL oder BI-Tools kann man:
     - Predictions des Champions ansehen
     - Predictions verschiedener Champions (in Zukunft) vergleichen
     - pro Modell Vorhersage-Statistiken berechnen

---

## 3. Datenbank-Schema

Das vollständige Schema ist in `sql/schema.sql` dokumentiert.

### 3.1 Tabelle `models`

Zweck: **Model Registry**

Wichtige Spalten:

- `id SERIAL PRIMARY KEY`
- `name TEXT`  
  z.B. `'HistGBR_log'`, `'LinearRegression_log'`, `'TorchMLP'`
- `version TEXT`  
  z.B. Timestamp-String `'20251204-171745'`
- `file_path TEXT`  
  Pfad zur `.joblib`-Datei des Modells
- `r2_test DOUBLE PRECISION`
- `rmse_test DOUBLE PRECISION`
- `mare_test DOUBLE PRECISION`
- `cv_rmse_mean DOUBLE PRECISION`
- `cv_rmse_std DOUBLE PRECISION`
- `hyperparams JSONB`  
  Hyperparameter des inneren Regressors (z.B. `max_depth`, `n_estimators`, Layergrößen)
- `is_champion BOOLEAN`
- `created_at TIMESTAMPTZ`

Typische Abfragen:

- „aktuellen Champion finden“
- „Verlauf der Modelle und ihrer Metriken“
- „Modell mit bestimmten Hyperparametern/Versionen anschauen“

### 3.2 Tabelle `predictions`

Zweck: **Prediction Store**

Wichtige Spalten:

- `id SERIAL PRIMARY KEY`
- `kaggle_id INTEGER`  
  die `Id` des Hauses aus den Kaggle-Testdaten
- `predicted_price DOUBLE PRECISION`  
  vorhergesagter Verkaufspreis
- `model_id INTEGER REFERENCES models(id)`  
  welches Modell hat vorhergesagt?
- `created_at TIMESTAMPTZ`

Typische Abfragen:

- „alle Predictions des aktuellen Champions“
- „Predictive Drift / Unterschied zwischen Champions (zukünftig)“
- „Vorhersage für ein bestimmtes Haus (`kaggle_id`)“

### 3.3 View `v_predictions_with_model`

Zweck: **Komfortabler Join** für Analysen & BI.

Definiert in `sql/schema.sql` als:

- Join von `predictions` und `models`
- liefert u.a.:
  - `kaggle_id`
  - `predicted_price`
  - `model_name`, `model_version`
  - `prediction_created_at`

Nützlich für:

- DBeaver-Analysen
- Tableau-Dashboards
- schnelle Übersicht, welches Modell welche Predictions erzeugt hat

---

## 4. Container & Cloud-Perspektive

### 4.1 Container / Docker

Aktuell:

- PostgreSQL läuft im Docker-Container:
  - Image: `postgres:16`
  - Start z.B. über:
    ```bash
    docker run --name house-price-postgres \
      -e POSTGRES_USER=house \
      -e POSTGRES_PASSWORD=house \
      -e POSTGRES_DB=house_prices \
      -p 5432:5432 \
      -d postgres:16
    ```
- Projektordner enthält ein `Dockerfile`:
  - Basis: `python:3.11-slim`
  - installiert `requirements.txt`
  - kopiert den Projekt-Code in `/app`
  - ermöglicht das Training im Container (z.B. `docker run house-price-model python train.py`)

Python-Skripte (`train.py`, `predict.py`) laufen derzeit primär lokal in einer venv, können aber in Zukunft ebenfalls über Docker orchestriert werden (z.B. via `docker-compose` oder Cloud-Orchestrierung).

### 4.2 Geplante Cloud-Architektur (AWS)

Die Architektur ist bereits so angelegt, dass eine Migration in die Cloud möglich ist:

- **Compute**  
  - Containerisiertes Training und Serving (z.B. AWS ECS/Fargate)
- **Datenbank**  
  - Amazon RDS für PostgreSQL anstelle des lokalen Docker-Containers
- **Speicher**  
  - S3-Bucket für Modell-Dateien (.joblib) und evtl. Logs/Artefakte
- **IaC (Infrastructure as Code)**  
  - Terraform-Konfigurationen (`infra/terraform/` geplant), um:
    - RDS-Instanz
    - Security Groups
    - ECS-Services / Tasks
    zu definieren

Die aktuelle Struktur (saubere Trennung von Code, Modellen, Datenbank-Schema) ist bewusst so gewählt, dass diese Komponenten später in Terraform-/Cloud-Skripte überführt werden können.