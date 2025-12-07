# Architektur – House Price Model

Dieses Dokument beschreibt die Architektur des Projekts:

- zentrale Komponenten
- Datenfluss (Training & Prediction)
- Datenbankschema
- Container-/Cloud-Setup (Azure)
- Lessons Learned bei der Cloud-DB

---

## 1. Komponentenübersicht

### 1.1 Core-Code (`src/`)

- `src/data.py`  
  Laden der Rohdaten (Train/Test) aus CSV-Dateien in Pandas-DataFrames.

- `src/features.py`  
  Feature-bezogene Transformationen:
  - Missing-Value-Treatment (None/Median/Modus, je nach Spalte)
  - Feature Engineering (z.B. `TotalSF`, `TotalBath`, `HouseAge`, `RemodAge`, `TotalPorchSF`)
  - Ordinal-Encodings (z.B. Ex/Gd/TA/Fa/Po → 5…1)

- `src/preprocessing.py`  
  - definiert `numeric_features` und `categorical_features`
  - baut einen `ColumnTransformer`:
    - `OneHotEncoder(handle_unknown="ignore")` für kategoriale Spalten
    - numerische/ordinale Features werden durchgereicht (`remainder="passthrough"`)
  - zentrale Funktion: Preprocessor für das aktuelle Feature-Set bauen.

- `src/models.py`  
  Builder-Funktionen für klassische scikit-learn-Modelle:

  - `build_linear_regression_model(preprocessor, use_log_target=False)`
  - `build_random_forest_model(preprocessor, use_log_target=False)`
  - `build_hist_gradient_boosting_model(preprocessor, use_log_target=False)`

  Die Modelle werden als Pipelines `preprocess → regressor` gebaut und bei Bedarf in einen `TransformedTargetRegressor` eingebettet (log-transformiertes Target).

- `src/nn_models.py`  
  PyTorch-MLP, sklearn-kompatibel:

  - `TorchMLPRegressor` (Feedforward-Netz mit ReLU, Adam, Early Stopping)
  - `build_torch_mlp_model(preprocessor, use_log_target=False)`:
    - Pipeline `preprocess → StandardScaler → TorchMLPRegressor`
    - optional mit log-Target über `TransformedTargetRegressor`.

- `src/db.py`  
  Kapselt alle DB-Zugriffe (PostgreSQL, lokal oder Azure):

  - `get_connection()`  
    liest Konfiguration aus Umgebungsvariablen:

    - `DB_HOST`
    - `DB_PORT`
    - `DB_NAME`
    - `DB_USER`
    - `DB_PASSWORD`
    - `DB_SSLMODE` (z.B. `"require"` für Azure, `"disable"` für lokale Docker-DB)

    Falls ENV-Variablen fehlen, werden lokale Defaults verwendet (Postgres auf `localhost`).

  - Schreibende Operationen:
    - `insert_model(...)` – legt einen Eintrag in `models` an
    - `insert_predictions(kaggle_ids, predictions, model_id)` – schreibt in `predictions`

  - Lesende Operationen:
    - `get_current_champion_id()` – ermittelt die aktuelle Champion-ID
    - `fetch_last_predictions(limit)` – hilft beim Debuggen / Analysieren

- `src/__init__.py`  
  Markiert `src` als Python-Paket.

---

### 1.2 Orchestrierung (Project Root)

- `train.py`  

  Orchestriert das komplette Training:

  1. Trainingsdaten laden (`src/data.py`).
  2. Feature Engineering & Preprocessing-Pipeline aufbauen (`src/features.py`, `src/preprocessing.py`).
  3. Modellfamilien definieren (`src/models.py`, `src/nn_models.py`).
  4. Für jedes Modell:
     - 5-fold Cross-Validation (CV-RMSE)
     - Test-Set-Evaluation (R², RMSE, relative Fehler)
  5. Champion anhand CV-RMSE auswählen.
  6. Champion auf allen Daten neu trainieren.
  7. Champion als `.joblib` unter `models/` speichern.
  8. Champion in der Tabelle `models` persistieren (inkl. Metriken, Hyperparametern, `is_champion`).

- `predict.py`  

  Orchestriert die Inferenz:

  1. Champion aus der Tabelle `models` (DB) und aus dem Dateisystem (Pfad zur `.joblib`) bestimmen.
  2. Testdaten (`data/raw/test.csv`) laden.
  3. Vorhersagen über die Modell-Pipeline erzeugen.
  4. `predictions/predictions.csv` schreiben.
  5. Vorhersagen in `predictions` in der DB speichern (inkl. `model_id` des Champions).

---

### 1.3 Utility-Skripte (`scripts/`)

- `scripts/test_db_connection.py`  
  Testet die DB-Verbindung (`get_connection()`) via einfachem:

    SELECT version(), current_database(), current_user;

- `scripts/init_schema.py`  
  - liest `sql/schema.sql`
  - führt das Schema gegen die aktuell konfigurierte DB aus (lokal oder Azure)
  - sorgt dafür, dass Tabellen, Indizes und View konsistent angelegt sind.

- `scripts/show_db_sample.py`  
  - gibt Beispielzeilen aus:
    - `models`
    - `v_predictions_with_model`
  - dient zum schnellen Check, ob Trainings-/Prediction-Läufe in der DB ankommen.

- `scripts/set_env_azure_db.ps1`  
  - PowerShell-Skript zum Setzen von:
    - `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_SSLMODE`
  - Werte stammen aus den Terraform-Outputs der Azure-DB.

- `scripts/__init__.py`  
  Markiert `scripts` als Paket, damit `python -m scripts.<name>` funktioniert.

---

### 1.4 Infrastruktur & Packaging

- `sql/schema.sql`  
  Enthält das vollständige Schema:

  - Tabelle `models`
  - Tabelle `predictions`
  - Foreign Key von `predictions.model_id` → `models.id`
  - Indizes auf typischen Query-Feldern
  - View `v_predictions_with_model`

- `terraform/main.tf`, `terraform/variables.tf`, `terraform/outputs.tf`  
  Terraform für Azure:

  - Resource Group `house-price-rg`
  - Azure Database for PostgreSQL – Flexible Server (`house-price-psql`, Region `northeurope`)
  - Datenbank `house_prices`
  - Firewall-Regel für die aktuelle öffentliche IP (Variable `client_ip`)
  - Outputs: FQDN, Admin-User, DB-Name, Beispiel-DSN

- `pyproject.toml`  
  Minimaler Packaging-Setup:

  - Projektname `house-price-model`
  - `setuptools` + `wheel`
  - `src` als Package
  - ermöglicht `pip install -e .` und saubere Importe (`from src.db import get_connection` etc.).

---

## 2. Datenfluss

### 2.1 Training

1. Input-Daten  
   - `data/raw/train.csv` (Kaggle-Train-Set, nicht versioniert im Repo)

2. Preprocessing & Feature Engineering  
   - Laden der Daten in `train.py` via `src/data.py`.
   - `src/features.py`:
     - Missing-Value-Treatment (je Spalte sinnvolle Strategie).
     - Feature Engineering: zusammengesetzte Flächen, Bäder, Altersmerkmale.
     - Ordinal-Encodings auf Integer-Skalen.
   - `src/preprocessing.py`:
     - `ColumnTransformer` mit One-Hot-Encoding für kategoriale Features.
     - numerische/ordinale Features werden durchgereicht (`passthrough`).

3. Modelle  
   - klassische Modelle:
     - `LinearRegression` / `LinearRegression_log`
     - `RandomForest` / `RandomForest_log`
     - `HistGradientBoostingRegressor` / `HistGBR_log`
   - neuronales Modell:
     - `TorchMLP` (in Pipeline mit `StandardScaler`)

   Alle Modelle sind als sklearn-Pipelines implementiert, optional mit `TransformedTargetRegressor` (log-Target).

4. Evaluation  
   - 5-fold Cross-Validation (CV-RMSE) pro Modell.
   - Test-Set-Evaluation:
     - R²
     - RMSE
     - MARE/RRMSE (relative Fehler).
   - alle Metriken werden im Log ausgegeben und in den DB-Record des Modells übernommen.

5. Champion-Auswahl & Persistenz  
   - Auswahlkriterium: minimaler CV-RMSE.
   - Champion wird auf allen Trainingsdaten neu trainiert.
   - Persistenz:
     - `.joblib` in `models/`.
     - Eintrag in `models` (Postgres) mit:
       - Name, Version (Timestamp-String), `file_path`
       - Test-Metriken, CV-Metriken
       - Hyperparametern (als JSONB)
       - `is_champion = TRUE` (vorherige Champions werden deaktiviert).

---

### 2.2 Prediction

1. Champion ermitteln  
   - `predict.py` ruft `get_current_champion_id()` aus `src/db.py` auf.
   - `file_path` des Champions wird aus der Tabelle `models` gelesen.
   - `.joblib`-Pipeline wird geladen.

2. Testdaten laden  
   - `data/raw/test.csv` (Kaggle-Test-Set, nicht im Repo).

3. Vorhersage  
   - identische Preprocessing-Pipeline wie im Training (Teil des Modells).
   - Modell erzeugt `SalePrice`-Vorhersagen für alle Test-Samples.

4. Persistenz  
   - `predictions/predictions.csv` wird geschrieben (für Kaggle).
   - `insert_predictions(...)` legt alle Vorhersagen in `predictions` ab:
     - `kaggle_id`
     - `predicted_price`
     - `model_id` (FK auf `models.id`)
     - `created_at`.

5. Nutzung / Analysen  
   - Zugriff auf:
     - `models` – Modell-Historie & Metriken
     - `predictions` – alle Vorhersagen
     - `v_predictions_with_model` – kombinierte Sicht für BI/Monitoring

---

## 3. Datenbankschema

### 3.1 Tabelle `models` (Model Registry)

- `id SERIAL PRIMARY KEY`
- `name TEXT NOT NULL`  
  (z.B. `HistGBR_log`, `LinearRegression_log`, `TorchMLP`)
- `version TEXT NOT NULL`  
  (z.B. `20251207-233208`)
- `file_path TEXT NOT NULL`  
  (Pfad zur `.joblib`-Datei)
- `r2_test DOUBLE PRECISION`
- `rmse_test DOUBLE PRECISION`
- `mare_test DOUBLE PRECISION`
- `cv_rmse_mean DOUBLE PRECISION`
- `cv_rmse_std DOUBLE PRECISION`
- `hyperparams JSONB`
- `is_champion BOOLEAN NOT NULL DEFAULT FALSE`
- `created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()`

### 3.2 Tabelle `predictions` (Prediction Store)

- `id SERIAL PRIMARY KEY`
- `kaggle_id INTEGER NOT NULL`  
  (Id aus `test.csv`)
- `predicted_price DOUBLE PRECISION NOT NULL`
- `model_id INTEGER`  
  (FK auf `models.id`)
- `created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()`

### 3.3 View `v_predictions_with_model`

- Join von `predictions` und `models`:

  - `p.id`
  - `p.kaggle_id`
  - `p.predicted_price`
  - `p.model_id`
  - `m.name        AS model_name`
  - `m.version     AS model_version`
  - `p.created_at  AS prediction_created_at`

- ideal für:
  - DBeaver-/psql-Abfragen
  - Tableau / Power BI
  - Monitoring/Reporting.

---

## 4. Container & Cloud

### 4.1 Docker (lokale Option)

- Lokale Postgres-Instanz kann in Docker laufen:

    docker run --name house-price-postgres ^
      -e POSTGRES_USER=house ^
      -e POSTGRES_PASSWORD=house ^
      -e POSTGRES_DB=house_prices ^
      -p 5432:5432 ^
      -d postgres:16

- Applikations-ENV für lokale DB:

    DB_HOST=localhost
    DB_PORT=5432
    DB_NAME=house_prices
    DB_USER=house
    DB_PASSWORD=house
    DB_SSLMODE=disable

- Das `Dockerfile` im Projekt baut ein Image, das `train.py` im Container ausführen kann (z.B. für CI oder späteres Cloud-Compute).

### 4.2 Azure-Architektur (aktuell im Einsatz)

Ziel: Datenbank in der Cloud, Compute zunächst lokal.

- Compute:
  - aktuell lokal (venv oder Docker).
  - perspektivisch Container auf Azure (z.B. Azure Container Apps / Azure Container Instances).

- Datenbank:
  - Azure Database for PostgreSQL – Flexible Server:
    - Region `northeurope`
    - DB `house_prices`
    - SKU `B_Standard_B1ms` (kleine, kostengünstige Entwicklungsinstanz)
    - 32 GB Storage.
  - Provisionierung via Terraform:
    - Resource Group `house-price-rg`
    - Server `house-price-psql`
    - DB `house_prices`
    - Firewall-Regel für die aktuelle öffentliche IP.

- Konfiguration:
  - Terraform-Outputs liefern:
    - `postgres_fqdn`
    - `postgres_admin_username`
    - `postgres_admin_password`
    - `postgres_database_name`
  - diese Werte werden in `scripts/set_env_azure_db.ps1` eingetragen.
  - `src/db.py` liest ausschließlich aus ENV (kein Hardcoding von Secrets).

- BI:
  - Tableau / Power BI können direkt auf die Azure-DB zugreifen:
    - Host: FQDN
    - Port: 5432
    - DB: `house_prices`
    - User: Admin oder Readonly-Benutzer
    - SSL aktiv
  - meist genutzte Basis: View `v_predictions_with_model`.

---

## 5. Cloud-DB auf Azure: Lessons Learned

### 5.1 Fehlgeschlagene erste Versuche

- Erste Terraform-Versuche mit Azure PostgreSQL – Flexible Server in den Regionen:
  - `germanywestcentral` (Frankfurt)
  - `westeurope` (Niederlande)

- Resultat: wiederholte Fehler von der Azure-API:

  - `LocationIsOfferRestricted`  
    → Subscription darf in der jeweiligen Region keinen Flexible Server provisionieren.

  - zeitweise zusätzlich `AvailabilityZoneNotAvailable`, wenn explizit `zone = "1"` gesetzt war.

- Ursachen:
  - Kombination aus Free-/Test-Subscription + Region + Angebot (Flexible Server-SKU).

### 5.2 Systematische Analyse & Lösung

- Statt zufällige Regionen zu probieren, wurden SKUs mit der Azure CLI geprüft:

    az postgres flexible-server list-skus -l <region>

- In `germanywestcentral` / `westeurope` waren Restriktionen sichtbar, in `northeurope` nicht.
- Anpassungen in Terraform:
  - `location = "northeurope"` in `variables.tf`
  - keine explizite Zone mehr gesetzt (Azure wählt intern eine passende Zone)
  - sicherstellen, dass die korrekte Subscription verwendet wird (via `subscription_id` im Provider oder `az account set`).

- Ergebnis:
  - `terraform apply` legt nun erfolgreich an:
    - Resource Group `house-price-rg`
    - PostgreSQL Flexible Server `house-price-psql`
    - Datenbank `house_prices`
    - Firewall-Regel für die eigene IP

### 5.3 Aktueller Stand

- Es gibt bewusst nur **eine** Terraform-Variante im Repo:
  - `terraform/main.tf`, `terraform/variables.tf`, `terraform/outputs.tf`
  - diese Variante provisioniert den Azure PostgreSQL Flexible Server in `northeurope`.
- Die Anwendung (Train/Prediction) ist so gebaut, dass sie:
  - lokal gegen eine Docker-Postgres-Instanz laufen kann,
  - und ohne Codeänderung gegen die Azure-Postgres-DB (nur ENV wechseln).
- Zukünftige Arbeiten (Webservice, BI-Anbindung, evtl. AWS-Portierung) bauen auf dieser stabilen Azure-DB-Basis auf.
