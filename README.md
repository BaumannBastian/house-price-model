# House Price Model

End-to-End Machine-Learning-Projekt zur Vorhersage von Hauspreisen  
(basierend auf dem Kaggle-Datensatz *House Prices – Advanced Regression Techniques*).

---

## TL;DR

- **Ziel**: Saubere ML-Pipeline (scikit-learn + PyTorch) für Hauspreise inkl. Model Registry & Prediction-Store in PostgreSQL.
- **Status**:
  - Training & Prediction laufen lokal (Python + venv).
  - Modell-Metadaten & Predictions werden in **Azure PostgreSQL Flexible Server** gespeichert (Region `northeurope`, via Terraform).
- **DB-Schema**:
  - `models` – Modell-Registry (Name, Version, Metriken, Hyperparameter, Champion-Flag).
  - `predictions` – Vorhersagen (`kaggle_id`, `predicted_price`, `model_id`, `created_at`).
  - `v_predictions_with_model` – Join aus `predictions` und `models` (für Analysen/BI).
- **Infra**:
  - Terraform-Konfiguration in `terraform/`.
  - ENV-basierte DB-Konfiguration in `src/db.py`.
  - Utility-Skripte in `scripts/` (Schema anwenden, DB testen, Sample anzeigen).
- **Nächste Schritte**:
  - BI-Tool (Tableau / Power BI) direkt auf die Azure-Postgres-DB hängen.
  - Optional: Webservice (z.B. FastAPI) für Online-Prediction.

---

## Projektstruktur

    .
    ├─ src/
    │  ├─ data.py            # Laden der Rohdaten (train/test)
    │  ├─ features.py        # Missing-Value-Treatment & Feature Engineering
    │  ├─ preprocessing.py   # ColumnTransformer & Feature-Listen
    │  ├─ models.py          # klassische ML-Modelle (Linear, RF, HistGBR)
    │  ├─ nn_models.py       # TorchMLPRegressor + Pipeline
    │  ├─ db.py              # PostgreSQL-Anbindung (models & predictions, via ENV)
    │  └─ __init__.py
    ├─ scripts/
    │  ├─ __init__.py
    │  ├─ test_db_connection.py  # Healthcheck der DB-Verbindung
    │  ├─ init_schema.py         # wendet sql/schema.sql auf die DB an
    │  ├─ show_db_sample.py      # zeigt Beispielzeilen aus models/View
    │  └─ set_env_azure_db.ps1   # setzt DB_* ENV-Variablen für Azure-Postgres
    ├─ sql/
    │  └─ schema.sql         # vollständiges DB-Schema (models, predictions, Indizes, View)
    ├─ terraform/
    │  ├─ main.tf            # Azure-Ressourcen (RG, PostgreSQL Flexible Server, Firewall)
    │  ├─ variables.tf       # Terraform-Variablen (Projektname, Region, DB-Name, etc.)
    │  ├─ outputs.tf         # Terraform-Outputs (FQDN, User, DB-Name, Beispiel-DSN)
    │  └─ .terraform.lock.hcl
    ├─ docs/
    │  ├─ architecture.md    # Architektur-Übersicht (Datenfluss, Komponenten, Cloud)
    │  └─ experiments.md     # Experiment-Log (Modelle & Ergebnisse)
    ├─ models/               # gespeicherte Champion-Modelle (.joblib)
    ├─ predictions/          # Predictions als CSV
    ├─ logs/                 # Logging (z.B. train.log)
    ├─ plots/                # Trainingsplots (z.B. TorchMLP-Losskurven)
    ├─ requirements.txt
    ├─ pyproject.toml
    ├─ Dockerfile
    ├─ .dockerignore
    └─ .gitignore

---

## Daten

### Dateibasiert

- Datensatz: Kaggle „House Prices: Advanced Regression Techniques“.
- Rohdaten (nicht im Repo):
  - `data/raw/train.csv`
  - `data/raw/test.csv`
- Artefakte:
  - `predictions/predictions.csv` (von `predict.py` erzeugt)
  - `models/*.joblib` (Champion-Modelle)
  - `logs/`, `plots/` (Training-Plots, Logs)

Folgende Verzeichnisse sind bewusst in `.gitignore` und `.dockerignore` eingetragen:

- `data/`
- `models/`
- `logs/`
- `plots/`
- `predictions/`

### Datenbank (PostgreSQL)

Das Projekt unterstützt zwei Modi:

1. **Lokal (Docker-Postgres)**  
   - DB: `house_prices`  
   - User/Passwort z.B.: `house` / `house`  
   - Host: `localhost`, Port: `5432`  
   - SSL: deaktiviert (`DB_SSLMODE=disable`)

2. **Cloud (Azure PostgreSQL Flexible Server)**  
   - Region: `northeurope`  
   - DB: `house_prices`  
   - Tabellen:
     - `models` – Modell-Registry (Name, Version, Pfad zur `.joblib`, Metriken, Hyperparameter, Champion-Flag)
     - `predictions` – Prediction-Store (`kaggle_id`, `predicted_price`, `model_id`, `created_at`)
   - View:
     - `v_predictions_with_model` – Join von `predictions` und `models` für Analysen/BI

Die Verbindung wird in `src/db.py` über Umgebungsvariablen gesteuert:

- `DB_HOST` – Hostname/FQDN (lokal: `localhost`, Azure: FQDN des Servers)
- `DB_PORT` – Standard: `5432`
- `DB_NAME` – z.B. `house_prices`
- `DB_USER` – DB-User (lokal: `house`, Azure: z.B. `hpadmin`)
- `DB_PASSWORD` – Passwort
- `DB_SSLMODE` – `"require"` (Azure) oder `"disable"` (lokal)

Wenn keine ENV-Variablen gesetzt sind, greifen lokale Defaults (Postgres auf `localhost` mit `house`/`house`, `DB_SSLMODE=require`).

---

## Pipeline

### Preprocessing & Features

- Laden der Trainingsdaten über `src/data.py`.
- Feature Engineering und Missing-Value-Treatment in `src/features.py`:
  - z.B. `TotalSF`, `TotalBath`, `HouseAge`, `RemodAge`, `TotalPorchSF`.
  - Missing Values je nach Typ (None/Median/Modus).
- Ordinale Features werden auf Integer gemappt (Ex/Gd/TA/Fa/Po → 5…1 usw.).
- `src/preprocessing.py` baut einen `ColumnTransformer`:
  - `OneHotEncoder(handle_unknown="ignore")` für kategoriale Features.
  - numerische/ordinale Features werden durchgereicht (`remainder="passthrough"`).

### Modelle

- `src/models.py`:
  - `LinearRegression`
  - `RandomForestRegressor`
  - `HistGradientBoostingRegressor`
- `src/nn_models.py`:
  - `TorchMLPRegressor` (PyTorch-MLP, sklearn-kompatibel)
- Alle Modelle werden als Pipelines gebaut:  
  `preprocess → (optional StandardScaler) → Regressor`  
  und optional in `TransformedTargetRegressor` gepackt (`log`-Target).

### Training (`train.py`)

- liest Trainingsdaten, baut Preprocessor.
- trainiert nacheinander:
  - `LinearRegression` (+ `_log`)
  - `RandomForest` (+ `_log`)
  - `HistGBR` (+ `_log`)
  - `TorchMLP`
- für jedes Modell:
  - 5-fold Cross-Validation (RMSE)
  - Test-Set-Evaluation (R², RMSE, MARE/RRMSE)
- wählt den **Champion** anhand CV-RMSE (aktuell: `HistGBR_log`).
- trainiert den Champion auf allen Daten.
- speichert das Champion-Modell als `.joblib` in `models/`.
- schreibt einen Eintrag in die Tabelle `models` (Postgres).

### Prediction (`predict.py`)

- ermittelt das aktuelle Champion-Modell (aus DB + Dateisystem).
- lädt das Modell (`.joblib`).
- liest `data/raw/test.csv`.
- erzeugt Predictions.
- schreibt:
  - `predictions/predictions.csv` (für Kaggle).
  - alle Predictions in die Tabelle `predictions` (inkl. `model_id` des Champions).

---

## Setup

### Python-Umgebung

    python -m venv .venv
    # Windows:
    .venv\Scripts\activate
    # Linux/Mac:
    source .venv/bin/activate

    pip install -r requirements.txt

Optional (für „sauberes“ Packaging):

    pip install -e .

Dadurch wird `pyproject.toml` genutzt und das Projekt als Package installiert; `src` ist dann als Package verfügbar und `python -m scripts.<name>` funktioniert wie geplant.

---

### Lokale PostgreSQL (Docker, optional)

    docker run --name house-price-postgres ^
      -e POSTGRES_USER=house ^
      -e POSTGRES_PASSWORD=house ^
      -e POSTGRES_DB=house_prices ^
      -p 5432:5432 ^
      -d postgres:16

(unter Linux/Mac `^` durch `\` ersetzen)

Beispiel-ENV für lokale DB:

    set DB_HOST=localhost
    set DB_PORT=5432
    set DB_NAME=house_prices
    set DB_USER=house
    set DB_PASSWORD=house
    set DB_SSLMODE=disable

(bzw. PowerShell: `$env:DB_HOST = "localhost"` usw.)

---

### Azure PostgreSQL (Terraform)

1. `az login` ausführen und Subscription auswählen.
2. In `terraform/`:

       terraform init
       terraform apply -var "client_ip=<DEINE_ÖFFENTLICHE_IP>"

   Terraform legt an:

   - Resource Group `house-price-rg`
   - PostgreSQL Flexible Server `house-price-psql` (Region `northeurope`)
   - Datenbank `house_prices`
   - Firewall-Regel für deine IP

3. Wichtige Outputs:

       terraform output -raw postgres_fqdn
       terraform output -raw postgres_admin_username
       terraform output -raw postgres_admin_password
       terraform output -raw postgres_database_name

4. Diese Werte werden in `scripts/set_env_azure_db.ps1` eingetragen.  
   Pro Session genügt dann:

       .\.venv\Scripts\Activate.ps1
       .\scripts\set_env_azure_db.ps1

5. Schema auf Azure anwenden:

       python -m scripts.init_schema

6. Verbindung / Inhalt prüfen:

       python -m scripts.test_db_connection
       python -m scripts.show_db_sample

---

## Usage

### Training

    python train.py

- trainiert alle konfigurierten Modelle,
- berechnet Metriken (Test + CV),
- wählt einen Champion,
- speichert das Modell (`models/`),
- trägt es in der DB-Tabelle `models` ein.

### Prediction

    python predict.py

- lädt den Champion,
- erzeugt Vorhersagen für den Testdatensatz,
- schreibt `predictions/predictions.csv`,
- speichert alle Predictions in der DB-Tabelle `predictions`.

---

## Roadmap

- **Kurzfristig**
  - Tableau / Power BI direkt auf Azure-Postgres (`models`, `predictions`, `v_predictions_with_model`).
  - Kleine Auswertungs-/Monitoring-Dashboards.

- **Mittelfristig**
  - FastAPI-Service als Web-API:
    - Endpunkte für aktuelle Champion-Infos.
    - Online-Predictions mit DB-Logging.
  - Containerisierung (Docker) und Deployment z.B. auf Azure Container Apps.

- **Langfristig**
  - Erweiterte Hyperparameter-Suche.
  - Weitere Modellklassen (z.B. XGBoost/LightGBM).
  - CI/CD-Pipeline für Training, Evaluation, Deployment.