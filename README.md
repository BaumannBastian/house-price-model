# House Price Model

End-to-End Machine-Learning-Projekt zur Vorhersage von Hauspreisen  
(basierend auf dem Kaggle-Datensatz *House Prices – Advanced Regression Techniques*).

---

## TL;DR

- **Ziel**: Saubere ML-Pipeline (scikit-learn + PyTorch) für House Prices inkl. Model Registry & Prediction-Store in PostgreSQL.
- **Default-Dev-Setup (kostenlos)**: Lokale PostgreSQL via Docker (docker compose).
- **Optional (Artifact)**: Azure PostgreSQL Flexible Server + Terraform (Cloud-Infra, kann Kosten verursachen).
- **Monitoring/BI**: Power BI Report auf den DB-Tabellen (Model-Overview + Error-Buckets + Drilldowns).

---

## Projektstruktur

    .
    ├─ src/
    │  ├─ data.py            # Laden der Rohdaten (train/test)
    │  ├─ features.py        # Missing-Value-Treatment & Feature Engineering
    │  ├─ preprocessing.py   # ColumnTransformer & Feature-Listen
    │  ├─ models.py          # klassische ML-Modelle (Linear, RF, HistGBR)
    │  ├─ nn_models.py       # TorchMLPRegressor + Pipeline
    │  ├─ db.py              # PostgreSQL-Anbindung (models/predictions/..., via ENV)
    │  └─ __init__.py
    ├─ scripts/
    │  ├─ __init__.py
    │  ├─ test_db_connection.py      # Healthcheck der DB-Verbindung
    │  ├─ init_schema.py             # wendet sql/schema.sql auf die DB an
    │  ├─ show_db_sample.py          # zeigt Beispielzeilen aus models/View
    │  ├─ set_env_azure_db.example.ps1  # Template (ohne Secrets)
    │  └─ set_env_azure_db.ps1       # lokal (mit Secrets, nicht committen)
    ├─ sql/
    │  └─ schema.sql         # vollständiges DB-Schema (models, predictions, train_* , Indizes, View)
    ├─ terraform/            # (Artifact) Azure-Ressourcen
    ├─ docs/
    ├─ models/               # gespeicherte Modelle (.joblib)
    ├─ predictions/          # Predictions als CSV
    ├─ logs/
    ├─ plots/
    ├─ docker-compose.yml    # lokale Dev-DB
    ├─ Dockerfile            # (optional) App-Container
    ├─ requirements.txt
    ├─ pyproject.toml
    ├─ start_dev.ps1         # Dev-Entry-Point (lokal/optional azure)
    ├─ .dockerignore
    └─ .gitignore

---

## Daten

### Dateibasiert

- Datensatz: Kaggle „House Prices: Advanced Regression Techniques“.
- Rohdaten (nicht im Repo):
  - `data/raw/train.csv`
  - `data/raw/test.csv`

Artefakte (lokal erzeugt):
- `models/*.joblib` (Model-Artefakte)
- `predictions/predictions.csv` (für Kaggle)
- `logs/`, `plots/`

---

## Datenbank (PostgreSQL)

### Tabellen / View

- `models`
  - Modell-Registry (Name, Version, Metriken, Hyperparameter, Champion-Flag, created_at)
- `predictions`
  - Kaggle-Testset-Predictions (id, kaggle_id, predicted_price, model_id, created_at)
- `train_predictions`
  - Fehler pro Train-Sample (Fit auf Full-Train; gut um Overfitting zu sehen)
- `train_cv_predictions`
  - OOF/CV-Fehler pro Train-Sample (für sinnvolle Bucket-Analyse)
- `v_predictions_with_model`
  - View: `predictions` ⨝ `models` (Komfort für BI)

### DB-Konfiguration (ENV)

Die Verbindung wird in `src/db.py` über ENV-Variablen gesteuert:

- `DB_HOST` – Hostname/FQDN (lokal: `localhost`)
- `DB_PORT` – Standard: `5432`
- `DB_NAME` – z.B. `house_prices`
- `DB_USER` – DB-User (lokal: `house`)
- `DB_PASSWORD` – Passwort
- `DB_SSLMODE` – `"disable"` (lokal) oder `"require"` (Azure)

---

## Power BI (Monitoring / Insights)

Der Power BI Report basiert direkt auf den DB-Tabellen:

- **Model Overview**
  - Tabelle/Matrix: `models` (name, version, cv_rmse_mean, rmse_test, r2_test, mare_test, is_champion, created_at)
  - Filter/Slicer: Modellname, Version, Zeitraum
- **Error Buckets**
  - Daten: `train_cv_predictions` (OOF Errors) + Join auf `models`
  - Buckets (z.B. 0–50k, 50–100k, ...): als DAX-Spalte oder Power Query
  - Visual: Avg Abs Error pro Bucket und Modell (Tooltip: Count, Max Error, etc.)

Hinweis:
- Für Bewerbungen reicht oft: `.pbix` + PDF/Screenshots im Repo unter `reports/` (wenn du es hinzufügen willst).

---

## Setup

### 1) Python-Umgebung

    python -m venv .venv
    # Windows:
    .venv\Scripts\activate
    pip install -r requirements.txt

Optional:

    pip install -e .

### 2) Lokale DB (Default)

    docker compose up -d
    python -m scripts.init_schema
    python -m scripts.test_db_connection

### 3) Training

Default (Train-only / schnell):

    python train.py

Analyse (CV/OOF + DB-Logging für Buckets):

    python train.py --mode analysis

### 4) Prediction

    python predict.py

---

## Azure (Artifact)

Optionaler Cloud-Teil:
- Terraform für Azure PostgreSQL Flexible Server in `terraform/`
- ENV-Setup via `scripts/set_env_azure_db.ps1` (lokal, nicht committen)

Hinweis:
- Cloud-Infrastruktur kann Kosten verursachen und ist nicht notwendig, um das Projekt lokal auszuführen.

---

## Roadmap (kurz)

- Data Engineering / Feature Engineering zur Verbesserung von CV-RMSE (HistGBR_log, TorchMLP)
- Hyperparameter-Tuning / bessere Regularisierung
- Optional: CI (GitHub Actions) für reproduzierbare Training-Runs