# House Price Model – Reproducible ML + Data Engineering Workflow

Dieses Repo ist ein lokal reproduzierbares End-to-End Projekt rund um das Kaggle-Dataset **House Prices: Advanced Regression Techniques**.

Fokus:
- mehrere Modellfamilien (sklearn + PyTorch) sauber vergleichbar trainieren
- Modell-Runs + Metriken + OOF/CV-Predictions in **PostgreSQL** persistieren
- **Power BI** für Monitoring / Exploration auf DB-Tabellen
- lokales Setup als Default (Docker + Flyway), Cloud optional

---

## Projektstruktur

    .
    ├─ data/
    │  └─ raw/                      # train.csv / test.csv, nicht im Repo
    ├─ scripts/
    │  ├─ db/
    │  │  ├─ set_env_local_db.ps1
    │  │  ├─ set_env_azure_db_example.ps1
    │  │  ├─ set_env_azure_db.ps1      # lokal (Secrets), nicht im Repo
    │  │  ├─ test_db_connection.py
    │  │  └─ show_db_sample.py
    │  └─ lakehouse/
    │     ├─ __init__.py
    │     ├─ build_bronze.py
    │     ├─ build_silver.py
    │     └─ build_gold.py
    ├─ sql/
    │  └─ migrations/
    │     ├─ V1__init.sql
    │     └─ V2__align_legacy_schema.sql
    ├─ src/
    │  ├─ data.py
    │  ├─ db.py
    │  ├─ features.py
    │  ├─ preprocessing.py
    │  ├─ models.py
    │  └─ nn_models.py
    ├─ models/                      # .joblib Artefakte (lokal)
    ├─ predictions/  
    ├─ docs/
    │  └─ powerbi/                   # Screenshots
    ├─ requirements.txt
    └─ start_dev.ps1

---

## Lokales Setup (Default)

### Voraussetzungen
- Windows + PowerShell
- Docker Desktop
- Python 3.11+

### DB-ENV Variablen

Erwartete Umgebungsvariablen:
- `DB_HOST` – lokal: `localhost`
- `DB_PORT` – lokal: `5432`
- `DB_NAME` – lokal: `house`
- `DB_USER` – lokal: `house`
- `DB_PASSWORD` – lokal: `house`
- `DB_SSLMODE` – `"disable"` (lokal) oder `"require"` (Azure)

Die ENV-Variablen werden per PowerShell-Skripten gesetzt:
- lokal (Docker): `scripts/db/set_env_local_db.ps1`
- cloud (Azure): `scripts/db/set_env_azure_db.ps1` (nicht im Repo; Template: `scripts/db/set_env_azure_db_example.ps1`)

---

## DB Schema Management (Flyway)

Das DB-Schema wird ausschließlich über Flyway verwaltet:

- Migrationen liegen unter `sql/migrations/`
- Flyway schreibt seine Historie in `flyway_schema_history`
- Schema wird nicht in Python erzeugt (keine init_schema.py / kein schema.sql)

---

## Start: One-Command Dev Setup

Der Entry-Point ist:

    .\start_dev.ps1

Was der Entry-Point macht (high level):
- aktiviert die Python-venv (falls vorhanden)
- startet die lokale PostgreSQL-DB via Docker Compose (persistentes Volume)
- setzt lokale DB-ENV-Variablen (`scripts/db/set_env_local_db.ps1`)
- führt Flyway Migrationen aus (`sql/migrations`)
- prüft die DB-Verbindung (`python -m scripts.db.test_db_connection`)

Optional: DB-Sample anzeigen

    python -m scripts.db.show_db_sample --limit 5

---

### Manuelles Setup (Debug / granular)

#### 1) Python-Umgebung

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt

#### 2) Docker DB starten

    docker compose up -d db

#### 3) DB-ENV setzen (lokal)

    . .\scripts\db\set_env_local_db.ps1

#### 4) Migrationen anwenden (Flyway)

    docker compose run --rm --no-deps flyway -connectRetries=60 migrate

#### 5) Verbindung testen

    python -m scripts.db.test_db_connection

---

### Optional: Azure (Artifact)

Der Azure-Teil ist optional und nicht erforderlich für lokale Runs.

1) Azure ENV setzen (lokal, nicht committen)

    . .\scripts\db\set_env_azure_db.ps1

2) Migrationen gegen Azure anwenden (Flyway Container, URL aus ENV)

    docker compose run --rm --no-deps flyway -connectRetries=60 migrate

3) Verbindung testen

    python -m scripts.db.test_db_connection

---

## Training

### Modus 1: train-only (Default)

Trainiert alle konfigurierten Modelle auf Full-Data, speichert Artefakte in `models/` und loggt Run-Infos in `models`.

    python train.py --mode train-only

### Modus 2: analysis (Cross-Validation + Champion)

- nutzt gemeinsame KFold-Splits
- schreibt OOF/CV-Predictions nach `train_cv_predictions`
- loggt Metriken nach `models`
- setzt Champion nach CV-RMSE

    python train.py --mode analysis

---

## Power BI

Power BI verbindet sich direkt an die Postgres-Tabellen (z.B. `models`, `train_cv_predictions`) und kann für Monitoring/Exploration genutzt werden.

Screenshots liegen unter:
- `docs/powerbi/`

---

## Troubleshooting

**DB-Verbindung prüfen**

    python -m scripts.db.test_db_connection

**DB Sample anzeigen**

    python -m scripts.db.show_db_sample --limit 10

**Flyway Historie**

In der DB existiert `flyway_schema_history`. Migrationen liegen in `sql/migrations/`.

---

## License

MIT (siehe `LICENSE`)