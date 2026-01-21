# Architektur – House Price Model

Dieses Dokument beschreibt die Architektur des Projekts: Komponenten, Datenfluss und Persistenzschicht.

---

## 1) Komponentenübersicht

### 1.1 Core (`src/`)

- `src/data.py`
  - lädt Kaggle-Rohdaten aus `data/raw/` und stellt Train/Test-DataFrames bereit
- `src/features.py`
  - Feature Engineering (ableitbare Features, Spalten-Handling)
- `src/preprocessing.py`
  - Preprocessing-Pipeline (Imputation, Encoding, Scaling; konsistent für Training/Inference)
- `src/models.py`
  - sklearn-Modelle / Wrapper / Factory-Logik
- `src/nn_models.py`
  - PyTorch MLP + Trainings-/Inference-Wrapper
- `src/db.py`
  - DB-Zugriffe (psycopg2): Inserts/Updates für Runs, Predictions, OOF/CV-Outputs

### 1.2 Orchestrierung (Project Root)

- `train.py`
  - `train-only`: Full-Fit + Artefakt speichern + Run in DB loggen
  - `analysis`: KFold-CV (gemeinsame Splits), OOF-Predictions in DB, Champion per CV-RMSE
- `predict.py`
  - lädt Champion und erzeugt Kaggle-Submission; optional Persistenz in DB
- `start_dev.ps1`
  - lokaler Entry-Point für Dev: Python-Env, Docker-DB, Migrationen, Connection-Test

### 1.3 Utility-Skripte (`scripts/`)

- `scripts/set_env_local_db.ps1`
  - setzt lokale DB-ENV-Variablen (Default: Docker Postgres)
- `scripts/set_env_azure_db_example.ps1`
  - Template für Azure ENV (Secrets lokal, nicht im Repo)
- `scripts/test_db_connection.py`
  - Smoke-Test: Verbindung herstellen und einfache Queries ausführen
- `scripts/show_db_sample.py`
  - kleine Stichprobe aus Tabellen (Debug/QA)

### 1.4 Schema-Management (Flyway)

- `sql/migrations/`
  - `V1__init.sql`: initiales Schema
  - `V2__align_legacy_schema.sql`: Alignment/Kompatibilität zu bestehender DB
- Flyway trackt den Stand in `flyway_schema_history`

### 1.5 Infrastruktur

- `docker-compose.yml`
  - `db` (Postgres 16)
  - `flyway` (Migration Runner)
- `Dockerfile`
  - optional für Containerisierung der Python-App (nicht zwingend fürs lokale Dev)
- `terraform/` (optional)
  - Azure PostgreSQL Flexible Server als Cloud-Artifact

---

## 2) Datenfluss

### 2.1 Training

1. `src/data.py` lädt `train.csv`
2. `src/features.py` + `src/preprocessing.py` bauen Transformationen
3. Modelltraining in `train.py`
   - `train-only`: Full-Fit (Train gesamt) und Artefakt (joblib) speichern
   - `analysis`: KFold-CV
     - pro Fold: Fit → Predictions → RMSE
     - OOF-Predictions in `train_cv_predictions`
4. Metriken + Metadaten pro Run in `models`
5. Champion-Flag in `models` wird gesetzt (nach CV-RMSE)

### 2.2 Prediction

1. `src/data.py` lädt `test.csv`
2. Preprocessing/Features wie beim Training (gleiche Pipeline)
3. Champion wird geladen
4. Predictions werden als Kaggle-Submission geschrieben
5. optional: Persistenz in `predictions` (inkl. `model_id`)

---

## 3) Datenmodell (PostgreSQL)

### 3.1 `models` (Model Registry)
- Run-Metadaten, Metriken, Hyperparameter/Metadata, Champion-Flag, Zeitstempel

### 3.2 `train_cv_predictions` (OOF/CV Output)
- pro Kaggle-ID die OOF-Prediction (und abgeleitete Error-Spalten)
- primäre Datenquelle für Power BI Error Analytics (z.B. Bucket-Auswertungen)

### 3.3 `train_predictions` (optional)
- Fehler pro Sample für Full-Fit (wenn aktiviert)

### 3.4 `predictions`
- Kaggle-Test-Predictions (z.B. Champion Runs)

### 3.5 `v_predictions_with_model`
- Convenience-Join (`predictions` ⨝ `models`) für Reporting

---

## 4) Laufzeit-Setups

### 4.1 Lokal (Default)
- Postgres via Docker Compose
- Schema via Flyway Migrationen
- ENV via `scripts/set_env_local_db.ps1`
- Entry-Point: `start_dev.ps1`

### 4.2 Azure (optional)
- Azure PostgreSQL Flexible Server via Terraform
- SSL/Firewall/ENV sind über `scripts/set_env_azure_db.ps1` steuerbar
- weiterhin Flyway Migrationen für reproduzierbares Schema

---

## 5) Notizen zur Cloud-DB (kurz)

- lokales Dev ist auf Docker fokussiert; Azure ist optionales Deployment-Artifact
- Verbindung nach Azure erfordert üblicherweise SSL (`sslmode=require`) und eine passende Firewall-Regel
- Migrationen bleiben identisch (Flyway), unabhängig vom Zielsystem
