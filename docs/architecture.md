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

### 1.2 Orchestrierung (Project Root)

- `train.py`
  - `train-only`: Full-Fit + Artefakt speichern
  - `analysis`: KFold-CV, Champion per CV-RMSE
  - exportiert Warehouse-RAW Tabellen als Parquet nach `data/warehouse/raw/`
- `predict.py`
  - lädt Champion und erzeugt Kaggle-Submission
  - exportiert Warehouse-RAW Tabellen als Parquet nach `data/warehouse/raw/`

### 1.3 Lakehouse (Databricks)

- `cloud/databricks/`
  - `01_bronze_ingest.py`: Ingestion (Bronze)
  - `02_silver_clean.py`: Cleaning (Silver)
  - `03_gold_features.py`: Feature Engineering (Gold)

Ziel: reproduzierbare Feature-Schichten (Bronze/Silver/Gold) im Databricks Catalog/Schema `house_prices`.

### 1.4 Warehouse (BigQuery)

- `cloud/bigquery/`
  - `load_raw_tables.py`: lädt RAW aus lokalen Parquet-Exports
  - `marts_views.sql`: Views für BI (MARTS)
  - `apply_views.py`: erstellt/updated Views

### 1.5 Utility-Skripte

- `scripts/databricks/`
  - kleine Helfer für lokale Dev-Workflows (z.B. Download aus Databricks Volumes)

---

## 2) Datenfluss

### 2.1 Lakehouse (Bronze → Silver → Gold)

1. Bronze: Rohdaten (CSV) werden in Databricks ingestiert
2. Silver: Cleaning / Missing Values / grundlegende Normalisierung
3. Gold: Feature Engineering (finale Trainingsfeatures)

### 2.2 Training (lokal)

1. Input:
   - `data-source=csv`: `data/raw/train.csv` (Preprocessing im Python-Code)
   - perspektivisch: Gold-Input (Feature Engineering in Databricks)
2. Modelltraining in `train.py`
   - `analysis`: KFold-CV → Champion nach CV-RMSE
   - `train-only`: Full-Fit
3. Modell-Artefakt wird nach `models/` gespeichert
4. Output-Tabellen werden als Parquet nach `data/warehouse/raw/` exportiert (Schnittstelle zum Warehouse)

### 2.3 Prediction

1. Input: `data/raw/test.csv`
2. Champion wird geladen
3. Predictions:
   - Kaggle-Submission (`data/submission.csv`)
   - Warehouse-Exports (`data/warehouse/raw/`)

---

## 3) Datenmodell (Warehouse)

Das Warehouse ist in Schichten gedacht:

- **RAW**: direkte Exports aus `train.py`/`predict.py` (Parquet)
- **CORE**: abgeleitete Tabellen für Analyse (z.B. Buckets/Aggregationen)
- **MARTS**: Views für BI (Power BI)

Die aktuelle Implementierung enthält:
- Loader-Skript (`cloud/bigquery/load_raw_tables.py`)
- MARTS-Views (`cloud/bigquery/marts_views.sql`) + Runner (`apply_views.py`)

---

## 4) Laufzeit-Setups

### 4.1 Lokal (Default)

- Python venv + `requirements.txt`
- Training/Inference läuft lokal (`train.py`, `predict.py`)
- Warehouse-Exports werden lokal geschrieben (`data/warehouse/raw/`)

### 4.2 Cloud

- **Databricks**: Lakehouse-Pipeline (Bronze/Silver/Gold)
- **BigQuery**: Warehouse-Schichten (RAW/CORE/MARTS)

---

## 5) Artifacts (ältere Infrastruktur)

Ein früherer Projektstand ist als Branch erhalten:

- lokale **PostgreSQL** in Docker
- **Flyway** Migrationen
- optional **Azure PostgreSQL** (Terraform)

Branch:
https://github.com/BaumannBastian/house-price-model/tree/artifact/postgres-flyway