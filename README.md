# House-Price-Model (Kaggle)

Dieses Repo enthält eine end-to-end ML-Pipeline für den Kaggle-Datensatz **House Prices (Ames)** – inkl. Feature Engineering im **Databricks Lakehouse**, lokalem Training (scikit-learn + PyTorch) und einem **BigQuery Warehouse** als Schnittstelle für Power BI.

---

## TL;DR (was läuft hier?)

- **Input (RAW):** `data/raw/train.csv` + `data/raw/test.csv` (lokal) **oder** Databricks Feature Store (Gold-Parquet)
- **Lakehouse (Databricks):** Bronze → Silver → Gold (Feature Engineering) als Parquet in einem Databricks Volume
- **Training (lokal):** `train.py` (Analysis / Train-only)
- **Inference (lokal):** `predict.py` (Submission + optional Warehouse-Export)
- **Warehouse (BigQuery):** RAW-Tabellen laden + Views/Marts erzeugen → Power BI drauf

---

## Projektstruktur

```text
.
├─ cloud/
│  ├─ databricks/
│  │  ├─ 01_bronze_ingest.py
│  │  ├─ 02_silver_clean.py
│  │  └─ 03_gold_features.py
│  └─ bigquery/
│     └─ marts_views.sql
├─ data/
│  ├─ raw/                      # Kaggle CSVs (train.csv / test.csv)
│  ├─ feature_store/            # Downloaded Gold-Parquet (optional)
│  └─ warehouse/
│     └─ raw/                   # reproduzierbare Exports (Parquet) für BigQuery RAW
├─ docs/
│  └─ architecture.md
├─ models/                      # gespeicherte Modelle (.joblib)
├─ predictions/                 # lokal erzeugte Predictions/Submission
├─ scripts/
│  ├─ databricks/
│  │  └─ download_feature_store.ps1
│  └─ bigquery/
│     ├─ load_raw_tables.py
│     └─ apply_views.py
├─ src/
│  ├─ data.py
│  ├─ features.py
│  ├─ preprocessing.py
│  ├─ models.py
│  └─ nn_models.py
├─ train.py
├─ predict.py
├─ start_dev.ps1
├─ requirements.txt
└─ requirements-dev.txt
```

---

## Entry Points (wie starte ich was?)

### 1) Lokal trainieren (CSV → Preprocessing in Python)

```powershell
# aus der venv heraus
python train.py --mode analysis --data-source csv
python predict.py --data-source csv
```

### 2) Databricks Lakehouse bauen (Bronze/Silver/Gold)

In Databricks (Repo-Integration) die drei Dateien unter `cloud/databricks/` ausführen:

1. `cloud/databricks/01_bronze_ingest.py`
2. `cloud/databricks/02_silver_clean.py`
3. `cloud/databricks/03_gold_features.py`

Ergebnis: **Gold-Parquet** (z. B. `train_gold.parquet`, `test_gold.parquet`) im Volume:

- `dbfs:/Volumes/workspace/house_prices/feature_store/`

### 3) Gold lokal herunterladen (Feature Store → `data/feature_store/`)

```powershell
# braucht Databricks CLI + auth login
.\scripts\databricks\download_feature_store.ps1
```

Wenn Gold lokal vorhanden ist, kann `train.py` im “Gold-only”-Modus laufen (ohne Preprocessing in Python).

### 4) BigQuery befüllen (RAW) + Views/Marts anwenden

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\service_account.json"
$env:BQ_PROJECT_ID="house-price-model"

python -m scripts.bigquery.load_raw_tables --dataset house_prices_raw
python -m scripts.bigquery.apply_views --raw house_prices_raw --core house_prices_core --marts house_prices_marts
```

---

## Setup

### Empfohlen: `start_dev.ps1` (Entry Point)

```powershell
# einmalig
python -m venv .venv

# Standard: venv aktivieren
.\start_dev.ps1

# optional: Dependencies installieren/aktualisieren
.\start_dev.ps1 -InstallDeps
```

### Alternative: Manuell

#### 1) Python Environment

```powershell
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

#### 2) Dependencies

```powershell
pip install -r requirements.txt
# optional (Dev/Cloud-Skripte)
pip install -r requirements-dev.txt
```

---

## Databricks CLI (für `scripts/databricks/*`)

Installieren: siehe Offizielle Anleitung.  
Login:

```bash
databricks auth login --host https://<your-workspace>
```

---

## BigQuery (für `scripts/bigquery/*`)

Setze die Env-Variable auf deinen Service-Account-Key (JSON):

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\service_account.json"
$env:BQ_PROJECT_ID="house-price-model"
```

---

## Artefakte / Historie (Postgres + Flyway)

Frühere Versionen dieses Projekts haben eine **PostgreSQL-Datenbank** lokal in Docker sowie optional in **Azure** genutzt, inkl. **Flyway-Migrations** und Views als “Warehouse”-Vorstufe.

Diese Variante ist als Artefakt-Branch dokumentiert:

- https://github.com/BaumannBastian/house-price-model/tree/artifact/postgres-flyway

---

## Dokumentation

- Architektur: `docs/architecture.md`