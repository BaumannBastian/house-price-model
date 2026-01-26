# House-Price-Model (Kaggle)

Dieses Repo enthält eine end-to-end ML-Pipeline für den Kaggle-Datensatz **House Prices** – inkl. Feature Engineering im **Databricks Lakehouse**, lokalem Training (**scikit-learn + PyTorch**) und einem **BigQuery Warehouse** als Schnittstelle für **Power BI**.

---

## TL;DR

- **Input (RAW):** `data/raw/train.csv` + `data/raw/test.csv` (lokal) **oder** Databricks Feature Store (Gold-Parquet)
- **Lakehouse (Databricks):** Bronze -> Silver -> Gold (Feature Engineering) als Parquet in einem Databricks Volume
- **Training (lokal):** `train.py` (Analysis / Train-only)
- **Inference (lokal):** `predict.py` (Submission + optional Warehouse-Export)
- **Warehouse (BigQuery):** RAW-Tabellen laden + Views/Marts erzeugen -> Power BI

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
│  │  ├─ sync_repo.py                 # Databricks Repo auf neuesten Git-Stand bringen
│  │  ├─ update_jobs_to_repo.py       # Jobs auf /Repos/... Notebooks umstellen (einmalig)
│  │  ├─ run_lakehouse_jobs.py        # Bronze/Silver/Gold Jobs sequenziell triggern
│  │  ├─ download_feature_store.ps1   # Gold Parquets lokal nach data/feature_store/ kopieren
│  │  └─ sync_feature_store.py        # optional: Manifest/Sync-Logic (falls genutzt)
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

## Training

### 1) Lokal trainieren (CSV -> Preprocessing und train/predict in Python)

```powershell
# aus der venv heraus
python train.py --mode analysis --data-source csv
python predict.py --data-source csv
```

### 2) Mit Gold-Data aus Databricks trainieren (Databricks Parquets -> train/predict in Python)

#### Databricks Repo: Git-Sync und Job-Ausführung

Ziel: Databricks Jobs sollen immer den aktuellen Code-Stand aus einem **Databricks Repo** verwenden.

Konventionen/Placeholders:

- `<DATABRICKS_PROFILE>`: Databricks CLI Profile-Name
- `<DATABRICKS_REPO_PATH>`: Repo-Pfad in Databricks (Repos Feature)
- `<GIT_REPO_URL>`: GitHub Repo URL

Beispielwerte (anpassen):

- `<GIT_REPO_URL>`: `https://github.com/<ORG_OR_USER>/house-price-model.git`
- `<DATABRICKS_REPO_PATH>`: `/Repos/<USERNAME>/house-price-model`

#### A) One-time Setup (Repo anlegen und Jobs auf Repo-Notebooks umstellen)

1) Databricks Repo erstellen:

```powershell
databricks repos create <GIT_REPO_URL> gitHub --path "<DATABRICKS_REPO_PATH>" --profile "<DATABRICKS_PROFILE>"
```

2) Jobs so umstellen, dass sie Notebooks aus `<DATABRICKS_REPO_PATH>` verwenden:

```powershell
python scripts/databricks/update_jobs_to_repo.py `
  --repo-path "<DATABRICKS_REPO_PATH>" `
  --profile "<DATABRICKS_PROFILE>"
```

Hinweis: Dieser Schritt ist nur beim Setup oder nach strukturellen Notebook-Pfadänderungen erforderlich.

#### B) Standard-Workflow nach `git push` (Repo syncen, Jobs laufen lassen)

1) Repo in Databricks auf den neuesten Git-Stand bringen:

```powershell
python scripts/databricks/sync_repo.py `
  --repo-path "<DATABRICKS_REPO_PATH>" `
  --branch main `
  --profile "<DATABRICKS_PROFILE>"
```

2) Lakehouse-Jobs triggern (Bronze -> Silver -> Gold):

```powershell
python scripts/databricks/run_lakehouse_jobs.py --stage all --profile "<DATABRICKS_PROFILE>"
```

#### C) Konflikte durch manuelle Änderungen in Databricks behandeln (Hard Reset)

Wenn `repos update` aufgrund lokaler Änderungen/Conflicts im Databricks Repo fehlschlägt, kann das Repo automatisiert zurückgesetzt werden:

- Backup des Workspace-Ordners (optional)
- Repo löschen
- Repo neu anlegen
- Branch neu auschecken

```powershell
python scripts/databricks/sync_repo.py `
  --repo-path "<DATABRICKS_REPO_PATH>" `
  --branch main `
  --profile "<DATABRICKS_PROFILE>" `
  --reset-if-conflict --backup
```

#### D) Ergebnis: Gold-Parquet im Databricks Volume

Nach erfolgreichem Gold-Job liegen die Artefakte im Volume, z. B.:

- `dbfs:/Volumes/workspace/house_prices/feature_store/`

### 3) Gold lokal herunterladen (Feature Store -> `data/feature_store/`)

```powershell
.\scripts\databricks\download_feature_store.ps1
```

Danach kann lokal im Gold-Modus trainiert werden (ohne Feature Engineering in Python):

```powershell
python train.py --mode analysis --data-source gold
python predict.py --data-source gold
```

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

#### Python Environment

```powershell
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

#### Dependencies

```powershell
pip install -r requirements.txt
# optional (Dev/Cloud-Skripte)
pip install -r requirements-dev.txt
```

---

## Databricks CLI (für `scripts/databricks/*`)

Installation und Authentifizierung erfolgen über die Databricks CLI.

Auth/Profiles prüfen:

```powershell
databricks auth profiles
databricks auth describe
```

Profil anlegen (Beispiel):

```powershell
databricks auth login --profile "<DATABRICKS_PROFILE>" --host "<DATABRICKS_HOST>"
```

Beispiele für Placeholders:

- `<DATABRICKS_PROFILE>`: `DEFAULT` oder ein Projektprofilname (z. B. `dev`)
- `<DATABRICKS_HOST>`: `https://<your-workspace>.cloud.databricks.com`

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