# ------------------------------------
# scripts/databricks/download_feature_store.ps1
#
# Laedt die Gold-Parquet-Dateien (train/test) aus dem Databricks Volume
# nach data/feature_store/ (lokal).
#
# Erwartete Dateien im Volume:
# - dbfs:/Volumes/workspace/house_prices/feature_store/train_gold.parquet
# - dbfs:/Volumes/workspace/house_prices/feature_store/test_gold.parquet
# ------------------------------------

$ErrorActionPreference = "Stop"

# Databricks CLI finden (falls nicht im PATH)
if (-not (Get-Command databricks -ErrorAction SilentlyContinue)) {
    $cliDir = Join-Path $env:LOCALAPPDATA "DatabricksCLI"
    $cliExe = Join-Path $cliDir "databricks.exe"

    if (Test-Path $cliExe) {
        $env:Path = "$cliDir;$env:Path"
    }
}

if (-not (Get-Command databricks -ErrorAction SilentlyContinue)) {
    throw "Databricks CLI nicht gefunden. Stelle sicher, dass 'databricks.exe' im PATH ist."
}

$RemoteBase = "dbfs:/Volumes/workspace/house_prices/feature_store"
$LocalDir   = "data/feature_store"

New-Item -ItemType Directory -Force $LocalDir | Out-Null

function Copy-OneFile {
    param(
        [Parameter(Mandatory=$true)][string]$RemotePath,
        [Parameter(Mandatory=$true)][string]$LocalPath
    )

    try {
        databricks fs ls $RemotePath | Out-Null
    }
    catch {
        throw "Remote-Datei existiert nicht: $RemotePath"
    }

    databricks fs cp $RemotePath $LocalPath --overwrite | Out-Null
}

Copy-OneFile "$RemoteBase/train_gold.parquet" (Join-Path $LocalDir "train_gold.parquet")
Copy-OneFile "$RemoteBase/test_gold.parquet"  (Join-Path $LocalDir "test_gold.parquet")

Write-Host "OK: Feature-Store geladen nach $LocalDir"