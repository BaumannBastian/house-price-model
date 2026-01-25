# ------------------------------------
# scripts/databricks/download_feature_store.ps1
#
# Lädt die aktuellen GOLD-Parquet-Dateien (+ manifest.json) aus Databricks Volumes
# nach data/feature_store.
#
# Usage
# ------------------------------------
#   # optional (wenn du mehrere Databricks-Profile nutzt)
#   .\scripts\databricks\download_feature_store.ps1 -Profile "basti.baumann@gmx.net"
#
#   # default
#   .\scripts\databricks\download_feature_store.ps1
# ------------------------------------

param(
    [string]$RemoteBase = "dbfs:/Volumes/workspace/house_prices/feature_store",
    [string]$LocalDir = "data/feature_store",
    [string]$Profile = ""
)

function Invoke-Databricks {
    param([string[]]$Args)

    if ([string]::IsNullOrWhiteSpace($Profile)) {
        & databricks @Args
    } else {
        & databricks --profile $Profile @Args
    }

    return $LASTEXITCODE
}

New-Item -ItemType Directory -Force -Path $LocalDir | Out-Null
$LocalDirFull = (Resolve-Path -Path $LocalDir).Path

$RemoteManifest = "$RemoteBase/manifest.json"
$RemoteTrain = "$RemoteBase/train_gold.parquet"
$RemoteTest  = "$RemoteBase/test_gold.parquet"

$LocalManifest = Join-Path $LocalDirFull "manifest.json"
$LocalManifestTmp = Join-Path $LocalDirFull "_manifest.remote.json"

$LocalTrain = Join-Path $LocalDirFull "train_gold.parquet"
$LocalTest  = Join-Path $LocalDirFull "test_gold.parquet"

# 1) Manifest zuerst (damit wir vergleichen können)
$rc = Invoke-Databricks @("fs","cp",$RemoteManifest,$LocalManifestTmp,"--overwrite")
if ($rc -ne 0) {
    throw "Manifest download failed: $RemoteManifest"
}

$remoteText = Get-Content -Raw -Path $LocalManifestTmp -Encoding UTF8
$localText = ""
if (Test-Path $LocalManifest) {
    $localText = Get-Content -Raw -Path $LocalManifest -Encoding UTF8
}

if ($localText -eq $remoteText) {
    Remove-Item -Force $LocalManifestTmp | Out-Null
    Write-Host "OK: Feature-Store ist bereits aktuell ($LocalDir)."
    exit 0
}

# 2) Daten (Parquet) herunterladen
$rc = Invoke-Databricks @("fs","cp",$RemoteTrain,$LocalTrain,"--overwrite")
if ($rc -ne 0) { throw "Databricks copy failed: $RemoteTrain -> $LocalTrain" }

$rc = Invoke-Databricks @("fs","cp",$RemoteTest,$LocalTest,"--overwrite")
if ($rc -ne 0) { throw "Databricks copy failed: $RemoteTest -> $LocalTest" }

# 3) Manifest atomar ersetzen
Move-Item -Force -Path $LocalManifestTmp -Destination $LocalManifest

Write-Host "OK: Feature-Store geladen nach $LocalDirFull"