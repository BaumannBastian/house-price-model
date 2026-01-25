# ------------------------------------
# start_dev.ps1
#
# Ein einziger Entry-Point fuer die lokale Dev-Umgebung.
# - aktiviert venv (wenn vorhanden)
# - optional: Dependencies installieren
# - optional: Databricks Feature Store lokal herunterladen
# - optional: BigQuery RAW Load + Views anwenden
# ------------------------------------

param(
    [switch]$InstallDeps,
    [switch]$DownloadFeatureStore,
    [switch]$BigQueryLoadRaw,
    [switch]$BigQueryApplyViews
)

# --------------------------------------------------------
# 0) UTF-8 / Console Robustheit
# --------------------------------------------------------
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
} catch { }
$env:PYTHONUTF8 = "1"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

Write-Host "Projektroot: $projectRoot" -ForegroundColor Cyan

# --------------------------------------------------------
# 1) Python venv aktivieren (wenn vorhanden)
# --------------------------------------------------------
$venvActivate = Join-Path $projectRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    . $venvActivate
    Write-Host "venv aktiviert [OK]" -ForegroundColor Green
} else {
    Write-Host "venv nicht gefunden." -ForegroundColor DarkYellow
}

# --------------------------------------------------------
# 2) Dependencies installieren (optional)
# --------------------------------------------------------
if ($InstallDeps) {
    $req = Join-Path $projectRoot "requirements.txt"
    $reqDev = Join-Path $projectRoot "requirements-dev.txt"

    if (-not (Test-Path $req)) {
        Write-Host "requirements.txt nicht gefunden: $req" -ForegroundColor Red
        exit 1
    }

    if (Test-Path $reqDev) {
        python -m pip install -r $req -r $reqDev
    } else {
        python -m pip install -r $req
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Host "pip install fehlgeschiscchlagen (ExitCode $LASTEXITCODE)." -ForegroundColor Red
        exit $LASTEXITCODE
    }

    Write-Host "Dependencies installiert [OK]" -ForegroundColor Green
}

# --------------------------------------------------------
# 3) Databricks CLI (PATH fuer diese Session)
# --------------------------------------------------------
$databricksInstallDir = Join-Path $env:LOCALAPPDATA "DatabricksCLI"
$databricksExe = Join-Path $databricksInstallDir "databricks.exe"

if (Test-Path $databricksExe) {
    if ($env:Path -notlike "*$databricksInstallDir*") {
        $env:Path = "$databricksInstallDir;$env:Path"
    }
}

# --------------------------------------------------------
# 4) Feature Store Download (optional)
# --------------------------------------------------------
if ($DownloadFeatureStore) {
    $dlScript = Join-Path $projectRoot "scripts\databricks\download_feature_store.ps1"
    if (-not (Test-Path $dlScript)) {
        Write-Host "Script nicht gefunden: $dlScript" -ForegroundColor Red
        exit 1
    }

    & $dlScript
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Feature Store Download fehlgeschlagen (ExitCode $LASTEXITCODE)." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# --------------------------------------------------------
# 5) BigQuery RAW Load / Views (optional)
# --------------------------------------------------------
if ($BigQueryLoadRaw) {
    python -m scripts.bigquery.load_raw_tables
    if ($LASTEXITCODE -ne 0) {
        Write-Host "BigQuery RAW Load fehlgeschlagen (ExitCode $LASTEXITCODE)." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

if ($BigQueryApplyViews) {
    python -m scripts.bigquery.apply_views
    if ($LASTEXITCODE -ne 0) {
        Write-Host "BigQuery Views fehlgeschlagen (ExitCode $LASTEXITCODE)." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Write-Host "start_dev abgeschlossen [OK]" -ForegroundColor Green