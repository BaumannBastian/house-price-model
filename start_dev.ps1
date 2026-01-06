# ------------------------------
# start_dev.ps1
#
# Ziel:
# Ein einziger Entry-Point, um lokal mit dem Projekt starten zu können.
#
# Default: local
# - startet Postgres via Docker (docker compose)
# - setzt lokale DB_* ENV-Variablen
# - aktiviert venv
# - wendet Schema an (python -m scripts.init_schema)
# - testet DB-Verbindung
#
# Optional: azure (Artifact)
# - (optional) Terraform Firewall Update
# - setzt Azure DB_* ENV-Variablen (aus lokalem, NICHT getracktem Script)
#
# Usage:
#   .\start_dev.ps1
#   .\start_dev.ps1 -Mode local
#   .\start_dev.ps1 -Mode azure
#   .\start_dev.ps1 -Mode azure -SkipTerraform
# ------------------------------

param(
    [ValidateSet("local", "azure")]
    [string]$Mode = "local",

    # Nur relevant im Azure-Mode
    [switch]$SkipTerraform
)

# --------------------------------------------------------
# 0) UTF-8 / Console Robustheit
# --------------------------------------------------------
# Wichtig:
# - Ohne das kann Windows bei Unicode-Ausgaben (z.B. Emojis) Probleme machen.
# - Zusätzlich setzen wir PYTHONUTF8, damit Python stabil UTF-8 nutzt.
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
} catch { }
$env:PYTHONUTF8 = "1"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot
Write-Host "Projektroot: $projectRoot" -ForegroundColor Cyan
Write-Host "Mode: $Mode" -ForegroundColor Cyan

# --------------------------------------------------------
# 1) Python venv aktivieren (falls vorhanden)
# --------------------------------------------------------
$venvActivate = Join-Path $projectRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    . $venvActivate
    Write-Host "venv aktiviert [OK]" -ForegroundColor Green
} else {
    Write-Host "Warnung: .venv nicht gefunden. (Optional: python -m venv .venv)" -ForegroundColor DarkYellow
}

# --------------------------------------------------------
# 2) DB ENV setzen + DB starten (local) bzw. Terraform (azure)
# --------------------------------------------------------
if ($Mode -eq "local") {

    Write-Host "Pruefe Docker ..." -ForegroundColor Yellow
    docker info *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Docker scheint nicht zu laufen. Bitte Docker Desktop starten." -ForegroundColor Red
        exit 1
    }

    $composeFile = Join-Path $projectRoot "docker-compose.yml"
    if (-not (Test-Path $composeFile)) {
        Write-Host "docker-compose.yml nicht gefunden: $composeFile" -ForegroundColor Red
        exit 1
    }

    Write-Host "Starte lokale Postgres-DB via docker compose ..." -ForegroundColor Yellow
    docker compose -f $composeFile up -d
    if ($LASTEXITCODE -ne 0) {
        Write-Host "docker compose up ist fehlgeschlagen (ExitCode $LASTEXITCODE)." -ForegroundColor Red
        exit $LASTEXITCODE
    }

    $setLocalEnv = Join-Path $projectRoot "scripts\set_env_local_db.ps1"
    if (-not (Test-Path $setLocalEnv)) {
        Write-Host "set_env_local_db.ps1 nicht gefunden: $setLocalEnv" -ForegroundColor Red
        exit 1
    }

    Write-Host "Setze lokale DB-Umgebungsvariablen ..." -ForegroundColor Yellow
    . $setLocalEnv

} else {

    # Azure Artifact-Flow
    try {
        Write-Host "Ermittele oeffentliche IP..." -ForegroundColor Yellow
        $clientIp = Invoke-RestMethod -Uri "https://api.ipify.org"
        Write-Host "Oeffentliche IP: $clientIp" -ForegroundColor Green
    }
    catch {
        Write-Host "Fehler beim Ermitteln der oeffentlichen IP:" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
        exit 1
    }

    if (-not $SkipTerraform) {
        $terraformDir = Join-Path $projectRoot "terraform"
        if (-not (Test-Path $terraformDir)) {
            Write-Host "Terraform-Verzeichnis nicht gefunden: $terraformDir" -ForegroundColor Red
            exit 1
        }

        Set-Location $terraformDir
        Write-Host "Wechsle ins Terraform-Verzeichnis: $terraformDir" -ForegroundColor Cyan

        terraform init
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

        terraform apply -auto-approve -var "client_ip=$clientIp"
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

        Set-Location $projectRoot
    } else {
        Write-Host "Terraform-Sektion uebersprungen (-SkipTerraform)." -ForegroundColor DarkYellow
    }

    $setAzureEnv = Join-Path $projectRoot "scripts\set_env_azure_db.ps1"
    if (-not (Test-Path $setAzureEnv)) {
        Write-Host "Azure ENV Script nicht gefunden: $setAzureEnv" -ForegroundColor Red
        Write-Host "Nutze scripts\set_env_azure_db.example.ps1 als Vorlage und erstelle lokal scripts\set_env_azure_db.ps1 (nicht committen!)." -ForegroundColor Yellow
        exit 1
    }

    Write-Host "Setze Azure-DB-Umgebungsvariablen..." -ForegroundColor Yellow
    . $setAzureEnv
}

# --------------------------------------------------------
# 3) Auf DB warten (gerade bei Docker-Start)
# --------------------------------------------------------
Write-Host "Warte auf DB (max 120s) ..." -ForegroundColor Yellow

$maxTries = 60
$ok = $false

for ($i = 1; $i -le $maxTries; $i++) {

    # Wichtig:
    # - nutzt --quiet, damit keine Ausgabe/Encoding-Probleme den ExitCode verfälschen
    python -m scripts.test_db_connection --quiet *> $null

    if ($LASTEXITCODE -eq 0) {
        Write-Host "DB ist erreichbar [OK] (Try $i/$maxTries)" -ForegroundColor Green
        $ok = $true
        break
    }

    Start-Sleep -Seconds 2
}

if (-not $ok) {
    Write-Host "DB ist nach 120s nicht erreichbar. Abbruch." -ForegroundColor Red

    if ($Mode -eq "local") {
        Write-Host "Debug: letzte DB-Logs (tail 50):" -ForegroundColor Yellow
        docker logs house-price-postgres --tail 50
    }

    exit 1
}

# --------------------------------------------------------
# 4) Schema anwenden (init_schema)
# --------------------------------------------------------
Write-Host "Wende DB-Schema an (python -m scripts.init_schema) ..." -ForegroundColor Yellow
python -m scripts.init_schema
if ($LASTEXITCODE -ne 0) {
    Write-Host "init_schema ist fehlgeschlagen (ExitCode $LASTEXITCODE)." -ForegroundColor Red
    exit $LASTEXITCODE
}

# --------------------------------------------------------
# 5) DB-Verbindung testen (sichtbar)
# --------------------------------------------------------
Write-Host "Teste DB-Verbindung (python -m scripts.test_db_connection) ..." -ForegroundColor Yellow
python -m scripts.test_db_connection
if ($LASTEXITCODE -eq 0) {
    Write-Host "Startup-Sequence erfolgreich abgeschlossen [OK]" -ForegroundColor Green
} else {
    Write-Host "DB-Test ist fehlgeschlagen (ExitCode $LASTEXITCODE)." -ForegroundColor Red
    exit $LASTEXITCODE
}
