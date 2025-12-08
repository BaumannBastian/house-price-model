# ------------------------------
# start_dev.ps1
#
# Dieses PowerShell-Skript richtet eine lokale Dev-Session für das
# House-Price-Projekt ein:
# - aktuelle öffentliche IP ermitteln
# - Terraform-Firewall-Regel für Azure-Postgres aktualisieren
# - zurück ins Projektroot wechseln
# - Python-venv aktivieren
# - Azure-DB-Umgebungsvariablen setzen
# - DB-Verbindung testen
#
# Hinweis:
# Vor dem Ausführen ggf. im aktuellen PowerShell-Fenster:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# ausführen, wenn Skripte blockiert werden.
# ------------------------------

param(
    [switch]$SkipTerraform
)

# Robust: Projektroot = Verzeichnis dieses Skripts
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

Write-Host "Projektroot: $projectRoot" -ForegroundColor Cyan

# --------------------------------------------------------
# 1) Aktuelle öffentliche IP ermitteln
# --------------------------------------------------------
try {
    Write-Host "Ermittele öffentliche IP..." -ForegroundColor Yellow
    $clientIp = Invoke-RestMethod -Uri "https://api.ipify.org"
    Write-Host "Öffentliche IP: $clientIp" -ForegroundColor Green
}
catch {
    Write-Host "Fehler beim Ermitteln der öffentlichen IP:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

# --------------------------------------------------------
# 2) Terraform-Firewall-Regel aktualisieren (optional)
# --------------------------------------------------------
if (-not $SkipTerraform) {
    $terraformDir = Join-Path $projectRoot "terraform"
    if (-not (Test-Path $terraformDir)) {
        Write-Host "Terraform-Verzeichnis nicht gefunden: $terraformDir" -ForegroundColor Red
        exit 1
    }

    Set-Location $terraformDir
    Write-Host "Wechsle ins Terraform-Verzeichnis: $terraformDir" -ForegroundColor Cyan

    # terraform apply mit aktueller IP
    $tfCommand = "terraform apply -auto-approve -var `"client_ip=$clientIp`""
    Write-Host "Starte: $tfCommand" -ForegroundColor Yellow

    try {
        terraform apply -auto-approve -var "client_ip=$clientIp"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "terraform apply ist fehlgeschlagen (ExitCode $LASTEXITCODE)." -ForegroundColor Red
            exit $LASTEXITCODE
        }
    }
    catch {
        Write-Host "Fehler bei terraform apply:" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
        exit 1
    }
}
else {
    Write-Host "Terraform-Sektion übersprungen (Parameter -SkipTerraform gesetzt)." -ForegroundColor DarkYellow
}

# --------------------------------------------------------
# 3) Zurück ins Projektroot
# --------------------------------------------------------
Set-Location $projectRoot
Write-Host "Zurück im Projektroot: $projectRoot" -ForegroundColor Cyan

# --------------------------------------------------------
# 4) Python-venv aktivieren
# --------------------------------------------------------
$venvActivate = Join-Path $projectRoot ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $venvActivate)) {
    Write-Host "Virtuelle Umgebung nicht gefunden: $venvActivate" -ForegroundColor Red
    Write-Host "Bitte venv erstellen mit: python -m venv .venv" -ForegroundColor Red
    exit 1
}

Write-Host "Aktiviere virtuelle Umgebung..." -ForegroundColor Yellow
. $venvActivate   # dot-sourcen, damit die Aktivierung im aktuellen Prozess bleibt

# --------------------------------------------------------
# 5) Azure-DB-Umgebungsvariablen setzen
# --------------------------------------------------------
$setEnvScript = Join-Path $projectRoot "scripts\set_env_azure_db.ps1"
if (-not (Test-Path $setEnvScript)) {
    Write-Host "Skript zum Setzen der Azure-DB-ENV nicht gefunden: $setEnvScript" -ForegroundColor Red
    Write-Host "Bitte 'scripts/set_env_azure_db.example.ps1' nach 'scripts/set_env_azure_db.ps1' kopieren und anpassen." -ForegroundColor Red
    exit 1
}

Write-Host "Setze Azure-DB-Umgebungsvariablen..." -ForegroundColor Yellow
. $setEnvScript   # ebenfalls dot-sourcen

# --------------------------------------------------------
# 6) DB-Verbindung testen
# --------------------------------------------------------
Write-Host "Teste DB-Verbindung mit python -m scripts.test_db_connection ..." -ForegroundColor Yellow
python -m scripts.test_db_connection

if ($LASTEXITCODE -eq 0) {
    Write-Host "Startup-Sequence erfolgreich abgeschlossen ✅" -ForegroundColor Green
} else {
    Write-Host "DB-Test ist fehlgeschlagen (ExitCode $LASTEXITCODE)." -ForegroundColor Red
}