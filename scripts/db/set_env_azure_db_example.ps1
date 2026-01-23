# ------------------------------
# scripts/set_env_azure_db.example.ps1
#
# Template für Azure-Postgres ENV-Variablen.
#
# Wichtig:
# - Diese Datei gehört INS Repo (ohne Secrets).
# - Erstelle lokal eine Kopie als:
#     scripts/set_env_azure_db.ps1
#   und trage dort DB_PASSWORD ein.
# - scripts/set_env_azure_db.ps1 darf NICHT committed werden.
# ------------------------------

$env:DB_HOST     = "YOUR_SERVER.postgres.database.azure.com"
$env:DB_PORT     = "5432"
$env:DB_NAME     = "house_prices"
$env:DB_USER     = "hpadmin"
$env:DB_PASSWORD = "PUT_PASSWORD_HERE"

$env:DB_SSLMODE  = "require"

Write-Host "Azure-DB-Umgebungsvariablen gesetzt (Example)." -ForegroundColor Green
Write-Host "DB_HOST = $env:DB_HOST"
Write-Host "DB_NAME = $env:DB_NAME"
Write-Host "DB_USER = $env:DB_USER"