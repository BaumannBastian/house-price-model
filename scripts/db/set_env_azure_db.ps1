# ------------------------------
# scripts/set_env_azure_db.ps1
#
# Azure-Postgres ENV-Variablen (lokal, enthält Secrets).
#
# Wichtig:
# - Diese Datei ist lokal und darf NICHT committed werden.
# - Im Repo liegt nur das Template:
#     scripts/set_env_azure_db.example.ps1
# ------------------------------

$env:DB_HOST     = "house-price-psql.postgres.database.azure.com"
$env:DB_PORT     = "5432"
$env:DB_NAME     = "house_prices"
$env:DB_USER     = "user"
$env:DB_PASSWORD = "password"

$env:DB_SSLMODE  = "require"

Write-Host "Azure-DB-Umgebungsvariablen gesetzt." -ForegroundColor Green
Write-Host "DB_HOST = $env:DB_HOST"
Write-Host "DB_NAME = $env:DB_NAME"
Write-Host "DB_USER = $env:DB_USER"
