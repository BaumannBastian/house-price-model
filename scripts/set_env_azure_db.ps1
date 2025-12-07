# ------------------------------
# scripts/set_env_azure_db.ps1
#
# Setzt die Umgebungsvariablen für die Azure-PostgreSQL-Datenbank
# (DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DB_SSLMODE).
#
# Hinweis:
# - Die Werte stammen aus `terraform output -raw ...`.
# - Dieses Skript sollte nach Aktivierung des venv ausgeführt werden.
# ------------------------------

$env:DB_HOST     = "house-price-psql.postgres.database.azure.com"
$env:DB_PORT     = "5432"
$env:DB_NAME     = "house_prices"
$env:DB_USER     = "hpadmin"
$env:DB_PASSWORD = "qZS2z7FZ2MuWs!E5Z1HPsYcz"
$env:DB_SSLMODE  = "require"

Write-Host "Azure-DB-Umgebungsvariablen gesetzt."
Write-Host "DB_HOST = $env:DB_HOST"
Write-Host "DB_NAME = $env:DB_NAME"
Write-Host "DB_USER = $env:DB_USER"