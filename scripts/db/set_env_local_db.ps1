# ------------------------------
# scripts/set_env_local_db.ps1
#
# Setzt die Umgebungsvariablen für die lokale Docker-Postgres-DB.
# ------------------------------

$env:DB_HOST     = "localhost"
$env:DB_PORT     = "5432"
$env:DB_NAME     = "house_prices"
$env:DB_USER     = "house"
$env:DB_PASSWORD = "house"
$env:DB_SSLMODE  = "disable"

Write-Host "Local-DB-Umgebungsvariablen gesetzt."
Write-Host "DB_HOST = $env:DB_HOST"
Write-Host "DB_NAME = $env:DB_NAME"
Write-Host "DB_USER = $env:DB_USER"