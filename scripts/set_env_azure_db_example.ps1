# ------------------------------
# scripts/set_env_azure_db.example.ps1
#
# Beispiel-Skript zum Setzen der Azure-DB-Umgebungsvariablen.
# Kopiere diese Datei nach `scripts/set_env_azure_db.ps1` und
# trage dort DEIN Passwort lokal ein. Die Datei `set_env_azure_db.ps1`
# wird nicht versioniert.
# ------------------------------

$env:DB_HOST = "house-price-psql.postgres.database.azure.com"
$env:DB_PORT = "5432"
$env:DB_NAME = "house_prices"
$env:DB_USER = "hpadmin"
$env:DB_PASSWORD = "<HIER_LOKAL_PASSWORT_EINTRAGEN>"
$env:DB_SSLMODE = "require"

Write-Host "Azure-DB-Umgebungsvariablen gesetzt."
Write-Host "DB_HOST = $env:DB_HOST"
Write-Host "DB_NAME = $env:DB_NAME"
Write-Host "DB_USER = $env:DB_USER"