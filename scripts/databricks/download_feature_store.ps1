# ------------------------------------
# scripts/databricks/download_feature_store.ps1
#
# Lädt die Gold-Feature-Store Artefakte (manifest.json + Parquets) aus Databricks
# nach lokal: data/feature_store/
#
# Usage
# ------------------------------------
#   .\scripts\databricks\download_feature_store.ps1
#   .\scripts\databricks\download_feature_store.ps1 -Force
#   .\scripts\databricks\download_feature_store.ps1 -Profile "basti.baumann@gmx.net"
# ------------------------------------

[CmdletBinding()]
param(
  [string]$RemoteDir = "dbfs:/Volumes/workspace/house_prices/feature_store",
  [switch]$Force,
  [string]$Profile = $env:DATABRICKS_CONFIG_PROFILE
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$LocalDir = Join-Path $RepoRoot "data\feature_store"

New-Item -ItemType Directory -Force -Path $LocalDir | Out-Null

$RemoteManifest = "$RemoteDir/manifest.json"
$RemoteTrain    = "$RemoteDir/train_gold.parquet"
$RemoteTest     = "$RemoteDir/test_gold.parquet"

$LocalManifest      = Join-Path $LocalDir "manifest.json"
$LocalManifestTmp   = Join-Path $LocalDir "manifest.remote.json"
$LocalTrain         = Join-Path $LocalDir "train_gold.parquet"
$LocalTest          = Join-Path $LocalDir "test_gold.parquet"

$ProfileArgs = @()
if ($Profile -and $Profile.Trim() -ne "") {
  $ProfileArgs = @("--profile", $Profile)
}

function Dbx {
  param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Args)
  databricks @ProfileArgs @Args | Out-Host
  if ($LASTEXITCODE -ne 0) { throw "Databricks CLI failed: databricks $($Args -join ' ')" }
}

function Copy-FromDbx([string]$src, [string]$dst) {
  Write-Host "Downloading: $src -> $dst"
  Dbx fs cp $src $dst --overwrite
}

function HashOrEmpty([string]$path) {
  if (-not (Test-Path $path)) { return "" }
  return (Get-FileHash $path -Algorithm SHA256).Hash
}

Write-Host "Profile  : $($Profile -or '(default)')"
Write-Host "RemoteDir: $RemoteDir"
Dbx fs ls $RemoteDir

Copy-FromDbx $RemoteManifest $LocalManifestTmp

if (-not $Force) {
  $hLocal  = HashOrEmpty $LocalManifest
  $hRemote = HashOrEmpty $LocalManifestTmp
  if ($hLocal -ne "" -and $hLocal -eq $hRemote) {
    Remove-Item -Force $LocalManifestTmp
    Write-Host "OK: Feature store up-to-date (manifest unverändert)."
    exit 0
  }
}

Copy-FromDbx $RemoteTrain $LocalTrain
Copy-FromDbx $RemoteTest  $LocalTest

Move-Item -Force $LocalManifestTmp $LocalManifest

Write-Host "DONE. Lokale Dateien:"
Get-ChildItem $LocalDir | Format-Table Name, Length, LastWriteTime