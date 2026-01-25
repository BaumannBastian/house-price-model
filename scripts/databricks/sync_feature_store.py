# ------------------------------------
# scripts/databricks/sync_feature_store.py
#
# Lädt Gold-Daten (train_gold.parquet / test_gold.parquet) aus Databricks
# nach data/feature_store/ runter.
#
# Es wird zuerst ein manifest.json gezogen und mit der lokalen Version
# verglichen. Nur wenn es Änderungen gibt, werden die Parquet-Files
# neu geladen.
# ------------------------------------

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple


DEFAULT_REMOTE_BASE = "dbfs:/Volumes/workspace/house_prices/feature_store"
DEFAULT_LOCAL_DIR = Path("data/feature_store")

MANIFEST_NAME = "manifest.json"
TRAIN_NAME = "train_gold.parquet"
TEST_NAME = "test_gold.parquet"


def _find_databricks_exe() -> str:
    exe = shutil.which("databricks")
    if exe:
        return exe

    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        candidate = Path(local_appdata) / "DatabricksCLI" / "databricks.exe"
        if candidate.exists():
            return str(candidate)

    raise RuntimeError(
        "Databricks CLI nicht gefunden. Stelle sicher, dass 'databricks' im PATH ist "
        "oder dass %LOCALAPPDATA%\\DatabricksCLI\\databricks.exe existiert."
    )


def _run(cmd: list[str]) -> Tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _cp_from_dbfs(databricks_exe: str, src: str, dst: Path, profile: Optional[str] = None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()

    cmd = [databricks_exe]
    if profile:
        cmd += ["--profile", profile]
    cmd += ["fs", "cp", src, str(dst)]
    code, out, err = _run(cmd)
    if code != 0:
        raise RuntimeError(f"Databricks copy failed: {src} -> {dst}\n{err or out}")


def sync_feature_store(
    remote_base: str = DEFAULT_REMOTE_BASE,
    local_dir: Path = DEFAULT_LOCAL_DIR,
    profile: Optional[str] = None,
) -> Path:
    """
    Synchronisiert den Feature-Store lokal und gibt den lokalen Ordner zurück.
    """
    databricks_exe = _find_databricks_exe()

    local_dir.mkdir(parents=True, exist_ok=True)

    local_manifest = local_dir / MANIFEST_NAME
    remote_manifest_tmp = local_dir / f"{MANIFEST_NAME}.remote"

    remote_manifest_src = f"{remote_base}/{MANIFEST_NAME}"
    _cp_from_dbfs(databricks_exe, remote_manifest_src, remote_manifest_tmp, profile=profile)

    remote_data = json.loads(remote_manifest_tmp.read_text(encoding="utf-8"))
    remote_stamp = json.dumps(remote_data, sort_keys=True)

    if local_manifest.exists():
        local_data = json.loads(local_manifest.read_text(encoding="utf-8"))
        local_stamp = json.dumps(local_data, sort_keys=True)
        if local_stamp == remote_stamp:
            remote_manifest_tmp.unlink(missing_ok=True)
            return local_dir

    # Manifest ist neu -> Parquet runterladen
    train_src = f"{remote_base}/{remote_data.get('files', {}).get('train', TRAIN_NAME)}"
    test_src = f"{remote_base}/{remote_data.get('files', {}).get('test', TEST_NAME)}"

    _cp_from_dbfs(databricks_exe, train_src, local_dir / TRAIN_NAME, profile=profile)
    _cp_from_dbfs(databricks_exe, test_src, local_dir / TEST_NAME, profile=profile)

    # Remote manifest als lokale "source of truth" übernehmen
    local_manifest.write_text(json.dumps(remote_data, ensure_ascii=False, indent=2), encoding="utf-8")
    remote_manifest_tmp.unlink(missing_ok=True)

    return local_dir