# ------------------------------------
# scripts/databricks/sync_feature_store.py
#
# Lädt Gold-Feature-Store Dateien aus Databricks (DBFS Volume) nach lokal:
#   data/feature_store/{train_gold.parquet, test_gold.parquet, manifest.json}
#
# Idee:
# - Databricks Gold-Job schreibt ein manifest.json (Stamp).
# - Dieses Script lädt zuerst das Remote-Manifest.
# - Wenn es identisch zum lokalen Manifest ist -> nichts tun.
# - Sonst -> Parquet-Dateien + Manifest aktualisieren.
#
# Usage:
#   python -m scripts.databricks.sync_feature_store
#   python -m scripts.databricks.sync_feature_store --profile <PROFILE_NAME>
# ------------------------------------

from __future__ import annotations

import argparse
import configparser
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple


REMOTE_BASE = "dbfs:/Volumes/workspace/house_prices/feature_store"
REMOTE_MANIFEST = f"{REMOTE_BASE}/manifest.json"
REMOTE_TRAIN = f"{REMOTE_BASE}/train_gold.parquet"
REMOTE_TEST = f"{REMOTE_BASE}/test_gold.parquet"

LOCAL_DIR = Path("data/feature_store")
LOCAL_MANIFEST = LOCAL_DIR / "manifest.json"
LOCAL_TRAIN = LOCAL_DIR / "train_gold.parquet"
LOCAL_TEST = LOCAL_DIR / "test_gold.parquet"


def _detect_profile() -> Optional[str]:
    """
    Best-effort Profile-Detection für Databricks CLI.

    - Wenn DATABRICKS_PROFILE oder DATABRICKS_CONFIG_PROFILE gesetzt ist -> verwenden
    - Sonst ~/.databrickscfg parsen:
        * Wenn [DEFAULT] existiert -> None (CLI nutzt Default)
        * Wenn genau 1 anderes Profil existiert -> dieses verwenden
        * Sonst None (User muss explizit --profile setzen)
    """
    for env_key in ("DATABRICKS_PROFILE", "DATABRICKS_CONFIG_PROFILE"):
        val = os.environ.get(env_key)
        if val:
            return val

    cfg_path = Path.home() / ".databrickscfg"
    if not cfg_path.exists():
        return None

    cp = configparser.ConfigParser()
    cp.read(cfg_path, encoding="utf-8")

    if cp.has_section("DEFAULT"):
        return None

    sections = [s for s in cp.sections() if s.strip()]
    if len(sections) == 1:
        return sections[0]

    return None


def _find_databricks_exe() -> str:
    env = os.environ.get("DATABRICKS_CLI_PATH")
    if env and Path(env).exists():
        return env

    exe = shutil.which("databricks")
    if exe:
        return exe

    raise RuntimeError(
        "Databricks CLI nicht gefunden. "
        "Bitte `databricks` in PATH oder ENV DATABRICKS_CLI_PATH setzen."
    )


def _run(cmd: list[str]) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def _cp_from_dbfs(databricks_exe: str, src: str, dst: str, profile: Optional[str]) -> None:
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [databricks_exe]
    if profile:
        cmd += ["--profile", profile]
    cmd += ["fs", "cp", src, str(dst_path), "--overwrite"]

    rc, out, err = _run(cmd)
    if rc != 0:
        raise RuntimeError(f"Databricks copy failed: {src} -> {dst}\n{err or out}")


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def sync_feature_store(profile: Optional[str] = None) -> None:
    databricks_exe = _find_databricks_exe()
    if profile is None:
        profile = _detect_profile()

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    tmp_manifest = LOCAL_DIR / "manifest.json.remote"

    # 1) Remote manifest holen
    _cp_from_dbfs(databricks_exe, REMOTE_MANIFEST, str(tmp_manifest), profile=profile)

    remote = _read_json(tmp_manifest)
    if remote is None:
        raise RuntimeError("Remote manifest konnte nicht gelesen werden.")

    local = _read_json(LOCAL_MANIFEST)

    # 2) Wenn identisch -> fertig
    if local is not None and local == remote and LOCAL_TRAIN.exists() and LOCAL_TEST.exists():
        print("Feature-Store: lokal aktuell (manifest unverändert).")
        tmp_manifest.unlink(missing_ok=True)
        return

    # 3) Sonst Parquet + Manifest laden
    _cp_from_dbfs(databricks_exe, REMOTE_TRAIN, str(LOCAL_TRAIN), profile=profile)
    _cp_from_dbfs(databricks_exe, REMOTE_TEST, str(LOCAL_TEST), profile=profile)

    LOCAL_MANIFEST.write_text(json.dumps(remote, indent=2), encoding="utf-8")
    tmp_manifest.unlink(missing_ok=True)

    print("Feature-Store: lokal aktualisiert.")
    print(f"- {LOCAL_TRAIN}")
    print(f"- {LOCAL_TEST}")
    print(f"- {LOCAL_MANIFEST}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default=None, help="Databricks CLI profile name (optional).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sync_feature_store(profile=args.profile)