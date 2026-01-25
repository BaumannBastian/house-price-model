# Databricks notebook source
import sys
from pathlib import Path

def _find_repo_root() -> str:
    """
    Find the Databricks repo root (folder that contains both 'src' and 'cloud')
    in a robust way, regardless of where the notebook is executed from.
    """
    # 1) Try to derive from notebook path in workspace context
    try:
        ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
        nb_path = ctx.notebookPath().get()  # e.g. /Users/<mail>/house-price-model/notebooks/lakehouse/01_bronze_job
        ws_abs = Path("/Workspace") / str(nb_path).lstrip("/")
        p = ws_abs
        for _ in range(12):
            if (p / "src").exists() and (p / "cloud").exists():
                return str(p)
            p = p.parent
    except Exception:
        pass

    # 2) Fallback: walk up from current working directory
    p = Path.cwd()
    for _ in range(12):
        if (p / "src").exists() and (p / "cloud").exists():
            return str(p)
        p = p.parent

    raise RuntimeError(
        "Konnte Repo-Root nicht finden (Ordner 'src' + 'cloud' nicht gefunden). "
        "Starte das Notebook im Repo-Kontext oder prüfe Repo-Pfad/Sync."
    )

REPO_ROOT = _find_repo_root()

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

print(f"REPO_ROOT={REPO_ROOT}")