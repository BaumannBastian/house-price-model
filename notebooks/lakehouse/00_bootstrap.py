# Databricks notebook source
import sys
from pathlib import Path


def _normalize_nb_path(nb_path: str) -> str:
    """
    Normalisiert ctx.notebookPath(), das je nach Kontext z.B. sein kann:
      - /Users/<mail>/...
      - /Workspace/Users/<mail>/...
      - /Repos/<user>/<repo>/...
      - /Workspace/Repos/<user>/<repo>/...
    Wir wollen daraus immer einen Pfad relativ unter /Workspace machen.
    """
    p = str(nb_path).replace("\\", "/").lstrip("/")

    # Manche Kontexte liefern bereits "Workspace/..."
    if p.startswith("Workspace/"):
        p = p[len("Workspace/") :]

    return p


def _find_repo_root() -> str:
    """
    Find the Databricks repo root (folder that contains both 'src' and 'cloud')
    in a robust way, regardless of where the notebook is executed from.
    """
    # 1) Try to derive from notebook path in workspace context
    try:
        ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
        nb_path = ctx.notebookPath().get()

        nb_rel = _normalize_nb_path(nb_path)
        ws_abs = Path("/Workspace") / nb_rel  # -> /Workspace/Users/... OR /Workspace/Repos/...

        p = ws_abs
        for _ in range(20):
            if (p / "src").exists() and (p / "cloud").exists():
                return str(p)
            p = p.parent
    except Exception:
        pass

    # 2) Fallback: walk up from current working directory
    p = Path.cwd()
    for _ in range(20):
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