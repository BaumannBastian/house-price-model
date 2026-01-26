# ------------------------------------
# scripts/databricks/update_jobs_to_repo.py
#
# Aktualisiert die 3 Lakehouse-Jobs so, dass sie Notebooks aus /Repos/... nutzen.
# Danach nehmen die Jobs automatisch den neuesten Git-Stand, sobald du
# `sync_repo.py` ausführst (repos update).
#
# Usage
# ------------------------------------
#   python scripts/databricks/update_jobs_to_repo.py --repo-path "/Repos/<user>/house-price-model" --profile "<profile>"
# ------------------------------------

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from typing import Optional


JOB_IDS = {
    "bronze": 1040997863835935,
    "silver": 870785997610023,
    "gold": 272073945184603,
}

TASKKEY_TO_NOTEBOOK = {
    "house_prices_bronze": "notebooks/lakehouse/01_bronze_job",
    "house_prices_silver": "notebooks/lakehouse/02_silver_job",
    "house_prices_gold": "notebooks/lakehouse/03_gold_job",
}


def _run_json(cmd: list[str]) -> dict:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or p.stdout.strip() or f"Command failed: {' '.join(cmd)}")
    out = (p.stdout or "").strip()
    if not out:
        raise RuntimeError(f"Empty JSON output: {' '.join(cmd)}")
    return json.loads(out)


def _run(cmd: list[str]) -> None:
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def _dbx(profile: Optional[str], *args: str) -> list[str]:
    cmd = ["databricks"]
    if profile:
        cmd += ["--profile", profile]
    cmd += list(args)
    return cmd


def update_job(job_id: int, repo_path: str, profile: Optional[str]) -> None:
    # 1) job settings holen
    job = _run_json(_dbx(profile, "jobs", "get", str(job_id), "--output", "json"))
    settings = job.get("settings")
    if not isinstance(settings, dict):
        raise RuntimeError(f"Unexpected job format for job_id={job_id}: missing settings")

    tasks = settings.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise RuntimeError(f"Unexpected job format for job_id={job_id}: missing tasks[]")

    # 2) notebook paths patchen
    changed = False
    for t in tasks:
        task_key = t.get("task_key")
        nb = t.get("notebook_task")
        if not isinstance(nb, dict) or "notebook_path" not in nb:
            continue

        if task_key in TASKKEY_TO_NOTEBOOK:
            target = f"{repo_path.rstrip('/')}/{TASKKEY_TO_NOTEBOOK[task_key]}"
            if nb["notebook_path"] != target:
                nb["notebook_path"] = target
                changed = True

    if not changed:
        print(f"job_id={job_id}: nichts zu ändern (Notebook-Pfade scheinen schon korrekt).")
        return

    # 3) reset payload bauen (reset überschreibt settings komplett – wir nehmen aber die aktuellen settings als Basis)
    payload = {
        "job_id": job_id,
        "new_settings": settings,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        tmp_path = f.name

    print(f"\nUpdating job_id={job_id} -> notebooks from {repo_path}")
    _run(_dbx(profile, "jobs", "reset", "--json", f"@{tmp_path}"))
    print("OK")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-path", required=True, help='z.B. "/Repos/basti.baumann@gmx.net/house-price-model"')
    p.add_argument("--profile", default=None)
    args = p.parse_args()

    for stage in ("bronze", "silver", "gold"):
        update_job(JOB_IDS[stage], args.repo_path, args.profile)


if __name__ == "__main__":
    main()