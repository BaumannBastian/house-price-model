# ------------------------------------
# scripts/lakehouse/run_lakehouse_jobs.py
#
# Triggert die Databricks Lakehouse Jobs (bronze/silver/gold) via Databricks CLI.
#
# Voraussetzungen
# ------------------------------------
# - Databricks CLI installiert (databricks)
# - Auth konfiguriert (databricks auth login / ~/.databrickscfg)
#
# Usage
# ------------------------------------
#   # alle Stages nacheinander
#   python scripts/lakehouse/run_lakehouse_jobs.py
#   python scripts/lakehouse/run_lakehouse_jobs.py --stage all
#
#   # einzelne Stage
#   python scripts/lakehouse/run_lakehouse_jobs.py --stage bronze
#   python scripts/lakehouse/run_lakehouse_jobs.py --stage silver
#   python scripts/lakehouse/run_lakehouse_jobs.py --stage gold
#
#   # optional: nicht warten / anderes CLI-Profil
#   python scripts/lakehouse/run_lakehouse_jobs.py --stage all --no-wait
#   python scripts/lakehouse/run_lakehouse_jobs.py --stage all --profile DEFAULT
# ------------------------------------

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Dict, List, Optional

# Diese Namen müssen zu "databricks jobs list" passen
JOB_NAME_BY_STAGE = {
    "bronze": "house_prices_bronze",
    "silver": "house_prices_silver",
    "gold": "house_prices_gold",
}

STAGE_ORDER = ["bronze", "silver", "gold"]


def _run_cmd(cmd: List[str]) -> str:
    """Run a command, raise on error, return stdout."""
    print(f"\n>> {' '.join(cmd)}")
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.stdout:
        print(p.stdout.strip())
    if p.returncode != 0:
        if p.stderr:
            print(p.stderr.strip(), file=sys.stderr)
        raise SystemExit(p.returncode)
    return p.stdout


def _databricks_base_cmd(profile: Optional[str]) -> List[str]:
    cmd = ["databricks"]
    # Global flag; klappt bei der neuen CLI auch nach dem Subcommand, aber so ist es sauber.
    if profile:
        cmd += ["--profile", profile]
    return cmd


def list_jobs_by_name(profile: Optional[str]) -> Dict[str, int]:
    """Return {job_name: job_id} from `databricks jobs list -o json`."""
    cmd = _databricks_base_cmd(profile) + ["jobs", "list", "-o", "json"]
    out = _run_cmd(cmd)

    data = json.loads(out) if out.strip() else None
    if data is None:
        raise RuntimeError("Konnte Jobs nicht laden: leere JSON-Antwort von 'databricks jobs list'.")

    # CLI kann je nach Version list oder dict mit "jobs" liefern
    jobs = data["jobs"] if isinstance(data, dict) and "jobs" in data else data
    if not isinstance(jobs, list):
        raise RuntimeError(f"Unerwartetes Format von 'jobs list': {type(jobs)}")

    mapping: Dict[str, int] = {}
    for j in jobs:
        if not isinstance(j, dict):
            continue
        job_id = j.get("job_id")
        name = (j.get("settings") or {}).get("name") or j.get("name")
        if name and job_id is not None:
            mapping[name] = int(job_id)

    return mapping


def run_job(job_id: int, profile: Optional[str], no_wait: bool, timeout: Optional[str]) -> None:
    cmd = _databricks_base_cmd(profile) + ["jobs", "run-now", str(job_id)]
    if no_wait:
        cmd.append("--no-wait")
    if timeout:
        # CLI erwartet duration, z.B. "20m0s" oder "45m0s"
        cmd += ["--timeout", timeout]
    _run_cmd(cmd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["all", "bronze", "silver", "gold"],
        default="all",
        help="Welche Stufe laufen soll. 'all' läuft bronze->silver->gold.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Optionaler Databricks CLI Profile-Name (sonst Default).",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Nicht auf Job-Ende warten (CLI gibt dann direkt Run-ID zurück).",
    )
    parser.add_argument(
        "--timeout",
        default=None,
        help="Max. Wartezeit bis TERMINATED/SKIPPED, z.B. '20m0s' (nur relevant ohne --no-wait).",
    )
    args = parser.parse_args()

    jobs_by_name = list_jobs_by_name(args.profile)

    def _job_id_for(stage: str) -> int:
        name = JOB_NAME_BY_STAGE[stage]
        if name not in jobs_by_name:
            available = ", ".join(sorted(jobs_by_name.keys()))
            raise SystemExit(f"Job '{name}' nicht gefunden. Verfügbare Jobs: {available}")
        return jobs_by_name[name]

    stages_to_run = STAGE_ORDER if args.stage == "all" else [args.stage]

    print("Stages:", " -> ".join(stages_to_run))
    for stg in stages_to_run:
        job_id = _job_id_for(stg)
        print(f"\n=== Running {stg.upper()} ({JOB_NAME_BY_STAGE[stg]} | id={job_id}) ===")
        run_job(job_id, args.profile, args.no_wait, args.timeout)

    print("\n✅ Fertig.")


if __name__ == "__main__":
    main()