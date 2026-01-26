# ------------------------------------
# scripts/databricks/run_lakehouse_jobs.py
#
# Triggert die Databricks Lakehouse Jobs (bronze/silver/gold) via Databricks CLI.
# Nutzt feste Job-IDs (stabil, kein jobs list nötig) und autodetectet CLI-Profile.
#
# Usage
# ------------------------------------
#   python scripts/databricks/run_lakehouse_jobs.py
#   python scripts/databricks/run_lakehouse_jobs.py --stage all
#   python scripts/databricks/run_lakehouse_jobs.py --stage bronze
#
#   # optional
#   python scripts/databricks/run_lakehouse_jobs.py --profile basti.baumann@gmx.net
#   python scripts/databricks/run_lakehouse_jobs.py --stage all --no-wait
# ------------------------------------

from __future__ import annotations

import argparse
import configparser
import os
import subprocess
from pathlib import Path
from typing import Optional


JOB_ID_BY_STAGE = {
    "bronze": 1040997863835935,
    "silver": 870785997610023,
    "gold": 272073945184603,
}

STAGE_ORDER = ["bronze", "silver", "gold"]


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


def _run(cmd: list[str]) -> None:
    print(f"\n>> {' '.join(cmd)}")
    # Kein capture_output => CLI kann sauber authen / logs ausgeben
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def run_job(job_id: int, profile: Optional[str], no_wait: bool, timeout: Optional[str]) -> None:
    cmd = ["databricks"]
    if profile:
        cmd += ["--profile", profile]
    cmd += ["jobs", "run-now", str(job_id)]

    if no_wait:
        cmd.append("--no-wait")
    if timeout:
        cmd += ["--timeout", timeout]

    _run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "bronze", "silver", "gold"], default="all")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--timeout", default=None, help="z.B. '45m0s'")
    args = parser.parse_args()

    profile = args.profile if args.profile is not None else _detect_profile()

    stages = STAGE_ORDER if args.stage == "all" else [args.stage]
    print("Stages:", " -> ".join(stages))
    print("Profile:", profile or "(default)")

    for stg in stages:
        job_id = JOB_ID_BY_STAGE[stg]
        print(f"\n=== Running {stg.upper()} (job_id={job_id}) ===")
        run_job(job_id, profile=profile, no_wait=args.no_wait, timeout=args.timeout)

    print("\n Fertig.")


if __name__ == "__main__":
    main()