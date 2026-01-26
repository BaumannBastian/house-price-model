# ------------------------------------
# scripts/databricks/sync_repo.py
#
# Sync (pull) a Databricks Repo to latest remote commit on a branch.
# Optional "reset if conflict": backup -> delete -> recreate -> checkout branch.
#
# Usage
# ------------------------------------
#   python scripts/databricks/sync_repo.py --repo-path "/Repos/<user>/house-price-model"
#   python scripts/databricks/sync_repo.py --repo-id 123456789
#
#   # if merge conflicts due to edits in Databricks:
#   python scripts/databricks/sync_repo.py --repo-path "/Repos/<user>/house-price-model" --reset-if-conflict --backup
# ------------------------------------

from __future__ import annotations

import argparse
import configparser
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


def _detect_profile() -> Optional[str]:
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


def _run(cmd: list[str], capture: bool = False) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=capture, text=True)
    out = (p.stdout or "").strip()
    err = (p.stderr or "").strip()
    return p.returncode, out, err


def _run_or_die(cmd: list[str], capture: bool = False) -> str:
    print(f"\n>> {' '.join(cmd)}")
    rc, out, err = _run(cmd, capture=capture)
    if rc != 0:
        raise RuntimeError(err or out or f"Command failed with rc={rc}")
    if capture:
        return out
    return ""


def _git_url_normalize(url: str) -> str:
    if url.startswith("https://github.com/") and not url.endswith(".git"):
        return url.rstrip("/") + ".git"
    return url


def repos_get(ref: str, profile: Optional[str]) -> Optional[dict]:
    cmd = ["databricks"]
    if profile:
        cmd += ["--profile", profile]
    cmd += ["--output", "json", "repos", "get", ref]
    rc, out, _ = _run(cmd, capture=True)
    if rc != 0 or not out:
        return None
    return json.loads(out)


def repos_update(ref: str, branch: str, profile: Optional[str]) -> None:
    cmd = ["databricks"]
    if profile:
        cmd += ["--profile", profile]
    cmd += ["repos", "update", ref, "--branch", branch]
    _run_or_die(cmd)


def repos_delete(ref: str, profile: Optional[str]) -> None:
    cmd = ["databricks"]
    if profile:
        cmd += ["--profile", profile]
    cmd += ["repos", "delete", ref]
    _run_or_die(cmd)


def repos_create(git_url: str, provider: str, repo_path: str, profile: Optional[str]) -> None:
    cmd = ["databricks"]
    if profile:
        cmd += ["--profile", profile]
    cmd += ["repos", "create", git_url, provider, "--path", repo_path]
    _run_or_die(cmd)


def backup_workspace_dir(source_path: str, target_dir: Path, profile: Optional[str]) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["databricks"]
    if profile:
        cmd += ["--profile", profile]
    cmd += ["workspace", "export-dir", source_path, str(target_dir), "--overwrite"]
    _run_or_die(cmd)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default=None)
    p.add_argument("--repo-path", default=None)
    p.add_argument("--repo-id", default=None)
    p.add_argument("--branch", default="main")
    p.add_argument("--git-url", default="https://github.com/BaumannBastian/house-price-model")
    p.add_argument("--provider", default="gitHub")
    p.add_argument("--reset-if-conflict", action="store_true")
    p.add_argument("--backup", action="store_true")
    p.add_argument("--backup-dir", default=None)
    args = p.parse_args()

    if not args.repo_path and not args.repo_id:
        raise SystemExit("Please provide --repo-path or --repo-id.")

    profile = args.profile if args.profile is not None else _detect_profile()
    ref = str(args.repo_id) if args.repo_id else args.repo_path
    git_url = _git_url_normalize(args.git_url)

    print(f"Ref     : {ref}")
    print(f"Branch  : {args.branch}")
    print(f"Profile : {profile or '(default)'}")
    print(f"Git URL : {git_url}")

    info = repos_get(ref, profile)
    if info is None:
        if not args.repo_path:
            raise SystemExit("Repo not found via --repo-id. Use --repo-path to allow create.")
        print("\nRepo not found -> creating it.")
        repos_create(git_url, args.provider, args.repo_path, profile)
        info = repos_get(args.repo_path, profile)

    repo_path = (info or {}).get("path") or args.repo_path

    try:
        print("\nSyncing repo (repos update)...")
        repos_update(ref, args.branch, profile)
        print("\n✅ Repo synced.")
        return
    except Exception as e:
        print("\n⚠️ repos update failed (likely local changes/merge conflict in Databricks).")
        print(str(e))
        if not args.reset_if_conflict:
            raise SystemExit("\nRun again with --reset-if-conflict (optionally --backup) to hard-reset the repo.")

    if args.backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = Path(args.backup_dir) if args.backup_dir else Path(".tmp/databricks_repo_backup") / ts
        print(f"\nBacking up workspace folder -> {backup_dir}")
        backup_workspace_dir(repo_path, backup_dir, profile)

    print("\nHard reset (delete + recreate)...")
    repos_delete(ref, profile)
    time.sleep(1)

    if not repo_path:
        raise SystemExit("Cannot recreate repo because repo_path is unknown. Provide --repo-path.")
    repos_create(git_url, args.provider, repo_path, profile)

    print("\nSyncing branch...")
    repos_update(repo_path, args.branch, profile)
    print("\n✅ Repo hard-reset + synced.")


if __name__ == "__main__":
    main()