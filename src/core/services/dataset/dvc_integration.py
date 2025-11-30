from __future__ import annotations
import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Dict

from src.core.settings import get_settings

log = logging.getLogger("dvc")

REPO_ROOT = Path(".")
DVC_DIR = REPO_ROOT / ".dvc"
REMOTE_NAME = "storage"


def _env() -> Dict[str, str]:
    s = get_settings()
    env = os.environ.copy()

    env.setdefault("AWS_ACCESS_KEY_ID", s.s3_access_key_id or "")
    env.setdefault("AWS_SECRET_ACCESS_KEY", s.s3_secret_access_key or "")
    env.setdefault("AWS_ENDPOINT_URL", s.s3_endpoint_url)
    env.setdefault("AWS_DEFAULT_REGION", "us-east-1")

    return env


def _run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    log.debug("exec: %s", cmd)

    return subprocess.run(
        shlex.split(cmd),
        cwd=str(REPO_ROOT),
        env=_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=check
    )


def setup_dvc():
    """
    Setup DVC
    """
    try:
        if not DVC_DIR.exists():
            _run("dvc init --no-scm -q")
            log.info("DVC initialized (--no-scm)")

        s = get_settings()
        out = _run("dvc remote list", check=False).stdout.strip()
    
        if REMOTE_NAME not in out:
            _run(f"dvc remote add -f {REMOTE_NAME} {s.dvc_remote}")
            _run(f"dvc remote modify {REMOTE_NAME} endpointurl {s.s3_endpoint_url}")

            if s.s3_access_key_id:
                _run(f"dvc remote modify {REMOTE_NAME} access_key_id {shlex.quote(s.s3_access_key_id)}")
            if s.s3_secret_access_key:
                _run(f"dvc remote modify {REMOTE_NAME} secret_access_key {shlex.quote(s.s3_secret_access_key)}")

            log.info("DVC remote configured: %s -> %s", REMOTE_NAME, s.dvc_remote)

    except Exception as e:
        log.warning("DVC setup failed: %s", e)


def add_and_push(path: Path):
    """
    Add dataset to DVC and push to remote
    """
    try:
        _run(f"dvc add -q {shlex.quote(str(path))}")
        _run(f"dvc push -q -r {REMOTE_NAME}")
        log.info("DVC pushed: %s", path)
    except Exception as e:
        log.warning(
            "DVC add/push failed for %s: %s", 
            path,
            e
        )


def remove_output(path: Path):
    """
    Remove dataset from DVC
    """
    try:
        dvc_meta = Path(str(path) + ".dvc")
        if dvc_meta.exists():
            _run(f"dvc remove -q --outs --force {shlex.quote(str(dvc_meta))}")
            log.info("DVC removed: %s", dvc_meta)
    except Exception as e:
        log.warning(
            "DVC remove failed for %s: %s",
            path,
            e
        )
