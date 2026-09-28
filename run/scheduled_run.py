"""
run/scheduled_run.py

Daily wrapper around run/main.py, launched by Windows Task Scheduler.

Automatically runs the model for the previous day, within the conda environment. On a fatal error, the main script is
retried using the checkpoint system managed in main.py. After cfg.max_retries failed attempts, it exits. If the same
error (same checkpoint & stderr response) is hit twice in a row, it exits early.
"""

import json
import logging
import os
import subprocess
import sys
import time
from datetime import timedelta
from datetime import datetime as dt
from pathlib import Path

# Task Scheduler starts in System32 by default, set cwd to ...\SWE_Fusion
# This has to be derived from the current filepath here rather than set in the .env since we need
# to know what the project root is initially to even find the .env and import Config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from utils import get_water_year

def setup_logging(date_str: str, cfg: Config) -> logging.Logger:
    """
    TODO: docs
    """
    log_dir = cfg.model_run_log_dir.format(water_year=get_water_year(int(date_str)))
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("scheduled_run")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    log_format = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(f"{log_dir}/run_{date_str}.log", encoding="utf-8")
    fh.setFormatter(log_format)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(log_format)
    logger.addHandler(ch)

    return logger


def checkpoint_file(date: int, cfg: Config) -> Path:
    ckpt_dir = Path(cfg.checkpoint_path.format(water_year=get_water_year(date)))
    return ckpt_dir / f"checkpoint_{date}.json"


def read_checkpoint(cp_path: Path):
    """
    Returns (completed_steps, last_updated), or (None, None) if missing/unreadable.
    On success main.py archives the file to completed/, so it's only expected to exist after a failed run.
    """
    if not cp_path.exists():
        return None, None
    try:
        with open(cp_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("completed_steps"), data.get("last_updated")
    except (json.JSONDecodeError, OSError):
        return None, None


def git_pull(logger: logging.Logger) -> bool:
    """
    Pull the latest code in PROJECT_ROOT before running. Returns True on success.
    A dirty working tree with throw an error which gets logged and treated as fatal.
    """
    cmd = ["git", "pull"]
    logger.info(f"Running: {' '.join(cmd)} (cwd={PROJECT_ROOT})")
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        shell=True,
    )
    if proc.stdout:
        logger.info(f"git pull stdout:\n{proc.stdout.strip()}")
    if proc.stderr:
        logger.warning(f"git pull stderr:\n{proc.stderr.strip()}")
    if proc.returncode != 0:
        logger.error(f"git pull failed (exit {proc.returncode}); aborting run.")
        return False
    return True


def stderr_tail(stderr_text: str, n_lines: int) -> str:
    if not stderr_text:
        return ""
    return "\n".join(stderr_text.strip().splitlines()[-n_lines:])


def flag_stuck(date_str: str, reason: str, logger: logging.Logger, cfg: Config):
    """
    Write a STUCK marker and log CRITICAL.
    """
    logger.critical(f"PIPELINE STUCK for {date_str}: {reason}")
    log_dir = cfg.model_run_log_dir.format(water_year=get_water_year(int(date_str)))
    (log_dir / f"STUCK_{date_str}.flag").write_text(reason, encoding="utf-8")


def run_once(date_str: str, logger: logging.Logger, cfg: Config):
    """
    Runs main.py once as a module from PROJECT_ROOT. Returns (returncode, stderr_text).
    Uses subprocess rather than direct import so that this script continues running even if main hard crashes (eg. OOM)
    """
    cmd = ["conda", "run", "-n", cfg.conda_env, "python", "-m", "run.main", "-d", date_str]
    logger.info(f"Running: {' '.join(cmd)} (cwd={PROJECT_ROOT})")

    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        shell=True,
    )

    if proc.stdout:
        logger.info(f"stdout:\n{proc.stdout.strip()}")
    if proc.stderr:
        logger.warning(f"stderr:\n{proc.stderr.strip()}")
    return proc.returncode, proc.stderr


def main():
    # Read config from .env
    cfg = Config()

    # Get yesterday's date
    run_date = dt.now() - timedelta(days=1)
    date_str = run_date.strftime("%Y%m%d")
    date_int = int(date_str)

    # Set up logger
    logger = setup_logging(date_str, cfg)
    logger.info(f"=== Starting SWE_Fusion pipeline for {date_str} ===")

    # Pull latest code
    if not git_pull(logger):
        return 1

    cp_path = checkpoint_file(date_int, cfg)
    logger.info(f"Checkpoint file expected at: {cp_path}")

    prev_steps, prev_updated, prev_stderr = None, None, None
    attempt = 0

    while attempt <= cfg.max_retries:
        if attempt > 0:
            logger.info(f"Retry attempt {attempt}/{cfg.max_retries}...")

        returncode, stderr_text = run_once(date_str, logger, cfg)

        # Success
        if returncode == 0:
            logger.info(f"Pipeline completed successfully for {date_str}.")
            return 0

        # Failure
        cur_steps, cur_updated = read_checkpoint(cp_path)
        cur_stderr = stderr_tail(stderr_text, cfg.stderr_tail_lines)
        logger.warning(
            f"Attempt {attempt} failed (exit {returncode}). "
            f"completed_steps={cur_steps}, last_updated={cur_updated}"
        )

        if attempt > 0:
            # Check if the error is the same as the previous error (stuck-detection)
            same_checkpoint = (cur_steps == prev_steps) and (cur_updated == prev_updated)
            same_error = (cur_stderr == prev_stderr) and cur_stderr != ""
            if same_checkpoint and same_error:
                flag_stuck(
                    date_str,
                    reason=(
                        "No checkpoint progress (same completed_steps/last_updated) "
                        "AND identical stderr tail on consecutive attempts. Likely a "
                        f"persistent failure on the step after "
                        f"'{cur_steps[-1] if cur_steps else 'the start of the run'}'.\n"
                        f"stderr tail:\n{cur_stderr}"
                    ),
                    logger=logger,
                    cfg=cfg,
                )
                return 1

        prev_steps, prev_updated, prev_stderr = cur_steps, cur_updated, cur_stderr
        attempt += 1

        # Retry from last checkpoint
        if attempt <= cfg.max_retries:
            logger.info(f"Waiting {cfg.retry_delay_s}s before retrying...")
            time.sleep(cfg.retry_delay_s)

    flag_stuck(
        date_str,
        reason=(
            f"Exceeded MAX_RETRIES ({cfg.max_retries}) without success and without a "
            f"repeated-error match. Last stderr tail:\n{prev_stderr}"
        ),
        logger=logger,
        cfg=cfg,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())