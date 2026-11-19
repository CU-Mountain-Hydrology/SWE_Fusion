# vetting/parse_log.py
"""
 Functions to read the latest output log file from a model run, and look for any errors or potential issues that warrant
 being notified about. See vetting/teams_alert.py for how to make the notification.

 Checks for the following issues:
  * Pipeline Stuck (High Severity)
"""

import os
import glob
import re

from config import Config
from run.utils import get_water_year
from vetting.teams_alert import send_teams_alert

_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \[")


def _notify(notify: bool, cfg: Config, severity: str, title: str, message: str = "",
            details: dict | None = None, code: str | None = None):
    """
    Wrapper for send_teams_alert() to handle alternately printing to terminal only when notify = False
    """
    if notify:
        send_teams_alert(cfg, severity, title, message, details, code=code)
    else:
        print(f"{severity} severity: {title}")
        print(f"{message}")
        print(f"{details}")
        if code:
            print(code)


def _pipeline_stuck(date, log_file, notify, cfg):
    low, med, high = [], [], []
    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    for i, line in enumerate(lines):
        if "PIPELINE STUCK" not in line:
            continue

        # The stderr tail is the untimestamped continuation of this log record
        tail = []
        for nxt in lines[i + 1:]:
            if _TS_RE.match(nxt):
                break
            tail.append(nxt)

        tail = [l for l in tail if l.strip() and l.strip() != "stderr tail:"]
        tail = tail[-cfg.std_err_tail_lines:]
        code = "\n".join(tail)

        _notify(notify, cfg,
                severity="high",
                title=f"{date} Model Stuck",
                message=line,
                code=code)
        high += [("Model Stuck", line, code)]
    return low, med, high


def parse_log(date: int, cfg: Config, notify: bool = True):
    """
    Main function to run all the parsing and send notifications

    :param date: YYYYMMDD Date to parse the logs for
    :param cfg: Configuration object containing environment variables set in the .env. Namely MODEL_RUN_LOG_DIR
    :param notify: True to send teams notifications about errors or potential issues

    """
    date_f = f"{str(date)[:4]}-{str(date)[4:6]}-{str(date)[6:]}"

    # Open output log
    log_dir = cfg.model_run_log_dir.format(water_year=get_water_year(date))
    logs = glob.glob(os.path.join(log_dir, f"run_{date}*"))
    if len(logs) == 0:
        print(f"No log files found for {date}. Cannot parse for errors.")
        return

    # Sort logs to get the latest one
    sorted_logs = sorted(
        logs, key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]
    )
    log_file = sorted_logs[-1]
    print(f"Parsing model status from {log_file}")

    # Keep running list of low, medium, and high severity issues
    low = []
    med = []
    high = []

    # Check if the pipeline is stuck
    l, m, h = _pipeline_stuck(date_f, log_file, notify=notify, cfg=cfg)
    low += l
    med += m
    high += h

    # TODO: so many more checks


    # TODO: Send alert if low or med warnings exist (high severity already have alerts)

    if not low and not med and not high:
        print("Run succeeded without warnings or issues.")
        if notify:
            send_teams_alert(cfg, severity="ok", title=f"{date_f} Run Succeeded", message="No warnings or issues detected.")

if __name__ == "__main__":
    config = Config()
    parse_log(20261115, config, notify=False)
