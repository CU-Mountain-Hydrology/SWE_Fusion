# run/utils.py

import os
from datetime import datetime

from config import Config

def make_directories(date: int, cfg: Config):
    """
    Makes the results and report directories, along with any necessary subdirectories.
    Unlike in the original SWE_FUSION.py, the MODIS directory does not need to be made here since it is now handled
    in download/download_modis.py

    :param date: int (YYYYMMDD) Date the model is run on.
    :param cfg: Configuration object containing environment variables from the .env.

    """
    print(f"Making results and report directories for {date}...",end="")

    # Make results directories
    os.makedirs(f"{cfg.ww_results_workspace}/{date}_results", exist_ok=True)
    os.makedirs(f"{cfg.snm_results_workspace}/{date}_results", exist_ok=True)

    # Make report directories
    year = datetime.strptime(f"{date}", "%Y%m%d").year
    ww_report_dir = f"{cfg.ww_reports_workspace.format(year=year)}/{date}_RT_report"
    snm_report_dir = f"{cfg.snm_reports_workspace.format(year=year)}/{date}_RT_report"
    os.makedirs(ww_report_dir, exist_ok=True)
    os.makedirs(snm_report_dir, exist_ok=True)

    # Make SNODAS directories
    os.makedirs(f"{ww_report_dir}/SNODAS", exist_ok=True)
    os.makedirs(f"{snm_report_dir}/SNODAS", exist_ok=True)

    print(". \033[32mDone.\033[0m")
