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


def get_water_year(date: int) -> int:
    """
    Calculates the water year for a given date.
    Water year 2026 starts October 1, 2025, and ends September 30, 2026.

    :param date: (YYYYMMDD) The date to calculate the water year for.
    :return: (int) The water year formatted as YYYY.
    """
    date_f = datetime.strptime(str(date), "%Y%m%d")
    return date_f.year + 1 if date_f >= datetime(date_f.year, 10, 1) else date_f.year


def get_previous_model_run(date: int, domain: str, cfg: Config) -> tuple[int, str]:
    """
    Gets the date and name of the model run prior to (not including) the given date.

    :param date: (YYYYMMDD) The date to get the model run for.
    :param domain: (str) The domain to get the model run for ("WW" or "SNM")
    :param cfg: Configuration object containing environment variables from the .env.

    :return: tuple[int,str] The date and name of the model run prior to the given date.
    """
    if domain == "SNM":
        # Get dates from all results subdirectories
        subdirs = [x for x in os.listdir(cfg.snm_results_workspace) if os.path.isdir(f"{cfg.snm_results_workspace}/{x}")]
        run_dates = [int(x[:8]) for x in subdirs if x.endswith("_results")]
        run_dates.sort()

        # Find the model run date prior to the given date
        previous_run_date = run_dates[0]
        for run_date in run_dates:
            if run_date < date:
                previous_run_date = run_date
            else:
                break

        # Access config logs to get model run name for that date
        water_year = get_water_year(previous_run_date)
        config_dir = f"{cfg.rmodel_config_log_dir}/WY{water_year}/{previous_run_date}"
        model_runs = os.listdir(config_dir)

        # TODO: How to determine which previous model run to use? woCCR? What if there are multiple woCCR?
        previous_model_run = model_runs[0].split(".")[0]

        return previous_run_date, previous_model_run


    elif domain == "WW":
        print("get_previous_model_run has not been implemented yet for the WW")
    else:
        print(f"Error - get_previous_model_run: domain {domain} not recognized!")
    return 0, ""


if __name__ == "__main__":
    # Quick dev tests
    config = Config()
    prev_date, prev_run = get_previous_model_run(20260518, "SNM", config)
    print(prev_date, prev_run)