# run/utils.py

import os
import glob
from datetime import datetime as dt

from config import Config

def make_directories(date: int, cfg: Config, report_dirs: bool = True):
    """
    Makes the results and report directories, along with any necessary subdirectories.
    Unlike in the original SWE_FUSION.py, the MODIS directory does not need to be made here since it is now handled
    in download/download_modis.py

    Only creates SNM directories if "SNM" is in the DOMAIN_LIST set in the .env

    :param date: int (YYYYMMDD) Date the model is run on.
    :param cfg: Configuration object containing environment variables from the .env.

    """
    print(f"Making results and report directories for {date}...",end="")

    # Make results directories
    os.makedirs(f"{cfg.ww_results_workspace}/{date}_results", exist_ok=True)
    if "SNM" in cfg.domain_list:
        os.makedirs(f"{cfg.snm_results_workspace}/{date}_results", exist_ok=True)

    # Make report directories
    if report_dirs:
        year = dt.strptime(f"{date}", "%Y%m%d").year
        ww_report_dir = f"{cfg.ww_reports_workspace.format(year=get_water_year(year))}/{date}_RT_report"
        snm_report_dir = f"{cfg.snm_reports_workspace.format(year=get_water_year(year))}/{date}_RT_report"
        os.makedirs(ww_report_dir, exist_ok=True)
        if "SNM" in cfg.domain_list:
            os.makedirs(snm_report_dir, exist_ok=True)

        # Make SNODAS directories
        os.makedirs(f"{ww_report_dir}/SNODAS", exist_ok=True)
        if "SNM" in cfg.domain_list:
            os.makedirs(f"{snm_report_dir}/SNODAS", exist_ok=True)

    print(". \033[32mDone.\033[0m")


def mark_usethis_useavg(date: int, cfg: Config):
    """
    Marks the different model run directories with UseThis and UseAvg in both the results and reports directories for
    both WW and SNM. This function is called before any ASO processing, so the directories may need to be renamed after
    that processing is complete.

    In the event there are multiple woCCR model runs with differing other parameters, the most recent will be used as
    the UseThis directory. The UseAvg directory will always use the same methods that have been used historically.

    This function is idempotent: any existing UseThis/UseAvg tags are stripped before the directories are re-tagged,
    so it is safe to rerun for a date that has already been processed. If stripping a tag would collide with an
    existing untagged directory (e.g. the model was rerun and regenerated it), the older of the two is moved to a
    sibling "{directory}_stale" folder rather than deleted.

    TODO: decide between using an error metric for wCCR vs woCCR or just always using woCCR

    :param date: (YYYYMMDD) Date to rename the directories for.
    :param cfg: Configuration object containing environment variables from the .env.
    """
    import shutil
    import time

    print(f"Marking directories with UseThis and UseAvg...", end="")

    year = dt.strptime(str(date), "%Y%m%d").year
    directories = {
        "ww_results": f"{cfg.ww_results_workspace}/{date}_results",
        "ww_report": f"{cfg.ww_reports_workspace.format(year=get_water_year(year))}/{date}_RT_report",
        "snm_results": f"{cfg.snm_results_workspace}/{date}_results",
        "snm_report": f"{cfg.snm_reports_workspace.format(year=get_water_year(year))}/{date}_RT_report",
    }

    if not "SNM" in cfg.domain_list:
        directories.pop("snm_results", None)
        directories.pop("snm_report", None)

    for d in directories.keys():
        base = directories[d]

        # Strip tags left over from any previous run so this function is safe to rerun.
        # Only the directory name is modified (not the parent path), and _UseThis is removed first so that
        # names like ..._UseAvg_UseThis are unwound in the right order.
        for tag in ("_UseThis", "_UseAvg"):
            for path in glob.glob(f"{base}/*{tag}*"):
                parent, name = os.path.split(path)
                clean = os.path.join(parent, name.replace(tag, ""))

                if os.path.exists(clean):
                    # Duplicate (e.g. the model was rerun and regenerated the untagged dir).
                    # Keep the newer one, move the older one aside rather than deleting it.
                    stale_root = f"{base}_stale"
                    os.makedirs(stale_root, exist_ok=True)
                    if os.path.getmtime(path) > os.path.getmtime(clean):
                        # Tagged copy is newer: shelve the untagged one, then untag this one
                        shutil.move(clean, os.path.join(stale_root, f"{os.path.basename(clean)}_{int(time.time())}"))
                        os.rename(path, clean)
                    else:
                        # Untagged copy is newer: shelve the tagged one
                        shutil.move(path, os.path.join(stale_root, f"{name}_{int(time.time())}"))
                else:
                    os.rename(path, clean)

        # UseAvg dir always uses the same methods (and therefore same directory name)
        useavg_dir = f"{base}/RT_CanAdj_rcn_woCCR_nofscamskSens"
        if not os.path.exists(useavg_dir):
            raise FileNotFoundError(f"No potential UseAvg directory found in {base}!")
        os.rename(useavg_dir, useavg_dir + "_UseAvg")

        # Find all woCCR directories (currently only woCCR model runs are considered for UseThis, see the to-do above)
        woCCR_dirs = glob.glob(f"{base}/*woCCR*")
        if not woCCR_dirs:
            raise FileNotFoundError(f"No woCCR directory found in {base}!")

        # Sort matches by time and use the most recent for UseThis
        usethis_dir = max(woCCR_dirs, key=os.path.getmtime)
        os.rename(usethis_dir, usethis_dir + "_UseThis")

    print(". \033[32mDone.\033[0m")


def get_water_year(date: int) -> int:
    """
    Calculates the water year for a given date.
    Water year 2026 starts October 1, 2025, and ends September 30, 2026.

    :param date: (YYYYMMDD) The date to calculate the water year for.
    :return: (int) The water year formatted as YYYY.
    """
    date_f = dt.strptime(str(date), "%Y%m%d")
    return date_f.year + 1 if date_f >= dt(date_f.year, 10, 1) else date_f.year


def get_previous_model_run(date: int, domain: str, cfg: Config) -> tuple[int, str]:
    """
    Gets the date and name of the model run prior to (not including) the given date.

    :param date: (YYYYMMDD) The date to get the model run for.
    :param domain: (str) The domain to get the model run for ("WW" or "SNM")
    :param cfg: Configuration object containing environment variables from the .env.

    :return: tuple[int,str] The date and name of the model run prior to the given date.
    """
    results_workspace = None
    match domain:
        case "WW":
            results_workspace = cfg.ww_results_workspace
        case "SNM":
            results_workspace = cfg.snm_results_workspace
        case _:
            print(f"Error - get_previous_model_run: domain {domain} not recognized!")
            return 0, ""

    # Get dates from all results subdirectories
    subdirs = [x for x in os.listdir(results_workspace) if os.path.isdir(f"{results_workspace}/{x}")]
    run_dates = [int(x[:8]) for x in subdirs if x.endswith("_results")]
    run_dates.sort()

    # Find the model run date prior to the given date
    previous_run_date = run_dates[0]
    for run_date in run_dates:
        if run_date < date:
            previous_run_date = run_date
        else:
            break
    # TODO: error handling for if run_dates is Null or if there is no previous date

    # Access config logs to get model run name for that date
    water_year = get_water_year(previous_run_date)
    config_dir = f"{cfg.rmodel_config_log_dir}/WY{water_year}/{previous_run_date}"
    if not os.path.exists(config_dir):
        print(f"No config logs found for previous run! {config_dir}")
        # TODO: documentation on how to create a JSON config for the previous model run
        # TODO: check results dir for a UseThis tag, then strip it
        return previous_run_date, ""
    model_runs = os.listdir(config_dir)
    print(f"DEBUG - Model runs: {model_runs}")

    # TODO: How to determine which previous model run to use? woCCR? What if there are multiple woCCR?
    previous_model_run = model_runs[0].split(".")[0]

    return previous_run_date, previous_model_run


from SWE_Fusion_functions import zero_CCR_sensors
def get_zero_sensors(date: int, domains: list[str], model_wCCR: str, cfg: Config):
    """
    Wrapper for calling zero_CCR_sensors
    TODO: refactor original function to incorporate into pipeline
    TODO: add error handling

    :param date: (YYYYMMDD) The date the model is run on (and also the date for the pillows).
    :param domains: (list[str]) The list of domains to get zero sensors for.
    :param model_wCCR: Name of the model run with CoCoRaHS sensors enabled
    :param cfg: Configuration object containing environment variables from the .env.
    """
    # When running on only SNM, use SNM results workspace
    if domains == ["SNM"]:
        results_workspace = cfg.snm_results_workspace
        sensors_path = "{results_workspace}/{date}_results/SNM_{date}_sensors_albn83.shp"
    else:
        results_workspace = cfg.ww_results_workspace
        sensors_path = "{results_workspace}/{date}_results/{date}_sensors_{domain}.shp"


    pillow_date = dt.strptime(str(date), "%Y%m%d").strftime("%d%b%Y")
    for domain in domains:
        zero_CCR_sensors(
            rundate=str(date), results_workspace=results_workspace, pillow_date=pillow_date, domain=domain,
            sensors=sensors_path.format(results_workspace=results_workspace, date=date, domain=domain), zero_sensors=True,
            CCR=False, model_workspace_domain=f"{cfg.regress_path}/{domain}/{cfg.rmodel_username}/StationSWERegressionV2/data/outputs/{model_wCCR}/"
        )


def get_usethis_dir(parent_dir: str):
    import glob
    matches = glob.glob(f"{parent_dir}/*UseThis*")
    if not matches:
        raise FileNotFoundError(f"Could not find a UseThis directory in {parent_dir}!")
    return matches[0]


if __name__ == "__main__":
    # Quick dev tests
    config = Config()
    prev_date, prev_run = get_previous_model_run(20260516, "WW", config)
    print(prev_date, prev_run)