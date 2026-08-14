# run/main.py

import argparse
import logging
import arcpy
from datetime import datetime

from config import Config
from run.checkpoint_manager import CheckpointManager

from download.download_fsca import download_fsca
from download.download_snodas import download_snodas
from download.download_snowtrax import download_snowtrax
from download.download_sensors import download_sensors
from run.fsca_processing import fsca_processing
from run.snodas_processing import snodas_processing
from run.sensor_processing import sensor_survey_processing
from run.run_R_model import write_simulation_date, run_R_model
from run.utils import make_directories, get_zero_sensors, get_water_year
from run.tables_and_layers import ww_tables_and_layers, snm_tables_and_layers
from vetting.sensor_plots import model_vs_sensor

from SWE_Fusion_functions import geopackage_to_shapefile, clear_arcpy_locks

logger = logging.getLogger(__name__)


def run_model(date: int, prompt_user: bool=False, reset_checkpoints: bool=False):
    """
    This is the master script to automatically run the whole model. Each step is called sequentially, with checkpoints
    in order to resume mid way through in case of a fatal error.
    TODO: time elapsed / progress indicator
    TODO: look into the root cause necessitating clear_arcpy_locks()
    TODO: add tests to confirm config/.env is set up correctly
    TODO: mark directories with UseThis, UseAvg for report code and vetting plots

    :param date: (YYYYMMDD) Date the model is run on.
    :param prompt_user: Ask the user for confirmation before downloading files. Default: False
    :param reset_checkpoints: Ignore any existing checkpoint for this date and rerun every step. Default: False

    """
    # Set up configuration object containing parameters from the .env
    cfg = Config()

    # Create checkpoint manager that saves the current state and variables to a JSON
    checkpoint_dir = cfg.checkpoint_path.format(water_year=get_water_year(date))
    ckpt = CheckpointManager(date, checkpoint_dir)
    if reset_checkpoints:
        ckpt.reset()

    # Download fSCA data
    ckpt.run_step("download_fsca", download_fsca, date, "ssh", cfg, prompt_user=prompt_user)

    # Download SNODAS data
    ckpt.run_step("download_snodas", download_snodas, date, cfg)

    # Download SnowTrax SWE data
    ckpt.run_step("download_snowtrax", download_snowtrax, cfg)

    # fSCA Processing
    ckpt.run_step("fsca_processing", fsca_processing, date, cfg)
    clear_arcpy_locks()

    # Set date in text file at SIMULATION_DATE_PATH from the .env
    # This is never read by the automated daily model, only kept for consistency and manual runs.
    ckpt.run_step("write_simulation_date", write_simulation_date, date, cfg)

    # Download snow sensor data
    ckpt.run_step("download_sensors", download_sensors, date, cfg)

    # Run model with CCR
    _, model_wCCR = ckpt.run_step("run_R_model_wCCR", run_R_model, date=date, isCCR=True, cfg=cfg)
    clear_arcpy_locks()

    # Run model without CCR
    _, model_woCCR = ckpt.run_step("run_R_model_woCCR", run_R_model, date=date, isCCR=False, cfg=cfg)
    clear_arcpy_locks()

    # Disable arcpy parallel processing for SWE Fusion steps
    # TODO: learn why this caused crash and hopefully fix issue so it can run faster
    arcpy.env.parallelProcessingFactor = "0"

    # Create results and report directories
    ckpt.run_step("make_directories", make_directories, date, cfg)

    # Download surveys on the first of the month
    # TODO: syrveys may come in throughout the week, only do this for first table of the month
    # TODO: function to download surveys and check if it is different than before in which case run with surveys again
    surveys_use="Y"

    # GPKG -> SHP for woCCR model run
    # Convert geopackage to shapefile for woCCR model run
    # TODO: restructure to just pass cfg directly
    pillow_date = datetime.strptime(str(date), "%Y%m%d").strftime("%d%b%Y")
    ckpt.run_step(
        "geopackage_to_shapefile",
        geopackage_to_shapefile,
        report_date=str(date), pillow_date=pillow_date, model_run=model_woCCR,
        user=cfg.rmodel_username, domainList=cfg.domain_list, model_workspace=cfg.regress_path,
        results_workspace=f"{cfg.ww_results_workspace}/{date}_results/",
    )
    clear_arcpy_locks()

    # Process and sort sensors and surveys for WW
    ckpt.run_step(
        "sensor_survey_processing_WW",
        sensor_survey_processing,
        date=date, domain="WW", surveys_use=surveys_use, cfg=cfg,
    )
    clear_arcpy_locks()

    # Run SNODAS for WW
    ckpt.run_step(
        "snodas_processing_WW",
        snodas_processing,
        date=date, domain="WW", model_woCCR=model_woCCR, cfg=cfg,
    )
    clear_arcpy_locks()

    # Tables and Layers WW wCCR/woCCR
    ckpt.run_step("ww_tables_and_layers", ww_tables_and_layers, date, model_wCCR, model_woCCR, cfg)
    clear_arcpy_locks()

    # Get zero sensors for all domains
    # TODO: why here instead of after sensor download
    ckpt.run_step(
        "get_zero_sensors_all",
        get_zero_sensors,
        date=date, domains=cfg.domain_list, model_wCCR=model_wCCR, cfg=cfg,
    )
    clear_arcpy_locks()

    # Process and sort sensors and surveys for SNM
    ckpt.run_step(
        "sensor_survey_processing_SNM",
        sensor_survey_processing,
        date=date, domain="SNM", surveys_use=surveys_use, cfg=cfg,
    )
    clear_arcpy_locks()

    # Run SNODAS for SNM
    ckpt.run_step(
        "snodas_processing_SNM",
        snodas_processing,
        date=date, domain="SNM", model_woCCR=model_woCCR, cfg=cfg,
    )
    clear_arcpy_locks()

    # Tables and Layers SNM wCCR/woCCR
    ckpt.run_step("snm_tables_and_layers", snm_tables_and_layers, date, model_woCCR, model_woCCR, cfg)
    clear_arcpy_locks()

    # Get zero sensors for SNM
    ckpt.run_step(
        "get_zero_sensors_SNM",
        get_zero_sensors,
        date=date, domains=["SNM"], model_wCCR=model_wCCR, cfg=cfg,
    )
    clear_arcpy_locks()

    ######### Vetting Starts Now #######
    domain_to_shapefile = {
        "SNM": r"W:/data/hydro/SNM_Region_albn83.shp",
        "SOCN": r"W:/data/hydro/SOCN_Region_albn83.shp",
        "NOCN": r"W:/data/hydro/NOCN_Region_albn83.shp",
        "INMT": r"W:/data/hydro/INMT_Region_albn83_woTahoe.shp",
        "PNW": r"W:/data/hydro/PNW_Region_albn83_v2.shp",
    }
    year = datetime.strptime(str(date), "%Y%m%d").year
    for domain in cfg.domain_list:
        if domain == "SNM":
            out_path = f"{cfg.snm_reports_workspace.format(year=year)}/{date}_RT_report/model_vs_sensor_{date}_{domain}.png"
        else:
            out_path = f"{cfg.ww_reports_workspace.format(year=year)}/{date}_RT_report/model_vs_sensor_{date}_{domain}.png"
        ckpt.run_step(
            f"model_vs_sensor_{domain}",
            model_vs_sensor,
            date=date, cfg=cfg, focal_sampling=True, extent_shapefile=domain_to_shapefile[domain], out_path=out_path,
            plot_title=f"Model vs Sensor SWE ({domain}) - {date}", save_stats=True
        )


    # Download MODIS true color imagery for SNM report
    # TODO

    # Run completed successfully. Archive the checkpoint so a future run for this same date starts fresh instead of
    # skipping every step. The checkpoint file is moved to {cfg.checkpoint_path}/completed
    ckpt.archive()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("date", type=int, help="YYYYMMDD Date the model is run on.")
    parser.add_argument("-u", "--prompt_user", action="store_true",
                        help="Prompt the user before overwriting or automatically selecting files")
    parser.add_argument("-r", "--reset_checkpoints", action="store_true",
                        help="Ignore any existing checkpoint for this date and rerun every step from the start")

    args = parser.parse_args()

    # Sanity checks
    date_d = args.date.strftime("%Y%m%d")
    if date_d > datetime.today():
        print(f"Date ({args.date} is in the future! Aborting.")
        exit(1)

    # Run Model
    run_model(args.date, args.prompt_user, args.reset_checkpoints)