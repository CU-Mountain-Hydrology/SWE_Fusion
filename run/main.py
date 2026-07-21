# run/main.py

import argparse
import arcpy
from datetime import datetime

from config import Config
from download.download_fsca import download_fsca
from download.download_snodas import download_snodas
from download.download_snowtrax import download_snowtrax
from download.download_sensors import download_sensors
from run.fsca_processing import fsca_processing
from run.run_R_model import write_simulation_date, run_R_model
from run.utils import make_directories
from run.tables_and_layers import ww_tables_and_layers, snm_tables_and_layers
from SWE_Fusion_functions import geopackage_to_shapefile

def run_model(date: int, prompt_user: bool=False):
    """
    This is the master script to automatically run the whole model. Each step is called sequentially, with checkpoints
    in order to resume mid way through in case of a fatal error.
    TODO: add checkpoints
    TODO: time elapsed / progress indicator
    TODO: clear arcpy locks between function calls <- look more into what this actually does

    :param date: (YYYYMMDD) Date the model is run on.
    :param prompt_user: Ask the user for confirmation before downloading files. Default: False

    """
    # Set up configuration object containing parameters from the .env
    cfg = Config()

    # Download fSCA data
    download_fsca(date, "ssh", cfg, prompt_user=prompt_user)

    # Download SNODAS data
    download_snodas(date, cfg)

    # Download SnowTrax SWE data
    download_snowtrax(cfg)

    # fSCA Processing
    fsca_processing(date, cfg)

    # Set date in text file at SIMULATION_DATE_PATH from the .env
    # This is never read by the automated daily model, only kept for consistency and manual runs.
    write_simulation_date(date, cfg)

    # Download snow sensor data
    download_sensors(date, cfg)

    # Run model with CCR
    _,model_wCCR = run_R_model(date=date, isCCR=True, cfg=cfg)

    # Run model without CCR
    _,model_woCCR = run_R_model(date=date, isCCR=False, cfg=cfg)

    # Disable arcpy parallel processing for SWE Fusion steps
    # TODO: learn why this caused crash and hopefully fix issue so it can run faster
    arcpy.env.parallelProcessingFactor = "0"

    # Create results and report directories
    make_directories(date, cfg)

    # Download surveys on the first of the month
    # TODO: syrveys may come in throughout the week, only do this for first table of the month
    # TODO: function to download surveys and check if it is different than before in which case run with surveys again

    # GPKG -> SHP for woCCR model run
    # Convert geopackage to shapefile for woCCR model run
    # TODO: restructure to just pass cfg directly
    pillow_date = datetime.strptime(str(date), "%Y%m%d").strftime("%d%b%Y")
    domainList = ["NOCN", "PNW", "SNM", "SOCN", "INMT"] # TODO: temporarily placed this here, move to config
    geopackage_to_shapefile(report_date=str(date), pillow_date=pillow_date, model_run=model_woCCR,
                            user=cfg.rmodel_username, domainList=domainList, model_workspace=cfg.regress_path,
                            results_workspace=f"{cfg.ww_results_workspace}/{date}_results/")

    ## ------- Should this whole section be moved to another function? --------------
    # Process & Sort WW sensors & surveys
    # TODO

    # Run SNODAS for WW
    # TODO
    # Add error handling for if SNODAS is not downloaded (err line 226)
    # Should this be run when SNODAS is downloaded instead of now?

    # Tables and Layers WW wCCR/woCCR
    ww_tables_and_layers(date, model_wCCR, model_woCCR, cfg)

    # Get zero sensors for all domains
    # TODO: why here
    ## ------- ^^^^^ Should this whole section be moved to another function? --------------


    # Process & Sort SNM sensors & surveys
    # TODO

    # Run SNODAS for SNM
    # TODO

    # Tables and Layers SNM wCCR/woCCR
    snm_tables_and_layers(date, model_woCCR, model_woCCR, cfg)

    # Get zero sensors for SNM
    # TODO: is this redundant?

    ######### Vetting Starts Now #######




    # Download MODIS true color imagery for SNM report
    # TODO


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("date", type=int, help="YYYYMMDD Date the model is run on.")
    parser.add_argument("-u", "--prompt_user", action="store_true",
                        help="Prompt the user before overwriting or automatically selecting files")

    args = parser.parse_args()

    # Sanity checks
    date_d = args.date.strftime("%Y%m%d")
    if date_d > datetime.today():
        print(f"Date ({args.date} is in the future! Aborting.")
        exit(1)

    # Run Model
    run_model(args.date, args.prompt_user)

