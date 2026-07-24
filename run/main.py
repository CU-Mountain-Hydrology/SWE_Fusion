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
from run.utils import make_directories, get_previous_model_run, get_water_year
from run.tables_and_layers import ww_tables_and_layers, snm_tables_and_layers
from SWE_Fusion_functions import geopackage_to_shapefile, merge_sort_sensors_surveys

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
    surveys_use="Y"

    # GPKG -> SHP for woCCR model run
    # Convert geopackage to shapefile for woCCR model run
    # TODO: restructure to just pass cfg directly
    pillow_date = datetime.strptime(str(date), "%Y%m%d").strftime("%d%b%Y")
    domainList = ["NOCN", "PNW", "SNM", "SOCN", "INMT"] # TODO: temporarily placed this here, move to config
    geopackage_to_shapefile(report_date=str(date), pillow_date=pillow_date, model_run=model_woCCR,
                            user=cfg.rmodel_username, domainList=domainList, model_workspace=cfg.regress_path,
                            results_workspace=f"{cfg.ww_results_workspace}/{date}_results/")

    ## ------- Should this whole section be moved to another function? --------------
    # Get previous run date to know if differencing should be used
    prev_ww_rundate, _ = get_previous_model_run(date=date, domain="WW", cfg=cfg)

    if get_water_year(prev_ww_rundate) == get_water_year(date):
        difference = "Y" # If the previous report date was from this water year, then difference="Y" (compare to last report)
    else:
        difference = "N"

    # Process & Sort WW sensors & surveys
    merge_sort_sensors_surveys(
        report_date=str(date), results_workspace=f"{cfg.ww_results_workspace}/{date}_results/",
        surveys=surveys_use, difference=difference,
        watershed_shapefile=cfg.ww_watershed_shapefile, case_field_wtrshd=cfg.case_field_watershed,
        case_field_band=cfg.case_field_band, band_shapefile=cfg.ww_band_shapefile, projOut=arcpy.SpatialReference(cfg.proj_alb), merge="Y",
        domainList=domainList, prev_report_date=str(prev_ww_rundate), prev_results_workspace=f"{cfg.ww_results_workspace}/{prev_ww_rundate}_results/")


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
    # Get previous run date to know if differencing should be used
    prev_snm_rundate, _ = get_previous_model_run(date=date, domain="SNM", cfg=cfg)

    if get_water_year(prev_snm_rundate) == get_water_year(date):
        difference = "Y"  # If the previous report date was from this water year, then difference="Y" (compare to last report)
    else:
        difference = "N"

    merge_sort_sensors_surveys(
        report_date=str(date), results_workspace=f"{cfg.snm_results_workspace}/{date}_results/",
        surveys=surveys_use, difference=difference,
        watershed_shapefile=cfg.snm_watershed_shapefile, band_shapefile=cfg.snm_band_shapefile,
        case_field_wtrshd=cfg.case_field_watershed, case_field_band=cfg.case_field_band,
        projOut=arcpy.SpatialReference(cfg.proj_alb), projIn=arcpy.SpatialReference(cfg.proj_geo),
        domain = "SNM", merge="N", domain_shapefile=cfg.snm_sensors_shp.format(date=date), prev_report_date=str(prev_snm_rundate),
        prev_results_workspace=f"{cfg.snm_results_workspace}/{prev_snm_rundate}_results/")

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

