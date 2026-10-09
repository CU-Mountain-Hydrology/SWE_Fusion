# run/fsca_processing.py

"""
This is a restructured version of fSCA_processing_alone.py along with the functions from SWE_Fusion_functions.py that
were called from within fSCA_processing_alone.py
"""

from datetime import timedelta
from datetime import datetime as dt
import arcpy
import os

from config import Config
from download.download_fsca import download_fsca
from run.utils import get_water_year
from SWE_Fusion_functions import fsca_processing_tif, calculate_dmfsca, create_mean_layer


def fsca_processing(date: int, cfg: Config):
    """
    This is a wrapper that handles automatically calling the functions from SWE_Fusion_functions.py that process
    the FSCA and create the DMFSCA and mean layers.

    :param date: (YYYYMMDD) Date the model is to be run on. FSCA data will be downloaded up through this date
    :param cfg: Configuration object containing environment variables from the .env
    """
    # Determine date of oldest fSCA image that's not processed
    water_year_start = dt(get_water_year(date) - 1, 10, 1)
    last_processed = water_year_start - timedelta(days=1)
    for file in os.listdir(cfg.processed_fsca_path):
        if not file.endswith(".tif"):
            continue
        filename = file.split(".")[0]
        try:
            file_date = dt.strptime(filename, "%Y%m%d")
        except ValueError:
            continue
        if file_date > last_processed:
            last_processed = file_date
    start_date = last_processed + timedelta(days=1)

    # Determine date of the newest fSCA image to process
    end_date = dt.strptime(str(date), "%Y%m%d")

    # One call per calendar year
    wy_start_int = int(water_year_start.strftime("%Y%m%d"))
    if start_date.year < end_date.year:
        download_fsca(int(f"{start_date.year}1231"), cfg, start_date=wy_start_int)
    download_fsca(date, cfg, start_date=wy_start_int)

    # Process fSCA Data
    print(f"Processing fSCA data from {start_date.strftime('%Y%m%d')} to {end_date.strftime('%Y%m%d')}...", end="")
    try:
        fsca_processing_tif(
            start_date=start_date,
            end_date=end_date,
            tile_list=cfg.fsca_tiles,
            netCDF_WS=cfg.local_fsca_path,
            output_fscaWS=cfg.processed_fsca_path,
            proj_in=arcpy.SpatialReference(cfg.sin_modis_proj),
            proj_out=arcpy.SpatialReference(4269),
            snap_raster=cfg.fsca_snap_raster,
            extent=cfg.fsca_extent
        )
        print(". \033[32mDone.\033[0m")
    except Exception as e:
        print(f"\n {e}")

    # Calculate DMFSCA
    print(f"Calculating DMFSCA data from {start_date.strftime('%Y%m%d')} to {end_date.strftime('%Y%m%d')}...", end="")
    try:
        calculate_dmfsca(
            fSCA_folder=cfg.processed_fsca_path,
            DMFSCA_folder=cfg.dmfsca_path,
            wateryear_start=water_year_start, # ex. Oct 1, 2025 for a model run in Jan 2026
            process_start_date=start_date,
            process_end_date=end_date,
        )
        print(". \033[32mDone.\033[0m")
    except Exception as e:
        print(f"\n {e}")

    # Create mean layer
    print(f"Creating mean layer for {end_date.strftime('%Y%m%d')}...", end="")
    try:
        create_mean_layer(
            input_workspace=cfg.mean_layer_workspace,
            output_folder=cfg.mean_layer_output,
            dateList=[end_date.strftime('%m%d')], # Date list is only the model run date when running daily
            start_year=cfg.mean_layer_start_year,
            end_year=cfg.mean_layer_end_year,
        )
        print(". \033[32mDone.\033[0m")
    except Exception as e:
        print(f"\n {e}")


if __name__ == "__main__":
    config = Config()
    fsca_processing(20260520, config)