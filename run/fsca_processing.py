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
from SWE_Fusion_functions import fsca_processing_tif, calculate_dmfsca, create_mean_layer


def fsca_processing(date: int, cfg: Config):
    """
    This is a wrapper that handles automatically calling the functions from SWE_Fusion_functions.py that process
    the FSCA and create the DMFSCA and mean layers.

    :param date: (YYYYMMDD) Date the model is to be run on. FSCA data will be downloaded up through this date
    :param cfg: Configuration object containing environment variables from the .env
    """
    # Determine date of oldest fSCA image that's not processed
    start_date = dt(2000,0,0)
    for file in os.listdir(cfg.processed_fsca_path):
        if not file.endswith(".tif"):
            continue
        filename = file.split(".")[0]
        try:
            file_date = dt.strptime(filename, "%Y%m%d")
        except ValueError:
            continue
        if file_date > start_date:
            start_date = file_date
    start_date += timedelta(days=1)

    # Determine date of the newest fSCA image to process
    end_date = dt.strptime(str(date), "%Y%m%d")

    # Check that the fsca data has been downloaded for all those dates
    date_list = [
        start_date.date() + timedelta(days=1) for _ in range((end_date.date() - start_date.date()).days + 1)
    ]
    for d in date_list:
        download_fsca(int(d.strftime("%Y%m%d")), cfg)

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
        # Get calendar year that the current water year starts in (current WY - 1)
        today = dt.now()
        prev_water_year = today.year if today >= dt(today.year, 10, 1) else today.year - 1

        calculate_dmfsca(
            fSCA_folder=cfg.processed_fsca_path,
            DMFSCA_folder=cfg.dmfsca_path,
            wateryear_start=dt(prev_water_year, 10, 1), # ex. Oct 1, 2025 for a model run in Jan 2026
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