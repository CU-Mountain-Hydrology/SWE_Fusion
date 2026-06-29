# run/fsca_processing.py

"""
This is a restructured version of fSCA_processing_alone.py along with the functions from SWE_Fusion_functions.py that
were called from within fSCA_processing_alone.py
"""

from datetime import datetime
import arcpy

from config import Config
from SWE_Fusion_functions import fsca_processing_tif, calculate_dmfsca


def fsca_processing(date: int, cfg: Config):
    """
    TODO: docs

    """
    # Determine date of oldest fSCA image that's not processed
    start_date = datetime(None)


    # Determine date of the newest fSCA image to process
    end_date = datetime.strptime(str(date), "%Y%m%d") # TODO: this may not always be true ?? when?

    # Check that the fsca data has been downloaded for all those dates
    # TODO: attempt to download if not found

    # Process fSCA Data
    print(f"Processing fSCA data from {start_date.strftime('%Y%m%d')} to {end_date.strftime('%Y%m%d')}...", end="")
    try:
        fsca_processing_tif(start_date=start_date,
                            end_date=end_date,
                            tile_list=cfg.fsca_tiles,
                            netCDF_WS=cfg.local_fsca_path,
                            output_fscaWS=cfg.processed_fsca_path,
                            proj_in=arcpy.SpatialReference(cfg.sin_modis_proj),
                            proj_out=arcpy.SpatialReference(4269),
                            snap_raster=cfg.fsca_snap_raster,
                            extent=cfg.fsca_extent)
        print(". \033[32mDone.\033[0m")
    except Exception as e:
        print(f"\n {e}")

if __name__ == "__main__":
    config = Config()
    fsca_processing(20260520, config)