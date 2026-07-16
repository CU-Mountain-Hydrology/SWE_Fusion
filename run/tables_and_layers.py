# run/tables_and_layer.py

from datetime import datetime
import arcpy

from config import Config
from SWE_Fusion_functions import tables_and_layers


def ww_tables_and_layers(date: int, model_wCCR: str, model_woCCR: str, cfg: Config):
    """
    TODO: docs
    TODO: print statements
    TODO: ww_watershed_zones is the same as ww_band_zones, is it always?

    """
    # Projections
    # TODO: Add arcpy names for the numbers
    projGEO = arcpy.SpatialReference(4269)
    projALB = arcpy.SpatialReference(102039)
    ProjOut_UTM = arcpy.SpatialReference(26911)

    year = datetime.strptime(str(date), "%Y%m%d").year
    mean_date = datetime.strptime(str(date), "%Y%m%d").strftime("%m%d")
    print(mean_date)
    # Run tables and layers WW wCCR
    tables_and_layers(user=cfg.rmodel_username, year=year, report_date=str(date), mean_date=mean_date,
                      meanWorkspace=cfg.mean_workspace, model_run=model_wCCR, masking="N",
                      watershed_zones=cfg.ww_watershed_zones, band_zones=cfg.ww_watershed_zones, HUC6_zones=cfg.huc6_zones, region_zones=cfg.ww_region_zones,
                      case_field_wtrshd=cfg.case_field_watershed,case_field_band=cfg.case_field_band,
                      watermask=cfg.water_mask, glacierMask=cfg.glacier_mask,
                      snapRaster_geon83=cfg.ww_snap_raster_geon83, snapRaster_albn83=cfg.ww_snap_raster_albn83,
                      projGEO=projGEO, projALB=projALB, ProjOut_UTM=ProjOut_UTM,
                      run_type="Normal")

    # Run tables and layers WW woCCR
    tables_and_layers(user=cfg.rmodel_username, year=year, report_date=str(date), mean_date=mean_date,
                      meanWorkspace=cfg.mean_workspace, model_run=model_woCCR, masking="N",
                      watershed_zones=cfg.ww_watershed_zones, band_zones=cfg.ww_watershed_zones,
                      HUC6_zones=cfg.huc6_zones, region_zones=cfg.ww_region_zones,
                      case_field_wtrshd=cfg.case_field_watershed, case_field_band=cfg.case_field_band,
                      watermask=cfg.water_mask, glacierMask=cfg.glacier_mask,
                      snapRaster_geon83=cfg.ww_snap_raster_geon83, snapRaster_albn83=cfg.ww_snap_raster_albn83,
                      projGEO=projGEO, projALB=projALB, ProjOut_UTM=ProjOut_UTM,
                      run_type="Normal")


# Run tables and layers SNM wCCR and SNM woCCR


if __name__ == "__main__":
    config = Config()
    ww_tables_and_layers(20260501, config)