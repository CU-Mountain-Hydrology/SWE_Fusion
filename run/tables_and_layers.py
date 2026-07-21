# run/tables_and_layer.py

from datetime import datetime
import arcpy

from config import Config
from utils import get_water_year, get_previous_model_run
from SWE_Fusion_functions import tables_and_layers, tables_and_layers_SNM

def ww_tables_and_layers(date: int, model_wCCR: str, model_woCCR: str, cfg: Config):
    """
    TODO: docs
    TODO: print statements
    TODO: ww_watershed_zones is the same as ww_band_zones, is it always?

    """
    # Projections
    projGEO = arcpy.SpatialReference(4269) # NAD83
    projALB = arcpy.SpatialReference(102039) # USA Contiguous Albers Equal Area Conic (USGS version)
    ProjOut_UTM = arcpy.SpatialReference(26911) # NAD 1983 / UTM Zone 11N (Nevada)

    # Format date
    year = datetime.strptime(str(date), "%Y%m%d").year
    mean_date = datetime.strptime(str(date), "%Y%m%d").strftime("%m%d")

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


def snm_tables_and_layers(date: int, model_wCCR: str, model_woCCR: str, cfg: Config):
    """
    TODO: docs
    TODO: print statements
    TODO: this should really all be one function, merge with above

    """
    # Format date
    year = datetime.strptime(str(date), "%Y%m%d").year
    mean_date = datetime.strptime(str(date), "%Y%m%d").strftime("%m%d")

    # Get previous report date
    prev_rundate, prev_model_run_SNM = get_previous_model_run(date=date, domain="SNM", cfg=cfg)

    # If the previous report date was from this water year, then difference="Y"
    if get_water_year(prev_rundate) == get_water_year(date):
        difference = "Y"
    else:
        difference = "N"

    # Run tables and layers SNM wCCR
    tables_and_layers_SNM(
        year=year, rundate=str(date), mean_date=mean_date, WW_model_run=model_wCCR, SNM_results_workspace=cfg.snm_results_workspace,
        watershed_zones=cfg.snm_watershed_zones, band_zones=cfg.snm_band_zones, region_zones=cfg.snm_region_zones,
        case_field_wtrshd=cfg.case_field_watershed, case_field_band=cfg.case_field_band,
        watermask=cfg.water_mask, glacier_mask=cfg.glacier_mask, domain_mask=cfg.snm_domain_mask, run_type="Normal",
        snap_raster=cfg.snm_snap_raster_albn83, WW_results_workspace=cfg.ww_results_workspace,
        Difference=difference, prev_report_date=prev_rundate, prev_model_run=prev_model_run_SNM)

    # Run tables and layers SNM woCCR
    tables_and_layers_SNM(
        year=year, rundate=str(date), mean_date=mean_date, WW_model_run=model_woCCR,
        SNM_results_workspace=cfg.snm_results_workspace,
        watershed_zones=cfg.snm_watershed_zones, band_zones=cfg.snm_band_zones, region_zones=cfg.snm_region_zones,
        case_field_wtrshd=cfg.case_field_watershed, case_field_band=cfg.case_field_band,
        watermask=cfg.water_mask, glacier_mask=cfg.glacier_mask, domain_mask=cfg.snm_domain_mask, run_type="Normal",
        snap_raster=cfg.snm_snap_raster_albn83, WW_results_workspace=cfg.ww_results_workspace,
        Difference=difference, prev_report_date=prev_rundate, prev_model_run=prev_model_run_SNM)

if __name__ == "__main__":
    config = Config()
    ww_tables_and_layers(20260501, config)