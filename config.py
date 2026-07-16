import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

class Config:
    """
    Configuration class for running the model

    Reads environment variables in from a .env file (searched for upwards from the current working directory). Stores
    all input/output paths, runtime parameters, and API keys/tokens. Any variable can be overridden by passing it as
    an argument, which takes priority over the value from the .env file.

    See .env.example for how each variable should be defined in the .env.
    """
    def __init__(self,
        ##### Results and Report Directories #####
        ww_results_workspace: str = None,
        snm_results_workspace: str = None,
        ww_reports_workspace: str = None,
        snm_reports_workspace: str = None,

        ##### fSCA Download Configs #####
        local_fsca_path: str = None,
        processed_fsca_path: str = None,
        dmfsca_path: str = None,
        fsca_tiles: list[str] = None,
        sin_modis_proj: str = None,
        fsca_snap_raster: str = None,
        fsca_extent: str = None,
        mean_layer_workspace: str = None,
        mean_layer_output: str = None,
        mean_layer_start_year: int = None,
        mean_layer_end_year: int = None,
         # PetaLibrary (ssh method)
        curc_fsca_path: str = None,
        curc_host: str = None,
        curc_identikey: str = None,
        # Snow Today (ftp method)
        snow_today_fsca_path: str = None,
        snow_today_host: str = None,
        snow_today_username: str = None,
        snow_today_password: str = None,

        ##### SNODAS Download Configs #####
        snodas_local_path: str = None,
        snodas_remote_path: str = None,
        snodas_host: str = None,

        ##### SnowTrax Download Configs #####
        snowtrax_path: str = None,
        snowtrax_filename: str = None,

        ##### MODIS Download Configs #####
        modis_bbox: list[float] = None,
        modis_path: str = None,

        ##### Snow Sensor Download Configs #####
        sensor_path: str = None,
        sensor_list: str = None,

        ##### R Model Configs #####
        rscript_exe_path: str = None,
        rmodel_script_path: str = None,
        rmodel_config_log_dir: str = None,
        rmodel_username: str = None,
        regress_path: str = None,
        simulation_date_path: str = None,
        modscag_path: str = None,
        rmodel_modscag_type: str = None,
        rmodel_modscag_file: str = None,
        rmodel_fveg_correction: str = None,
        rmodel_sens_per: float = None,
        rmodel_fsca_type: str = None,
        rmodel_is_fsca_flag: str = None,
        rmodel_is_fsca_mask: str = None,
        rmodel_is_gpkg: str = None,
        rmodel_is_his_day: str = None,
        rmodel_best_his_date: str = None,
        rmodel_snow_var: str = None,

        ##### Tables and Layers Config #####
        mean_workspace: str = None,
        ww_watershed_zones: str = None,
        huc6_zones: str = None,
        ww_region_zones: str = None,
        case_field_watershed: str = None,
        case_field_band: str = None,
        water_mask: str = None,
        glacier_mask: str = None,
        ww_snap_raster_geon83: str = None,
        ww_snap_raster_albn83: str = None,
        snm_snap_raster_albn83: str = None,
    ):
        # Load .env file into os.environment
        load_dotenv(find_dotenv(usecwd=True))

        #####  Results and Report Directories #####
        self.ww_results_workspace = ww_results_workspace or str(os.environ.get("WW_RESULTS_WORKSPACE"))
        self.snm_results_workspace = snm_results_workspace or str(os.environ.get("SNM_RESULTS_WORKSPACE"))
        self.ww_reports_workspace = ww_reports_workspace or str(os.environ.get("WW_REPORTS_WORKSPACE"))
        self.snm_reports_workspace = snm_reports_workspace or str(os.environ.get("SNM_REPORTS_WORKSPACE"))

        ##### fSCA Download Configs #####
        self.local_fsca_path = local_fsca_path or str(os.environ.get("LOCAL_FSCA_PATH"))
        self.processed_fsca_path = processed_fsca_path or str(os.environ.get("PROCESSED_FSCA_PATH"))
        self.dmfsca_path = dmfsca_path or str(os.environ.get("DMFSCA_PATH"))
        self.fsca_tiles = fsca_tiles or [str(x) for x in str(os.environ.get("FSCA_TILES")).split(",")]
        self.sin_modis_proj = sin_modis_proj or str(os.environ.get("SIN_MODIS_PROJ"))
        self.fsca_snap_raster = fsca_snap_raster or str(os.environ.get("FSCA_SNAP_RASTER"))
        self.fsca_extent = fsca_extent or str(os.environ.get("FSCA_EXTENT"))
        self.mean_layer_workspace = mean_layer_workspace or str(os.environ.get("MEAN_LAYER_WORKSPACE"))
        self.mean_layer_output = mean_layer_output or str(os.environ.get("MEAN_LAYER_OUTPUT"))
        self.mean_layer_start_year = mean_layer_start_year or int(os.environ.get("MEAN_LAYER_START_YEAR"))
        self.mean_layer_end_year = mean_layer_end_year or int(os.environ.get("MEAN_LAYER_END_YEAR"))
        # PetaLibrary (ssh method)
        self.curc_fsca_path = curc_fsca_path or str(os.environ.get("CURC_FSCA_PATH"))
        self.curc_host = curc_host or str(os.environ.get("CURC_HOST"))
        self.curc_identikey = curc_identikey or str(os.environ.get("CURC_IDENTIKEY"))
        # Snow Today (ftp method)
        self.snow_today_fsca_path = snow_today_fsca_path or str(os.environ.get("SNOW_TODAY_FSCA_PATH"))
        self.snow_today_host = snow_today_host or str(os.environ.get("SNOW_TODAY_HOST"))
        self.snow_today_username = snow_today_username or str(os.environ.get("SNOW_TODAY_USERNAME"))
        self.snow_today_password = snow_today_password or str(os.environ.get("SNOW_TODAY_PASSWORD"))

        ##### SNODAS Download Configs #####
        self.snodas_local_path = snodas_local_path or str(os.environ.get("SNODAS_LOCAL_PATH"))
        self.snodas_remote_path = snodas_remote_path or str(os.environ.get("SNODAS_REMOTE_PATH"))
        self.snodas_host = snodas_host or str(os.environ.get("SNODAS_HOST"))

        ##### SnowTrax Download Configs #####
        self.snowtrax_path = snowtrax_path or str(os.environ.get("SNOWTRAX_PATH"))
        self.snowtrax_filename = snowtrax_filename or str(os.environ.get("SNOWTRAX_FILENAME"))

        ##### MODIS Download Configs #####
        self.modis_bbox = modis_bbox or [float(x) for x in str(os.environ.get("MODIS_BBOX")).split(",")]
        self.modis_path = modis_path or str(os.environ.get("MODIS_PATH"))

        ##### Snow Sensor Download Configs #####
        self.sensor_path = sensor_path or str(os.environ.get("SENSOR_PATH"))
        self.sensor_list = sensor_list or str(os.environ.get("SENSOR_LIST"))

        ##### R Model Configs #####
        self.rscript_exe_path = rscript_exe_path or str(os.environ.get("RSCRIPT_EXE_PATH"))
        self.rmodel_script_path = rmodel_script_path or str(os.environ.get("RMODEL_SCRIPT_PATH"))
        self.rmodel_config_log_dir = rmodel_config_log_dir or str(os.environ.get("RMODEL_CONFIG_LOG_DIR"))
        self.rmodel_username = rmodel_username or str(os.environ.get("RMODEL_USERNAME"))
        self.regress_path = regress_path or str(os.environ.get("REGRESS_PATH"))
        self.simulation_date_path = simulation_date_path or str(os.environ.get("SIMULATION_DATE_PATH"))
        self.modscag_path = modscag_path or str(os.environ.get("MODSCAG_PATH"))
        self.rmodel_modscag_type = rmodel_modscag_type or str(os.environ.get("RMODEL_MODSCAG_TYPE", "NRT"))
        self.rmodel_modscag_file = rmodel_modscag_file or str(os.environ.get("RMODEL_MODSCAG_FILE", "snow_fraction_canadj"))
        self.rmodel_fveg_correction = rmodel_fveg_correction or str(os.environ.get("RMODEL_FVEG_CORRECTION", "F"))
        self.rmodel_sens_per = rmodel_sens_per or float(os.environ.get("RMODEL_SENS_PER", 0.3))
        self.rmodel_fsca_type = rmodel_fsca_type or str(os.environ.get("RMODEL_FSCA_TYPE", "Rittger"))
        self.rmodel_is_fsca_flag = rmodel_is_fsca_flag or str(os.environ.get("RMODEL_IS_FSCA_FLAG", "T"))
        self.rmodel_is_fsca_mask = rmodel_is_fsca_mask or str(os.environ.get("RMODEL_IS_FSCA_MASK", "T"))
        self.rmodel_is_gpkg = rmodel_is_gpkg or str(os.environ.get("RMODEL_IS_GPKG", "T"))
        self.rmodel_is_his_day = rmodel_is_his_day or str(os.environ.get("RMODEL_IS_HIS_DAY", "F"))
        self.rmodel_best_his_date = rmodel_best_his_date or str(os.environ.get("RMODEL_BEST_HIS_DATE", "1989-06-12"))
        self.rmodel_snow_var = rmodel_snow_var or str(os.environ.get("RMODEL_SNOW_VAR", "rcn"))

        ##### Tables and Layers Config #####
        self.mean_workspace = mean_workspace or str(os.environ.get("MEAN_WORKSPACE"))
        self.ww_watershed_zones = ww_watershed_zones or str(os.environ.get("WW_WATERSHED_ZONES"))
        self.huc6_zones = huc6_zones or str(os.environ.get("HUC6_ZONES"))
        self.ww_region_zones = ww_region_zones or str(os.environ.get("WW_REGION_ZONES"))
        self.case_field_watershed = case_field_watershed or str(os.environ.get("CASE_FIELD_WATERSHED"))
        self.case_field_band = case_field_band or str(os.environ.get("CASE_FIELD_BAND"))
        self.water_mask = water_mask or str(os.environ.get("WATER_MASK"))
        self.glacier_mask = glacier_mask or str(os.environ.get("GLACIER_MASK"))
        self.ww_snap_raster_geon83 = ww_snap_raster_geon83 or str(os.environ.get("WW_SNAP_RASTER_GEON83"))
        self.ww_snap_raster_albn83 = ww_snap_raster_albn83 or str(os.environ.get("WW_SNAP_RASTER_ALBN83"))
        self.snm_snap_raster_albn83 = snm_snap_raster_albn83 or str(os.environ.get("SNM_SNAP_RASTER_ALBN83"))

if __name__ == "__main__":
    config = Config()