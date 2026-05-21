# THIS IS THE CODE THAT'S THE OVERALL FUNCTION FOR MODEL POST PROCESSING
from time import sleep

# import modules
# import modules
import arcpy
import arcpy
from arcpy import env
from arcpy.ra import ZonalStatisticsAsTable
from arcpy.sa import *
import pandas as pd
import geopandas as gpd
import os
# from tables_layers_testing_code import *
from SWE_Fusion_functions import *
# from ASO_Bias_Correction_Functions import *
from ASO_Bias_Correction_Functions import *
from Vetting_functions import *
print('modules imported')
start = time.time()
print(f"\n START TIME: {start}")

# TURN OFF PROCESSING SO IT DOESN'T CRASH FOR NO REASON WHATSOEVER
arcpy.env.parallelProcessingFactor = "0"

######################################
# RUNDATE VARIABLES
######################################
## date info
user = "Olaf"
year = 2026
rundate = "20260320"
survey_date = "20260301"
pillow_date = "20Mar2026"
mean_date = "0320"
prev_rundate = "20260308"

# flags
difference = "N" # should be Y if you want to compare against a previous model run
biasCorrection = "Y"
surveys_use = "N"

# model run information
domainList = ["NOCN", "PNW", "SNM", "SOCN", "INMT"]
model_wCCR = "RT_CanAdj_rcn_wCCR_nofscamskSens"
model_woCCR = "RT_CanAdj_rcn_woCCR_nofscamskSens_dailyTest"
modelRuns = [model_woCCR, model_wCCR]
model_labels = ["woCCR", "wCCR"]
prev_model_run_WW = "RT_CanAdj_rcn_woCCR_nofscamskSens_UseAvg"
prev_model_run_SNM = "RT_CanAdj_rcn_woCCR_nofscamskSens_UseAvg"
prev_model_run_WW_tables = "ASO_BiasCorrect_RT_CanAdj_rcn_woCCR_nofscamskSens_UseThis"
prev_model_run_SNM_tables = "ASO_BiasCorrect_RT_CanAdj_rcn_woCCR_nofscamskSens_UseThis"
aso_symbol = "§"
SNM_prev_tables_workspace = rf"M:/SWE/Sierras/paperwork/0_UCSB_DWR_Project/{year}_RT_Reports/{prev_rundate}_RT_report/{prev_model_run_SNM_tables}/Tables/"
WW_prev_tables_workspace = rf"M:/SWE/WestWide/documents/{year}_RT_Reports/{prev_rundate}_RT_report/{prev_model_run_WW_tables}/Tables/"
######################################
# VETTING VARIABLES
######################################
# bias correction
aso_snotel_data = r"W:/Spatial_SWE/ASO/2026/data/ASO_SNOTEL_DifferenceStats.csv"
currentYear = True
error_metric = "Avg.Abs.Perc.Error"
# current_year = datetime.now().year
# year = 2026
methods = ["RECENT", "GRADE", "SENSOR_PATTERN", "GRADES_SPECF"]
grade = "negative"
grade_range = False
grade_amount = -15
sensorTrend = "Mixed"
SNOTEL = "Decreasing"
output_csv = "Y"
csv_outFile = fr"W:/Spatial_SWE/ASO/{year}/data/FracError_data.csv"
asoCatalog = fr"W:/Spatial_SWE/ASO/{year}/data/ASO_SNOTEL_DifferenceStats.csv"
basin_List = r"W:/Spatial_SWE/ASO/ASO_Metadata/ASO_in_Basin.txt"
fracErrorWorkspace = f"W:/Spatial_SWE/ASO/{year}/data/"
domain_textFile = r"M:\SWE\WestWide\Spatial_SWE\ASO\ASO_Metadata\State_Basin.txt"
SNM_shapefile_workspace = "M:/SWE/WestWide/data/hydro/SNM/Basin_Shapefiles/"
WW_shapefile_workspace = "M:/SWE/WestWide/data/hydro/WW/ASO_Basin_Shapefiles/"

# vetting
snowTrax_csv = r"J:\Spatial_SWE\SNM_regression\RT_report_data\wsfr_snow.csv"
elevation_tif =r"M:\SWE\WestWide\data\topo\ww_DEM_albn83_feet_banded_100.tif"
aspect_path = r"M:\SWE\WestWide\data\topo\ww_ASPECT_albn83.tif"
reference_col = "SNOW17_SWE_AF"
elev_bins = np.arange(1500, 15000, 100, dtype=float)
swe_col_surv = 'SWE_m'
id_col_surv = 'Station_Na'
# error_statistic = ""

######################################
# WORKSPACES
######################################
meanWorkspace = fr"M:/SWE/WestWide/Spatial_SWE/WW_regression/mean_2000_2025_WY26_glacMask/"
clipbox_WS = "M:/SWE/WestWide/data/boundaries/Domains/DomainShapefiles/"

## results workspaces
SNM_results_workspace = "J:/Spatial_SWE/SNM_regression/RT_report_data/"
WW_results_workspace = "W:/Spatial_SWE/WW_regression/RT_report_data/"

## reporting workspaces
WW_reports_workspace = f"W:/documents/{year}_RT_Reports/"
SNM_reports_workspace = f"J:/paperwork/0_UCSB_DWR_Project/{year}_RT_Reports/"
model_workspace = fr"H:/WestUS_Data/Regress_SWE/"

######################################
# GEOSPATIAL PARAMETERS
######################################
# projections
projGEO = arcpy.SpatialReference(4269)
projALB = arcpy.SpatialReference(102039)
ProjOut_UTM = arcpy.SpatialReference(26911)

# snap rasters
snapRaster_geon83 = "M:/SWE/WestWide/data/boundaries/SnapRaster_geon83.tif"
snapRaster_albn83 = "M:/SWE/WestWide/data/boundaries/SnapRaster_albn83.tif"
SNM_snapRaster_albn83 = "M:/SWE/WestWide/data/boundaries/SNM_SnapRaster_albn83.tif"

# basins tifs and shapefiles
case_field_wtrshd = "SrtName"
case_field_band = "SrtNmeBand"
statelines_file = r"W:\data\basemap\USA_statelines.shp"
## sierras
SNM_band_zones = "M:/SWE/WestWide/data/hydro/SNM/dwr_band_basins_geoSort_albn83_delin.tif"
SNM_watershed_zones = "M:/SWE/WestWide/data/hydro/SNM/dwr_basins_geoSort_albn83.tif"
SNM_watershed_shapefile = "M:/SWE/WestWide/data/hydro/SNM/dwr_basins_albn83_GT5000.shp"
SNM_band_shapefile = "M:/SWE/WestWide/data/hydro/SNM/dwr_band_basins_albn83_delin.shp"
SNM_regions = "M:/SWE/WestWide/data/hydro/SNM/dwr_regions_albn83.tif"
SNM_domain_msk = "M:/SWE/WestWide/data/hydro/SNM/dwr_mask_null_albn83.tif"
SNM_clipbox = "M:/SWE/WestWide/data/boundaries/Domains/DomainShapefiles/WW_SNM_Clipbox_albn83.shp"
## west wide
HUC6_zones = "M:/SWE/WestWide/data/hydro/WW_HUC6_albn83_ras_msked.tif"
region_zones = "M:/SWE/WestWide/data/hydro/WW_Regions_albn83_v2.tif"
# WW_band_zones = r"M:\SWE\WestWide\data\hydro\outdated\WW_BasinBanded_noSNM_notahoe_albn83_sel.tif"
# WW_watershed_zones = r"M:\SWE\WestWide\data\hydro\outdated\WW_BasinBanded_noSNM_notahoe_albn83_sel.tif"
# WW_watershed_shapefile = r"M:\SWE\WestWide\data\hydro\outdated\WW_Basins_noSNM_notahoe_sel_albn83.shp"
# WW_band_shapefile = r"M:\SWE\WestWide\data\hydro\outdated\WW_BasinsBanded_noSNM_notahoe_sel_albn83.shp"
WW_band_zones = r"W:\data\hydro\20260113_BasinUpdatesNoSNM\WW_BasinBanded_noSNM_notahoe_albn83_sel_v2_updated.tif"
WW_watershed_zones = r"W:\data\hydro\20260113_BasinUpdatesNoSNM\WW_BasinBanded_noSNM_notahoe_albn83_sel_v2_updated.tif"
WW_watershed_shapefile = r"W:\data\hydro\20260113_BasinUpdatesNoSNM\WW_Basins_noSNM_notahoe_albn83_sel_v2_dis.shp"
WW_band_shapefile = r"W:\data\hydro\20260113_BasinUpdatesNoSNM\WW_BasinsBanded_noSNM_notahoe_albn83_sel_v2_updated_20260208.shp"


# masks
watermask = "M:/SWE/WestWide//data/mask/watermask_outdated/outdated/WW/Albers/WW_watermaks_10_albn83_clp_final.tif"
glacierMask = "M:/SWE/WestWide/data/mask/glacierMask/glims_glacierMask_null_GT10_final.tif"

######################################
#SENSOR AND SURVEY DATA
######################################
# surveys paths
survey_workspace = "W:/data/Snow_Instruments/surveys_courses/"
cdec_shapefile = survey_workspace + "20240207_Courses.shp"
basin_list = ["SHASTA RIVER", "SCOTT RIVER", "TRINITY RIVER", "SACRAMENTO RIVER", "MC CLOUD RIVER", "STONY CREEK",
             "FEATHER RIVER", "YUBA RIVER", "AMERICAN RIVER", "MOKELUMNE RIVER", "STANISLAUS RIVER", "TUOLUMNE RIVER",
             "MERCED RIVER", "SAN JOAQUIN RIVER", "KINGS RIVER", "KAWEAH RIVER", "TULE RIVER", "KERN RIVER"]
WW_url_file = survey_workspace + r"stateSurveyURLS.txt"
NRCS_shp = survey_workspace + "20240114_nrcs_course.shp"
WW_state_list = ["AZ", "CO", "ID", "MT", "NV", "NM", "OR", "SD", "UT", "WA", "WY"]
WW_surveys = WW_results_workspace + f"{rundate}_results/{rundate}_surveys_albn83.shp"

# sensors
SNM_sensors = rf"M:/SWE/WestWide/Spatial_SWE/WW_regression/RT_report_data/{rundate}_results/{rundate}_sensors_SNM.shp"

######################################
## SNODAS Variables
######################################
SNODAS_Type = "masked"
WW_NOHRSC_workspace = r"M:/SWE/WestWide/Spatial_SWE/NOHRSC/"
SNM_NOHRSC_workspace = r"M:/SWE/Sierras/Spatial_SWE/NOHRSC/"
Cellsize = "500"

############################################################################################################
# Processing Starts Now
############################################################################################################

# make results and reports directory
os.makedirs(WW_results_workspace + f"/{rundate}_results", exist_ok=True)
print(WW_results_workspace + f"/{rundate}_results")
# os.makedirs(SNM_results_workspace + f"/{rundate}_results", exist_ok=True)
# print("\nResults directories made")

# os.makedirs(WW_reports_workspace + f"/{rundate}_RT_report", exist_ok=True)
# os.makedirs(SNM_reports_workspace + f"/{rundate}_RT_report", exist_ok=True)
# os.makedirs(SNM_reports_workspace + f"/{rundate}_RT_report/SNODAS/", exist_ok=True)
# os.makedirs(SNM_reports_workspace + f"/{rundate}_RT_report/MODIS/", exist_ok=True)
# os.makedirs(WW_reports_workspace + f"/{rundate}_RT_report/SNODAS/", exist_ok=True)
print("\nReports directories made")

# download surveys if requested
if surveys_use == "Y":
    print("\nGetting SNM Surveys")
    download_cdec_snow_surveys(report_date=rundate, survey_date=survey_date, survey_workspace=survey_workspace,
                               SNM_results_workspace=SNM_results_workspace,
                               cdec_shapefile=cdec_shapefile, basin_list=basin_list)

    print("\nGetting WW Surveys")
    download_snow_surveys(report_date=rundate, survey_date=survey_date, survey_workspace=survey_workspace, results_workspace=WW_results_workspace,
                          WW_url_file=WW_url_file, NRCS_shp=NRCS_shp, WW_state_list=WW_state_list)

# clear memory
sleep(30)
clear_arcpy_locks()

# get geopackage converted to shapefile
print("\nProcessing GeoPackage")
geopackage_to_shapefile(report_date=rundate, pillow_date=pillow_date, model_run=model_woCCR,
                        user=user, domainList=domainList, model_workspace=model_workspace,
                        results_workspace=WW_results_workspace + f"/{rundate}_results/")
# clear memory
sleep(30)
clear_arcpy_locks()

# organize and reprocess the sensors for West Wide
print('\nProcessing and sorting the sensors for West Wide ... ')
merge_sort_sensors_surveys(report_date=rundate, results_workspace=WW_results_workspace + f"/{rundate}_results/", surveys=surveys_use, difference=difference,
                           watershed_shapefile=WW_watershed_shapefile, case_field_wtrshd=case_field_wtrshd,
                           case_field_band=case_field_band, band_shapefile=WW_band_shapefile, projOut=projALB, merge="Y",
                           domainList=domainList, prev_report_date=prev_rundate, prev_results_workspace=WW_results_workspace + f"/{prev_rundate}_results/")

# clear memory
sleep(30)
clear_arcpy_locks()

# run tables and layers code for the woCCR model run for West Wide
print(f'\nRunning Tables and Layers Code for all domains for {model_woCCR}')
tables_and_layers_NYT(user=user, year=year, report_date=rundate, mean_date = mean_date, meanWorkspace = meanWorkspace, model_run=model_woCCR,
                  masking="N", watershed_zones=WW_watershed_zones, band_zones=WW_band_zones, HUC6_zones=HUC6_zones,
                  region_zones=region_zones, case_field_wtrshd=case_field_wtrshd,case_field_band=case_field_band,
                  watermask=watermask, glacierMask=glacierMask, snapRaster_geon83=snapRaster_geon83, snapRaster_albn83=snapRaster_albn83,
                  projGEO=projGEO, projALB=projALB, ProjOut_UTM=ProjOut_UTM, run_type="Normal")

