# THIS IS THE CODE THAT'S THE OVERALL FUNCTION FOR MODEL POST PROCESSING

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
from tables_layers_testing_code import *
from SWE_Fusion_functions import *
print('modules imported')

# tables and layers -- establish paths
user = "Leanne"
year = "2025"
report_date = "20250525"
mean_date = "0528"
prev_report_date = "20250517"
model_run = "RT_CanAdj_rcn_noSW_woCCR_nofscamsk"
prev_model_run = "ASO_FixLayers_fSCA_RT_CanAdj_rcn_noSW_woCCR_UseThis"
masking = "N"
bias = "N"
surveys = "N"
snapRaster_geon83 = "M:/SWE/WestWide/data/boundaries/SnapRaster_geon83.tif"
snapRaster_albn83 = "M:/SWE/WestWide/data/boundaries/SnapRaster_albn83.tif"
projGEO = arcpy.SpatialReference(4269)
projALB = arcpy.SpatialReference(102039)
ProjOut_UTM = arcpy.SpatialReference(26911)
watermask = "M:/SWE/WestWide//data/mask/watermask_outdated/outdated/WW/Albers/WW_watermaks_10_albn83_clp_final.tif"
glacierMask = "M:/SWE/WestWide/data/mask/glacierMask/glims_glacierMask_null_GT10_final.tif"
band_zones = "M:/SWE/WestWide/data/hydro/WW_BasinBanded_noSNM_notahoe_albn83_sel.tif"
watershed_zones = "M:/SWE/WestWide/data/hydro/WW_BasinBanded_noSNM_notahoe_albn83_sel.tif"

# surveys -- establish paths
survey_workspace = "W:/data/Snow_Instruments/surveys_courses/"
SNM_results_workspace = "J:/Spatial_SWE/SNM_regression/RT_report_data/"
cdec_shapefile = survey_workspace + "20240207_Courses.shp"
basin_list = ["SHASTA RIVER", "SCOTT RIVER", "TRINITY RIVER", "SACRAMENTO RIVER", "MC CLOUD RIVER", "STONY CREEK",
             "FEATHER RIVER", "YUBA RIVER", "AMERICAN RIVER", "MOKELUMNE RIVER", "STANISLAUS RIVER", "TUOLUMNE RIVER",
             "MERCED RIVER", "SAN JOAQUIN RIVER", "KINGS RIVER", "KAWEAH RIVER", "TULE RIVER", "KERN RIVER"]
WW_results_workspace = "M:/SWE/WestWide/Spatial_SWE/WW_regression/RT_report_data/"
WW_url_file = survey_workspace + r"stateSurveyURLS.txt"
NRCS_shp = survey_workspace + "20240114_nrcs_course.shp"
WW_state_list = ["AZ", "CO", "ID", "MT", "NV", "NM", "OR", "SD", "UT", "WA", "WY"]

# make results and reports directory
os.makedirs(WW_results_workspace + f"{report_date}_results_ET", exist_ok=True)
os.makedirs(SNM_results_workspace + f"{report_date}_results_ET", exist_ok=True)
print("\nResults directories made")

#process fSCA
print("\nProcessing fSCA data...")
# fsca_processing_tif(start_date, end_date, netCDF_WS, tile_list, output_fscaWS, proj_in, snap_raster, extent, proj_out)

# run DMFSCA (look into r version of this code)

# download sensors (look into r version of this code and see if we can have a python version)

if surveys == "Y":
    print("\nGetting CDEC Surveys")
    download_cdec_snow_surveys(report_date=report_date, survey_workspace=survey_workspace,
                               SNM_results_workspace=SNM_results_workspace,
                               cdec_shapefile=cdec_shapefile, basin_list=basin_list)

    print("\nGetting WW Surveys")
    download_snow_surveys(report_date=report_date, surveyWorkspace=survey_workspace, results_workspace=WW_results_workspace,
                          WW_url_file=WW_url_file, NRCS_shp=NRCS_shp, WW_state_list=WW_state_list)

# run SNODAS code --> need to make a function
report_date = "20250525"

## Use this run name when creating SNODAS for fSCA, nt for final model output
RunName = "RT_CanAdj_rcn_noSW_woCCR_nofscamsk"

## Is SNODAS masked or unmasked? "masked" or "unmasked"
SNODAS_Type = "masked"
WW_workspaceBase = r"M:/SWE/WestWide/Spatial_SWE/WW_regression/"
## Set workspaces
WW_NOHRSC_workspace = r"M:/SWE/WestWide/Spatial_SWE/NOHRSC/"
SNM_NOHRSC_workspace = r"M:/SWE/Sierras/Spatial_SWE/NOHRSC/"
WW_results_workspace = WW_workspaceBase + "RT_report_data/"
projin = arcpy.SpatialReference(4269) #GCS NAD
projout = arcpy.SpatialReference(102039) #Albers
Cellsize = "500"
unzip_SNODAS = "Y"


snapRaster = "M:/SWE/WestWide/data/boundaries/SnapRaster_albn83.tif"
watermask = "M:/SWE/WestWide//data/mask/watermask_outdated/outdated/WW/Albers/WW_watermaks_10_albn83_clp_final.tif"
glacierMask = "M:/SWE/WestWide/data/mask/glacierMask/glims_glacierMask_null_GT10_final.tif"
band_zones = "M:/SWE/WestWide/data/hydro/WW_BasinBanded_noSNM_notahoe_albn83_sel.tif"
watershed_zones = "M:/SWE/WestWide/data/hydro/WW_BasinBanded_noSNM_notahoe_albn83_sel.tif"
## run SNODAS for WW
SNODAS_Processing(report_date=report_date, RunName=RunName, NOHRSC_workspace=WW_NOHRSC_workspace, results_workspace=WW_results_workspace,
                     projin=projin, projout=projout, Cellsize=Cellsize, snapRaster=snapRaster, watermask=watermask, glacierMask=glacierMask,
                     band_zones=band_zones, watershed_zones=watershed_zones, unzip_SNODAS="Y")
# Run SNODAS for SNM
SNODAS_Processing(report_date=report_date, RunName=RunName, NOHRSC_workspace=SNM_NOHRSC_workspace, results_workspace=SNM_results_workspace,
                     projin=projin, projout=projout, Cellsize=Cellsize, snapRaster=snapRaster, watermask=watermask, glacierMask=glacierMask,
                     band_zones=band_zones, watershed_zones=watershed_zones, unzip_SNODAS="N")

## Maybe this section should be looped through CCR and woCCR
# run R model

# run sensors code

# run tables and layers
print('\nRunning Tables and Layers Code for all domains')
tables_and_layers(user=user, year=year, report_date=report_date, mean_date=mean_date, prev_report_date=prev_report_date, model_run=model_run,
                  prev_model_run=prev_model_run, masking=masking, bias=bias)

print('\nRunning Tables and Layers Code for Sierra')

# run vetting code

# ASO Bias correction
# if biasCorrection = TK
    # list of methods
    # loops through methods

    # runs vetting code again