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

# surveys -- establish paths
survey_workspace = "W:/data/Snow_Instruments/surveys_courses/"
SNM_results_workspace = "J:/Spatial_SWE/SNM_regression/RT_report_data/"
cdec_shapefile = survey_workspace + "20240207_Courses.shp"
basin_list = ["SHASTA RIVER", "SCOTT RIVER", "TRINITY RIVER", "SACRAMENTO RIVER", "MC CLOUD RIVER", "STONY CREEK",
             "FEATHER RIVER", "YUBA RIVER", "AMERICAN RIVER", "MOKELUMNE RIVER", "STANISLAUS RIVER", "TUOLUMNE RIVER",
             "MERCED RIVER", "SAN JOAQUIN RIVER", "KINGS RIVER", "KAWEAH RIVER", "TULE RIVER", "KERN RIVER"]
results_workspace = "M:/SWE/WestWide/Spatial_SWE/WW_regression/RT_report_data/"
WW_url_file = survey_workspace + r"stateSurveyURLS.txt"
NRCS_shp = survey_workspace + "20240114_nrcs_course.shp"
WW_state_list = ["AZ", "CO", "ID", "MT", "NV", "NM", "OR", "SD", "UT", "WA", "WY"]

# set paths

#process fSCA
print("\nProcessing fSCA data...")
fsca_processing_tif(start_date, end_date, netCDF_WS, tile_list, output_fscaWS, proj_in, snap_raster, extent, proj_out)

# run DMFSCA (look into r version of this code)

# download sensors (look into r version of this code and see if we can have a python version)

if surveys == "Y":
    print("\nGetting CDEC Surveys")
    download_cdec_snow_surveys(report_date=report_date, survey_workspace=survey_workspace,
                               SNM_results_workspace=SNM_results_workspace,
                               cdec_shapefile=cdec_shapefile, basin_list=basin_list)

    print("\nGetting WW Surveys")
    download_snow_surveys(report_date=report_date, surveyWorkspace=survey_workspace, results_workspace=results_workspace,
                          WW_url_file=WW_url_file, NRCS_shp=NRCS_shp, WW_state_list=WW_state_list)

# run SNODAS code --> need to make a function
## run SNODAS for WW
## clip snodas extent

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