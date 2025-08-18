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
print('modules imported')

# establish paths
user = "Leanne"
year = "2025"

# set dates
report_date = "20250525"
mean_date = "0528"
prev_report_date = "20250517"

# set model run info
model_run = "RT_CanAdj_rcn_noSW_woCCR_nofscamsk"
prev_model_run = "ASO_FixLayers_fSCA_RT_CanAdj_rcn_noSW_woCCR_UseThis"
masking = "N"
bias = "N"

tables_and_layers(user=user, year=year, report_date=report_date, mean_date=mean_date, prev_report_date=prev_report_date, model_run=model_run,
                  prev_model_run=prev_model_run, masking=masking, bias=bias)

# set paths

# download fSCA

# run DMFSCA (look into r version of this code)

# download sensors (look into r version of this code and see if we can have a python version)

# if surveys = True:
    # download surveys

# run SNODAS code

# run R model

# run sensors code

# run tables and layers

# run vetting code