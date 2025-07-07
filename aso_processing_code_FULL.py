# import modules
import sys
from ASO_Processing import *
from SWE_FusionProcessing import *
import arcpy
import rasterio
import matplotlib.pyplot as plt
from arcpy.sa import *
import zipfile
import os
import shutil
import ast
from datetime import datetime
from shapefile import NODATA

print("modules imported")

## model run variables
rundate = "20250413"
modelRun = "fSCA_RT_CanAdj_rcn_noSW_woCCR"

# set parameters for zip extraction
zip_file_path = r"M:\SWE\WestWide\Spatial_SWE\ASO\2025\ASO_BoulderCreek_2025Apr09-10_AllData_and_Reports.zip"
search_tag = "swe_50m.tif"
data_folder = r"M:/SWE/WestWide/Spatial_SWE/ASO/2025/data_testing/"
basin_textFile = r"M:\SWE\WestWide\Spatial_SWE\ASO\SNM_basins.txt"
basinList = ["BoulderCreek"]

# open basin file for list
with open(basin_textFile, "r") as f:
    content = f.read()
    region_list = ast.literal_eval(content)

# # get SWE file from zip folder
extract_zip(zip_path=zip_file_path, ext=search_tag, output_folder=data_folder)
print("file moved")

# get basin and basin info from aso folder
asoSWE = os.listdir(data_folder)
for file in asoSWE:
    if file.endswith(".tif"):
        basinName = file.split("_")[1] # gets the value between the first and second "_"
        if basinName in basinList:
            dates = file.split("_")[2]
            startDate = dates.split("-")[0]
            date_obj = datetime.strptime(startDate, "%Y%b%d")
            format_date = date_obj.strftime("%Y%m%d")

            # getting input coordinate system
            desc = arcpy.Describe(data_folder + file)
            spatial_ref = desc.spatialReference
            projIn = arcpy.SpatialReference(spatial_ref.factoryCode)

            # checking for domains
            if basinName in region_list:
                domain = "SNM"
                fullDomain = "Sierras"
                compareWS = "M:/SWE/WestWide/Spatial_SWE/ASO/2025/data_testing/SNM_comparison_testing/"
                snapRaster = r"M:\SWE\WestWide\data\boundaries\SNM_SnapRaster_albn83.tif"
                zonalRaster = r"M:\SWE\WestWide\data\hydro\SNM\dwr_band_basins_geoSort_albn83_delin.tif"
            else:
                domain = "WW"
                fullDomain = "WestWide"
                compareWS = "M:/SWE/WestWide/Spatial_SWE/ASO/2025/data_testing/WW_comparison_testing/"
                snapRaster = r"M:\SWE\WestWide\data\boundaries\SnapRaster_albn83.tif"
                zonalRaster = r"M:\SWE\WestWide\data\hydro\WW_BasinBanded_noSNM_notahoe_albn83_sel.tif"

            print(domain)
            modelRunWorkspace = rf"M:/SWE/{fullDomain}/Spatial_SWE/{domain}_regression/RT_report_data/{rundate}_results/{modelRun}/"

            # process ASO comparison
            process_aso_comparison(file, rundate, modelRun, data_folder, modelRunWorkspace, compareWS, snapRaster,
                                   projIn, zonalRaster)

            # download snotel function
            print(startDate)
            # load in shp
            # isolate shapefile that are just within the ASO area
            # determine the first value and the last value
            # determine the % grade
            # add that to the csv.

## Determine the percent difference in the two points
## fill csv with that information and create the csv if it doesn't already exists


## __FUNCTION: WW fractional error
# calculate fractional error layer.




