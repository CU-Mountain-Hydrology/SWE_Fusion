# import modules
import sys
from ASO_Processing import *
from SWE_FusionProcessing import *
import arcpy
import csv
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
basin_textFile = r"M:\SWE\WestWide\Spatial_SWE\ASO\ASO_Metadata\State_Basin.txt"
basinList = ["BoulderCreek"]
snotel_shp = r"W:\Spatial_SWE\ASO\ASO_Metadata\WW_CDEC_SNOTEL_geon83.shp"
cdec_shp = ""

# open basin file for list
basin_state_map = {}
with open(basin_textFile, 'r') as f:
    reader = csv.reader(f, delimiter='|')
    for row in reader:
        basin = row[0].strip('"')
        state = row[1].strip('"')
        basin_state_map[basin] = state

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

            # get state associated with basin
            basin_state = basin_state_map.get(basinName, None)
            # checking for domains
            if basin_state == "CA":
                domain = "SNM"
                fullDomain = "Sierras"
                compareWS = "M:/SWE/WestWide/Spatial_SWE/ASO/2025/data_testing/SNM_comparison_testing/"
                snapRaster = r"M:\SWE\WestWide\data\boundaries\SNM_SnapRaster_albn83.tif"
                zonalRaster = r"M:\SWE\WestWide\data\hydro\SNM\dwr_band_basins_geoSort_albn83_delin.tif"
                snotelWorkspace = "W:/Spatial_SWE/ASO/2025/data_testing/snotel_comparisons/cdec_metaData.csv"
                pillow_shp = cdec_shp
            else:
                domain = "WW"
                fullDomain = "WestWide"
                compareWS = "M:/SWE/WestWide/Spatial_SWE/ASO/2025/data_testing/WW_comparison_testing/"
                snapRaster = r"M:\SWE\WestWide\data\boundaries\SnapRaster_albn83.tif"
                zonalRaster = r"M:\SWE\WestWide\data\hydro\WW_BasinBanded_noSNM_notahoe_albn83_sel.tif"
                snotelWS = data_folder + "snotel_comparisons/"
                pillowMeta = r"W:\Spatial_SWE\ASO\ASO_Metadata\snotel_metaData.csv"
                pillow_shp = snotel_shp

            print(f"Basin: {basinName}, State: {basin_state}, Domain: {domain}")
            modelRunWorkspace = rf"M:/SWE/{fullDomain}/Spatial_SWE/{domain}_regression/RT_report_data/{rundate}_results/{modelRun}/"

            # process ASO comparison
            process_aso_comparison(file, rundate, modelRun, data_folder, modelRunWorkspace, compareWS, snapRaster,
                                   projIn, zonalRaster)

            # find the snotels that are within a raster file
            gdf_final, site_id_list = get_points_within_raster(pillow_shp, data_folder + file, id_column="site_id")

            # download snotel function
            print(startDate)
            if domain =="WW":
                #check to see if a csv exists, maybe check on the csv and the state list just be the state of the ASO flight
                end_snotel = datetime.strptime(startDate, "%Y%b%d")
                start_snotel = end_snotel - timedelta(days=7)
                start = start_snotel.strftime("%Y-%m-%d")
                end = end_snotel.strftime("%Y-%m-%d")
                snotel_df = pd.read_csv(pillowMeta)
                if len(site_id_list) > 0:
                    # Use the site IDs from points within raster
                    filtered_snotel = snotel_df[snotel_df["site_id"].isin(site_id_list)]
                    id_list = filtered_snotel["site_id"].tolist()
                    state_list = filtered_snotel["state_id"].tolist()
                    output_filename = f"merged_snotel_{basinName}_{end}.csv"

                    # Check if file already exists
                    if os.path.exists(snotelWS + output_filename):
                        print("sensor file already downloaded")

                    else:
                        print(f"Downloading SNOTEL data for {len(id_list)} sites within raster extent")
                        # Download snotel values
                        merged_snotel_df = download_and_merge_snotel_data(
                            id_list=id_list,
                            state_list=state_list,
                            start_date=start,
                            end_date=end,
                            output_dir=snotelWS,
                            output_filename=output_filename
                        )
                else:
                    print("No SNOTEL sites found within raster extent")



## __FUNCTION: WW fractional error
# calculate fractional error layer.




