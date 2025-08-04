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
rundate = "20250526"
modelRun = "fSCA_RT_CanAdj_rcn_noSW_woCCR"

# set parameters for zip extraction
toProcessFolder = r"M:/SWE/WestWide/Spatial_SWE/ASO/2025/toProcess/"
# zip_file_path = r"W:\Spatial_SWE\ASO\2025\ASO_BlueRiver_2025May24_AllData_and_Reports.zip"
search_tag = "swe_50m.tif"
data_folder = r"M:/SWE/WestWide/Spatial_SWE/ASO/2025/data_testing/"
basin_textFile = r"M:\SWE\WestWide\Spatial_SWE\ASO\ASO_Metadata\State_Basin.txt"
basinList = []
snotel_shp = r"W:\Spatial_SWE\ASO\ASO_Metadata\WW_CDEC_SNOTEL_geon83.shp"
cdec_shp = ""
modelStatsCSV = f"M:/SWE/WestWide/Spatial_SWE/ASO/2025/data_testing/ASO_SNOTEL_DifferenceStats.csv"

# open basin file for list
basin_state_map = {}
with open(basin_textFile, 'r') as f:
    reader = csv.reader(f, delimiter='|')
    for row in reader:
        basin = row[0].strip('"')
        state = row[1].strip('"')
        basin_state_map[basin] = state

zips_to_process = os.listdir(toProcessFolder)
for zip_file in zips_to_process:
    if zip_file.endswith(".zip"):
        zip_file_path = os.path.join(toProcessFolder, zip_file)

        # # get SWE file from zip folder
        extract_zip(zip_path=zip_file_path, ext=search_tag, output_folder=data_folder)
        print("file moved")

        name = zip_file.split("_")[1]
        basinList.append(name)
        print(basinList)

# get basin and basin info from aso folder
all_stats = []
asoSWE = os.listdir(data_folder)
for file in asoSWE:
    if file.endswith(".tif"):
        print(file)
        basinName = file.split("_")[1] # gets the value between the first and second "_"
        print(basinName)
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
                    if not os.path.exists(snotelWS + output_filename):
                        merged_snotel_df = download_and_merge_snotel_data(
                            id_list=id_list,
                            state_list=state_list,
                            start_date=start,
                            end_date=end,
                            output_dir=snotelWS,
                            output_filename=output_filename
                        )
                    ## get % grade of snotel
                    basin_snotel_df = pd.read_csv(snotelWS + output_filename)
                    basin_snotel_df['MEAN'] = basin_snotel_df.drop('Date', axis=1).mean(axis=1)
                    first_mean = basin_snotel_df['MEAN'].iloc[0]
                    last_mean = basin_snotel_df['MEAN'].iloc[-1]
                    SWE_Difference = (last_mean - first_mean)
                    percent_change = ((last_mean - first_mean) / first_mean) * 100
                    direction = "positive" if percent_change > 0 else "negative" if percent_change < 0 else "no change"

                    station_cols = basin_snotel_df.columns.drop("Date")
                    trends = {col: classify_trend(basin_snotel_df[col]) for col in station_cols}

                    if all(t == "Increasing" for t in trends.values()):
                        overall_trend = "Increasing"
                    elif all(t == "Decreasing" for t in trends.values()):
                        overall_trend = "Decreasing"
                    elif all(t == "No Trend" for t in trends.values()):
                        overall_trend = "No Trend"
                    else:
                        overall_trend = "Mixed"

                    # add metrics to csv
                    all_stats.append({
                        'Basin': basinName,
                        'Date': startDate,
                        'Domain': domain,
                        'Year': startDate[:4],
                        'GradeDirection': direction,
                        'GradeDifference': percent_change,
                        'SWEDifference_in': SWE_Difference,
                        'modelRun': modelRun,
                        'RunDate': rundate,
                        'OverallTrend': overall_trend
                    })

                ## __FUNCTION: WW fractional error layer
                fractional_error(filename=file, input_folder=data_folder, output_folder=compareWS + f"{rundate}_{modelRun}/",
                                 snapRaster=snapRaster, projIn=projIn, modelRunWorkspace=modelRunWorkspace,
                                 rundate=rundate, delete=False)
                print("completed thanks")
if all_stats:
    stats_df = pd.DataFrame(all_stats)
    if os.path.exists(modelStatsCSV):
        stats_df.to_csv(modelStatsCSV, mode='a', header=False, index=False)
    else:
        stats_df.to_csv(modelStatsCSV, index=False)
df = pd.DataFrame(all_stats)



