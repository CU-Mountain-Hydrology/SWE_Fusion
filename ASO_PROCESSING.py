# import modules
import sys
from ASO_Processing_functions import *
from SWE_Fusion_functions import *
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
rundate = "20250420"
SNM_modelRun = "RT_CanAdj_rcn_noSW_woCCR_UseAvg"
WW_modelRun = "fSCA_RT_CanAdj_rcn_noSW_woCCR"

# set parameters for zip extraction
toProcessFolder = r"M:/SWE/WestWide/Spatial_SWE/ASO/2025/toProcess/"
# zip_file_path = r"W:\Spatial_SWE\ASO\2025\ASO_BlueRiver_2025May24_AllData_and_Reports.zip"
search_tag = "swe_50m.tif"
basin_textFile = r"M:\SWE\WestWide\Spatial_SWE\ASO\ASO_Metadata\State_Basin.txt"
basinList = []
snotel_shp = r"W:\Spatial_SWE\ASO\ASO_Metadata\WW_CDEC_SNOTEL_geon83.shp"
cdec_shp = r"W:\data\Snow_Instruments\pillows\SNM_CDEC_SNOTEL_geow84_20241125.shp"
modelStatsCSV = f"M:/SWE/WestWide/Spatial_SWE/ASO/2025/data_testing/ASO_SNOTEL_DifferenceStats.csv"

# open basin file for list
basin_state_map = {}
with open(basin_textFile, 'r') as f:
    reader = csv.reader(f, delimiter='|')
    for row in reader:
        basin = row[0].strip('"')
        state = row[1].strip('"')
        basin_state_map[basin] = state
        # print(state)

zips_to_process = os.listdir(toProcessFolder)
for zip_file in zips_to_process:
    if zip_file.endswith(".zip"):
        zip_file_path = os.path.join(toProcessFolder, zip_file)
        print(zip_file_path)
        name = zip_file.split("_")[1]
        basinList.append(name)
        print(name)
        basin_state = basin_state_map.get(name, None)


        data_folder = r"M:/SWE/WestWide/Spatial_SWE/ASO/2025/data_testing/"
        ## add something here that is only unzips if the file doesn't already exist
        extract_zip(zip_path=zip_file_path, ext=search_tag, output_folder=data_folder)
        # print("file moved")

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
            print(basin_state)
            # checking for domains
            if basin_state == "CA":
                domain = "SNM"
                fullDomain = "Sierras"
                compareWS = "M:/SWE/WestWide/Spatial_SWE/ASO/2025/data_testing/SNM_comparison_testing/"
                snapRaster = r"M:\SWE\WestWide\data\boundaries\SNM_SnapRaster_albn83.tif"
                zonalRaster = r"M:\SWE\WestWide\data\hydro\SNM\dwr_band_basins_geoSort_albn83_delin.tif"
                snotelWorkspace = "W:/Spatial_SWE/ASO/2025/data_testing/snotel_comparisons/cdec_metaData.csv"
                pillow_shp = cdec_shp

                print(f"\nBasin: {basinName}, State: {basin_state}, Domain: {domain}")
                modelRunWorkspace = rf"M:/SWE/{fullDomain}/Spatial_SWE/{domain}_regression/RT_report_data/{rundate}_results/{SNM_modelRun}/"
                print(f"Workspace: {modelRunWorkspace}")

                # process ASO comparison
                process_aso_comparison(file, rundate, SNM_modelRun, data_folder, modelRunWorkspace, compareWS, snapRaster,
                                       projIn, zonalRaster)

                # find the snotels that are within a raster file
                gdf_final, site_id_list = get_points_within_raster(pillow_shp, data_folder + file, id_column="Site_ID")

                end_cdec = datetime.strptime(startDate, "%Y%b%d")
                start_cdec = end_cdec - timedelta(days=7)
                start_SNM = start_cdec.strftime("%Y%m%d")
                end_SNM = end_cdec.strftime("%Y%m%d")
                cdec_ws = "W:/Spatial_SWE/ASO/2025/data_testing/cdec_comparisons/"
                cdec_merged = download_and_merge_cdec_pillow_data(start_date=start_SNM, end_date=end_SNM,
                                                                  cdec_ws=cdec_ws,
                                                                  output_csv_filename=f"cdec_snowPillows_{end_SNM}.csv")

                cdec_subset = cdec_merged[cdec_merged["ID"].isin(site_id_list)]
                cdec_subset.to_csv(cdec_ws + f"cdec_snowPillows_{end_SNM}_{basinName}.csv")

                swe_cols = [col for col in cdec_subset.columns if col.endswith("_SWE")]
                daily_avg = cdec_subset[swe_cols].mean(axis=0, skipna=True)
                avg_df = daily_avg.reset_index()
                avg_df.columns = ["date", "avg_SWE"]
                avg_df["date"] = avg_df["date"].str.replace("_SWE", "")
                first_mean = avg_df["avg_SWE"].iloc[0]
                last_mean = avg_df["avg_SWE"].iloc[-1]
                SWE_Difference = (last_mean - first_mean)
                percent_change = ((last_mean - first_mean) / first_mean) * 100
                direction = "positive" if percent_change > 0 else "negative" if percent_change < 0 else "no change"

                trends = {}
                for i, row in cdec_subset.iterrows():
                    station = row["Station"]
                    valid_values = row[swe_cols].dropna()

                    if valid_values.empty:
                        trend = "No Data"

                    else:
                        first_val = valid_values.iloc[0]
                        last_val = valid_values.iloc[-1]

                        if last_val > first_val:
                            trend = "Increasing"
                        elif last_val < first_val:
                            trend = "Decreasing"
                        else:
                            trend = "No Trend"

                    trends[station] = trend

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
                    'modelRun': SNM_modelRun,
                    'RunDate': rundate,
                    'OverallTrend': overall_trend
                })

                ## __FUNCTION: WW fractional error layer
                fractional_error(filename=file, input_folder=data_folder,
                                 output_folder=compareWS + f"{rundate}_{SNM_modelRun}/",
                                 snapRaster=snapRaster, projIn=projIn, modelRunWorkspace=modelRunWorkspace,
                                 rundate=rundate, delete=False)

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
                modelRunWorkspace = rf"M:/SWE/{fullDomain}/Spatial_SWE/{domain}_regression/RT_report_data/{rundate}_results/{WW_modelRun}/"

                # process ASO comparison
                process_aso_comparison(file, rundate, WW_modelRun, data_folder, modelRunWorkspace, compareWS, snapRaster,
                                       projIn, zonalRaster)

                # find the snotels that are within a raster file
                gdf_final, site_id_list = get_points_within_raster(pillow_shp, data_folder + file, id_column="site_id")


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
                        'modelRun': WW_modelRun,
                        'RunDate': rundate,
                        'OverallTrend': overall_trend
                    })

                ## __FUNCTION: SNM fractional error layer
                fractional_error(filename=file, input_folder=data_folder, output_folder=compareWS + f"{rundate}_{WW_modelRun}/",
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



