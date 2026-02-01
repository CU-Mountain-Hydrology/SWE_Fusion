# This fuction is used to process Snow Today fSCA data from MODIS Sinusodial tiles [h08v04, h08v05, h09v04, h09v05, h10v04].
# This code needs the following folder structures in order to run properly:
### netCDF_WS = f"{root_directory}/{tile_from_tileList}/{year}", for example: C:\fSCA_processing\netcdf\h10v04\2025
### output_fscaWS needs subdirectories as well including: intermediary, outTifs, projected, for example: H:\WestUS_Data\NRT_FSCA_WW_N83\2025\intermediary, outTifs, projected.
#### if these folders aren't created, the code will create them for you. The final output should be in the root of output_fSCAWS.
### this is an arcpy code and the projections should be in the arcpy format, i.e.: projOut = arcpy.SpatialReference(4269)

import arcpy
import os
from arcpy.sa import *
import time
import rasterio
from rasterio.merge import merge
from datetime import datetime, timedelta
import shutil

def fsca_processing_tif(start_date, end_date, netCDF_WS, tile_list, output_fscaWS, proj_in, snap_raster, extent, proj_out):
    """
        This code requires the user to download the NetCDF tiles from SnowToday and place them in the correct folder.
        TKTK This code should add a prompt that if the files are not in the correct folder, it will add a prompt. Ask Eric

        Once the folders are in the correct places, it will reprojects, mosiac, and process to a GeoTIFF for use in the
        CU Boulder LRM model. This will go from a Sinusodial MODIS projection to a GCS NAD83 projection. This function
        uses ArcPy and requires the proper license.

        Parameters:
            start_date (datetime object): e.g. datetime(2025, 5, 26)
            start_date (datetime object): e.g. datetime(2025, 5, 28)
            netCDF_WS (str): Directory where the NetCDFs files are located. This should contain the tile folders
            tile_list (list, str): This is the list of the strings og the various tiles from MODIS. This includes ['h08v04', 'h09v04... etc.]
            output_fscaWS (str): Directory where the processing is output is held. This function creates three sub-directories to hold the intermediary processed data.
            proj_in (arcpy proj str): This is the MODIS Sinusodial projection string that will be used to define the projection of the tiles.
            snap_raster (str): This is a reference raster that has the same cell size, cell alignment, extent, and CRS as the desired output.
            extent (str): This is the extent of the final raster
            proj_out (arcpy proj Spatial Reference): This is the projection string that will be used to define the output. This would likely be GCS NAD83 for the CU Boulder LRM model.
        """

    proj_out = proj_out
    proj_in = proj_in

    while start_date <= end_date:
        print("\n")
        time.sleep(10)
        arcpy.env.workspace = None
        arcpy.ResetEnvironments()

        yyyymmddd = start_date.strftime("%Y%m%d")
        yyyymmddd_str = str(yyyymmddd)
        print(f"Working on {yyyymmddd} fSCA file")
        current_year = start_date.year

        server = os.path.join(output_fscaWS, str(current_year))
        out_intermediary = os.path.join(server, "intermediary")
        out_files_mos = os.path.join(server, "outTifs")
        out_projected = os.path.join(server, "projected")

        os.makedirs(out_intermediary, exist_ok=True)
        os.makedirs(out_files_mos, exist_ok=True)
        os.makedirs(out_projected, exist_ok=True)

        # NetCDF and output paths
        netCDF_list = [os.path.join(netCDF_WS, tile, str(current_year),
                        f"SPIRES_NRT_{tile}_MOD09GA061_{yyyymmddd_str}_V2.0.nc")
                       for tile in tile_list]

        outTIF_list = [os.path.join(out_intermediary, f"{tile}_Terra_{yyyymmddd_str}.v2024.0d.tif")
                       for tile in tile_list]

        # Convert NetCDF to GeoTIFF
        for netCDF, geotif in zip(netCDF_list, outTIF_list):
            print(f"Processing {netCDF}")
            layer_name = f"{geotif[:-4]}_layer"
            arcpy.MakeNetCDFRasterLayer_md(netCDF, "snow_fraction", "x", "y", layer_name, "", "", "BY_VALUE", "CENTER")
            arcpy.CopyRaster_management(layer_name, geotif, pixel_type="32_BIT_FLOAT", format="TIFF")
            if arcpy.Exists(layer_name):
                arcpy.Delete_management(layer_name)
            arcpy.DefineProjection_management(geotif, proj_in)

        # Mosaic
        tif_files_to_mosaic = [rasterio.open(fp) for fp in outTIF_list]
        mosaic_array, out_transform = merge(tif_files_to_mosaic)
        out_meta = tif_files_to_mosaic[0].meta.copy()
        out_meta.update({'driver': 'GTiff', 'count': 1,
                         'height': mosaic_array.shape[1],
                         'width': mosaic_array.shape[2],
                         'transform': out_transform})

        out_mosaic = os.path.join(out_files_mos, f"WUS_Terra_{yyyymmddd_str}_sinu.v2024.0d.tif")
        with rasterio.open(out_mosaic, 'w', **out_meta) as dest:
            dest.write(mosaic_array[0], 1)

        for tif in tif_files_to_mosaic:
            tif.close()

        # Project raster
        arcpy.env.snapRaster = snap_raster
        arcpy.env.extent = snap_raster
        arcpy.env.outputCoordinateSystem = snap_raster
        arcpy.env.cellSize = snap_raster

        projected_tif = os.path.join(out_projected, f"WUS_Terra_{yyyymmddd_str}_sinu.v2024.0d_geon83.tif")
        arcpy.ProjectRaster_management(out_mosaic, projected_tif, proj_out, "NEAREST", ".005")

        # Clip and mask
        clipped = ExtractByMask(projected_tif, extent, "INSIDE")
        clipped_path = os.path.join(out_projected, f"WUS_Terra_{yyyymmddd_str}_sinu.v2024.0d_geon83_GT100.tif")
        clipped.save(clipped_path)

        # fSCA > 100 masking
        final_fsca = Con(Raster(clipped_path) > 100, 100, Raster(clipped_path))
        final_output = os.path.join(server, f"{yyyymmddd_str}.tif")
        final_fsca.save(final_output)
        print("fSCA file created")

        # Cleanup
        deletes = []
        for folder in [out_projected, out_files_mos, out_intermediary]:
            for file in os.listdir(folder):
                if not file.endswith(".tif"):
                    deletes.append(os.path.join(folder, file))
        for f in deletes:
            os.remove(f)

        # Move to next day
        start_date += timedelta(days=1)

        # # delete intermediary directories
        # shutil.rmtree(out_intermediary)
        # shutil.rmtree(out_files_mos)
        # shutil.rmtree(out_projected)



import arcpy
import rasterio
import numpy as np
import os


def create_mean_layer(input_workspace, output_folder, dateList, start_year, end_year):
    years = list(range(start_year, (end_year + 1)))

    for date in dateList:
        file_list = []
        for year in years:
            print('year ', year)
            folder = input_workspace + str(year) + "/"
            file = f"WW_phvrcn_{year}{date}_fscamsk_glacMask.tif"

            # append to list
            if file in os.listdir(folder):
                file_list.append(folder + file)

        arrays = []
        meta = None

        for r in file_list:
            with rasterio.open(r) as src:
                arr = src.read(1).astype("float32")

                # Convert NoData to NaN
                if src.nodata is not None:
                    arr[arr == src.nodata] = np.nan

                arrays.append(arr)

                if meta is None:
                    meta = src.meta.copy()

        # Stack → (n_rasters, rows, cols)
        stack = np.stack(arrays)
        mean_raster = np.nanmean(stack, axis=0)
        meta.update(dtype="float32", nodata=np.nan)
        out_raster = output_folder + f"WW_{date}_fscamsk_glacMask_mean.tif"

        with rasterio.open(out_raster, "w", **meta) as dst:
            dst.write(mean_raster, 1)

        print("Mean raster written:", out_raster)

def calculate_dmfsca(
        fSCA_folder,
        DMFSCA_folder,
        wateryear_start,
        process_start_date,
        process_end_date
):
    """
    Calculate Daily Mean fSCA using incremental weighted average.
    Matches R code logic for efficiency.
    """

    # Build date -> filepath mapping (checking ALL necessary calendar years)
    raster_dict = {}

    # Determine which calendar years we need to scan
    years_to_check = set()
    current = wateryear_start
    while current <= process_end_date:
        years_to_check.add(current.year)
        current += timedelta(days=1)

    print(f"Water year: {wateryear_start.year}")
    print(f"Processing: {process_start_date.strftime('%Y-%m-%d')} to {process_end_date.strftime('%Y-%m-%d')}")
    print(f"Scanning calendar years: {sorted(years_to_check)}\n")

    # Scan all relevant year folders for INPUT files
    for year in sorted(years_to_check):
        year_folder = os.path.join(fSCA_folder, str(year))

        if not os.path.exists(year_folder):
            print(f"Input folder not found: {year_folder}")
            continue

        print(f"canning input: {year_folder}")
        raster_files = sorted([f for f in os.listdir(year_folder) if f.endswith(".tif")])

        for f in raster_files:
            try:
                date_str = os.path.splitext(f)[0]
                file_date = datetime.strptime(date_str, "%Y%m%d")
                raster_dict[file_date] = os.path.join(year_folder, f)
            except ValueError:
                continue

    # Filter to water year dates
    sorted_dates = sorted([d for d in raster_dict.keys()
                           if wateryear_start <= d <= process_end_date])

    if not sorted_dates:
        print("\nERROR: No valid input files found!")
        return

    print(f"\nFound {len(sorted_dates)} input files")
    print(f"Date range: {sorted_dates[0].strftime('%Y-%m-%d')} to {sorted_dates[-1].strftime('%Y-%m-%d')}\n")

    files_created = 0
    current_output_year = None
    current_output_folder = None

    # LOOP THROUGH EACH DATE
    for current_date in sorted_dates:
        if current_date < process_start_date:
            continue

        # DYNAMIC OUTPUT FOLDER: Changes when year changes
        output_year = current_date.year

        # If year changed, create new output folder
        if output_year != current_output_year:
            current_output_year = output_year
            current_output_folder = os.path.join(DMFSCA_folder, str(output_year))
            os.makedirs(current_output_folder, exist_ok=True)
            print(f"YEAR CHANGED → Now processing {output_year}")
            print(f"Output folder: {current_output_folder}")

        current_str = current_date.strftime('%Y%m%d')
        output_file = os.path.join(current_output_folder, f"dmfsca_{current_str}.tif")

        # Skip if exists
        if os.path.exists(output_file):
            print(f"Already exists: {current_str}")
            continue

        # =====================================================================
        # CASE 1: Oct 1 (dmfsca = fsca)
        # =====================================================================
        if current_date == wateryear_start:
            with rasterio.open(raster_dict[current_date]) as src:
                profile = src.profile
                data = src.read(1)

            with rasterio.open(output_file, 'w', **profile) as dst:
                dst.write(data, 1)

            print(f"Oct 1 (dmfsca=fsca): {current_str}")
            files_created += 1
            continue

        # =====================================================================
        # CASE 2: All other days (weighted average)
        # =====================================================================
        yesterday = current_date - timedelta(days=1)
        yesterday_year = yesterday.year
        yesterday_str = yesterday.strftime('%Y%m%d')

        # Yesterday's dmfsca might be in DIFFERENT YEAR FOLDER!
        yesterday_output_folder = os.path.join(DMFSCA_folder, str(yesterday_year))
        yesterday_dmfsca = os.path.join(yesterday_output_folder, f"dmfsca_{yesterday_str}.tif")

        if not os.path.exists(yesterday_dmfsca):
            print(f"Skipping {current_str} (missing {yesterday_str})")
            print(f"Expected: {yesterday_dmfsca}")
            continue

        # Read yesterday's dmfsca
        with rasterio.open(yesterday_dmfsca) as src:
            dmfsca_yesterday = src.read(1).astype(np.float32)
            profile = src.profile
            nodata = src.nodata if src.nodata is not None else -9999

        # Read today's fsca
        with rasterio.open(raster_dict[current_date]) as src:
            fsca_today = src.read(1).astype(np.float32)

        # Calculate: dmfsca = (dmfsca_yesterday × i + fsca_today) / (i + 1)
        days_from_start = (current_date - wateryear_start).days

        # Handle nodata
        mask_yesterday = (dmfsca_yesterday == nodata) | np.isnan(dmfsca_yesterday)
        mask_today = (fsca_today == nodata) | np.isnan(fsca_today)

        dmfsca_calc = np.where(mask_yesterday, 0, dmfsca_yesterday)
        fsca_calc = np.where(mask_today, 0, fsca_today)

        # Weighted average
        dmfsca_today = (dmfsca_calc * days_from_start + fsca_calc) / (days_from_start + 1)

        # Restore nodata
        dmfsca_today = np.where(mask_yesterday & mask_today, nodata, dmfsca_today)

        # Write
        profile.update(dtype=rasterio.float32, nodata=nodata)
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(dmfsca_today.astype(np.float32), 1)

        print(f"Created: {current_str} (day {days_from_start + 1})")
        files_created += 1

    print(f"COMPLETE: Created {files_created} DMFSCA files")

# downloading snow surveys
import os
import requests
import pandas as pd
import glob
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime
from shapely.geometry import Point

# def download_snow_surveys(report_date, survey_workspace, results_workspace, WW_url_file, NRCS_shp, WW_state_list):
#     """
#     Downloads, cleans, merges, and georeferences NRCS survey data for a given rundate.
#
#     Parameters:
#         rundate (str): e.g. "20250401"
#         surveyWorkspace (str): Base directory for output folder
#         resultsWorkspace (str): Directory to save final results
#         url_file (str): Path to .txt file with URLs formatted as "STATE_ABBR|URL"
#         NRCS_shp (str): Path to NRCS course shapefile
#         state_list (list): List of 2-letter state abbreviations (e.g. ["CO", "UT", ...])
#     """
#
#     # ---- Set up workspace ----
#     path = os.path.join(survey_workspace, report_date)
#     os.makedirs(path, exist_ok=True)
#     snowCourseWorkspace = os.path.join(survey_workspace, report_date)
#     date_obj = datetime.strptime(report_date, "%Y%m%d")
#     month = date_obj.strftime("%B")
#     year = date_obj.year
#
#     # ---- Read URLs from file ----
#     state_url_dict = {}
#     with open(WW_url_file, "r") as f:
#         for line in f:
#             if "|" in line:
#                 state, url = line.strip().split("|", 1)
#                 state_url_dict[state] = url
#
#     state_url_list = [state_url_dict[state] for state in WW_state_list]
#     state_text_list = [os.path.join(snowCourseWorkspace, f"{state}_original.txt") for state in WW_state_list]
#     state_edit_list = [os.path.join(snowCourseWorkspace, f"{state}.txt") for state in WW_state_list]
#
#     for url, text, edit, state in zip(state_url_list, state_text_list, state_edit_list, WW_state_list):
#         # Download text data
#         state_data = requests.get(url)
#         with open(text, 'w', encoding='utf-8') as out_f:
#             out_f.write(state_data.text)
#
#         # Remove headers and blank lines
#         with open(text, "r", encoding='utf-8') as file:
#             content = file.readlines()
#
#         marker = [i for i, line in enumerate(content) if line.startswith("#") or line == "\n"]
#
#         with open(edit, "w") as file:
#             for i, line in enumerate(content):
#                 if i not in marker:
#                     file.write(line)
#
#         with open(edit, "r", encoding='utf-8') as file:
#             cleaned_content = file.read().strip()
#
#         if not cleaned_content:
#             print(f"Warning: No data found for {state} after cleaning. Skipping...")
#             continue
#
#         # Convert cleaned text to CSV
#         df = pd.read_csv(edit, sep=",")
#         df.to_csv(f"{edit[:-4]}.csv", index=False)
#
#         # Clean and structure CSV
#         df = pd.read_csv(f"{edit[:-4]}.csv")
#         df = df[~df[month].astype(str).str.contains('Snow Water', na=False)]
#         df.drop(columns=[col for col in df.columns if "Unnamed" in col], inplace=True)
#         df['State'] = state
#         df = df[['Station Id', 'Station Name', 'Water Year', month, 'State']]
#         df.to_csv(f"{edit[:-4]}_update.csv", index=False)
#
#     # Merge all updated CSVs
#     all_update_csvs = glob.glob(os.path.join(snowCourseWorkspace, "*_update.csv"))
#     df_list = [pd.read_csv(csv) for csv in all_update_csvs]
#     merged_df = pd.concat(df_list, ignore_index=True)
#     merged_df["SWE_in"] = merged_df[month]
#     merged_df["SWE_m"] = merged_df["SWE_in"] * 0.0254
#     merged_df.to_csv(os.path.join(snowCourseWorkspace, f"{report_date}_WestWide_surveys.csv"), index=False)
#
#     # Merge with shapefile
#     gdf = gpd.read_file(NRCS_shp)
#     df = pd.read_csv(os.path.join(snowCourseWorkspace, f"{report_date}_WestWide_surveys.csv"))
#
#     df = df[["Station Name", "Station Id", month, "SWE_in", "SWE_m"]]
#     gdf = gdf[["Station_Na", "Station_Id", "State_Code", "Network_Co", "Elevation", "Latitude", "Longitude", "geometry"]]
#
#     merged_df = pd.merge(df, gdf, left_on="Station Name", right_on="Station_Na", how="right")
#     merged_df = merged_df.dropna(subset=[month]).drop_duplicates(subset=["Station Id"])
#
#     # Export as shapefile
#     geometry = [Point(xy) for xy in zip(merged_df["Longitude"], merged_df["Latitude"])]
#     gdf_stateSurvey = gpd.GeoDataFrame(merged_df, geometry=geometry, crs="EPSG:4326")
#
#     results_dir = os.path.join(results_workspace, f"{report_date}_results_ET")
#     os.makedirs(results_dir, exist_ok=True)
#
#     gdf_stateSurvey.to_file(os.path.join(results_dir, f"{report_date}_surveys.shp"), driver="ESRI Shapefile")
#
#     print(f"Snow Courses Downloaded")
def download_snow_surveys(report_date, survey_date, survey_workspace, results_workspace, WW_url_file, NRCS_shp, WW_state_list):
    """
    Downloads, cleans, merges, and georeferences NRCS survey data for a given rundate.

    Parameters:
        report_date (str): e.g. "20250401"
        survey_workspace (str): Base directory for output folder
        results_workspace (str): Directory to save final results
        WW_url_file (str): Path to .txt file with URLs formatted as "STATE_ABBR|URL"
        NRCS_shp (str): Path to NRCS course shapefile
        WW_state_list (list): List of 2-letter state abbreviations (e.g. ["CO", "UT", ...])
    """

    # ---- Set up workspace ----
    path = os.path.join(survey_workspace, report_date)
    os.makedirs(path, exist_ok=True)
    snowCourseWorkspace = os.path.join(survey_workspace, report_date)
    date_obj = datetime.strptime(survey_date, "%Y%m%d")
    month = date_obj.strftime("%b")
    year = date_obj.year

    # ---- Read URLs from file ----
    state_url_dict = {}
    with open(WW_url_file, "r") as f:
        for line in f:
            if "|" in line:
                state, url = line.strip().split("|", 1)
                state_url_dict[state] = url

    state_url_list = [state_url_dict[state] for state in WW_state_list]
    state_text_list = [os.path.join(snowCourseWorkspace, f"{state}_original.txt") for state in WW_state_list]
    state_edit_list = [os.path.join(snowCourseWorkspace, f"{state}.txt") for state in WW_state_list]

    # Track valid CSVs for merging
    valid_csvs = []

    for url, text, edit, state in zip(state_url_list, state_text_list, state_edit_list, WW_state_list):
        try:
            # Download text data
            state_data = requests.get(url)
            with open(text, 'w', encoding='utf-8') as out_f:
                out_f.write(state_data.text)

            # Remove headers and blank lines
            with open(text, "r", encoding='utf-8') as file:
                content = file.readlines()

            marker = [i for i, line in enumerate(content) if line.startswith("#") or line == "\n"]

            with open(edit, "w") as file:
                for i, line in enumerate(content):
                    if i not in marker:
                        file.write(line)

            with open(edit, "r", encoding='utf-8') as file:
                cleaned_content = file.read().strip()

            if not cleaned_content:
                print(f"Warning: No data found for {state} after cleaning. Skipping...")
                continue

            # Convert cleaned text to CSV
            df = pd.read_csv(edit, sep=",")

            # Check if month column exists
            if month not in df.columns:
                print(f"Warning: Column '{month}' not found for {state}. Skipping...")
                continue

            df.to_csv(f"{edit[:-4]}.csv", index=False)

            # Clean and structure CSV
            df = pd.read_csv(f"{edit[:-4]}.csv")
            df = df[~df[month].astype(str).str.contains('Snow Water', na=False)]
            df.drop(columns=[col for col in df.columns if "Unnamed" in col], inplace=True)
            df['State'] = state
            df = df[['Station Id', 'Station Name', 'Water Year', month, 'State']]

            csv_path = f"{edit[:-4]}_update.csv"
            df.to_csv(csv_path, index=False)
            valid_csvs.append(csv_path)

        except Exception as e:
            print(f"Error processing {state}: {e}. Skipping...")
            continue

    # Check if we have any valid CSVs to merge
    if not valid_csvs:
        print("ERROR: No valid survey data found for any state!")
        return

    # Merge all valid updated CSVs
    df_list = [pd.read_csv(csv) for csv in valid_csvs]
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df["SWE_in"] = merged_df[month]
    merged_df["SWE_m"] = merged_df["SWE_in"] * 0.0254
    merged_df.to_csv(os.path.join(snowCourseWorkspace, f"{report_date}_WestWide_surveys.csv"), index=False)

    # Merge with shapefile
    gdf = gpd.read_file(NRCS_shp)
    df = pd.read_csv(os.path.join(snowCourseWorkspace, f"{report_date}_WestWide_surveys.csv"))

    df = df[["Station Name", "Station Id", month, "SWE_in", "SWE_m"]]
    gdf = gdf[
        ["Station_Na", "Station_Id", "State_Code", "Network_Co", "Elevation", "Latitude", "Longitude", "geometry"]]

    merged_df = pd.merge(df, gdf, left_on="Station Name", right_on="Station_Na", how="left")
    merged_df = merged_df.dropna(subset=[month]).drop_duplicates(subset=["Station Id"])

    merged_df["Longitude"] = pd.to_numeric(merged_df["Longitude"], errors="coerce")
    merged_df["Latitude"] = pd.to_numeric(merged_df["Latitude"], errors="coerce")
    merged_df = merged_df.dropna(subset=["Longitude", "Latitude"])
    merged_df = merged_df[
        merged_df["Longitude"].apply(lambda x: isinstance(x, (int, float))) &
        merged_df["Latitude"].apply(lambda x: isinstance(x, (int, float)))
        ]

    # Export as shapefile
    # geometry = [
    #     Point(float(x), float(y))
    #     for x, y in zip(merged_df["Longitude"], merged_df["Latitude"])
    # ]
    # gdf_stateSurvey = gpd.GeoDataFrame(merged_df, geometry=geometry, crs="EPSG:4326")

    gdf_stateSurvey = gpd.GeoDataFrame(
        merged_df,
        geometry="geometry",
        crs=gdf.crs
    )

    results_dir = os.path.join(results_workspace, f"{report_date}_results_ET")
    os.makedirs(results_dir, exist_ok=True)

    gdf_stateSurvey.to_file(os.path.join(results_dir, f"{report_date}_surveys.shp"), driver="ESRI Shapefile")

    print(f"Snow Courses Downloaded from {len(valid_csvs)} states")

def download_cdec_snow_surveys(report_date, survey_date, survey_workspace, SNM_results_workspace, cdec_shapefile, basin_list):
    print("Starting CDEC snow survey download...")

    # Parse date
    date_obj = datetime.strptime(survey_date, "%Y%m%d")
    month = date_obj.strftime("%B")
    year = date_obj.year
    other_date = survey_date[:-2]

    # Set paths
    snow_course_workspace = os.path.join(survey_workspace, report_date)
    os.makedirs(snow_course_workspace, exist_ok=True)

    cdec_url = f"https://cdec.water.ca.gov/reportapp/javareports?name=COURSES.{other_date}"
    original_csv = os.path.join(survey_workspace, f"{report_date}_SnowCourseMeasurements_original.csv")
    v1_csv = os.path.join(survey_workspace, f"{report_date}_SnowCourseMeasurements_v1.csv")
    clean_csv = os.path.join(survey_workspace, f"{report_date}_SnowCourseMeasurements.csv")
    shapefile_out = os.path.join(snow_course_workspace, f"{report_date}_surveys_cdec.shp")
    merged_csv = os.path.join(survey_workspace, report_date, f"{report_date}_surveys_cdec.csv")
    final_shapefile = os.path.join(SNM_results_workspace, f"{report_date}_results_ET", f"{report_date}_surveys.shp")

    # Download HTML table from CDEC
    print(f"Downloading survey from: {cdec_url}")
    df_list = pd.read_html(cdec_url)
    df = df_list[0]
    df.to_csv(original_csv, index=False)
    print(f"Original CSV saved: {original_csv}")

    # Clean raw CSV
    df = pd.read_csv(original_csv)
    df.drop(columns=["Unnamed: 0", "Depth", "Density", "April 1", "Average"], inplace=True, errors="ignore")

    # Remove unwanted rows
    df = df[~df["Num"].astype(str).str.contains("Basin Average", na=False)]
    df = df[~df["Num"].astype(str).isin(basin_list)]
    df = df[~df["Date"].astype(str).str.contains("Scheduled", na=False)]
    df.to_csv(v1_csv, index=False)

    # Rename headers and drop bad rows
    # Read CSV WITH header
    file = pd.read_csv(v1_csv)

    print("Raw columns:", file.columns.tolist())

    # Explicitly rename CDEC columns
    file = file.rename(columns={
        "Name": "Station_Na",
        "Elev.": "Elev_Sur",
        "Water Content": "SWE_in"
    })

    # Keep only expected columns
    file = file[["Station_Na", "Elev_Sur", "Date", "SWE_in"]]

    # Drop header-like / non-numeric rows
    file["SWE_in"] = pd.to_numeric(file["SWE_in"], errors="coerce")
    file["Elev_Sur"] = pd.to_numeric(file["Elev_Sur"], errors="coerce")

    file = file.dropna(subset=["SWE_in"])

    # Convert SWE
    file["SWE_in"] = pd.to_numeric(file["SWE_in"], errors="coerce")
    file["SWE_m"] = file["SWE_in"] * 0.0254
    file.to_csv(clean_csv, index=False)
    print(file.head(5))
    print(f"Cleaned CSV saved: {clean_csv}")

    # Read shapefile and attach survey data
    gdf = gpd.read_file(cdec_shapefile)
    gdf.to_file(shapefile_out)
    df = pd.read_csv(clean_csv)

    df_sub = df[["Station_Na", "Elev_Sur", "Date", "SWE_in", "SWE_m"]]
    merged = gdf.merge(df_sub, on="Station_Na", how="left")
    merged.to_csv(merged_csv, index=False)

    # Drop empty values and save final shapefile
    merged.dropna(subset=["SWE_in"], inplace=True)
    os.makedirs(os.path.dirname(final_shapefile), exist_ok=True)
    merged.to_file(final_shapefile)
    print(f"Final shapefile saved: {final_shapefile}")


## function for downloading snotel for a time series
from rasterio.merge import merge
import pandas as pd
import io
import re
import requests
from shapely.geometry import box
from datetime import datetime, timedelta
import numpy as np
import time

def download_and_merge_snotel_data(id_list, state_list, start_date, end_date, output_dir, output_filename):
    merged_csv_path = os.path.join(output_dir, output_filename)
    
    # Skip if already downloaded
    if os.path.exists(merged_csv_path):
        print("Sensors already downloaded.")
        return pd.read_csv(merged_csv_path)
    
    print("Downloading SNOTEL data...")

    for ids, state in zip(id_list, state_list):
        if ids == 0:
            continue

        url = (
            f"https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customMultiTimeSeriesGroupByStationReport/daily/"
            f"start_of_period/{ids}:{state}:SNTL%257Cid=%2522%2522%257Cname/"
            f"{start_date},{end_date}/stationId,name,WTEQ::value?fitToScreen=false"
        )
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Filter lines
            lines = response.text.splitlines()
            filtered_lines = [line for line in lines if not line.lstrip().startswith('#')]
            filtered_csv = "\n".join(filtered_lines)
            
            df = pd.read_csv(io.StringIO(filtered_csv))

            # Find SWE column
            matching_cols = [col for col in df.columns if "Snow Water Equivalent" in col]
            if matching_cols:
                col = matching_cols[0]
                match = re.search(r'\(([^)]+)\)', col)
                new_col_name = match.group(1) if match else col
                df = df.rename(columns={col: new_col_name})

            # Save to temp CSV
            temp_csv_path = os.path.join(output_dir, f"snotel_{ids}_{state}_{end_date}.csv")
            df.to_csv(temp_csv_path, index=False)

        except Exception as ex:
            print(f"Error downloading {ids}, {state}: {ex}")
            continue

    # Merge downloaded CSVs
    csv_files = [f for f in os.listdir(output_dir) if f.startswith("snotel") and f.endswith(".csv")]
    merged_df = None

    for file in csv_files:
        file_path = os.path.join(output_dir, file)
        try:
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            print(f"Skipping empty file: {file}")
            continue

        df.columns = [col.strip() for col in df.columns]
        if 'Date' not in df.columns:
            print(f"Skipping file without 'Date' column: {file}")
            continue

        sensor_name = os.path.splitext(file)[0].split("_")[1]
        data_cols = [col for col in df.columns if col != 'Date']
        if not data_cols:
            print(f"No SWE column in file: {file}")
            continue

        df = df[['Date', data_cols[0]]].rename(columns={data_cols[0]: sensor_name})

        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on='Date', how='outer')

    # Save and clean up
    if merged_df is not None:
        merged_df.to_csv(merged_csv_path, index=False)
        print(f"Merged CSV saved to: {merged_csv_path}")
    else:
        print("No dataframes merged.")
        return pd.DataFrame()  # return empty df

    # Delete intermediate files
    for file in os.listdir(output_dir):
        if file.startswith("snotel") and not file.startswith("merged"):
            os.remove(os.path.join(output_dir, file))

    return merged_df


import pandas as pd
from datetime import datetime, timedelta
def download_and_merge_cdec_pillow_data(start_date, end_date, cdec_ws, output_csv_filename):
    """
    Download CDEC PAGE6 snow pillow data from start_date to end_date
    and merge into a single CSV.

    Parameters:
    start_date (str): YYYYMMDD
    end_date (str): YYYYMMDD
    cdec_ws (str): folder path to save CSV
    output_csv_filename (str): output CSV filename
    """

    # generate prior 7 dates + target
    start_obj = datetime.strptime(start_date, "%Y%m%d")
    end_obj = datetime.strptime(end_date, "%Y%m%d")

    # generate list of all dates from start to end (inclusive)
    delta_days = (end_obj - start_obj).days
    all_dates = [(start_obj + timedelta(days=i)).strftime("%Y%m%d") for i in range(delta_days + 1)]
    print(f'\nProcessing cdec pillows for: {all_dates}')

    all_dfs = []

    for dt in all_dates:
        print(f'Downloading {dt}')
        sensor_url = f"https://cdec.water.ca.gov/reportapp/javareports?name=PAGE6.{dt}"

        # read tables from url
        tables = pd.read_html(sensor_url)

        cleaned_tables = []
        for t in tables:
            if isinstance(t.columns, pd.MultiIndex):
                t.columns = t.columns.droplevel(0)
            cleaned_tables.append(t)

        # concat all tables for this date
        df = pd.concat(cleaned_tables, ignore_index=True)

        # clean stray characters in 'Today (IN)'
        df['Today (IN)'] = (
            df['Today (IN)']
            .astype(str)
            .str.replace(r"[^0-9.\-]", "", regex=True)
        )
        df['Today (IN)'] = pd.to_numeric(df['Today (IN)'], errors="coerce")

        # drop rows where ID is NaN
        df = df.dropna(subset=["ID"])

        # keep relevant columns
        df_clean = df[['Station', 'ID', 'Elev (FT)', 'Today (IN)']].copy()

        # rename 'Today (IN)' to include the date
        df_clean[f'{dt}_SWE'] = df_clean['Today (IN)']
        df_clean = df_clean.drop(columns=["Today (IN)"])

        all_dfs.append(df_clean)

    # merge all dataframes on 'ID' (or ['Station','ID'] if you prefer)
    from functools import reduce
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=['Station', 'ID', 'Elev (FT)'], how='outer'),
                       all_dfs)

    print(merged_df.head(10))
    merged_df.to_csv(cdec_ws + output_csv_filename)
    return merged_df

import shutil
import zipfile
def extract_zip(zip_path, ext, output_folder):
    """
    Extracts files with a specific tag from a zip archive and moves them to a new folder.

    Parameters:
        zip_path (str): Path to the .zip file.
        tag (str): Substring to match in filenames (e.g., "maps_aso_bestpred2014").
        destination_folder (str): Directory to move matching files to.
    """
    # Create the destination folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Create a temporary extraction folder
    temp_extract_path = os.path.join(os.path.dirname(zip_path), "temp_extract")
    os.makedirs(temp_extract_path, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract all to temporary location
        zip_ref.extractall(temp_extract_path)

        # Iterate through extracted files
        for root, _, files in os.walk(temp_extract_path):
            for file in files:
                if ext in file:
                    src_file = os.path.join(root, file)
                    dst_file = os.path.join(output_folder, file)

                    # Move file to destination
                    shutil.move(src_file, dst_file)
                    print(f"Moved: {src_file} → {dst_file}")

    # Clean up temp folder
    try:
        shutil.rmtree(temp_extract_path)
    except OSError:
        print(f"Temp folder not removed (may be locked): {temp_extract_path}")

# safely read in a shapefile
import geopandas as gpd
import fiona

def safe_read_shapefile(shapefile_path):
    gdf = gpd.read_file(shapefile_path)

    # Drop rows with missing or empty geometry
    gdf = gdf[gdf.geometry.notnull()]

    if gdf.empty:
        raise ValueError(f"No valid geometries found in {shapefile_path}")

    return gdf

def read_raster_values(path, remove_zeros=True):
    with rasterio.open(path) as src:
        arr = src.read(1)
        # remove no-data
        if src.nodata is not None:
            arr = arr[arr != src.nodata]
        # remove zeros if needed
        if remove_zeros:
            arr = arr[arr > 0]
    return arr

import rasterio
import os
##add  buffer
def get_points_within_raster(shapefile_path, raster_path, id_column="site_id"):
    """
    Find points from a shapefile that are within a raster and have actual data values.

    Parameters:
    shapefile_path (str): Path to the input shapefile
    raster_path (str): Path to the input raster file
    id_column (str): Name of the ID column to extract (default: "site_id")

    Returns:
    tuple: (gdf_final, site_id_list) - filtered GeoDataFrame and list of unique IDs
    """
    # read in shapefile
    gdf = safe_read_shapefile(shapefile_path)

    # Read the raster
    with rasterio.open(raster_path) as src:
        # Get raster bounds and CRS
        raster_bounds = src.bounds
        raster_crs = src.crs

        # Create bounding box polygon from raster extent
        raster_bbox = box(raster_bounds.left, raster_bounds.bottom,
                          raster_bounds.right, raster_bounds.top)

        # Ensure same CRS
        if gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)

        # Filter points within raster extent
        gdf_within_extent = gdf[gdf.geometry.within(raster_bbox) | gdf.geometry.intersects(raster_bbox)]

        # Optional: Further filter to only points with actual raster data (not NoData)
        points_with_data = []

        for idx, point in gdf_within_extent.iterrows():
            # Sample raster at point location
            coords = [(point.geometry.x, point.geometry.y)]
            try:
                sampled_values = list(src.sample(coords))
                raster_value = sampled_values[0][0]  # Get first band value

                # Check if value is not NoData
                if not np.isnan(raster_value) and raster_value != src.nodata:
                    points_with_data.append(idx)
            except:
                continue

        # Filter to points with actual data
        gdf_final = gdf_within_extent.loc[points_with_data]

    gdf_final = gdf_final[gdf_final.geometry.notnull()]
    site_id_list = gdf_final[id_column].unique().tolist()

    return gdf_final, site_id_list


# def tables_and_layers(user, year, report_date, mean_date, meanWorkspace, model_run, masking, watershed_zones,
#                       band_zones, HUC6_zones, region_zones, case_field_wtrshd, case_field_band, watermask, glacierMask, snapRaster_geon83,
#                       snapRaster_albn83, projGEO, projALB, ProjOut_UTM, bias, prev_report_date=None, prev_model_run=None):
#
#     # set code parameters
#     where_clause = """"POLY_AREA" > 100"""
#     part_area = "100 SquareKilometers"
#
#
#
#     #######################################################################
#     # End of Setting Variables
#     #######################################################################
#     workspaceBase = fr"M:/SWE/WestWide/Spatial_SWE/WW_regression/"
#     resultsWorkspace = workspaceBase + f"RT_report_data/{report_date}_results_ET/"
#
#     os.makedirs(resultsWorkspace, exist_ok=True)
#
#     # create directory for model run
#     if masking == "Y":
#         RunNameMod = f"fSCA_{model_run}"
#     else:
#         RunNameMod = model_run
#
#     # create directory
#     if bias == "N":
#         arcpy.CreateFolder_management(resultsWorkspace, RunNameMod)
#         outWorkspace = resultsWorkspace + RunNameMod + "/"
#         print("model run workspace created")
#
#     if bias == "Y":
#         outWorkspace = resultsWorkspace + RunNameMod + "/"
#
#     # meanWorkspace = workspaceBase + "mean_2001_2021_Nodmfsca/"
#     prevRepWorkspace = workspaceBase + f"RT_report_data/{prev_report_date}_results/{prev_model_run}/"
#     meanMask = outWorkspace + f"{mean_date}_mean_msk.tif"
#     # MODSCAG_tif_plus = f"H:/WestUS_Data/Rittger_data/fsca_v2025.0.1_ops/NRT_FSCA_WW_N83/{year}/{report_date}.tif"
#     MODSCAG_tif_plus_proj = outWorkspace + f"fSCA_{report_date}_albn83.tif"
#     MODSCAG_tif_plus = f"H:/WestUS_Data/Rittger_data/fsca_v2024.0d/NRT_FSCA_WW_N83/{year}/{report_date}.tif"
#
#     ## project and clip SNODAS
#     SNODASWorkspace = resultsWorkspace + "SNODAS/"
#     ClipSNODAS = SNODASWorkspace + "SWE_" + report_date + "_Cp_m_albn83_clp.tif"
#     SWE_Diff = outWorkspace + "SNODAS_Regress_" + report_date + ".tif"
#     SWE_both = outWorkspace + f"SWE_{report_date}_both.tif"
#
#     # define snow-no snow layer
#     modscag_0_1 = outWorkspace + f"modscag_0_1_{report_date}.tif"
#     modscag_per = outWorkspace + f"modscag_per_{report_date}.tif"
#     modscag_per_msk = outWorkspace + f"modscag_per_{report_date}_msk.tif"
#
#     # define snow/no snow null layer
#     mod_null = outWorkspace + f"modscag_0_1_{report_date}msk_null.tif"
#     mod_poly = outWorkspace + f"modscag_0_1_{report_date}msk_null_poly.shp"
#     ### ASK LEANNE ABOUT UTM
#     mod_poly_utm = outWorkspace + f"modscag_0_1_{report_date}_msk_null_poly_utm.shp"
#
#     snowPolySel = outWorkspace + f"modscag_{report_date}_snowline_Sel.shp"
#     snowPolyElim = outWorkspace + f"modscag_{report_date}_snowline_Sel_elim.shp"
#
#     # define snow pillow gpkg
#     meanMap = meanWorkspace + f"WW_{mean_date}_fscamsk_glacMask_mean.tif"
#     meanMap_copy = outWorkspace + f"WW_{mean_date}_mean_geon83.tif"
#     meanMap_proj = outWorkspace + f"WW_{mean_date}_mean_albn83.tif"
#     meanMapMask = outWorkspace + f"WW_{mean_date}_mean_msk_albn83.tif"
#     lastRast = prevRepWorkspace + f"p8_{prev_report_date}_noneg.tif"
#     DiffRaster = outWorkspace + f"Diff_{report_date}_{prev_report_date}.tif"
#
#     ## define rasters
#     rcn_raw = outWorkspace + f"WW_{report_date}_phvrcn_mos_noMask.tif"
#     rcn_glacMask = outWorkspace + f"WW_{report_date}_phvrcn_mos_masked.tif"
#     rcn_raw_proj = outWorkspace + f"WW_{report_date}_phvrcn_albn83.tif"
#     rcnFinal = outWorkspace + f"phvrcn_{report_date}_final.tif"
#     product7 = outWorkspace + f"p7_{report_date}.tif"
#     product7_noFsca = outWorkspace + f"p7_{report_date}_nofsca.tif"
#     product8 = outWorkspace + f"p8_{report_date}_noneg.tif"
#     prod8msk = outWorkspace + f"p8_{report_date}_noneg_msk.tif"
#     product9 = outWorkspace + f"p9_{report_date}.tif"
#     product10 = outWorkspace + f"p10_{report_date}.tif"
#     product11 = outWorkspace + f"p11_{report_date}.tif"
#     product12 = outWorkspace + f"p12_{report_date}.tif"
#
#     # output Tables
#     SWEbandtable = outWorkspace + f"{report_date}band_swe_table.dbf"
#     SWEtable = outWorkspace + f"{report_date}swe_table.dbf"
#     SWEbandtable100 = outWorkspace + f"{report_date}swe_table_100.dbf"
#     SWEbandtable_save = outWorkspace + f"{report_date}band_swe_table_save.dbf"
#     SWEtable_save = outWorkspace + f"{report_date}swe_table_save.dbf"
#     SWEbandtable100_save = outWorkspace + f"{report_date}swe_table_100_save.dbf"
#
#     # anomoly tables
#     anombandTable = outWorkspace + f"{report_date}band_anom_table.dbf"
#     anomTable = outWorkspace + f"{report_date}anom_table.dbf"
#     anomHuc6Table = outWorkspace + f"{report_date}huc6_anom_table.dbf"
#     anomHuc6Table_save = outWorkspace + f"{report_date}huc6_anom_table_save.dbf"
#     meanTable = outWorkspace + f"{report_date}mean_table.dbf"
#     anombandTable_save = outWorkspace + f"{report_date}band_anom_table_save.dbf"
#     anomTable_save = outWorkspace + f"{report_date}anom_table_save.dbf"
#     meanTable_save = outWorkspace + f"{report_date}mean_table_save.dbf"
#
#     # region tables
#     anomRegionTable = outWorkspace + f"{report_date}anomRegion_table.dbf"
#     anomRegionTable_save = outWorkspace + f"{report_date}anomRegion_table_save.dbf"
#
#     # Modscag 0/1 tables and % tables
#     scabandtable = outWorkspace + f"{report_date}band_sca_table.dbf"
#     scatable = outWorkspace + f"{report_date}sca_table.dbf"
#     scabandtable_save = outWorkspace + f"{report_date}band_sca_table_save.dbf"
#     scatable_save = outWorkspace + f"{report_date}_sca_table_save.dbf"
#     perbandtable = outWorkspace + f"{report_date}band_per_table.dbf"
#     pertable = outWorkspace + f"{report_date}_per_table.dbf"
#
#     # create tempoary view for join
#     SWEbandtableView = outWorkspace + f"{report_date}band_swe_table_view.dbf"
#     SWEtableView = outWorkspace + f"{report_date}swe_table_view.dbf"
#
#     # create joined tables
#     BandtableJoin = outWorkspace + f"{report_date}band_table.dbf"
#     WtshdTableJoin = outWorkspace + f"{report_date}Wtshd_table.dbf"
#
#     # Anomaly maps
#     anomMap = outWorkspace + f"{report_date}_anom.tif"
#     anom0_100map = outWorkspace + f"{report_date}anom0_200.tif"
#     anom0_100msk = outWorkspace + f"{report_date}anom0_200_msk.tif"
#
#     #SWE maps
#     SWEzoneMap = outWorkspace + f"{report_date}_swe_wshd.tif"
#     SWEHuc6Map = outWorkspace + f"{report_date}_swe_huc6.tif"
#     MeanHuc6Map = outWorkspace + f"{report_date}_mean_huc6.tif"
#     MeanzoneMap = outWorkspace + f"{report_date}_mean_wshd.tif"
#     SWEbandzoneMap = outWorkspace + f"{report_date}_swe_band_wshd.tif"
#     MeanBandZoneMap = outWorkspace + f"{report_date}_mean_band_wshd.tif"
#     SWEregionMap = outWorkspace + f"{report_date}_swe_region.tif"
#     MeanRegionMap = outWorkspace + f"{report_date}_mean_region.tif"
#
#     # mean layer masked for use in creating anomly map
#     anomMask = outWorkspace + f"{report_date}_anom_mask.tif"
#
#     # statistic
#     statisticType = "MEAN"
#
#     # final output csv tables
#     WtshdTableJoinCSV = outWorkspace + f"{report_date}Wtshd_table.csv"
#     BandtableJoinCSV = outWorkspace + f"{report_date}band_table.csv"
#     anomRegionTableCSV = outWorkspace + f"{report_date}anomRegion_table.csv"
#     Band100TableCSV = outWorkspace + f"{report_date}band_table_100.csv"
#     SCATableJoinCSV = outWorkspace + f"{report_date}sca_Wtshd_table.csv"
#     BandSCAtableJoinCSV = outWorkspace + f"{report_date}sca_band_table.csv"
#     anomWtshdTableCSV = outWorkspace + f"{report_date}anomWtshd_table.csv"
#     anomBandTableCSV = outWorkspace + f"{report_date}anomBand_table.csv"
#     anomHUC6TableCSV = outWorkspace + f"{report_date}anomHUC6_table.csv"
#     print("file paths established")
#
#     # domain model runs
#     if bias == "N":
#         print("Starting process for clipping files....")
#
#         domains = ["SNM", "PNW", "INMT", "SOCN", "NOCN"]
#         clipFilesWorkspace = "M:/SWE/WestWide/data/boundaries/Domains/DomainCutLines/complete/"
#
#         print("making clip workspace...")
#         arcpy.CreateFolder_management(outWorkspace, "cutlines")
#         cutLinesWorkspace = outWorkspace + "cutlines/"
#
#         for domain in domains:
#             MODWorkspace = fr"H:/WestUS_Data/Regress_SWE/{domain}/{user}/StationSWERegressionV2/"
#             arcpy.env.snapRaster = snapRaster_geon83
#             arcpy.env.cellSize = snapRaster_geon83
#             modelTIF = MODWorkspace + f"data/outputs/{model_run}/{domain}_phvrcn_{report_date}_nofscamsk.tif"
#
#             # extract by mask
#             outCut = ExtractByMask(modelTIF, clipFilesWorkspace + f"WW_{domain}_cutline_v2.shp", 'INSIDE')
#             outCut.save(cutLinesWorkspace + f"{domain}_{report_date}_clp.tif")
#             print(f"{domain} clipped")
#
#         # mosaic all tifs together
#         arcpy.env.snapRaster = snapRaster_geon83
#         arcpy.env.cellSize = snapRaster_geon83
#         outCutsList = [os.path.join(cutLinesWorkspace, f) for f in os.listdir(cutLinesWorkspace) if f.endswith(".tif")]
#         arcpy.MosaicToNewRaster_management(outCutsList, outWorkspace, f"WW_{report_date}_phvrcn_mos_noMask.tif",
#                                            projGEO, "32_BIT_FLOAT", ".005 .005", "1", "LAST")
#         print('mosaicked raster created. ')
#
#         ## apply glacier mask
#         outGlaciers = Raster(rcn_raw) * Raster(glacierMask)
#         outGlaciers.save(rcn_glacMask)
#         print("data glaciers masks")
#
#     ########################
#     print(f"Processing begins...")
#     ## copy in mean map
#
#     print("Project both fSCA and phvRaster...")
#     # project fSCA image
#     arcpy.env.snapRaster = snapRaster_albn83
#     arcpy.env.cellSize = snapRaster_albn83
#     arcpy.env.extent = snapRaster_albn83
#     # arcpy.DefineProjection_management(meanMap, projGEO)
#     arcpy.ProjectRaster_management(meanMap, meanMapMask, projALB,
#                                    "NEAREST", "500 500",
#                                    "", "", projGEO)
#
#     if bias == "N":
#         arcpy.ProjectRaster_management(rcn_glacMask, rcn_raw_proj, projALB,
#                                        "NEAREST", "500 500",
#                                        "", "")
#         arcpy.ProjectRaster_management(MODSCAG_tif_plus, MODSCAG_tif_plus_proj, projALB,
#                                        "NEAREST", "500 500",
#                                        "", "")
#         print("fSCA and rcn raw image and mean map projected")
#
#         mod_01 = Con((Raster(MODSCAG_tif_plus_proj) < 101) & (Raster(MODSCAG_tif_plus_proj) > 0),
#                      1, 0)
#         mod_01_Wtrmask = mod_01 * Raster(watermask)
#         mod_01_AllMaks = mod_01_Wtrmask * Raster(glacierMask)
#         mod_01_AllMaks.save(modscag_0_1)
#         print(f"fSCA mask tif saved")
#
#         # create fSCA percent layer
#         Mod_per = (Float(SetNull(Raster(MODSCAG_tif_plus_proj) > 100, Raster(MODSCAG_tif_plus_proj))) / 100)
#         Mod_per.save(modscag_per)
#         print(f"fSCA percent layer saved")
#
#         # create fsca percent layer ASK LEANNE, WHAT'S THE DIFFERENT BETWEEN LAKES MASK AND WATER MASK
#         mod_01_mask = Con(Raster(modscag_per) > 0.0001, 1, 0)
#         mod_per_msk = Raster(watermask) * mod_01_mask
#         mod_per_Allmsk = Raster(glacierMask) * mod_per_msk
#         mod_per_Allmsk.save(modscag_per_msk)
#         print("fSCA percent layer created")
#
#         rcn_final = Raster(rcn_raw_proj) * Raster(watermask)
#         rcn_final_wtshd = (Con((IsNull(rcn_final)) & (Raster(modscag_per_msk) >= 0), 0, rcn_final))
#         rcn_final_wtshd.save(rcnFinal)
#         print("rcn final created")
#
#     # ASK LEANNE, WHAT'S THE DIFFERENCE BETWEEN TIF MASK AND WATER MASK
#     print(f"Creating snowline shapefile: {snowPolyElim}")
#     mod_01_mask = Raster(modscag_0_1) * Raster(watermask)
#     mod_01_mask_glacier = Raster(modscag_0_1) * Raster(glacierMask)
#     mod_01_msk_null = SetNull(mod_01_mask_glacier == 0, mod_01_mask_glacier)
#     mod_01_msk_null.save(mod_null)
#
#     # Convert raster to polygon
#     arcpy.RasterToPolygon_conversion(mod_null, mod_poly, "NO_SIMPLIFY", "Value")
#     arcpy.Project_management(mod_poly, mod_poly_utm, ProjOut_UTM)
#     arcpy.AddGeometryAttributes_management(mod_poly_utm, "AREA", "", "SQUARE_KILOMETERS", "")
#     arcpy.Select_analysis(mod_poly_utm, snowPolySel, where_clause)
#     arcpy.EliminatePolygonPart_management(snowPolySel, snowPolyElim, "AREA", part_area, "0", "CONTAINED_ONLY")
#
#     print(f"creating masked SWE product")
#     if bias == "N":
#         rcn_LT_200 = SetNull(Raster(rcnFinal) > 200, rcnFinal)
#         rcn_GT_0 = Con(rcn_LT_200 < 0.0001, 0, rcn_LT_200)
#         rcn_GT_0.save(product7)
#         # ASK LEANNE ABOUT MASK VS WATERMASK
#         rcn_mask = rcn_GT_0 * Raster(watermask)
#         rcn_allMask = rcn_mask * Raster(glacierMask)
#
#         if masking == "Y":
#             rcn_mask_final = rcn_allMask * modscag_per
#         else:
#             rcn_mask_final = rcn_allMask
#         rcn_mask_final.save(product8)
#
#
#     # Create GT 0 mean blended swe and make mask
#     con01 = Con(Raster(meanMapMask) > 0.00, 1, 0)
#     con01.save(anomMask)
#     #
#     # # make anomoly mask
#     AnomProd = (Raster(product8) / Raster(meanMapMask)) * 100
#     AnomProd.save(anomMap)
#     print(f"anomaly map made")
#
#     # # make noneg anomoly map ## ASK LEANNE, DOES THIS NEED TO BE ADJUSTED?
#     connoeg = Con(Raster(anomMap) > 200, 200, Raster(anomMap))
#     connoeg.save(anom0_100map)
#
#     # # mask with watermaks
#     anomnoneg = connoeg * Raster(watermask)
#     anomnoneg_Mask = anomnoneg * Raster(glacierMask)
#     anomnoneg_Mask.save(anom0_100msk)
#
#     ## add SNODAS
#     # create difference with SNODAS
#     Diff_SNODAS = Raster(ClipSNODAS) - Raster(product8)
#     Diff_SNODAS.save(SWE_Diff)
#
#     # create overlap layers
#     print("Creating SNODAS and Regress diff layers ...")
#     SNODAS1000 = Con(Raster(ClipSNODAS) > 0.001, 1000, 0)
#     RSWE100 = Con(Raster(product8) > 0.001, 100, 0)
#
#     ## Then add them together to create a layer showing where they overlap and
#     ## where they're different
#     SWEboth = SNODAS1000 + RSWE100
#
#     ## Then save both layers
#     SWEboth.save(SWE_both)
#
#     print("create zonal stats and tables")
#     # outBandTable = ZonalStatisticsAsTable(band_zones, case_field_band, product8, SWEbandtable, "DATA",
#     #                                       "MEAN")
#     # outSWETable = ZonalStatisticsAsTable(watershed_zones, case_field_wtrshd, product8, SWEtable, "DATA",
#     #                                      "MEAN")
#     # outSCABand = ZonalStatisticsAsTable(band_zones, case_field_band, modscag_per, scabandtable, "DATA",
#     #                                     "ALL")
#     # outSCAWtshd = ZonalStatisticsAsTable(watershed_zones, case_field_wtrshd, modscag_per, scatable, "DATA",
#     #                                      "ALL")
#     ZonalStatisticsAsTable(band_zones, case_field_band, product8, SWEbandtable, "DATA",
#                                           "MEAN")
#     arcpy.Delete_management("in-memory")
#     import gc
#     gc.collect()
#
#     ZonalStatisticsAsTable(watershed_zones, case_field_wtrshd, product8, SWEtable, "DATA",
#                                          "MEAN")
#     arcpy.Delete_management("in-memory")
#     gc.collect()
#
#     ZonalStatisticsAsTable(band_zones, case_field_band, modscag_per, scabandtable, "DATA",
#                                         "ALL")
#     arcpy.Delete_management("in-memory")
#     gc.collect()
#
#     ZonalStatisticsAsTable(watershed_zones, case_field_wtrshd, modscag_per, scatable, "DATA",                                   "ALL")
#     arcpy.Delete_management("in-memory")
#     gc.collect()
#
#     arcpy.AddField_management(SWEbandtable, "SWE_IN", "DOUBLE", "", "", "",
#                               "", "NULLABLE", "NON_REQUIRED")
#     # arcpy.AddField_management(SWEbandtable100, "SWE_IN", "DOUBLE", "", "", "",
#     #                           "", "NULLABLE", "NON_REQUIRED")
#     arcpy.Delete_management("in-memory")
#     gc.collect()
#
#     arcpy.AddField_management(SWEtable, "SWE_IN", "DOUBLE", "#", "#", "#",
#                               "#", "NULLABLE", "NON_REQUIRED", "#")
#     arcpy.Delete_management("in-memory")
#     gc.collect()
#
#     arcpy.AddField_management(SWEbandtable, "AREA_MI2", "DOUBLE", "#", "#", "#",
#                               "#", "NULLABLE", "NON_REQUIRED", "#")
#     arcpy.Delete_management("in-memory")
#     gc.collect()
#
#     arcpy.AddField_management(SWEtable, "AREA_MI2", "DOUBLE", "#", "#", "#",
#                               "#", "NULLABLE", "NON_REQUIRED", "#")
#     arcpy.Delete_management("in-memory")
#     gc.collect()
#
#     # arcpy.AddField_management(SWEbandtable100, "VOL_M3", "DOUBLE", "#", "#", "#",
#     #                           "#", "NULLABLE", "NON_REQUIRED", "#")
#     arcpy.AddField_management(SWEbandtable, "VOL_M3", "DOUBLE", "#", "#", "#",
#                               "#", "NULLABLE", "NON_REQUIRED", "#")
#     arcpy.Delete_management("in-memory")
#     gc.collect()
#
#     arcpy.AddField_management(SWEtable, "VOL_M3", "DOUBLE", "#", "#", "#",
#                               "#", "NULLABLE", "NON_REQUIRED", "#")
#     arcpy.Delete_management("in-memory")
#     gc.collect()
#
#     # arcpy.AddField_management(SWEbandtable100, "VOL_AF", "DOUBLE", "#", "#", "#",
#     #                           "#", "NULLABLE", "NON_REQUIRED", "#")
#     arcpy.AddField_management(SWEbandtable, "VOL_AF", "DOUBLE", "#", "#", "#",
#                               "#", "NULLABLE", "NON_REQUIRED", "#")
#     arcpy.Delete_management("in-memory")
#     gc.collect()
#
#     arcpy.AddField_management(SWEtable, "VOL_AF", "DOUBLE", "#", "#", "#",
#                               "#", "NULLABLE", "NON_REQUIRED", "#")
#     arcpy.Delete_management("in-memory")
#     gc.collect()
#
#     print("fields added")
#     # calculate fields
#     arcpy.CalculateField_management(SWEbandtable, "SWE_IN", "!MEAN! * 39.370079", "PYTHON")
#     arcpy.CalculateField_management(SWEtable, "SWE_IN", "!MEAN! * 39.370079", "PYTHON")
#     arcpy.Delete_management("in-memory")
#     gc.collect()
#
#     # Calculate area in sq miles
#     arcpy.CalculateField_management(SWEbandtable, "AREA_MI2", "0.00000038610216 * !AREA!", "PYTHON")
#     arcpy.CalculateField_management(SWEtable, "AREA_MI2", "0.00000038610216 * !AREA!", "PYTHON")
#     arcpy.Delete_management("in-memory")
#     gc.collect()
#
#     # Calculate volume in cubic meters
#     arcpy.CalculateField_management(SWEbandtable, "VOL_M3", "!MEAN! * !AREA!", "PYTHON")
#     arcpy.CalculateField_management(SWEtable, "VOL_M3", "!MEAN! * !AREA!", "PYTHON")
#     arcpy.Delete_management("in-memory")
#     gc.collect()
#     # arcpy.CalculateField_management(SWEbandtable100, "VOL_M3", "!MEAN! * !AREA!", "PYTHON")
#
#     # Calculate volume in acre feet
#     # arcpy.CalculateField_management(SWEbandtable100, "VOL_AF", "!VOL_M3! * 0.000810714", "PYTHON")
#     arcpy.CalculateField_management(SWEbandtable, "VOL_AF", "!VOL_M3! * 0.000810714", "PYTHON")
#     arcpy.CalculateField_management(SWEtable, "VOL_AF", "!VOL_M3! * 0.000810714", "PYTHON")
#     arcpy.Delete_management("in-memory")
#     gc.collect()
#
#     ### Sort by bandname and watershed name, 2 tables
#     arcpy.Sort_management(SWEbandtable, SWEbandtable_save, [[case_field_band, "ASCENDING"]])
#     arcpy.Sort_management(SWEtable, SWEtable_save, [[case_field_wtrshd, "ASCENDING"]])
#     arcpy.Delete_management("in-memory")
#     gc.collect()
#     # arcpy.Sort_management(SWEbandtable100, SWEbandtable100_save, [["Value", "ASCENDING"]])
#
#     ## work on SCA tables
#     arcpy.AddField_management(scabandtable, "Percent", "DOUBLE", "", "", "",
#                               "", "NULLABLE", "NON_REQUIRED")
#     arcpy.AddField_management(scatable, "Percent", "DOUBLE", "", "", "",
#                               "", "NULLABLE", "NON_REQUIRED")
#     arcpy.Delete_management("in-memory")
#     gc.collect()
#     # calculate percent
#     arcpy.CalculateField_management(scabandtable, "Percent", "( !SUM! / !COUNT! ) * 100", "PYTHON", "")
#     arcpy.CalculateField_management(scatable, "Percent", "( !SUM! / !COUNT! ) * 100", "PYTHON", "")
#     arcpy.Delete_management("in-memory")
#     gc.collect()
#
#     # sort
#     arcpy.Sort_management(scabandtable, scabandtable_save, [[case_field_band, "ASCENDING"]])
#     arcpy.Sort_management(scatable, scatable_save, [[case_field_wtrshd, "ASCENDING"]])
#
#     print("Create SWE and mean zonal maps...")
#     # NEED TO ADD IN MEAN MASK
#     swezmap = ZonalStatistics(watershed_zones, case_field_wtrshd, product8, "MEAN", "DATA")
#     meanzmap = ZonalStatistics(watershed_zones, case_field_wtrshd, meanMapMask, "MEAN", "DATA")
#     swezmap.save(SWEzoneMap)
#     meanzmap.save(MeanzoneMap)
#     arcpy.Delete_management("in-memory")
#     gc.collect()
#
#     # NEED TO ADD IN MEAN MASK
#     print("creating product 9...")
#     proj9 = (Raster(SWEzoneMap) / Raster(MeanzoneMap)) * 100
#     proj9.save(product9)
#
#     # creating banded watershed mean and swe
#     # NEED TO ADD IN MEAN MASK
#     tswebzmap = ZonalStatistics(band_zones, case_field_band, product8, statisticType, "DATA")
#     tmeanbzmap = ZonalStatistics(band_zones, case_field_band, meanMapMask, statisticType, "DATA")
#     tswebzmap.save(SWEbandzoneMap)
#     tmeanbzmap.save(MeanBandZoneMap)
#
#     # NEED TO ADD IN MEAN MASK
#     print("creating product 10 = " + product10)
#     prod10 = (Raster(SWEbandzoneMap) / Raster(MeanBandZoneMap)) * 100
#     prod10.save(product10)
#
#     print("created product 11 = HUC 6 percent of average")
#     swezmap = ZonalStatistics(HUC6_zones, "name", product8, "MEAN", "DATA")
#     meanzmap = ZonalStatistics(HUC6_zones, "name", meanMapMask, "MEAN", "DATA")
#     swezmap.save(SWEHuc6Map)
#     meanzmap.save(MeanHuc6Map)
#
#     prod11 = (Raster(SWEHuc6Map) / Raster(MeanHuc6Map)) * 100
#     prod11.save(product11)
#
#     print("created product 12 = region percent of average")
#     swezmap = ZonalStatistics(region_zones, "RegionAll", product8, "MEAN", "DATA")
#     meanzmap = ZonalStatistics(region_zones, "RegionAll", meanMapMask, "MEAN", "DATA")
#     swezmap.save(SWEregionMap)
#     meanzmap.save(MeanRegionMap)
#
#     prod11 = (Raster(SWEregionMap) / Raster(MeanRegionMap)) * 100
#     prod11.save(product12)
#
#
#     print("create anomaly layer table = " + anomTable)
#     # NEED TO ADD IN MEAN MASK
#     anomt = ZonalStatisticsAsTable(watershed_zones, case_field_wtrshd, product9, anomTable, "DATA", "MEAN")
#     arcpy.Delete_management("in-memory")
#     gc.collect()
#
#     anombt = ZonalStatisticsAsTable(band_zones, case_field_band, product10, anombandTable, "DATA", "MEAN")
#     arcpy.Delete_management("in-memory")
#     gc.collect()
#
#     anomh6 = ZonalStatisticsAsTable(HUC6_zones, "name", product11, anomHuc6Table, "DATA", "MEAN")
#     arcpy.Delete_management("in-memory")
#     gc.collect()
#
#     anomreg = ZonalStatisticsAsTable(region_zones, "RegionAll", product12, anomRegionTable, "DATA", "MEAN")
#     arcpy.Delete_management("in-memory")
#     gc.collect()
#
#     # NEED TO ADD IN MEAN MASK
#     # Sort by bandname and watershed name, 3 tables
#     arcpy.Sort_management(anombandTable, anombandTable_save, [[case_field_band, "ASCENDING"]])
#     arcpy.Sort_management(anomTable, anomTable_save, [[case_field_wtrshd, "ASCENDING"]])
#     arcpy.Sort_management(anomHuc6Table, anomHuc6Table_save, [["name", "ASCENDING"]])
#     arcpy.Sort_management(anomRegionTable, anomRegionTable_save, [["RegionAll", "ASCENDING"]])
#
#     # add field for anom
#     arcpy.AddField_management(anombandTable_save, "Average", "DOUBLE", "", "", "",
#                               "", "NULLABLE", "NON_REQUIRED")
#     arcpy.AddField_management(anomTable_save, "Average", "DOUBLE", "", "", "",
#                               "", "NULLABLE", "NON_REQUIRED")
#     arcpy.AddField_management(anomHuc6Table_save, "Average", "DOUBLE", "", "", "",
#                               "", "NULLABLE", "NON_REQUIRED")
#     arcpy.AddField_management(anomRegionTable_save, "Average", "DOUBLE", "", "", "",
#                               "", "NULLABLE", "NON_REQUIRED")
#
#     # calculate field
#     arcpy.CalculateField_management(anombandTable_save, "Average", f"!MEAN!", "PYTHON3")
#     arcpy.CalculateField_management(anomTable_save, "Average", f"!MEAN!", "PYTHON3")
#     arcpy.CalculateField_management(anomHuc6Table_save, "Average", f"!MEAN!", "PYTHON3")
#     arcpy.CalculateField_management(anomRegionTable_save, "Average", f"!MEAN!", "PYTHON3")
#
#     print("Joining sorted tables ... ")
#     ## Delete extra fields from tables before joining them
#     ## Banded Tables
#     arcpy.DeleteField_management(SWEbandtable_save, "ZONE_CODE")
#     arcpy.DeleteField_management(anombandTable_save, "ZONE_CODE;COUNT")
#     arcpy.DeleteField_management(scabandtable_save,
#                                  "ZONE_CODE;COUNT;MIN;MAX;RANGE;MEAN;STD;SUM;VARIETY;MAJORITY;MINORITY;MEDIAN")
#     ## Watershed tables
#     arcpy.DeleteField_management(SWEtable_save, "ZONE_CODE")
#     arcpy.DeleteField_management(anomTable_save, "ZONE_CODE;COUNT")
#     arcpy.DeleteField_management(scatable_save,
#                                  "ZONE_CODE;COUNT;MIN;MAX;RANGE;MEAN;STD;SUM;VARIETY;MAJORITY;MINORITY;MEDIAN")
#
#
#     ## Make tables into table views for joins
#     arcpy.MakeTableView_management(SWEbandtable_save, SWEbandtableView)
#     arcpy.MakeTableView_management(SWEtable_save, SWEtableView)
#
#     arcpy.JoinField_management(SWEtable_save, case_field_wtrshd, scatable_save, case_field_wtrshd, "Percent")
#     arcpy.JoinField_management(SWEbandtable_save, case_field_band, scabandtable_save, case_field_band, "Percent")
#
#     print("Making csvs...")
#     # wtshd_dbf = gpd.read_file(SWEtableView)
#     wtshd_dbf = gpd.read_file(SWEtable_save)
#     wtshd_df = pd.DataFrame(wtshd_dbf)
#     wtshd_df.to_csv(WtshdTableJoinCSV, index=False)
#
#     band_dbf = gpd.read_file(SWEbandtable_save)
#     band_df = pd.DataFrame(band_dbf)
#     band_df.to_csv(BandtableJoinCSV, index=False)
#
#     anom_dbf = gpd.read_file(anomTable_save)
#     anom_df = pd.DataFrame(anom_dbf)
#     anom_df.to_csv(anomWtshdTableCSV, index=False)
#
#     anom_band_dbf = gpd.read_file(anombandTable_save)
#     anom_band_df = pd.DataFrame(anom_band_dbf)
#     anom_band_df.to_csv(anomBandTableCSV, index=False)
#
#     anom_huc_dbf = gpd.read_file(anomHuc6Table_save)
#     anom_huc_df = pd.DataFrame(anom_huc_dbf)
#     anom_huc_df.to_csv(anomHUC6TableCSV, index=False)
#
#     anom_region_dbf = gpd.read_file(anomRegionTable_save)
#     anom_region_df = pd.DataFrame(anom_region_dbf)
#     anom_region_dbf.to_csv(anomRegionTableCSV, index=False)

def tables_and_layers(user, year, report_date, mean_date, meanWorkspace, model_run, masking, watershed_zones,
                      band_zones, HUC6_zones, region_zones, case_field_wtrshd, case_field_band, watermask, glacierMask, snapRaster_geon83,
                      snapRaster_albn83, projGEO, projALB, ProjOut_UTM, run_type="Normal", bias_model_run=None, prev_report_date=None, prev_model_run=None):

    # set code parameters
    where_clause = """"POLY_AREA" > 100"""
    part_area = "100 SquareKilometers"



    #######################################################################
    # End of Setting Variables
    #######################################################################
    workspaceBase = fr"M:/SWE/WestWide/Spatial_SWE/WW_regression/"
    resultsWorkspace = workspaceBase + f"RT_report_data/{report_date}_results_ET/"

    os.makedirs(resultsWorkspace, exist_ok=True)

    # create directory for model run
    if masking == "Y":
        RunNameMod = f"fSCA_{model_run}"
    else:
        RunNameMod = model_run

    # Determine workspace based on run_type
    if run_type == "Normal":
        arcpy.CreateFolder_management(resultsWorkspace, RunNameMod)
        outWorkspace = resultsWorkspace + RunNameMod + "/"
        print("model run workspace created")

    elif run_type == "Vetting":
        outWorkspace = resultsWorkspace + RunNameMod + "/"

    elif run_type == "Bias":
        outWorkspace = resultsWorkspace + bias_model_run + "/"

    # meanWorkspace = workspaceBase + "mean_2001_2021_Nodmfsca/"
    prevRepWorkspace = workspaceBase + f"RT_report_data/{prev_report_date}_results/{prev_model_run}/"
    meanMask = outWorkspace + f"{mean_date}_mean_msk.tif"
    # MODSCAG_tif_plus = f"H:/WestUS_Data/Rittger_data/fsca_v2025.0.1_ops/NRT_FSCA_WW_N83/{year}/{report_date}.tif"
    MODSCAG_tif_plus_proj = outWorkspace + f"fSCA_{report_date}_albn83.tif"
    MODSCAG_tif_plus = f"H:/WestUS_Data/Rittger_data/fsca_v2024.0d/NRT_FSCA_WW_N83/{year}/{report_date}.tif"

    ## project and clip SNODAS
    SNODASWorkspace = resultsWorkspace + "SNODAS/"
    ClipSNODAS = SNODASWorkspace + "SWE_" + report_date + "_Cp_m_albn83_clp.tif"
    SWE_Diff = outWorkspace + "SNODAS_Regress_" + report_date + ".tif"
    SWE_both = outWorkspace + f"SWE_{report_date}_both.tif"

    # define snow-no snow layer
    modscag_0_1 = outWorkspace + f"modscag_0_1_{report_date}.tif"
    modscag_per = outWorkspace + f"modscag_per_{report_date}.tif"
    modscag_per_msk = outWorkspace + f"modscag_per_{report_date}_msk.tif"

    # define snow/no snow null layer
    mod_null = outWorkspace + f"modscag_0_1_{report_date}msk_null.tif"
    mod_poly = outWorkspace + f"modscag_0_1_{report_date}msk_null_poly.shp"
    ### ASK LEANNE ABOUT UTM
    mod_poly_utm = outWorkspace + f"modscag_0_1_{report_date}_msk_null_poly_utm.shp"

    snowPolySel = outWorkspace + f"modscag_{report_date}_snowline_Sel.shp"
    snowPolyElim = outWorkspace + f"modscag_{report_date}_snowline_Sel_elim.shp"

    # define snow pillow gpkg
    meanMap = meanWorkspace + f"WW_{mean_date}_fscamsk_glacMask_mean.tif"
    meanMap_copy = outWorkspace + f"WW_{mean_date}_mean_geon83.tif"
    meanMap_proj = outWorkspace + f"WW_{mean_date}_mean_albn83.tif"
    meanMapMask = outWorkspace + f"WW_{mean_date}_mean_msk_albn83.tif"
    lastRast = prevRepWorkspace + f"p8_{prev_report_date}_noneg.tif"
    DiffRaster = outWorkspace + f"Diff_{report_date}_{prev_report_date}.tif"

    ## define rasters
    rcn_raw = outWorkspace + f"WW_{report_date}_phvrcn_mos_noMask.tif"
    rcn_glacMask = outWorkspace + f"WW_{report_date}_phvrcn_mos_masked.tif"
    rcn_raw_proj = outWorkspace + f"WW_{report_date}_phvrcn_albn83.tif"
    rcnFinal = outWorkspace + f"phvrcn_{report_date}_final.tif"
    product7 = outWorkspace + f"p7_{report_date}.tif"
    product7_noFsca = outWorkspace + f"p7_{report_date}_nofsca.tif"
    product8 = outWorkspace + f"p8_{report_date}_noneg.tif"
    prod8msk = outWorkspace + f"p8_{report_date}_noneg_msk.tif"
    product9 = outWorkspace + f"p9_{report_date}.tif"
    product10 = outWorkspace + f"p10_{report_date}.tif"
    product11 = outWorkspace + f"p11_{report_date}.tif"
    product12 = outWorkspace + f"p12_{report_date}.tif"

    # output Tables
    SWEbandtable = outWorkspace + f"{report_date}band_swe_table.dbf"
    SWEtable = outWorkspace + f"{report_date}swe_table.dbf"
    SWEbandtable100 = outWorkspace + f"{report_date}swe_table_100.dbf"
    SWEbandtable_save = outWorkspace + f"{report_date}band_swe_table_save.dbf"
    SWEtable_save = outWorkspace + f"{report_date}swe_table_save.dbf"
    SWEbandtable100_save = outWorkspace + f"{report_date}swe_table_100_save.dbf"

    # anomoly tables
    anombandTable = outWorkspace + f"{report_date}band_anom_table.dbf"
    anomTable = outWorkspace + f"{report_date}anom_table.dbf"
    anomHuc6Table = outWorkspace + f"{report_date}huc6_anom_table.dbf"
    anomHuc6Table_save = outWorkspace + f"{report_date}huc6_anom_table_save.dbf"
    meanTable = outWorkspace + f"{report_date}mean_table.dbf"
    anombandTable_save = outWorkspace + f"{report_date}band_anom_table_save.dbf"
    anomTable_save = outWorkspace + f"{report_date}anom_table_save.dbf"
    meanTable_save = outWorkspace + f"{report_date}mean_table_save.dbf"

    # region tables
    anomRegionTable = outWorkspace + f"{report_date}anomRegion_table.dbf"
    anomRegionTable_save = outWorkspace + f"{report_date}anomRegion_table_save.dbf"

    # Modscag 0/1 tables and % tables
    scabandtable = outWorkspace + f"{report_date}band_sca_table.dbf"
    scatable = outWorkspace + f"{report_date}sca_table.dbf"
    scabandtable_save = outWorkspace + f"{report_date}band_sca_table_save.dbf"
    scatable_save = outWorkspace + f"{report_date}_sca_table_save.dbf"
    perbandtable = outWorkspace + f"{report_date}band_per_table.dbf"
    pertable = outWorkspace + f"{report_date}_per_table.dbf"

    # create tempoary view for join
    SWEbandtableView = outWorkspace + f"{report_date}band_swe_table_view.dbf"
    SWEtableView = outWorkspace + f"{report_date}swe_table_view.dbf"

    # create joined tables
    BandtableJoin = outWorkspace + f"{report_date}band_table.dbf"
    WtshdTableJoin = outWorkspace + f"{report_date}Wtshd_table.dbf"

    # Anomaly maps
    anomMap = outWorkspace + f"{report_date}_anom.tif"
    anom0_100map = outWorkspace + f"{report_date}anom0_200.tif"
    anom0_100msk = outWorkspace + f"{report_date}anom0_200_msk.tif"

    #SWE maps
    SWEzoneMap = outWorkspace + f"{report_date}_swe_wshd.tif"
    SWEHuc6Map = outWorkspace + f"{report_date}_swe_huc6.tif"
    MeanHuc6Map = outWorkspace + f"{report_date}_mean_huc6.tif"
    MeanzoneMap = outWorkspace + f"{report_date}_mean_wshd.tif"
    SWEbandzoneMap = outWorkspace + f"{report_date}_swe_band_wshd.tif"
    MeanBandZoneMap = outWorkspace + f"{report_date}_mean_band_wshd.tif"
    SWEregionMap = outWorkspace + f"{report_date}_swe_region.tif"
    MeanRegionMap = outWorkspace + f"{report_date}_mean_region.tif"

    # mean layer masked for use in creating anomly map
    anomMask = outWorkspace + f"{report_date}_anom_mask.tif"

    # statistic
    statisticType = "MEAN"

    # final output csv tables
    WtshdTableJoinCSV = outWorkspace + f"{report_date}Wtshd_table.csv"
    BandtableJoinCSV = outWorkspace + f"{report_date}band_table.csv"
    anomRegionTableCSV = outWorkspace + f"{report_date}anomRegion_table.csv"
    Band100TableCSV = outWorkspace + f"{report_date}band_table_100.csv"
    SCATableJoinCSV = outWorkspace + f"{report_date}sca_Wtshd_table.csv"
    BandSCAtableJoinCSV = outWorkspace + f"{report_date}sca_band_table.csv"
    anomWtshdTableCSV = outWorkspace + f"{report_date}anomWtshd_table.csv"
    anomBandTableCSV = outWorkspace + f"{report_date}anomBand_table.csv"
    anomHUC6TableCSV = outWorkspace + f"{report_date}anomHUC6_table.csv"
    print("file paths established")

    # domain model runs
    if run_type == "Normal":
        print("Starting process for clipping files....")

        domains = ["SNM", "PNW", "INMT", "SOCN", "NOCN"]
        clipFilesWorkspace = "M:/SWE/WestWide/data/boundaries/Domains/DomainCutLines/complete/"

        print("making clip workspace...")
        arcpy.CreateFolder_management(outWorkspace, "cutlines")
        cutLinesWorkspace = outWorkspace + "cutlines/"

        for domain in domains:
            MODWorkspace = fr"H:/WestUS_Data/Regress_SWE/{domain}/{user}/StationSWERegressionV2/"
            arcpy.env.snapRaster = snapRaster_geon83
            arcpy.env.cellSize = snapRaster_geon83
            modelTIF = MODWorkspace + f"data/outputs/{model_run}/{domain}_phvrcn_{report_date}_nofscamsk.tif"

            # extract by mask
            outCut = ExtractByMask(modelTIF, clipFilesWorkspace + f"WW_{domain}_cutline_v2.shp", 'INSIDE')
            outCut.save(cutLinesWorkspace + f"{domain}_{report_date}_clp.tif")
            print(f"{domain} clipped")

        # mosaic all tifs together
        arcpy.env.snapRaster = snapRaster_geon83
        arcpy.env.cellSize = snapRaster_geon83
        outCutsList = [os.path.join(cutLinesWorkspace, f) for f in os.listdir(cutLinesWorkspace) if f.endswith(".tif")]
        arcpy.MosaicToNewRaster_management(outCutsList, outWorkspace, f"WW_{report_date}_phvrcn_mos_noMask.tif",
                                           projGEO, "32_BIT_FLOAT", ".005 .005", "1", "LAST")
        print('mosaicked raster created. ')

        ## apply glacier mask
        outGlaciers = Raster(rcn_raw) * Raster(glacierMask)
        outGlaciers.save(rcn_glacMask)
        print("data glaciers masks")

    ########################
    print(f"Processing begins...")
    ## copy in mean map

    print("Project both fSCA and phvRaster...")
    # project fSCA image
    arcpy.env.snapRaster = snapRaster_albn83
    arcpy.env.cellSize = snapRaster_albn83
    arcpy.env.extent = snapRaster_albn83
    # arcpy.DefineProjection_management(meanMap, projGEO)
    arcpy.ProjectRaster_management(meanMap, meanMapMask, projALB,
                                   "NEAREST", "500 500",
                                   "", "", projGEO)
    # project fSCA
    arcpy.ProjectRaster_management(MODSCAG_tif_plus, MODSCAG_tif_plus_proj, projALB,
                                   "NEAREST", "500 500",
                                   "", "")
    print("fSCA and rcn raw image and mean map projected")

    mod_01 = Con((Raster(MODSCAG_tif_plus_proj) < 101) & (Raster(MODSCAG_tif_plus_proj) > 0),
                 1, 0)
    mod_01_Wtrmask = mod_01 * Raster(watermask)
    mod_01_AllMaks = mod_01_Wtrmask * Raster(glacierMask)
    mod_01_AllMaks.save(modscag_0_1)
    print(f"fSCA mask tif saved")

    # create fSCA percent layer
    Mod_per = (Float(SetNull(Raster(MODSCAG_tif_plus_proj) > 100, Raster(MODSCAG_tif_plus_proj))) / 100)
    Mod_per.save(modscag_per)
    print(f"fSCA percent layer saved")

    # create fsca percent layer ASK LEANNE, WHAT'S THE DIFFERENT BETWEEN LAKES MASK AND WATER MASK
    mod_01_mask = Con(Raster(modscag_per) > 0.0001, 1, 0)
    mod_per_msk = Raster(watermask) * mod_01_mask
    mod_per_Allmsk = Raster(glacierMask) * mod_per_msk
    mod_per_Allmsk.save(modscag_per_msk)
    print("fSCA percent layer created")

    # ASK LEANNE, WHAT'S THE DIFFERENCE BETWEEN TIF MASK AND WATER MASK
    print(f"Creating snowline shapefile: {snowPolyElim}")
    mod_01_mask = Raster(modscag_0_1) * Raster(watermask)
    mod_01_mask_glacier = Raster(modscag_0_1) * Raster(glacierMask)
    mod_01_msk_null = SetNull(mod_01_mask_glacier == 0, mod_01_mask_glacier)
    mod_01_msk_null.save(mod_null)

    if run_type == "Normal":
        arcpy.ProjectRaster_management(rcn_glacMask, rcn_raw_proj, projALB,
                                       "NEAREST", "500 500",
                                       "", "")

        rcn_final = Raster(rcn_raw_proj) * Raster(watermask)
        rcn_final_wtshd = (Con((IsNull(rcn_final)) & (Raster(modscag_per_msk) >= 0), 0, rcn_final))
        rcn_final_wtshd.save(rcnFinal)
        print("rcn final created")

    # Convert raster to polygon
    arcpy.RasterToPolygon_conversion(mod_null, mod_poly, "NO_SIMPLIFY", "Value")
    arcpy.Project_management(mod_poly, mod_poly_utm, ProjOut_UTM)
    arcpy.AddGeometryAttributes_management(mod_poly_utm, "AREA", "", "SQUARE_KILOMETERS", "")
    arcpy.Select_analysis(mod_poly_utm, snowPolySel, where_clause)
    arcpy.EliminatePolygonPart_management(snowPolySel, snowPolyElim, "AREA", part_area, "0", "CONTAINED_ONLY")

    print(f"creating masked SWE product")
    if run_type == "Normal":
        rcn_LT_200 = SetNull(Raster(rcnFinal) > 200, rcnFinal)
        rcn_GT_0 = Con(rcn_LT_200 < 0.0001, 0, rcn_LT_200)
        rcn_GT_0.save(product7)
        # ASK LEANNE ABOUT MASK VS WATERMASK
        rcn_mask = rcn_GT_0 * Raster(watermask)
        rcn_allMask = rcn_mask * Raster(glacierMask)

        if masking == "Y":
            rcn_mask_final = rcn_allMask * modscag_per
        else:
            rcn_mask_final = rcn_allMask
        rcn_mask_final.save(product8)


    # Create GT 0 mean blended swe and make mask
    con01 = Con(Raster(meanMapMask) > 0.00, 1, 0)
    con01.save(anomMask)
    #
    # # make anomoly mask
    AnomProd = (Raster(product8) / Raster(meanMapMask)) * 100
    AnomProd.save(anomMap)
    print(f"anomaly map made")

    # # make noneg anomoly map ## ASK LEANNE, DOES THIS NEED TO BE ADJUSTED?
    connoeg = Con(Raster(anomMap) > 200, 200, Raster(anomMap))
    connoeg.save(anom0_100map)

    # # mask with watermaks
    anomnoneg = connoeg * Raster(watermask)
    anomnoneg_Mask = anomnoneg * Raster(glacierMask)
    anomnoneg_Mask.save(anom0_100msk)

    ## add SNODAS
    # create difference with SNODAS
    Diff_SNODAS = Raster(ClipSNODAS) - Raster(product8)
    Diff_SNODAS.save(SWE_Diff)

    # create overlap layers
    print("Creating SNODAS and Regress diff layers ...")
    SNODAS1000 = Con(Raster(ClipSNODAS) > 0.001, 1000, 0)
    RSWE100 = Con(Raster(product8) > 0.001, 100, 0)

    ## Then add them together to create a layer showing where they overlap and
    ## where they're different
    SWEboth = SNODAS1000 + RSWE100

    ## Then save both layers
    SWEboth.save(SWE_both)

    print("create zonal stats and tables")
    # outBandTable = ZonalStatisticsAsTable(band_zones, case_field_band, product8, SWEbandtable, "DATA",
    #                                       "MEAN")
    # outSWETable = ZonalStatisticsAsTable(watershed_zones, case_field_wtrshd, product8, SWEtable, "DATA",
    #                                      "MEAN")
    # outSCABand = ZonalStatisticsAsTable(band_zones, case_field_band, modscag_per, scabandtable, "DATA",
    #                                     "ALL")
    # outSCAWtshd = ZonalStatisticsAsTable(watershed_zones, case_field_wtrshd, modscag_per, scatable, "DATA",
    #                                      "ALL")
    ZonalStatisticsAsTable(band_zones, case_field_band, product8, SWEbandtable, "DATA",
                                          "MEAN")
    arcpy.Delete_management("in-memory")
    import gc
    gc.collect()

    ZonalStatisticsAsTable(watershed_zones, case_field_wtrshd, product8, SWEtable, "DATA",
                                         "MEAN")
    arcpy.Delete_management("in-memory")
    gc.collect()

    ZonalStatisticsAsTable(band_zones, case_field_band, modscag_per, scabandtable, "DATA",
                                        "ALL")
    arcpy.Delete_management("in-memory")
    gc.collect()

    ZonalStatisticsAsTable(watershed_zones, case_field_wtrshd, modscag_per, scatable, "DATA",                                   "ALL")
    arcpy.Delete_management("in-memory")
    gc.collect()

    arcpy.AddField_management(SWEbandtable, "SWE_IN", "DOUBLE", "", "", "",
                              "", "NULLABLE", "NON_REQUIRED")
    # arcpy.AddField_management(SWEbandtable100, "SWE_IN", "DOUBLE", "", "", "",
    #                           "", "NULLABLE", "NON_REQUIRED")
    arcpy.Delete_management("in-memory")
    gc.collect()

    arcpy.AddField_management(SWEtable, "SWE_IN", "DOUBLE", "#", "#", "#",
                              "#", "NULLABLE", "NON_REQUIRED", "#")
    arcpy.Delete_management("in-memory")
    gc.collect()

    arcpy.AddField_management(SWEbandtable, "AREA_MI2", "DOUBLE", "#", "#", "#",
                              "#", "NULLABLE", "NON_REQUIRED", "#")
    arcpy.Delete_management("in-memory")
    gc.collect()

    arcpy.AddField_management(SWEtable, "AREA_MI2", "DOUBLE", "#", "#", "#",
                              "#", "NULLABLE", "NON_REQUIRED", "#")
    arcpy.Delete_management("in-memory")
    gc.collect()

    # arcpy.AddField_management(SWEbandtable100, "VOL_M3", "DOUBLE", "#", "#", "#",
    #                           "#", "NULLABLE", "NON_REQUIRED", "#")
    arcpy.AddField_management(SWEbandtable, "VOL_M3", "DOUBLE", "#", "#", "#",
                              "#", "NULLABLE", "NON_REQUIRED", "#")
    arcpy.Delete_management("in-memory")
    gc.collect()

    arcpy.AddField_management(SWEtable, "VOL_M3", "DOUBLE", "#", "#", "#",
                              "#", "NULLABLE", "NON_REQUIRED", "#")
    arcpy.Delete_management("in-memory")
    gc.collect()

    # arcpy.AddField_management(SWEbandtable100, "VOL_AF", "DOUBLE", "#", "#", "#",
    #                           "#", "NULLABLE", "NON_REQUIRED", "#")
    arcpy.AddField_management(SWEbandtable, "VOL_AF", "DOUBLE", "#", "#", "#",
                              "#", "NULLABLE", "NON_REQUIRED", "#")
    arcpy.Delete_management("in-memory")
    gc.collect()

    arcpy.AddField_management(SWEtable, "VOL_AF", "DOUBLE", "#", "#", "#",
                              "#", "NULLABLE", "NON_REQUIRED", "#")
    arcpy.Delete_management("in-memory")
    gc.collect()

    print("fields added")
    # calculate fields
    arcpy.CalculateField_management(SWEbandtable, "SWE_IN", "!MEAN! * 39.370079", "PYTHON")
    arcpy.CalculateField_management(SWEtable, "SWE_IN", "!MEAN! * 39.370079", "PYTHON")
    arcpy.Delete_management("in-memory")
    gc.collect()

    # Calculate area in sq miles
    arcpy.CalculateField_management(SWEbandtable, "AREA_MI2", "0.00000038610216 * !AREA!", "PYTHON")
    arcpy.CalculateField_management(SWEtable, "AREA_MI2", "0.00000038610216 * !AREA!", "PYTHON")
    arcpy.Delete_management("in-memory")
    gc.collect()

    # Calculate volume in cubic meters
    arcpy.CalculateField_management(SWEbandtable, "VOL_M3", "!MEAN! * !AREA!", "PYTHON")
    arcpy.CalculateField_management(SWEtable, "VOL_M3", "!MEAN! * !AREA!", "PYTHON")
    arcpy.Delete_management("in-memory")
    gc.collect()
    # arcpy.CalculateField_management(SWEbandtable100, "VOL_M3", "!MEAN! * !AREA!", "PYTHON")

    # Calculate volume in acre feet
    # arcpy.CalculateField_management(SWEbandtable100, "VOL_AF", "!VOL_M3! * 0.000810714", "PYTHON")
    arcpy.CalculateField_management(SWEbandtable, "VOL_AF", "!VOL_M3! * 0.000810714", "PYTHON")
    arcpy.CalculateField_management(SWEtable, "VOL_AF", "!VOL_M3! * 0.000810714", "PYTHON")
    arcpy.Delete_management("in-memory")
    gc.collect()

    ### Sort by bandname and watershed name, 2 tables
    arcpy.Sort_management(SWEbandtable, SWEbandtable_save, [[case_field_band, "ASCENDING"]])
    arcpy.Sort_management(SWEtable, SWEtable_save, [[case_field_wtrshd, "ASCENDING"]])
    arcpy.Delete_management("in-memory")
    gc.collect()
    # arcpy.Sort_management(SWEbandtable100, SWEbandtable100_save, [["Value", "ASCENDING"]])

    ## work on SCA tables
    arcpy.AddField_management(scabandtable, "Percent", "DOUBLE", "", "", "",
                              "", "NULLABLE", "NON_REQUIRED")
    arcpy.AddField_management(scatable, "Percent", "DOUBLE", "", "", "",
                              "", "NULLABLE", "NON_REQUIRED")
    arcpy.Delete_management("in-memory")
    gc.collect()
    # calculate percent
    arcpy.CalculateField_management(scabandtable, "Percent", "( !SUM! / !COUNT! ) * 100", "PYTHON", "")
    arcpy.CalculateField_management(scatable, "Percent", "( !SUM! / !COUNT! ) * 100", "PYTHON", "")
    arcpy.Delete_management("in-memory")
    gc.collect()

    # sort
    arcpy.Sort_management(scabandtable, scabandtable_save, [[case_field_band, "ASCENDING"]])
    arcpy.Sort_management(scatable, scatable_save, [[case_field_wtrshd, "ASCENDING"]])

    print("Create SWE and mean zonal maps...")
    # NEED TO ADD IN MEAN MASK
    swezmap = ZonalStatistics(watershed_zones, case_field_wtrshd, product8, "MEAN", "DATA")
    meanzmap = ZonalStatistics(watershed_zones, case_field_wtrshd, meanMapMask, "MEAN", "DATA")
    swezmap.save(SWEzoneMap)
    meanzmap.save(MeanzoneMap)
    arcpy.Delete_management("in-memory")
    gc.collect()

    # NEED TO ADD IN MEAN MASK
    print("creating product 9...")
    proj9 = (Raster(SWEzoneMap) / Raster(MeanzoneMap)) * 100
    proj9.save(product9)

    # creating banded watershed mean and swe
    # NEED TO ADD IN MEAN MASK
    tswebzmap = ZonalStatistics(band_zones, case_field_band, product8, statisticType, "DATA")
    tmeanbzmap = ZonalStatistics(band_zones, case_field_band, meanMapMask, statisticType, "DATA")
    tswebzmap.save(SWEbandzoneMap)
    tmeanbzmap.save(MeanBandZoneMap)

    # NEED TO ADD IN MEAN MASK
    print("creating product 10 = " + product10)
    prod10 = (Raster(SWEbandzoneMap) / Raster(MeanBandZoneMap)) * 100
    prod10.save(product10)

    print("created product 11 = HUC 6 percent of average")
    swezmap = ZonalStatistics(HUC6_zones, "name", product8, "MEAN", "DATA")
    meanzmap = ZonalStatistics(HUC6_zones, "name", meanMapMask, "MEAN", "DATA")
    swezmap.save(SWEHuc6Map)
    meanzmap.save(MeanHuc6Map)

    prod11 = (Raster(SWEHuc6Map) / Raster(MeanHuc6Map)) * 100
    prod11.save(product11)

    print("created product 12 = region percent of average")
    swezmap = ZonalStatistics(region_zones, "RegionAll", product8, "MEAN", "DATA")
    meanzmap = ZonalStatistics(region_zones, "RegionAll", meanMapMask, "MEAN", "DATA")
    swezmap.save(SWEregionMap)
    meanzmap.save(MeanRegionMap)

    prod11 = (Raster(SWEregionMap) / Raster(MeanRegionMap)) * 100
    prod11.save(product12)


    print("create anomaly layer table = " + anomTable)
    # NEED TO ADD IN MEAN MASK
    anomt = ZonalStatisticsAsTable(watershed_zones, case_field_wtrshd, product9, anomTable, "DATA", "MEAN")
    arcpy.Delete_management("in-memory")
    gc.collect()

    anombt = ZonalStatisticsAsTable(band_zones, case_field_band, product10, anombandTable, "DATA", "MEAN")
    arcpy.Delete_management("in-memory")
    gc.collect()

    anomh6 = ZonalStatisticsAsTable(HUC6_zones, "name", product11, anomHuc6Table, "DATA", "MEAN")
    arcpy.Delete_management("in-memory")
    gc.collect()

    anomreg = ZonalStatisticsAsTable(region_zones, "RegionAll", product12, anomRegionTable, "DATA", "MEAN")
    arcpy.Delete_management("in-memory")
    gc.collect()

    # NEED TO ADD IN MEAN MASK
    # Sort by bandname and watershed name, 3 tables
    arcpy.Sort_management(anombandTable, anombandTable_save, [[case_field_band, "ASCENDING"]])
    arcpy.Sort_management(anomTable, anomTable_save, [[case_field_wtrshd, "ASCENDING"]])
    arcpy.Sort_management(anomHuc6Table, anomHuc6Table_save, [["name", "ASCENDING"]])
    arcpy.Sort_management(anomRegionTable, anomRegionTable_save, [["RegionAll", "ASCENDING"]])

    # add field for anom
    arcpy.AddField_management(anombandTable_save, "Average", "DOUBLE", "", "", "",
                              "", "NULLABLE", "NON_REQUIRED")
    arcpy.AddField_management(anomTable_save, "Average", "DOUBLE", "", "", "",
                              "", "NULLABLE", "NON_REQUIRED")
    arcpy.AddField_management(anomHuc6Table_save, "Average", "DOUBLE", "", "", "",
                              "", "NULLABLE", "NON_REQUIRED")
    arcpy.AddField_management(anomRegionTable_save, "Average", "DOUBLE", "", "", "",
                              "", "NULLABLE", "NON_REQUIRED")

    # calculate field
    arcpy.CalculateField_management(anombandTable_save, "Average", f"!MEAN!", "PYTHON3")
    arcpy.CalculateField_management(anomTable_save, "Average", f"!MEAN!", "PYTHON3")
    arcpy.CalculateField_management(anomHuc6Table_save, "Average", f"!MEAN!", "PYTHON3")
    arcpy.CalculateField_management(anomRegionTable_save, "Average", f"!MEAN!", "PYTHON3")

    print("Joining sorted tables ... ")
    ## Delete extra fields from tables before joining them
    ## Banded Tables
    arcpy.DeleteField_management(SWEbandtable_save, "ZONE_CODE")
    arcpy.DeleteField_management(anombandTable_save, "ZONE_CODE;COUNT")
    arcpy.DeleteField_management(scabandtable_save,
                                 "ZONE_CODE;COUNT;MIN;MAX;RANGE;MEAN;STD;SUM;VARIETY;MAJORITY;MINORITY;MEDIAN")
    ## Watershed tables
    arcpy.DeleteField_management(SWEtable_save, "ZONE_CODE")
    arcpy.DeleteField_management(anomTable_save, "ZONE_CODE;COUNT")
    arcpy.DeleteField_management(scatable_save,
                                 "ZONE_CODE;COUNT;MIN;MAX;RANGE;MEAN;STD;SUM;VARIETY;MAJORITY;MINORITY;MEDIAN")


    ## Make tables into table views for joins
    arcpy.MakeTableView_management(SWEbandtable_save, SWEbandtableView)
    arcpy.MakeTableView_management(SWEtable_save, SWEtableView)

    arcpy.JoinField_management(SWEtable_save, case_field_wtrshd, scatable_save, case_field_wtrshd, "Percent")
    arcpy.JoinField_management(SWEbandtable_save, case_field_band, scabandtable_save, case_field_band, "Percent")

    print("Making csvs...")
    # wtshd_dbf = gpd.read_file(SWEtableView)
    wtshd_dbf = gpd.read_file(SWEtable_save)
    wtshd_df = pd.DataFrame(wtshd_dbf)
    wtshd_df.to_csv(WtshdTableJoinCSV, index=False)

    band_dbf = gpd.read_file(SWEbandtable_save)
    band_df = pd.DataFrame(band_dbf)
    band_df.to_csv(BandtableJoinCSV, index=False)

    anom_dbf = gpd.read_file(anomTable_save)
    anom_df = pd.DataFrame(anom_dbf)
    anom_df.to_csv(anomWtshdTableCSV, index=False)

    anom_band_dbf = gpd.read_file(anombandTable_save)
    anom_band_df = pd.DataFrame(anom_band_dbf)
    anom_band_df.to_csv(anomBandTableCSV, index=False)

    anom_huc_dbf = gpd.read_file(anomHuc6Table_save)
    anom_huc_df = pd.DataFrame(anom_huc_dbf)
    anom_huc_df.to_csv(anomHUC6TableCSV, index=False)

    anom_region_dbf = gpd.read_file(anomRegionTable_save)
    anom_region_df = pd.DataFrame(anom_region_dbf)
    anom_region_dbf.to_csv(anomRegionTableCSV, index=False)

import arcpy
from arcpy import env
from arcpy.ra import ZonalStatisticsAsTable
from arcpy.sa import *
import pandas as pd
import geopandas as gpd


def tables_and_layers_SNM(year, rundate, mean_date, WW_model_run, SNM_results_workspace, watershed_zones, band_zones, region_zones,
                          case_field_wtrshd, case_field_band, watermask, glacier_mask, domain_mask, run_type, snap_raster, WW_results_workspace,
                          Difference, bias_model_run=None, prev_report_date=None, previous_model_run=None):
    # create directory
    prevRepWorkspace = SNM_results_workspace + f"{prev_report_date}_results_ET/{previous_model_run}/"
    where_clause = """"POLY_AREA" > 100"""
    part_area = "100 SquareKilometers"
    ProjOut_UTM = arcpy.SpatialReference(26911)

    # raster paths
    # watershed_zones = "M:/SWE/WestWide/data/hydro/SNM/dwr_basins_geoSort_albn83.tif"
    # band_zones = "M:/SWE/WestWide/data/hydro/SNM/dwr_band_basins_geoSort_albn83_delin.tif"
    # region = "M:/SWE/WestWide/data/hydro/SNM/dwr_regions_albn83.tif"
    SNM_clipbox_alb = "M:/SWE/WestWide/data/boundaries/Domains/DomainShapefiles/WW_SNM_Clipbox_albn83.shp"
    # case_field_wtrshd = "SrtName"
    # case_field_band = "SrtNmeBand"

    if run_type == "Normal":
        arcpy.CreateFolder_management(SNM_results_workspace + f"/{rundate}_results_ET/", WW_model_run)
        outWorkspace = SNM_results_workspace + f"/{rundate}_results_ET/" + WW_model_run + "/"
        print("model run workspace created")

    if run_type == "Vetting":
        outWorkspace = SNM_results_workspace + f"/{rundate}_results_ET/" + WW_model_run + "/"

    if run_type == "Bias":
        outWorkspace = SNM_results_workspace + f"/{rundate}_results_ET/" + bias_model_run + "/"
    SNM_results_workspace + f"/{rundate}_results_ET/"

    ## project and clip SNODAS
    SNODASWorkspace = SNM_results_workspace + f"/{rundate}_results_ET/" + "SNODAS/"
    ClipSNODAS = SNODASWorkspace + "SWE_" + rundate + "_Cp_m_albn83_clp.tif"
    SWE_Diff = outWorkspace + "SNODAS_Regress_" + rundate + ".tif"
    SWE_both = outWorkspace + f"SWE_{rundate}_both.tif"

    meanMask = outWorkspace + f"{mean_date}_mean_msk.tif"
    MODSCAG_tif_plus = f"H:/WestUS_Data/Rittger_data/fsca_v2025.0.1_ops/NRT_FSCA_WW_N83/{year}/{rundate}.tif"
    MODSCAG_tif_plus_proj_WW = WW_results_workspace + f"{rundate}_results_ET/{WW_model_run}/" + f"fSCA_{rundate}_albn83.tif"
    MODSCAG_tif_plus_proj = outWorkspace + f"SNM_fSCA_{rundate}_albn83.tif"

    # define snow-no snow layer
    modscag_0_1 = outWorkspace + f"modscag_0_1_{rundate}.tif"
    modscag_per = outWorkspace + f"modscag_per_{rundate}.tif"
    modscag_per_msk = outWorkspace + f"modscag_per_{rundate}_msk.tif"

    # define snow/no snow null layer
    mod_null = outWorkspace + f"modscag_0_1_{rundate}msk_null.tif"
    mod_poly = outWorkspace + f"modscag_0_1_{rundate}msk_null_poly.shp"
    ### ASK LEANNE ABOUT UTM
    mod_poly_utm = outWorkspace + f"modscag_0_1_{rundate}_msk_null_poly_utm.shp"

    snowPolySel = outWorkspace + f"modscag_{rundate}_snowline_Sel.shp"
    snowPolyElim = outWorkspace + f"modscag_{rundate}_snowline_Sel_elim.shp"

    # define snow pillow gpkg
    meanMap_proj_WW = WW_results_workspace + f"{rundate}_results_ET/{WW_model_run}/" + f"WW_{mean_date}_mean_msk_albn83.tif"
    meanMap_proj = outWorkspace + f"SNM_{mean_date}_mean_albn83.tif"
    meanMapMask = outWorkspace + f"SNM_{mean_date}_mean_msk_albn83.tif"
    lastRast = prevRepWorkspace + f"p8_{prev_report_date}_noneg.tif"
    DiffRaster = outWorkspace + f"Diff_{rundate}_{prev_report_date}.tif"

    ## define rasters
    WW_product8 = WW_results_workspace + f"{rundate}_results_ET/{WW_model_run}/" + f"p8_{rundate}_noneg.tif"
    rcn_glacMask_WW = WW_results_workspace + f"{rundate}_results_ET/{WW_model_run}/" + f"WW_{rundate}_phvrcn_mos_masked.tif"
    rcn_raw_proj_WW = WW_results_workspace + f"{rundate}_results_ET/{WW_model_run}/" + f"WW_{rundate}_phvrcn_albn83.tif"
    WW_p8_SNM = outWorkspace + f"WW_p8_{rundate}_noneg.tif"
    rcnFinal = outWorkspace + f"phvrcn_{rundate}_final.tif"
    product7 = outWorkspace + f"p7_{rundate}.tif"
    product7_noFsca = outWorkspace + f"p7_{rundate}_nofsca.tif"
    prod7msk = outWorkspace + f"p7_{rundate}_msk.tif"
    product8 = outWorkspace + f"p8_{rundate}_noneg.tif"
    prod8msk = outWorkspace + f"p8_{rundate}_noneg_msk.tif"
    product9 = outWorkspace + f"p9_{rundate}.tif"
    prod9msk = outWorkspace + f"p9_{rundate}_msk.tif"
    prod10msk = outWorkspace + f"p10_{rundate}_msk.tif"
    product10 = outWorkspace + f"p10_{rundate}.tif"
    rcn_raw_proj = outWorkspace + f"SNM_{rundate}_phvrcn_albn83.tif"

    # output Tables
    SWEbandtable = outWorkspace + f"{rundate}band_swe_table.dbf"
    SWEtable = outWorkspace + f"{rundate}swe_table.dbf"
    SWEbandtable100 = outWorkspace + f"{rundate}swe_table_100.dbf"
    SWEbandtable_save = outWorkspace + f"{rundate}band_swe_table_save.dbf"
    SWEtable_save = outWorkspace + f"{rundate}swe_table_save.dbf"
    SWEbandtable100_save = outWorkspace + f"{rundate}swe_table_100_save.dbf"

    # anomoly tables
    anombandTable = outWorkspace + f"{rundate}band_anom_table.dbf"
    anomTable = outWorkspace + f"{rundate}anom_table.dbf"
    meanTable = outWorkspace + f"{rundate}mean_table.dbf"
    anombandTable_save = outWorkspace + f"{rundate}band_anom_table_save.dbf"
    anomTable_save = outWorkspace + f"{rundate}anom_table_save.dbf"
    meanTable_save = outWorkspace + f"{rundate}mean_table_save.dbf"

    # region tables
    anomRegionTable = outWorkspace + f"{rundate}anomRegion_table.dbf"
    anomRegionTable_save = outWorkspace + f"{rundate}anomRegion_table_save.dbf"

    # Modscag 0/1 tables and % tables
    scabandtable = outWorkspace + f"{rundate}band_sca_table.dbf"
    scatable = outWorkspace + f"{rundate}sca_table.dbf"
    scabandtable_save = outWorkspace + f"{rundate}band_sca_table_save.dbf"
    scatable_save = outWorkspace + f"{rundate}_sca_table_save.dbf"
    perbandtable = outWorkspace + f"{rundate}band_per_table.dbf"
    pertable = outWorkspace + f"{rundate}_per_table.dbf"

    # create tempoary view for join
    SWEbandtableView = outWorkspace + f"{rundate}band_swe_table_view.dbf"
    SWEtableView = outWorkspace + f"{rundate}swe_table_view.dbf"

    # create joined tables
    BandtableJoin = outWorkspace + f"{rundate}band_table.dbf"
    WtshdTableJoin = outWorkspace + f"{rundate}Wtshd_table.dbf"

    # Anomaly maps
    anomMap = outWorkspace + f"{rundate}_anom.tif"
    anom0_100map = outWorkspace + f"{rundate}anom0_200.tif"
    anom0_100msk = outWorkspace + f"{rundate}anom0_200_msk.tif"

    # SWE maps
    SWEzoneMap = outWorkspace + f"{rundate}_swe_wshd.tif"
    MeanzoneMap = outWorkspace + f"{rundate}_mean_wshd.tif"
    SWEbandzoneMap = outWorkspace + f"{rundate}_swe_band_wshd.tif"
    MeanBandZoneMap = outWorkspace + f"{rundate}_mean_band_wshd.tif"

    # mean layer masked for use in creating anomly map
    anomMask = outWorkspace + f"{rundate}_anom_mask.tif"

    # statistic
    statisticType = "MEAN"

    # final output csv tables
    WtshdTableJoinCSV = outWorkspace + f"{rundate}Wtshd_table.csv"
    BandtableJoinCSV = outWorkspace + f"{rundate}band_table.csv"
    anomRegionTableCSV = outWorkspace + f"{rundate}anomRegion_table.csv"
    Band100TableCSV = outWorkspace + f"{rundate}band_table_100.csv"
    SCATableJoinCSV = outWorkspace + f"{rundate}sca_Wtshd_table.csv"
    BandSCAtableJoinCSV = outWorkspace + f"{rundate}sca_band_table.csv"
    anomWtshdTableCSV = outWorkspace + f"{rundate}anomWtshd_table.csv"
    anomBandTableCSV = outWorkspace + f"{rundate}anomBand_table.csv"
    print("file paths established")

    # start with envirnonment settings
    arcpy.env.snapRaster = snap_raster
    arcpy.env.cellSize = snap_raster
    arcpy.env.extent = snap_raster
    # arcpy.env.snapRaster = snapRaster_albn83
    print("starting")
    # domain model runs
    if run_type == "Normal":
        print('clip and bring over the other files')
        # set snap raster
        # copy over files
        arcpy.CopyRaster_management(WW_product8, WW_p8_SNM)

        outSNM_phvrcn = ExtractByMask(WW_p8_SNM, SNM_clipbox_alb, "INSIDE")
        print('clipped')
        outSNM_phvrcn.save(product8)
        print("saved SWE")

    # fsca
    outSNM_fsca = ExtractByMask(MODSCAG_tif_plus_proj_WW, SNM_clipbox_alb, "INSIDE")
    outSNM_fsca.save(MODSCAG_tif_plus_proj)
    print("saved fSCA")
    # mean
    outSNM_mean = ExtractByMask(meanMap_proj_WW, SNM_clipbox_alb, "INSIDE")
    outSNM_mean.save(meanMap_proj)
    print("clipped domain")

    ########################
    print(f"Processing begins...")
    # create snow/no snow layer
    mod_01 = Con((Raster(MODSCAG_tif_plus_proj) < 101) & (Raster(MODSCAG_tif_plus_proj) > 0),
                 1, 0)
    mod_01_Wtrmask = mod_01 * Raster(watermask)
    mod_01_AllMaks = mod_01_Wtrmask * Raster(glacier_mask)
    mod_01_AllMaks.save(modscag_0_1)
    print(f"fSCA mask tif saved")

    # create fSCA percent layer
    Mod_per = (Float(SetNull(Raster(MODSCAG_tif_plus_proj) > 100, Raster(MODSCAG_tif_plus_proj))) / 100)
    Mod_per.save(modscag_per)
    print(f"fSCA percent layer saved")

    # create fsca percent layer ASK LEANNE, WHAT'S THE DIFFERENT BETWEEN LAKES MASK AND WATER MASK
    mod_01_mask = Con(Raster(modscag_per) > 0.0001, 1, 0)
    mod_per_msk = Raster(watermask) * mod_01_mask
    mod_per_Allmsk = Raster(glacier_mask) * mod_per_msk
    mod_per_Allmsk.save(modscag_per_msk)
    print("fSCA percent layer created")

    # ASK LEANNE, WHAT'S THE DIFFERENCE BETWEEN TIF MASK AND WATER MASK
    print(f"Creating snowline shapefile: {snowPolyElim}")
    mod_01_mask = mod_01 + Raster(watermask)
    mod_01_mask_glacier = mod_01 + Raster(glacier_mask)
    mod_01_msk_null = SetNull(mod_01_mask_glacier == 0, mod_01_mask_glacier)
    mod_01_msk_null.save(mod_null)

    # Convert raster to polygon
    arcpy.RasterToPolygon_conversion(mod_null, mod_poly, "NO_SIMPLIFY", "Value")
    arcpy.Project_management(mod_poly, mod_poly_utm, ProjOut_UTM)
    arcpy.AddGeometryAttributes_management(mod_poly_utm, "AREA", "", "SQUARE_KILOMETERS", "")
    arcpy.Select_analysis(mod_poly_utm, snowPolySel, where_clause)
    arcpy.EliminatePolygonPart_management(snowPolySel, snowPolyElim, "AREA", part_area, "0", "CONTAINED_ONLY")

    print(f"creating masked SWE product")
    if Difference == "Y":
        Diff_rgrs = Raster(product8) - Raster(lastRast)
        Diff_rgrs.save(DiffRaster)

    # create difference with SNODAS
    Diff_SNODAS = Raster(ClipSNODAS) - Raster(product8)
    Diff_SNODAS.save(SWE_Diff)

    # create overlap layers
    print("Creating SNODAS and Regress diff layers ...")
    SNODAS1000 = Con(Raster(ClipSNODAS) > 0.001, 1000, 0)
    RSWE100 = Con(Raster(product8) > 0.001, 100, 0)

    ## Then add them together to create a layer showing where they overlap and
    ## where they're different
    SWEboth = SNODAS1000 + RSWE100

    ## Then save both layers
    SWEboth.save(SWE_both)

    print("creating mean mask")
    # MeanMapMsk = Raster(meanMap_proj) * Raster(watermask)
    # MeanMapALlMsk = MeanMapMsk * Raster(glacier_mask)
    MeanMapALlMsk_2 = Raster(meanMap_proj) * Raster(domain_mask)
    MeanMapALlMsk_2.save(meanMapMask)

    # Create GT 0 mean blended swe and make mask
    con01 = Con(Raster(meanMap_proj) > 0.00, 1, 0)
    con01.save(anomMask)
    #
    # # make anomoly mask
    AnomProd = (Raster(product8) / Raster(meanMap_proj)) * 100
    AnomProd.save(anomMap)
    print(f"anomaly map made")

    # # make noneg anomoly map ## ASK LEANNE, DOES THIS NEED TO BE ADJUSTED?
    connoeg = Con(Raster(anomMap) > 200, 200, Raster(anomMap))
    connoeg.save(anom0_100map)

    # # mask with watermaks
    anomnoneg = connoeg * Raster(watermask)
    anomnoneg_Mask = anomnoneg * Raster(glacier_mask)
    prod8Msk = anomnoneg_Mask / Raster(domain_mask)
    prod8Msk.save(anom0_100msk)

    arcpy.Delete_management("in-memory")
    gc.collect()
    ZonalStatisticsAsTable(band_zones, case_field_band, product8, SWEbandtable, "DATA",
                                          "MEAN")
    arcpy.Delete_management("in-memory")
    gc.collect()

    ZonalStatisticsAsTable(watershed_zones, case_field_wtrshd, product8, SWEtable, "DATA",
                                         "MEAN")
    arcpy.Delete_management("in-memory")
    gc.collect()

    ZonalStatisticsAsTable(band_zones, case_field_band, modscag_per, scabandtable, "DATA",
                                        "ALL")
    arcpy.Delete_management("in-memory")
    gc.collect()

    ZonalStatisticsAsTable(watershed_zones, case_field_wtrshd, modscag_per, scatable, "DATA",
                                         "ALL")
    arcpy.Delete_management("in-memory")
    gc.collect()


    # outBandTable = ZonalStatisticsAsTable(band_zones, case_field_band, product8, SWEbandtable, "DATA",
    #                                       "MEAN")
    # outSWETable = ZonalStatisticsAsTable(watershed_zones, case_field_wtrshd, product8, SWEtable, "DATA",
    #                                      "MEAN")
    # outSCABand = ZonalStatisticsAsTable(band_zones, case_field_band, modscag_per, scabandtable, "DATA",
    #                                     "ALL")
    # outSCAWtshd = ZonalStatisticsAsTable(watershed_zones, case_field_wtrshd, modscag_per, scatable, "DATA",
    #                                      "ALL")
    arcpy.AddField_management(SWEbandtable, "SWE_IN", "DOUBLE", "", "", "",
                              "", "NULLABLE", "NON_REQUIRED")
    arcpy.AddField_management(SWEtable, "SWE_IN", "DOUBLE", "#", "#", "#",
                              "#", "NULLABLE", "NON_REQUIRED", "#")
    arcpy.AddField_management(SWEbandtable, "AREA_MI2", "DOUBLE", "#", "#", "#",
                              "#", "NULLABLE", "NON_REQUIRED", "#")
    arcpy.AddField_management(SWEtable, "AREA_MI2", "DOUBLE", "#", "#", "#",
                              "#", "NULLABLE", "NON_REQUIRED", "#")
    arcpy.AddField_management(SWEbandtable, "VOL_M3", "DOUBLE", "#", "#", "#",
                              "#", "NULLABLE", "NON_REQUIRED", "#")
    arcpy.AddField_management(SWEtable, "VOL_M3", "DOUBLE", "#", "#", "#",
                              "#", "NULLABLE", "NON_REQUIRED", "#")
    arcpy.AddField_management(SWEbandtable, "VOL_AF", "DOUBLE", "#", "#", "#",
                              "#", "NULLABLE", "NON_REQUIRED", "#")
    arcpy.AddField_management(SWEtable, "VOL_AF", "DOUBLE", "#", "#", "#",
                              "#", "NULLABLE", "NON_REQUIRED", "#")

    # calculate fields
    arcpy.CalculateField_management(SWEbandtable, "SWE_IN", "!MEAN! * 39.370079", "PYTHON")
    arcpy.CalculateField_management(SWEtable, "SWE_IN", "!MEAN! * 39.370079", "PYTHON")

    # Calculate area in sq miles
    arcpy.CalculateField_management(SWEbandtable, "AREA_MI2", "0.00000038610216 * !AREA!", "PYTHON")
    arcpy.CalculateField_management(SWEtable, "AREA_MI2", "0.00000038610216 * !AREA!", "PYTHON")

    # Calculate volume in cubic meters
    arcpy.CalculateField_management(SWEbandtable, "VOL_M3", "!MEAN! * !AREA!", "PYTHON")
    arcpy.CalculateField_management(SWEtable, "VOL_M3", "!MEAN! * !AREA!", "PYTHON")

    # Calculate volume in acre feet
    arcpy.CalculateField_management(SWEbandtable, "VOL_AF", "!VOL_M3! * 0.000810714", "PYTHON")
    arcpy.CalculateField_management(SWEtable, "VOL_AF", "!VOL_M3! * 0.000810714", "PYTHON")

    ### Sort by bandname and watershed name, 2 tables
    arcpy.Sort_management(SWEbandtable, SWEbandtable_save, [[case_field_band, "ASCENDING"]])
    arcpy.Sort_management(SWEtable, SWEtable_save, [[case_field_wtrshd, "ASCENDING"]])

    ## work on SCA tables
    arcpy.AddField_management(scabandtable, "Percent", "DOUBLE", "", "", "",
                              "", "NULLABLE", "NON_REQUIRED")
    arcpy.AddField_management(scatable, "Percent", "DOUBLE", "", "", "",
                              "", "NULLABLE", "NON_REQUIRED")
    # calculate percent
    arcpy.CalculateField_management(scabandtable, "Percent", "( !SUM! / !COUNT! ) * 100", "PYTHON", "")
    arcpy.CalculateField_management(scatable, "Percent", "( !SUM! / !COUNT! ) * 100", "PYTHON", "")

    # sort
    arcpy.Sort_management(scabandtable, scabandtable_save, [[case_field_band, "ASCENDING"]])
    arcpy.Sort_management(scatable, scatable_save, [[case_field_wtrshd, "ASCENDING"]])

    print("Create SWE and mean zonal maps...")
    # NEED TO ADD IN MEAN MASK
    swezmap = ZonalStatistics(watershed_zones, case_field_wtrshd, product8, "MEAN", "DATA")
    meanzmap = ZonalStatistics(watershed_zones, case_field_wtrshd, meanMapMask, "MEAN", "DATA")
    swezmap.save(SWEzoneMap)
    meanzmap.save(MeanzoneMap)

    # NEED TO ADD IN MEAN MASK
    print("creating product 9...")
    proj9 = (Raster(SWEzoneMap) / Raster(MeanzoneMap)) * 100
    proj9.save(product9)

    # # make anomoly mask
    prod8Msk = (Raster(product8) / Raster(domain_mask))
    prod8Msk.save(prod8msk)

    # creating banded watershed mean and swe
    tswebzmap = ZonalStatistics(band_zones, case_field_band, product8, statisticType, "DATA")
    tmeanbzmap = ZonalStatistics(band_zones, case_field_band, meanMapMask, statisticType, "DATA")
    tswebzmap.save(SWEbandzoneMap)
    tmeanbzmap.save(MeanBandZoneMap)

    print("creating product 10 = " + product10)
    prod10 = (Raster(SWEbandzoneMap) / Raster(MeanBandZoneMap)) * 100
    prod10.save(product10)

    # # # make anomoly mask
    # prod7Msk = (Raster(product7) / Raster(domain_mask)) * 100
    # prod7Msk.save(prod7msk)
    # print(f"anomaly map made")

    print("create anomaly layer table = " + anomTable)
    anomt = ZonalStatisticsAsTable(watershed_zones, case_field_wtrshd, product9, anomTable, "DATA", "MEAN")
    anomRt = ZonalStatisticsAsTable(region_zones, "Regions", product9, anomRegionTable, "DATA", "MEAN")
    anombt = ZonalStatisticsAsTable(band_zones, case_field_band, product10, anombandTable, "DATA", "MEAN")
    arcpy.Delete_management("in-memory")
    gc.collect()
    # Sort by bandname and watershed name, 3 tables
    arcpy.Sort_management(anombandTable, anombandTable_save, [[case_field_band, "ASCENDING"]])
    arcpy.Sort_management(anomTable, anomTable_save, [[case_field_wtrshd, "ASCENDING"]])
    arcpy.Delete_management("in-memory")
    gc.collect()
    # add field for anom
    arcpy.AddField_management(anombandTable_save, "Average", "DOUBLE", "", "", "",
                              "", "NULLABLE", "NON_REQUIRED")
    arcpy.AddField_management(anomTable_save, "Average", "DOUBLE", "", "", "",
                              "", "NULLABLE", "NON_REQUIRED")
    arcpy.Delete_management("in-memory")
    gc.collect()

    # calculate field
    arcpy.CalculateField_management(anombandTable_save, "Average", f"!MEAN!", "PYTHON3")
    arcpy.CalculateField_management(anomTable_save, "Average", f"!MEAN!", "PYTHON3")

    print("Joining sorted tables ... ")
    ## Delete extra fields from tables before joining them
    ## Banded Tables
    arcpy.DeleteField_management(SWEbandtable_save, "ZONE_CODE")
    arcpy.DeleteField_management(anombandTable_save, "ZONE_CODE;COUNT")
    arcpy.DeleteField_management(scabandtable_save,
                                 "ZONE_CODE;COUNT;MIN;MAX;RANGE;MEAN;STD;SUM;VARIETY;MAJORITY;MINORITY;MEDIAN")
    ## Watershed tables
    arcpy.DeleteField_management(SWEtable_save, "ZONE_CODE")
    arcpy.DeleteField_management(anomTable_save, "ZONE_CODE;COUNT")
    arcpy.DeleteField_management(scatable_save,
                                 "ZONE_CODE;COUNT;MIN;MAX;RANGE;MEAN;STD;SUM;VARIETY;MAJORITY;MINORITY;MEDIAN")

    ## Make tables into table views for joins
    arcpy.MakeTableView_management(SWEbandtable_save, SWEbandtableView)
    arcpy.MakeTableView_management(SWEtable_save, SWEtableView)

    arcpy.JoinField_management(SWEtable_save, case_field_wtrshd, scatable_save, case_field_wtrshd, "Percent")
    arcpy.JoinField_management(SWEbandtable_save, case_field_band, scabandtable_save, case_field_band, "Percent")

    print("Making csvs...")
    # wtshd_dbf = gpd.read_file(SWEtableView)
    wtshd_dbf = gpd.read_file(SWEtable_save)
    wtshd_df = pd.DataFrame(wtshd_dbf)
    wtshd_df.to_csv(WtshdTableJoinCSV, index=False)

    band_dbf = gpd.read_file(SWEbandtable_save)
    band_df = pd.DataFrame(band_dbf)
    band_df.to_csv(BandtableJoinCSV, index=False)

    anom_dbf = gpd.read_file(anomTable_save)
    anom_df = pd.DataFrame(anom_dbf)
    anom_df.to_csv(anomWtshdTableCSV, index=False)

    anom_band_dbf = gpd.read_file(anombandTable_save)
    anom_band_df = pd.DataFrame(anom_band_dbf)
    anom_band_df.to_csv(anomBandTableCSV, index=False)

import os, sys, string
import arcinfo
import arcpy
import geopandas as gpd
import pandas as pd
from arcpy import env
import gzip
import shutil
from arcpy.sa import *


def clear_arcpy_locks():
    """Clear all ArcPy file locks and reset environment"""
    import gc
    import time

    # Clear all environment settings
    arcpy.ClearEnvironment("workspace")
    arcpy.ClearEnvironment("scratchWorkspace")
    arcpy.ClearEnvironment("extent")
    arcpy.ClearEnvironment("snapRaster")
    arcpy.ClearEnvironment("cellSize")
    arcpy.ClearEnvironment("mask")
    arcpy.ClearEnvironment("outputCoordinateSystem")

    # Reset workspace to None
    arcpy.env.workspace = None
    arcpy.env.scratchWorkspace = None
    arcpy.env.extent = None
    arcpy.env.snapRaster = None
    arcpy.env.cellSize = None

    # clear in-memory
    arcpy.Delete_management("in_memory")

    # Force Python garbage collection multiple times
    for _ in range(3):
        gc.collect()

    # Small delay to allow file system to release locks
    time.sleep(3)

    # Reset geoprocessing environment
    arcpy.ResetEnvironments()

def SNODAS_Processing(report_date, RunName, NOHRSC_workspace, results_workspace,
                         projin, projout, Cellsize, snapRaster, watermask, glacierMask, band_zones, watershed_zones, unzip_SNODAS):
    SNODASWorkspace = NOHRSC_workspace + f"SNODAS_{report_date}/"
    SWEWorkspaceBase = results_workspace + f"{report_date}_results_ET/{RunName}/"
    resultsWorkspace = results_workspace +f"{report_date}_results_ET/"
    SWEWorkspace = results_workspace + f"{report_date}_results_ET/SNODAS/"

    ## Set regression SWE image for the same date
    # RegressSWE = SWEWorkspaceBase + f"p8_{report_date}_noneg.tif"

    ##### Set automatic local variables
    arcpy.CreateFolder_management(resultsWorkspace, "SNODAS")
    # product8 = SWEWorkspace + f"p8_{report_date}_noneg.tif"
    # arcpy.CopyRaster_management(RegressSWE, product8)

    OutSNODAS = f"SWE_{report_date}.tif"
    OutSNODASplus = SNODASWorkspace + OutSNODAS
    FloatSNODAS = SWEWorkspace + f"SWE_{report_date}_Cp.tif"
    MeterSNODAS = SWEWorkspace + f"SWE_{report_date}_Cp_m.tif"
    ProjSNODAS = SWEWorkspace + f"SWE_{report_date}_Cp_m_albn83.tif"
    ClipSNODAS = SWEWorkspace + f"SWE_{report_date}_Cp_m_albn83_clp.tif"
    SCA_SNODAS = SWEWorkspace + f"SWE_{report_date}_fSCA.tif"

    SWEbandtable = SWEWorkspace + f"{report_date}_band_SNODAS_swe_table.dbf"
    SWEtable = SWEWorkspace + f"{report_date}_SNODAS_swe_table.dbf"
    SWEbandtable_save = SWEWorkspace + f"{report_date}_band_SNODAS_swe_table_save.dbf"
    SWEtable_save = SWEWorkspace + f"{report_date}_SNODAS_swe_table_save.dbf"
    SWEbandtableCSV = SWEWorkspace + f"{report_date}_band_SNODAS_swe_table.csv"
    SWEtableCSV = SWEWorkspace +f"{report_date}_SNODAS_swe_table.csv"

    ###### End of setting up variables


    # unzip and move HDR file
    if unzip_SNODAS == "Y":
        arcpy.env.workspace = SNODASWorkspace
        gz_datFile = SNODASWorkspace + f"us_ssmv11034tS__T0001TTNATS{report_date}05HP001.dat.gz"
        gz_unzipDat = SNODASWorkspace + f"us_ssmv11034tS__T0001TTNATS{report_date}05HP001.dat"

        print("\nUnzipping SNODAS file...")
        with gzip.open(gz_datFile, "rb") as f_in:
            with open(gz_unzipDat, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        print("SNODAS file unzipped")

        hdrSNODAS = NOHRSC_workspace + f"us_ssmv11034tS_masked.hdr"
        hdrSNODAS_copy = f"us_ssmv11034tS__T0001TTNATS{report_date}05HP001.hdr"
        shutil.copy(hdrSNODAS, os.path.join(SNODASWorkspace, hdrSNODAS_copy))
        print("HDR file moved")

        ## Add .dat file to file list
        dats = arcpy.ListFiles("*.dat")

        ## Process all applicable .dat files
        for dat in dats:
            ## Create geoTIF file from .dat file
            OutTif = dat[0:-4] + ".tif"
            print("Creating: " + OutSNODAS)

            ## Check to see if geoTIF file exists, if not create it.
            if arcpy.Exists(OutTif):
                print(" ")
            else:
                ## Create a geotif from the .dat file
                arcpy.RasterToOtherFormat_conversion(dat, SNODASWorkspace, "TIFF")

            # define projection
            arcpy.DefineProjection_management(OutTif, projin)

            ## Get rid of -9999 values and change to NODATA values
            NoData = SetNull(Raster(OutTif) == -9999, OutTif)
            NoData.save(OutSNODASplus)

        arcpy.env.workspace = None
    clear_arcpy_locks()
    arcpy.env.workspace = None
    arcpy.ClearEnvironment("workspace")
    arcpy.ClearEnvironment("extent")

    import gc
    gc.collect()
    import time
    time.sleep(2)
    clear_arcpy_locks()
    # Verify source file exists
    print(f"Checking source file: {OutSNODASplus}")
    if not arcpy.Exists(OutSNODASplus):
        raise FileNotFoundError(f"ERROR: Source SNODAS file not found: {OutSNODASplus}")

    ## Copy to floating point raster
    # print(f"Copying to: {FloatSNODAS}")
    ## Copy to floating point raster
    if os.path.exists(FloatSNODAS):
        print(f"{FloatSNODAS} exists")
    if not os.path.exists(FloatSNODAS):
        print(f"Copying to: {FloatSNODAS}")
        arcpy.CopyRaster_management(OutSNODASplus, FloatSNODAS)
    # arcpy.CopyRaster_management(OutSNODASplus, FloatSNODAS)
    clear_arcpy_locks()
    # Verify source file exists
    print(f"Checking source file: {OutSNODASplus}")
    if not arcpy.Exists(OutSNODASplus):
        raise FileNotFoundError(f"ERROR: Source SNODAS file not found: {OutSNODASplus}")

    clear_arcpy_locks()
    ## Copy to floating point raster
    # print(FloatSNODAS)
    # # arcpy.CopyRaster_management(OutSNODASplus, FloatSNODAS, "", "", "-2147483648", "NONE", "NONE", "32_BIT_FLOAT",
    # #                             "NONE", "NONE")
    # arcpy.CopyRaster_management(OutSNODASplus, FloatSNODAS)

    print("Creating SWE in meters ...")
    clear_arcpy_locks()
    ## Divide by 1000 to get value in meters not mm
    SWEm = Raster(FloatSNODAS) / 1000
    SWEm.save(MeterSNODAS)

    print("Projecting and snapping to regression SWE ...")

    ## Define projection again b/c arcpy can't deal
    arcpy.DefineProjection_management(MeterSNODAS, projin)

    ## Project to WGS84, match to UCRB domain cellsize, extent and snapraster
    arcpy.env.snapRaster = snapRaster
    arcpy.env.extent = snapRaster
    arcpy.env.cellSize = snapRaster

    arcpy.ProjectRaster_management(MeterSNODAS, ProjSNODAS, projout, "NEAREST", Cellsize,
                                   "", "", projin)

    # set extent and apply masks
    # arcpy.env.extent = snapRaster
    SNODASwatMsk = Raster(ProjSNODAS) * Raster(watermask)
    SNODASallMsk = SNODASwatMsk * Raster(glacierMask)

    SNODASmsk = ExtractByMask(ProjSNODAS, snapRaster, "INSIDE")
    SNODASmsk.save(ClipSNODAS)
    clear_arcpy_locks()
    ## If test run previously then SCA_SNODAS will exist, delete and then create
    if arcpy.Exists(SCA_SNODAS):
        arcpy.Delete_management(SCA_SNODAS, "#")
        SNODASfSCA = Con(SNODASallMsk > .001, 100, 0)
        SNODASfSCA.save(SCA_SNODAS)
    ## Else if this is a test run create it
    else:
        SNODASfSCA = Con(SNODASallMsk > .001, 100, 0)
        SNODASfSCA.save(SCA_SNODAS)

    clear_arcpy_locks()
    # Do zonal stats for real time swe layer table
    print("creating zonal stats for SNODAS swe = " + SWEtable)
    ZonalStatisticsAsTable(band_zones, "SrtNmeBand", ClipSNODAS, SWEbandtable, "DATA", "MEAN")
    ZonalStatisticsAsTable(watershed_zones, "SrtName", ClipSNODAS, SWEtable, "DATA", "MEAN")
    arcpy.Delete_management("in-memory")
    gc.collect()
    clear_arcpy_locks()
    # Add SWE in inches fields to 2 tables above
    arcpy.AddField_management(SWEbandtable, "SWE_IN", "FLOAT", "#", "#", "#",
                              "#", "NULLABLE", "NON_REQUIRED", "#")
    arcpy.AddField_management(SWEtable, "SWE_IN", "FLOAT", "#", "#", "#",
                              "#", "NULLABLE", "NON_REQUIRED", "#")
    arcpy.Delete_management("in-memory")
    gc.collect()

    # Calculate SWE in inches from meters
    arcpy.CalculateField_management(SWEbandtable, "SWE_IN", "!MEAN! * 39.370079", "PYTHON")
    arcpy.CalculateField_management(SWEtable, "SWE_IN", "!MEAN! * 39.370079", "PYTHON")
    arcpy.Delete_management("in-memory")
    gc.collect()

    # Sort by bandname and watershed name, 2 tables
    arcpy.Sort_management(SWEbandtable, SWEbandtable_save, [["SrtNmeBand", "ASCENDING"]])
    arcpy.Sort_management(SWEtable, SWEtable_save, [["SrtName", "ASCENDING"]])
    arcpy.Delete_management("in-memory")
    gc.collect()

    # print("Creating SNODAS and Regress diff layers ...")
    # SNODAS1000 = Con(Raster(ClipSNODAS) > 0.001, 1000, 0)
    # RSWE100 = Con(Raster(RegressSWE) > 0.001, 100, 0)
    #
    # ## Then add them together to create a layer showing where they overlap and
    # ## where they're different
    # SWEboth = SNODAS1000 + RSWE100
    #
    # ## Then save both layers
    # SWEboth.save(SWE_both)

    print("Creating CSV tables ...")

    snodas_wtshd_dbf = gpd.read_file(SWEtable_save)
    snodas_wtshd_df = pd.DataFrame(snodas_wtshd_dbf)
    snodas_wtshd_df.to_csv(SWEtableCSV, index=False)
    arcpy.Delete_management("in-memory")
    gc.collect()

    snodas_band_dbf = gpd.read_file(SWEbandtable_save)
    snodas_band_df = pd.DataFrame(snodas_band_dbf)
    snodas_band_df.to_csv(SWEbandtableCSV, index=False)

    arcpy.env.workspace = None

# sensors section
import os
import shutil
import geopandas as gpd

def geopackage_to_shapefile(report_date, pillow_date, model_run, user, domainList, model_workspace, results_workspace):
    for domain in domainList:
        model_workspace_domain = f"{model_workspace}/{domain}/{user}/StationSWERegressionV2/data/outputs/{model_run}/"
        pillow_gpkg = model_workspace_domain + f"{domain}_pillow-{pillow_date}.gpkg"
        snowPillow = results_workspace + f"{report_date}_sensors_{domain}_corrupt.shp"
        pillow_copy = results_workspace + f"{domain}_pillow-{pillow_date}.gpkg"
        pillow_copy_rnme = results_workspace + f"pillow-{pillow_date}_{domain}.gpkg"
        pillowGPKG = results_workspace + f"{domain}_pillow-{pillow_date}.gpkg/pillow_data"

        # # Copy the .gpkg file to the new folder
        shutil.copy2(pillow_gpkg, results_workspace)

        # Path to the copied .gpkg file
        new_gpkg_path = os.path.join(results_workspace, f"{domain}_pillow-{pillow_date}.gpkg")

        # Read the .gpkg file (you can list the layers or specify the one you want)
        gdf = gpd.read_file(new_gpkg_path)
        gdf = gdf.rename(columns={
            "nwbDistance": "nwbDist",
            "regionaleastness": "regEast",
            "regionalnorthness": "regNorth",
            "regionalzness": "regZn",
            "northness4km" : "north4km",
            "eastness4km" : "east4km"
        })

        # Specify the path for the shapefile output
        shapefile_output = os.path.join(results_workspace, f"{report_date}_sensors_{domain}.shp")

        # Convert the GeoDataFrame to a shapefile
        gdf.to_file(shapefile_output)

        print(f"Shapefile saved to: {shapefile_output}")

import arcpy
import pandas as pd
import geopandas as gpd
import warnings
import gc

def merge_sort_sensors_surveys(report_date, results_workspace, surveys, difference, watershed_shapefile, case_field_wtrshd,
                               band_shapefile, case_field_band, merge, projOut, projIn=None, domainList=None, domain= None, domain_shapefile=None,
                               prev_report_date=None, prev_results_workspace=None):

    # Set up snow pillow and snow survey shapefiles
    snowPillow_merge = results_workspace + f"{report_date}_sensors_WW_merge.shp"
    snowSurveys = results_workspace + f"{report_date}_surveys.shp"
    snowSurveys_proj = results_workspace + f"{report_date}_surveys_albn83.shp"

    # Create temp view for a join
    snowPillowView = results_workspace + f"{report_date}_sensors_view.dbf"

    # Create joined tables
    snowPillowsJoin = results_workspace + f"{report_date}_sensors_join.dbf"
    calcField = f"{report_date}_sensors.Diff_In"

    # snow and survey file names
    SensorWtshdInt = results_workspace + f"{report_date}_sensors_Wtshd_Intersect.shp"
    SnwSurvWtshdInt = results_workspace + f"{report_date}_surveys_Wtshd_Intersect.shp"
    SensorBandWtshdInt = results_workspace + f"{report_date}_sensors_BandWtshd_Intersect.shp"
    SnwSurvBandWtshdInt = results_workspace + f"{report_date}_surveys_BandWtshd_Intersect.shp"
    SensorWtshdIntStat = f"{SensorWtshdInt[:-4]}_stat.dbf"
    SnwSurvWtshdIntStat = f"{SnwSurvWtshdInt[:-4]}_stat.dbf"
    SensorBandWtshdIntStat = f"{SensorBandWtshdInt[:-4]}_stat.dbf"
    SnwSurvBandWtshdIntStat = f"{SnwSurvBandWtshdInt[:-4]}_stat.dbf"
    SensorBandWtshdIntStat_save = f"{SensorBandWtshdInt[:-4]}_save.dbf"
    SensorWtshdIntStat_save = f"{SensorWtshdIntStat[:-4]}_save.dbf"
    SnwSurvBandWtshdIntStat_save = f"{SnwSurvBandWtshdInt[:-4]}_save.dbf"
    SnwSurvWtshdIntStat_save = f"{SnwSurvWtshdIntStat[:-4]}_save.dbf"

    # final outputs
    SensorWtshdIntStat_CSV = f"{SensorWtshdIntStat[:-4]}.csv"
    SensorBandWtshdIntStat_CSV = f"{SensorBandWtshdInt[:-4]}.csv"
    SnwSurvWtshdIntStat_CSV = f"{SnwSurvWtshdInt[:-4]}.csv"
    SnwSurvBandWtshdIntStat_CSV = f"{SnwSurvBandWtshdInt[:-4]}.csv"
    SnwPillowsJoin_CSV = results_workspace + f"{report_date}_sensors_Join.csv"

    # set up intersect lists

    IntersctLstSurvey = [snowSurveys_proj, watershed_shapefile]
    IntersctLstBandSurvey = [snowSurveys_proj, band_shapefile]

    ############################################################################
    # Processing begins
    ############################################################################
    # ## set paths
    # merge and delete duplicates
    if merge == "Y":
        snowPillow_proj = results_workspace + f"{report_date}_sensors_albn83.shp"
        IntersctLst = [snowPillow_proj, watershed_shapefile]
        IntersctLstBand = [snowPillow_proj, band_shapefile]
        arcpy.Merge_management([results_workspace + f"{report_date}_sensors_{domainList[0]}.shp",
                                results_workspace + f"{report_date}_sensors_{domainList[1]}.shp",
                                results_workspace + f"{report_date}_sensors_{domainList[2]}.shp",
                                results_workspace + f"{report_date}_sensors_{domainList[3]}.shp",
                                results_workspace + f"{report_date}_sensors_{domainList[4]}.shp"], snowPillow_merge)

        # delete duplicates
        arcpy.DeleteIdentical_management(snowPillow_merge, "Site_ID")

        # reproject to Albers
        arcpy.Project_management(snowPillow_merge, snowPillow_proj, projOut)

    if merge == "N":
        snowPillow_proj = results_workspace + f"SNM_{report_date}_sensors_albn83.shp"
        IntersctLst = [snowPillow_proj, watershed_shapefile]
        IntersctLstBand = [snowPillow_proj, band_shapefile]
        arcpy.Project_management(domain_shapefile, snowPillow_proj, projOut, "", projIn)

    ## first add SWE inches, don't need to do this for surveys, it's already in there and then calculate field
    arcpy.AddField_management(snowPillow_proj, "SWE_In", "DOUBLE", "#", "#", "#",
                              "#", "NULLABLE", "NON_REQUIRED", "#")
    arcpy.CalculateField_management(snowPillow_proj, "SWE_In", "!pillowswe! * 39.370079", "PYTHON")

    ## Intersect with watersheds
    arcpy.Intersect_analysis(IntersctLst, SensorWtshdInt, "ALL", "-1 Unknown", "POINT")

    ## Create statistics
    arcpy.Statistics_analysis(SensorWtshdInt, SensorWtshdIntStat, "SWE_In MEAN", case_field_wtrshd)
    arcpy.Delete_management("in-memory")
    gc.collect()

    arcpy.AddField_management(SensorWtshdIntStat, "SWE_freq", "TEXT", "#", "#",
                              "#", "#", "NULLABLE", "NON_REQUIRED", "#")
    arcpy.Delete_management("in-memory")
    gc.collect()

    arcpy.CalculateField_management(SensorWtshdIntStat, "SWE_freq",
                                    '"{} ( {} )".format(round( !MEAN_SWE_I! ,1) , !FREQUENCY! )', "PYTHON", "")
    if surveys == "Y":
        arcpy.Project_management(snowSurveys, snowSurveys_proj, projOut)
        arcpy.Intersect_analysis(IntersctLstSurvey, SnwSurvWtshdInt, "ALL", "-1 Unknown", "POINT")
        arcpy.Statistics_analysis(SnwSurvWtshdInt, SnwSurvWtshdIntStat, "SWE_in MEAN", case_field_wtrshd)
        arcpy.AddField_management(SnwSurvWtshdIntStat, "SWE_freq", "TEXT", "#", "#",
                                  "#", "#", "NULLABLE", "NON_REQUIRED",
                                  "#")
        arcpy.CalculateField_management(SnwSurvWtshdIntStat, "SWE_freq",
                                        '"{} ( {} )".format(round( !MEAN_SWE_i! ,1) , !FREQUENCY! )', "PYTHON", "")

    arcpy.Intersect_analysis(IntersctLstBand, SensorBandWtshdInt, "ALL", "-1 Unknown", "POINT")
    if surveys == "Y":
        arcpy.Intersect_analysis(IntersctLstBandSurvey, SnwSurvBandWtshdInt, "ALL", "-1 Unknown", "POINT")

    arcpy.Statistics_analysis(SensorBandWtshdInt, SensorBandWtshdIntStat, "SWE_In MEAN", case_field_band)
    arcpy.Delete_management("in-memory")
    gc.collect()

    arcpy.AddField_management(SensorBandWtshdIntStat, "SWE_freq", "TEXT", "#", "#",
                              "#", "#", "NULLABLE", "NON_REQUIRED",
                              "#")
    arcpy.Delete_management("in-memory")
    gc.collect()
    ## Calculate Field
    arcpy.CalculateField_management(SensorBandWtshdIntStat, "SWE_freq",
                                    '"{} ( {} )".format(round( !MEAN_SWE_I! ,1) , !FREQUENCY! )', "PYTHON", "")
    if surveys == "Y":
        arcpy.Statistics_analysis(SnwSurvBandWtshdInt, SnwSurvBandWtshdIntStat, "SWE_in MEAN", case_field_band)
        arcpy.AddField_management(SnwSurvBandWtshdIntStat, "SWE_freq", "TEXT", "#",
                                  "#", "#", "#", "NULLABLE",
                                  "NON_REQUIRED", "#")
        arcpy.CalculateField_management(SnwSurvBandWtshdIntStat, "SWE_freq",
                                        '"{} ( {} )".format(round( !MEAN_SWE_I! ,1) , !FREQUENCY! )', "PYTHON", "")
    # Sort by bandname and watershed name, 2 tables
    arcpy.Sort_management(SensorBandWtshdIntStat, SensorBandWtshdIntStat_save, [[case_field_band, "ASCENDING"]])
    arcpy.Sort_management(SensorWtshdIntStat, SensorWtshdIntStat_save, [[case_field_wtrshd, "ASCENDING"]])
    if surveys == "Y":
        arcpy.Sort_management(SnwSurvBandWtshdIntStat, SnwSurvBandWtshdIntStat_save, [[case_field_band, "ASCENDING"]])
        arcpy.Sort_management(SnwSurvWtshdIntStat, SnwSurvWtshdIntStat_save, [[case_field_wtrshd, "ASCENDING"]])

    ## Make tables into table views for joins
    arcpy.MakeTableView_management(snowPillow_proj, snowPillowView)

    # creating a data frame of just the last SWE inches
    if difference == "Y":

        if domain == "SNM":

            lastPillowView = prev_results_workspace + f"SNM_{prev_report_date}_sensors_view.dbf"
            lastPillow = prev_results_workspace + f"SNM_{prev_report_date}_sensors_albn83.shp"

        else:
            lastPillowView = prev_results_workspace + f"{prev_report_date}_sensors_view.dbf"
            lastPillow = prev_results_workspace + f"{prev_report_date}_sensors_albn83.shp"

        arcpy.MakeTableView_management(lastPillow, lastPillowView)
        arcpy.TableToTable_conversion(lastPillowView, results_workspace, f"{report_date}_temp.csv")
        temp_df = pd.read_csv(results_workspace + f"{report_date}_temp.csv")
        temp_df = temp_df[["Site_ID", "SWE_In"]]
        temp_df.rename(columns={"SWE_In": "LastSWE_in"}, inplace=True)

        arcpy.TableToTable_conversion(snowPillowView, results_workspace, f"{report_date}_sensors.csv")
        curr_df = pd.read_csv(results_workspace + f"{report_date}_sensors.csv")
        merged_df = pd.merge(curr_df, temp_df[["Site_ID", "LastSWE_in"]], how="left", on="Site_ID")
        merged_df.to_csv(results_workspace + f"{report_date}_sensors_Join.csv", index=False)

    sensorBand_dbf = gpd.read_file(SensorBandWtshdIntStat_save)
    sensorBand_dbf = pd.DataFrame(sensorBand_dbf)
    sensorBand_dbf.to_csv(SensorBandWtshdIntStat_CSV, index=False)

    sensorWtshd_dbf = gpd.read_file(SensorWtshdIntStat_save)
    sensorWtshd_dbf = pd.DataFrame(sensorWtshd_dbf)
    sensorWtshd_dbf.to_csv(SensorWtshdIntStat_CSV, index=False)

    if surveys == "Y":
        surveyBand_dbf = gpd.read_file(SnwSurvBandWtshdIntStat_save)
        surveyBand_dbf = pd.DataFrame(surveyBand_dbf)
        surveyBand_dbf.to_csv(SnwSurvBandWtshdIntStat_CSV, index=False)

        surveyWtshd_dbf = gpd.read_file(SnwSurvWtshdIntStat_save)
        surveyWtshd_dbf = pd.DataFrame(surveyWtshd_dbf)
        surveyWtshd_dbf.to_csv(SnwSurvWtshdIntStat_CSV, index=False)


def zero_CCR_sensors(rundate, results_workspace, pillow_date, domain, sensors, zero_sensors, CCR, model_workspace_domain=None):

    pillow_gpkg = model_workspace_domain + f"{domain}_pillow-{pillow_date}.gpkg"

    # # Copy the .gpkg file to the new folder
    if CCR:

        shutil.copy(pillow_gpkg, results_workspace + f"/{rundate}_results_ET/{os.path.basename(pillow_gpkg)}")
        if os.path.exists(results_workspace + f"/{rundate}_results_ET/{os.path.basename(pillow_gpkg)}"):
            print('copied successfully')

        # Path to the copied .gpkg file
        new_gpkg_path = results_workspace + f"/{rundate}_results_ET/{domain}_pillow-{pillow_date}.gpkg"

        # Read the .gpkg file (you can list the layers or specify the one you want)
        gdf_CCR = gpd.read_file(new_gpkg_path)
        gdf_CCR = gdf_CCR.rename(columns={
            "nwbDistance": "nwbDist",
            "regionaleastness": "regEast",
            "regionalnorthness": "regNorth",
            "regionalzness": "regZn",
            "northness4km" : "north4km",
            "eastness4km" : "east4km"
        })

        # Specify the path for the shapefile output
        gdf_CCR["Site_ID"] = gdf_CCR["Site_ID"].astype(str)

        # Filter by length > 4
        gdf_filtered = gdf_CCR[gdf_CCR["Site_ID"].str.len() > 4]
        gdf_filtered.to_file(results_workspace + f"/{rundate}_results_ET/{rundate}_CCR_sensors_albn83.shp")

    if zero_sensors:
        # SNM_geopackage_CCR =
        gdf = gpd.read_file(sensors)
        gdf_zero = gdf[gdf['pillowswe'] == 0]
        gdf_zero.to_file(results_workspace + f"/{rundate}_results_ET/{rundate}_Zero_sensors_albn83.shp")

import pandas as pd
import os
import shutil
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
def parse_sensors(s):
    # Handle NA or missing values
    if s is None or s == 'NA' or (isinstance(s, float) and np.isnan(s)):
        return np.nan, 0  # no sensors
    # match "4.7 ( 18 )"
    match = re.match(r'([\d\.]+)\s*\(\s*(\d+)\s*\)', s)
    if match:
        return float(match.group(1)), int(match.group(2))
    else:
        try:
            return float(s), 1
        except:
            return np.nan, 0

def clean_numeric(val):
    """Convert a value that might be a formatted string (with commas) back to float"""
    if isinstance(val, str):
        return float(val.replace(',', ''))
    return float(val)

def WW_tables_for_report(rundate, modelRunName, averageRunName, results_workspace, reports_workspace, difference, aso_bc_basins,
                         aso_symbol, prev_tables_workspace=None, survey_date=None, prev_rundate=None, surveys_use=False):

    # dictionaries
    elevationBands = {
        "-1000": "< 0", "00000": "0", "01000": "1,000-2,000'", "02000": "2,000-3,000'", "03000": "3,000-4,000'",
        "04000": "4,000-5,000'",
        "05000": "5,000-6,000'", "06000": "6,000-7,000'", "07000": "7,000-8,000'", "08000": "8,000-9,000'",
        "09000": "9,000-10,000'",
        "10000": "10,000-11,000'", "11000": "11,000-12,000'", "12000": "12,000-13,000'", "13000": "13,000-14,000'",
        "14000": "14,000-15,000'",
        "14000GT": ">14,000'", "13000GT": ">13,000'", "12000GT": ">12,000'", "11000GT": ">11,000'",
        "10000GT": ">10,000'", "09000GT": ">9,000'", "08000GT": ">8,000'",
        "07000GT": ">7,000'", "06000GT": ">6,000'", "05000GT": ">5,000'"}

    states = {"0PNW0": "Pacific Northwest", "INMT1": "Intermountain", "INMT2": "Intermountain",
              "SOCN0": "South Continental", "NOCN0": "North Continental"}
    bandTableIndex = {"0PNW0": "06", "INMT1": "09a", "INMT2": "09b", "SOCN0": "08", "NOCN0": "07"}
    wtshTableIndex = {"0PNW0": "01", "INMT1": "04a", "INMT2": "04b", "SOCN0": "03", "NOCN0": "02"}

    abbrevs = ["0PNW0", "INMT1", "INMT2", "NOCN0", "SOCN0"]
    domain_tab = {"0PNW0": "PNW", "INMT1": "INMT", "INMT2": "INMT", "NOCN0": "NOCN", "SOCN0": "SOCN"}

    # Add a helper function to append the symbol
    def add_special_symbol(basin_name, aso_bc_basins, aso_symbol):
        """Add symbol to basin name if it's in the special list"""
        if aso_bc_basins is None or len(aso_bc_basins) == 0:
            return basin_name
        if basin_name in aso_bc_basins:
            return f"{basin_name}{aso_symbol}"
        return basin_name

    ## set new date structure
    date_obj = datetime.strptime(rundate, "%Y%m%d")
    formatted_date = f"{date_obj.month}/{date_obj.day}/{date_obj.year}"
    if difference == "Y":
        prev_date_obj = datetime.strptime(prev_rundate, "%Y%m%d")
        prev_formatted_date = f"{prev_date_obj.month}/{prev_date_obj.day}/{prev_date_obj.year}"
        prev_date_abrev = f"{prev_date_obj.month}/{prev_date_obj.day}"

    date_abrev = f"{date_obj.month}/{date_obj.day}"

    if surveys_use:
        surv_date_obj = datetime.strptime(survey_date, "%Y%m%d")
        surv_date_abrev = f"{surv_date_obj.month}/{surv_date_obj.day}"

    # copy over files
    shutil.copy(results_workspace + f"{modelRunName}/{rundate}anomBand_table.csv",
                reports_workspace + f"{modelRunName}/{rundate}anomBand_table.csv")
    shutil.copy(results_workspace + f"{modelRunName}/{rundate}anomHUC6_table.csv",
                reports_workspace + f"{modelRunName}/{rundate}anomHUC6_table.csv")
    shutil.copy(results_workspace + f"{modelRunName}/{rundate}anomRegion_table.csv",
                reports_workspace + f"{modelRunName}/{rundate}anomRegion_table.csv")
    shutil.copy(results_workspace + f"{modelRunName}/{rundate}anomWtshd_table.csv",
                reports_workspace + f"{modelRunName}/{rundate}anomWtshd_table.csv")
    shutil.copy(results_workspace + f"{modelRunName}/{rundate}anomWtshd_table.csv",
                reports_workspace + f"{modelRunName}/{rundate}anomWtshd_table.csv")
    shutil.copy(results_workspace + f"{modelRunName}/{rundate}band_table.csv",
                reports_workspace + f"{modelRunName}/{rundate}band_table.csv")
    shutil.copy(results_workspace + f"{modelRunName}/{rundate}Wtshd_table.csv",
                reports_workspace + f"{modelRunName}/{rundate}Wtshd_table.csv")
    shutil.copy(results_workspace + f"{averageRunName}/{rundate}anomBand_table.csv",
                reports_workspace + f"{averageRunName}/{rundate}anomBand_table.csv")
    shutil.copy(results_workspace + f"{averageRunName}/{rundate}anomHUC6_table.csv",
                reports_workspace + f"{averageRunName}/{rundate}anomHUC6_table.csv")
    shutil.copy(results_workspace + f"{averageRunName}/{rundate}anomRegion_table.csv",
                reports_workspace + f"{averageRunName}/{rundate}anomRegion_table.csv")
    shutil.copy(results_workspace + f"{averageRunName}/{rundate}anomWtshd_table.csv",
                reports_workspace + f"{averageRunName}/{rundate}anomWtshd_table.csv")
    shutil.copy(results_workspace + f"{averageRunName}/{rundate}anomWtshd_table.csv",
                reports_workspace + f"{averageRunName}/{rundate}anomWtshd_table.csv")
    shutil.copy(results_workspace + f"{averageRunName}/{rundate}band_table.csv",
                reports_workspace + f"{averageRunName}/{rundate}band_table.csv")
    shutil.copy(results_workspace + f"{averageRunName}/{rundate}Wtshd_table.csv",
                reports_workspace + f"{averageRunName}/{rundate}Wtshd_table.csv")
    shutil.copy(results_workspace + f"SNODAS/{rundate}_band_SNODAS_swe_table.csv",
                reports_workspace + f"{rundate}_band_SNODAS_swe_table.csv")
    shutil.copy(results_workspace + f"SNODAS/{rundate}_SNODAS_swe_table.csv",
                reports_workspace + f"{rundate}_SNODAS_swe_table.csv")
    shutil.copy(results_workspace + f"{rundate}_sensors_Wtshd_Intersect_stat.csv",
                reports_workspace + f"{rundate}_sensors_Wtshd_Intersect_stat.csv")
    shutil.copy(results_workspace + f"{rundate}_sensors_BandWtshd_Intersect.csv",
                reports_workspace + f"{rundate}_sensors_BandWtshd_Intersect.csv")

    if surveys_use:
        shutil.copy(results_workspace + f"{rundate}_surveys_BandWtshd_Intersect.csv",
                    reports_workspace + f"{rundate}_surveys_BandWtshd_Intersect.csv")
        shutil.copy(results_workspace + f"{rundate}_surveys_Wtshd_Intersect.csv",
                    reports_workspace + f"{rundate}_surveys_Wtshd_Intersect.csv")

    # make tables folder
    tables_workspace = reports_workspace + f"/{modelRunName}/Tables/"
    os.makedirs(tables_workspace, exist_ok=True)

    # open band table and sort
    df_band = pd.read_csv(reports_workspace + f"{modelRunName}/{rundate}band_table.csv")
    df_band['region'] = df_band["SrtNmeBand"].str[:5]
    df_band['VOL_AF'] = df_band['VOL_AF'].apply(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)
    df_band['AREA_MI2'] = df_band['AREA_MI2'].apply(lambda x: f"{x:,.1f}" if isinstance(x, (int, float)) else x)
    df_band['SWE_IN'] = df_band['SWE_IN'].round(1)
    df_band['Percent'] = df_band['Percent'].round(1)

    # open and sort the sensors table
    df_bnd_sens = pd.read_csv(reports_workspace + f"{rundate}_sensors_BandWtshd_Intersect.csv")
    df_bnd_sens = df_bnd_sens[["SrtNmeBand", "SWE_freq"]]
    df_bnd_sens = df_bnd_sens.rename(columns={"SWE_freq": "sensors"})

    # open and sort the banded percent of average table
    df_band_avg = pd.read_csv(reports_workspace + f"{averageRunName}/{rundate}anomBand_table.csv")
    df_band_avg = df_band_avg[["SrtNmeBand", "Average"]]
    df_band_avg = df_band_avg.rename(columns={"Average": "Avg"})

    # open and sort SNODAS code
    df_bnd_snodas = pd.read_csv(reports_workspace + f"{rundate}_band_SNODAS_swe_table.csv")
    df_bnd_snodas = df_bnd_snodas[["SrtNmeBand", "SWE_IN"]]
    df_bnd_snodas = df_bnd_snodas.rename(columns={"SWE_IN": "SNODAS"})

    # merge tables together
    merged_df = pd.merge(df_band, df_band_avg, on="SrtNmeBand", how="left")
    merged_df = pd.merge(merged_df, df_bnd_sens, on="SrtNmeBand", how="left")
    merged_df = pd.merge(merged_df, df_bnd_snodas, on="SrtNmeBand", how="left")

    if surveys_use:
        df_bnd_surv = pd.read_csv(reports_workspace + f"{rundate}_surveys_BandWtshd_Intersect.csv")
        df_bnd_surv = df_bnd_surv[["SrtNmeBand", "SWE_freq"]]
        df_bnd_surv = df_bnd_surv.rename(columns={"SWE_freq": "surveys"})
        merged_df = pd.merge(merged_df, df_bnd_surv, on="SrtNmeBand", how="left")
        merged_df['surveys'] = merged_df['surveys'].fillna('NA')

    # merge to include NAs -- Check to see if this is done multiple times
    merged_df['Avg'] = merged_df['Avg'].fillna("NA")
    merged_df['Avg'] = merged_df['Avg'].apply(lambda x: int(round(x)) if x != "NA" else x)
    merged_df['sensors'] = merged_df['sensors'].fillna('NA')

    # separate into tables for domain
    for abbrev in abbrevs:
        state_df = merged_df[merged_df['region'] == abbrev]
        state_df['Basin_rw'] = state_df['SrtNmeBand'].apply(lambda x: x[9:-8] if x[-2:] == "GT" else x[9:-6])
        state_df['Basin_rw'] = state_df['Basin_rw'].apply(lambda x: add_special_symbol(x, aso_bc_basins, aso_symbol))
        state_df['Num'] = state_df['SrtNmeBand'].str[6:8].astype(int).astype(str) + '.'
        state_df['Basin'] = state_df['Num'] + " " + state_df['Basin_rw']
        state_df['Elevation Band'] = state_df['SrtNmeBand'].str[-5:]
        state_df['Elevation Band'] = state_df['SrtNmeBand'].apply(lambda x: x[-7:] if x[-2:] == "GT" else x[-5:])
        state_df['Elevation Band'] = state_df['Elevation Band'].map(elevationBands)

        if abbrev == "SOCN0":
            bands = ["7,000-8,000'", "8,000-9,000'", "9,000-10,000'", "10,000-11,000'",
                     "11,000-12,000'", "12,000-13,000'", "13,000-14,000'"]
            rows = ['33. Animas', '21. San Juan']

            # loop through elevation bands
            for band in bands:
                subset = state_df.loc[state_df['Basin'].isin(rows) & state_df['Elevation Band'].isin([band])]

                # Get San Juan row for this band and its index
                san_juan_band_mask = (state_df['Basin'] == '21. San Juan') & (state_df['Elevation Band'] == band)

                if not san_juan_band_mask.any():
                    continue

                san_juan_index = state_df[san_juan_band_mask].index[0]

                subset['VOL_AF'] = subset['VOL_AF'].apply(clean_numeric)
                subset['AREA_MI2'] = subset['AREA_MI2'].apply(clean_numeric)
                subset['SWE_IN'] = pd.to_numeric(subset['SWE_IN'], errors='coerce')
                subset['Avg'] = pd.to_numeric(subset['Avg'], errors='coerce')
                subset['Percent'] = pd.to_numeric(subset['Percent'], errors='coerce')
                subset['SNODAS'] = pd.to_numeric(subset['SNODAS'], errors='coerce')

                # Summations
                sum_vals = subset[['VOL_AF', 'AREA_MI2']].sum()

                # Weighted averages by AREA_MI2
                # weights = subset['AREA_MI2']
                # swe_weighted = (subset['SWE_IN'].mul(weights).sum() / weights.sum())
                # snodas_weighted = (subset['SNODAS'].mul(weights).sum() / weights.sum())
                # pct_weighted = (subset['Avg'].mul(weights).sum() / weights.sum())
                # sca_weighted = (subset['Percent'].mul(weights).sum() / weights.sum())

                weights = subset['AREA_MI2']

                # SWE_IN
                valid_swe = subset['SWE_IN'].notna()
                swe_weighted = (subset.loc[valid_swe, 'SWE_IN'].mul(subset.loc[valid_swe, 'AREA_MI2']).sum() /
                                subset.loc[valid_swe, 'AREA_MI2'].sum()) if valid_swe.any() else np.nan

                # SNODAS
                valid_snodas = subset['SNODAS'].notna()
                snodas_weighted = (subset.loc[valid_snodas, 'SNODAS'].mul(subset.loc[valid_snodas, 'AREA_MI2']).sum() /
                                   subset.loc[valid_snodas, 'AREA_MI2'].sum()) if valid_snodas.any() else np.nan

                # Avg (percent of average)
                valid_avg = subset['Avg'].notna()
                pct_weighted = (subset.loc[valid_avg, 'Avg'].mul(subset.loc[valid_avg, 'AREA_MI2']).sum() /
                                subset.loc[valid_avg, 'AREA_MI2'].sum()) if valid_avg.any() else np.nan

                # Percent (SCA)
                valid_pct = subset['Percent'].notna()
                sca_weighted = (subset.loc[valid_pct, 'Percent'].mul(subset.loc[valid_pct, 'AREA_MI2']).sum() /
                                subset.loc[valid_pct, 'AREA_MI2'].sum()) if valid_pct.any() else np.nan

                subset[['SWE_from_sensors', 'Num_sensors']] = subset['sensors'].apply(
                    lambda x: pd.Series(parse_sensors(x)))

                # Weighted average SWE for sensors
                if subset['Num_sensors'].sum() > 0:
                    weighted_swe = (subset['SWE_from_sensors'] * subset['Num_sensors']).sum() / subset[
                        'Num_sensors'].sum()
                    total_sensors = subset['Num_sensors'].sum()
                    sensors_weighted_str = f"{weighted_swe:.1f} ( {total_sensors:.0f} )"
                else:
                    sensors_weighted_str = "NA"

                # Update the San Juan row IN PLACE with combined weighted values
                state_df.at[san_juan_index, 'Basin'] = '21. San Juan**'
                state_df.at[san_juan_index, 'VOL_AF'] = sum_vals['VOL_AF']
                state_df.at[san_juan_index, 'AREA_MI2'] = sum_vals['AREA_MI2']
                state_df.at[san_juan_index, 'SWE_IN'] = swe_weighted
                state_df.at[san_juan_index, 'Avg'] = pct_weighted
                state_df.at[san_juan_index, 'SNODAS'] = snodas_weighted
                state_df.at[san_juan_index, 'Percent'] = sca_weighted
                state_df.at[san_juan_index, 'sensors'] = sensors_weighted_str

                if surveys_use:
                    subset[['SWE_from_surveys', 'Num_surveys']] = subset['surveys'].apply(
                        lambda x: pd.Series(parse_sensors(x)))

                    # Weighted average SWE for sensors
                    if subset['Num_surveys'].sum() > 0:
                        weighted_swe = (subset['SWE_from_surveys'] * subset['Num_surveys']).sum() / subset[
                            'Num_surveys'].sum()
                        total_surveys = subset['Num_surveys'].sum()
                        surveys_weighted_str = f"{weighted_swe:.1f} ( {total_surveys:.0f} )"
                    else:
                        surveys_weighted_str = "NA"

                    state_df.at[san_juan_index, 'surveys'] = surveys_weighted_str


                # assert combined_row['AREA_MI2'] == subset['AREA_MI2'].sum()
        state_df['VOL_AF'] = state_df['VOL_AF'].apply(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)
        state_df['AREA_MI2'] = state_df['AREA_MI2'].apply(lambda x: f"{x:,.1f}" if isinstance(x, (int, float)) else x)
        state_df['Avg'] = state_df['Avg'].apply(lambda x: int(round(x)) if x != "NA" else x)
        state_df['sensors'] = state_df['sensors'].fillna('NA')
        state_df['VOL_AF'] = state_df['VOL_AF'].apply(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)
        state_df['AREA_MI2'] = state_df['AREA_MI2'].apply(lambda x: f"{x:,.1f}" if isinstance(x, (int, float)) else x)
        state_df['SWE_IN'] = state_df['SWE_IN'].round(1)
        state_df['Percent'] = state_df['Percent'].round(1)

        if difference == "Y":
            difference_cols = ['prev_SWE_IN', 'prev_sensors', 'prev_Avg']
            df_band_prev = pd.read_csv(
                prev_tables_workspace + f"{abbrev}_{prev_rundate}_table{bandTableIndex[f'{abbrev}']}_raw.csv")
            df_band_prev = df_band_prev.rename(
                columns={"SWE_IN": "prev_SWE_IN", "sensors": "prev_sensors", "Avg": "prev_Avg"})
            df_band_prev = df_band_prev[['Basin', 'Elevation Band', 'prev_SWE_IN', 'prev_sensors', 'prev_Avg']]
            state_df = pd.merge(state_df, df_band_prev, on=['Basin', 'Elevation Band'], how='inner')

            # edit and export
            if not surveys_use:
                df_band_tbl = state_df[
                    ['Basin', 'Elevation Band', 'SWE_IN', "Percent", "AREA_MI2", 'VOL_AF', "sensors", "Avg", "SNODAS",
                     'prev_SWE_IN', 'prev_sensors', 'prev_Avg']]

                # for export
                df_band_export = state_df[
                    ["Basin", 'Elevation Band', 'prev_Avg', "Avg", "prev_SWE_IN", "SWE_IN", "Percent", "VOL_AF",
                     "AREA_MI2", 'prev_sensors', "sensors", "SNODAS"]]
                df_band_export = df_band_export.rename(
                    columns={"prev_Avg": f"%{prev_date_abrev} Avg.", "Avg": f"%{date_abrev} Avg.",
                             "prev_SWE_IN": "SWE (in)", "SWE_IN": "SWE (in)", "Percent": "%SCA",
                             "VOL_AF": "Vol (AF)", "AREA_MI2": "Area (mi2)", "prev_sensors": "Pillows",
                             "sensors": "Pillows", "SNODAS": "SNODAS*"})
                top_header = ["", "", prev_formatted_date, formatted_date, prev_formatted_date, formatted_date,
                              formatted_date, formatted_date, formatted_date, prev_formatted_date, formatted_date,
                              formatted_date]
                df_band_export.columns = pd.MultiIndex.from_arrays(
                    [top_header, df_band_export.columns]
                )
                df_band_export.to_csv(
                    tables_workspace + f"{abbrev}_{rundate}_table{bandTableIndex[f'{abbrev}']}_final.csv")

            if surveys_use:
                df_band_tbl = state_df[
                    ['Basin', 'Elevation Band', 'SWE_IN', "Percent", "AREA_MI2", 'VOL_AF', "sensors", "Avg", "surveys",
                     "SNODAS", 'prev_SWE_IN', 'prev_sensors', 'prev_Avg']]

                # for export
                df_band_export = state_df[
                    ["Basin", 'Elevation Band', 'prev_Avg', "Avg", "SWE_IN", "Percent", "VOL_AF", "AREA_MI2", "sensors",
                     "surveys", "SNODAS"]]
                df_band_export = df_band_export.rename(
                    columns={"prev_Avg": f"%{prev_date_abrev} Avg.", "Avg": f"%{date_abrev} Avg.",
                             "prev_SWE_IN": "SWE (in)", "SWE_IN": "SWE (in)", "Percent": "%SCA",
                             "VOL_AF": "Vol (AF)", "AREA_MI2": "Area (mi2)", "prev_sensors": "Pillows",
                             "sensors": "Pillows", "surveys": "Surveys", "SNODAS": "SNODAS*"})
                top_header = ["", "", prev_formatted_date, formatted_date, prev_formatted_date, formatted_date,
                              formatted_date, formatted_date, formatted_date, prev_formatted_date, formatted_date,
                              survey_date, formatted_date]
                df_band_export.columns = pd.MultiIndex.from_arrays(
                    [top_header, df_band_export.columns]
                )
                df_band_export.to_csv(
                    tables_workspace + f"{abbrev}_{rundate}_table{bandTableIndex[f'{abbrev}']}_final.csv")

        if difference == "N":
            # edit and export
            if not surveys_use:
                df_band_tbl = state_df[
                    ['Basin', 'Elevation Band', 'SWE_IN', "Percent", "AREA_MI2", 'VOL_AF', "sensors", "Avg", "SNODAS"]]

                # for export
                df_band_export = state_df[
                    ["Basin", 'Elevation Band', "Avg", "SWE_IN", "Percent", "VOL_AF", "AREA_MI2", "sensors", "SNODAS"]]
                df_band_export = df_band_export.rename(
                    columns={"Avg": f"%{date_abrev} Avg.", "SWE_IN": "SWE (in)", "Percent": "%SCA",
                             "VOL_AF": "Vol (AF)", "AREA_MI2": "Area (mi2)", "sensors": "Pillows", "SNODAS": "SNODAS*"})
                top_header = ["", "", formatted_date, formatted_date, formatted_date, formatted_date, formatted_date,
                              formatted_date, formatted_date]
                df_band_export.columns = pd.MultiIndex.from_arrays(
                    [top_header, df_band_export.columns]
                )
                df_band_export.to_csv(
                    tables_workspace + f"{abbrev}_{rundate}_table{bandTableIndex[f'{abbrev}']}_final.csv")

            if surveys_use:
                df_band_tbl = state_df[
                    ['Basin', 'Elevation Band', 'SWE_IN', "Percent", "AREA_MI2", 'VOL_AF', "sensors", "Avg", "surveys",
                     "SNODAS"]]

                # for export
                df_band_export = state_df[
                    ["Basin", 'Elevation Band', "Avg", "SWE_IN", "Percent", "VOL_AF", "AREA_MI2", "sensors", "surveys",
                     "SNODAS"]]
                df_band_export = df_band_export.rename(
                    columns={"Avg": f"%{date_abrev} Avg.", "SWE_IN": "SWE (in)", "Percent": "%SCA",
                             "VOL_AF": "Vol (AF)", "AREA_MI2": "Area (mi2)", "sensors": "Pillows", "surveys": "Surveys",
                             "SNODAS": "SNODAS*"})
                top_header = ["", "", formatted_date, formatted_date, formatted_date, formatted_date, formatted_date,
                              formatted_date, survey_date, formatted_date]
                df_band_export.columns = pd.MultiIndex.from_arrays(
                    [top_header, df_band_export.columns]
                )
                df_band_export.to_csv(
                    tables_workspace + f"{abbrev}_{rundate}_table{bandTableIndex[f'{abbrev}']}_final.csv")

        df_band_tbl.to_csv(tables_workspace + f"{abbrev}_{rundate}_table{bandTableIndex[f'{abbrev}']}_raw.csv",
                           index=False)

    ###
    print('Moving on to watershed table')
    # getting watershed table
    df_wtshd = pd.read_csv(reports_workspace + f"{modelRunName}/{rundate}Wtshd_table.csv")
    df_wtshd['SWE_IN'] = df_wtshd['SWE_IN'].round(1)
    df_wtshd['Percent'] = df_wtshd['Percent'].round(1)
    df_wtshd['region'] = df_wtshd["SrtName"].str[:5]

    # get and sort the sensors table
    df_wtshd_sens = pd.read_csv(reports_workspace + f"{rundate}_sensors_Wtshd_Intersect_stat.csv")
    df_wtshd_sens = df_wtshd_sens[["SrtName", "SWE_freq"]]
    df_wtshd_sens = df_wtshd_sens.rename(columns={"SWE_freq": "sensors"})

    # get and sort the percent of average table
    df_wtshd_avg = pd.read_csv(reports_workspace + f"{averageRunName}/{rundate}anomWtshd_table.csv")
    df_wtshd_avg = df_wtshd_avg[["SrtName", "Average"]]
    df_wtshd_avg = df_wtshd_avg.rename(columns={"Average": "Avg"})

    # open and sort SNODAS code
    df_wtshd_snodas = pd.read_csv(reports_workspace + f"{rundate}_SNODAS_swe_table.csv")
    df_wtshd_snodas = df_wtshd_snodas[["SrtName", "SWE_IN"]]
    df_wtshd_snodas = df_wtshd_snodas.rename(columns={"SWE_IN": "SNODAS"})

    # merge tables together
    merged_wtshd_df = pd.merge(df_wtshd, df_wtshd_avg, on="SrtName", how="left")
    merged_wtshd_df = pd.merge(merged_wtshd_df, df_wtshd_sens, on="SrtName", how="left")
    merged_wtshd_df = pd.merge(merged_wtshd_df, df_wtshd_snodas, on="SrtName", how="left")

    if surveys_use:
        df_wtshd_surv = pd.read_csv(reports_workspace + f"{rundate}_surveys_Wtshd_Intersect.csv")
        df_wtshd_surv = df_wtshd_surv[["SrtName", "SWE_freq"]]
        df_wtshd_surv = df_wtshd_surv.rename(columns={"SWE_freq": "surveys"})
        merged_wtshd_df = pd.merge(merged_wtshd_df, df_wtshd_surv, on="SrtName", how="left")
        merged_wtshd_df['surveys'] = merged_wtshd_df['surveys'].fillna('NA')

    # merge to include NAs -- Check to see if this is done multiple times
    merged_wtshd_df['Avg'] = merged_wtshd_df['Avg'].fillna("NA")
    merged_wtshd_df['Avg'] = merged_wtshd_df['Avg'].apply(lambda x: int(round(x)) if x != "NA" else x)
    merged_wtshd_df['sensors'] = merged_wtshd_df['sensors'].fillna('NA')

    # separate into tables for domain
    for abbrev in abbrevs:
        state_wtshd_df = merged_wtshd_df[merged_wtshd_df['region'] == abbrev]
        state_wtshd_df['Basin_rw'] = state_wtshd_df['SrtName'].str[9:]
        state_wtshd_df['Basin_rw'] = state_wtshd_df['Basin_rw'].apply(lambda x: add_special_symbol(x, aso_bc_basins, aso_symbol))
        state_wtshd_df['Num'] = state_wtshd_df['SrtName'].str[6:8].astype(int).astype(str) + '.'
        state_wtshd_df['Basin'] = state_wtshd_df['Num'] + " " + state_wtshd_df['Basin_rw']

        if abbrev == "SOCN0":
            for c in ['VOL_AF', 'AREA_MI2']:
                state_wtshd_df[c] = (
                    state_wtshd_df[c]
                    .astype(str)
                    .str.replace(',', '', regex=False)
                    .astype(float)
                )

            rows = ['33. Animas', '21. San Juan']
            subset = state_wtshd_df.loc[state_wtshd_df['Basin'].isin(rows)]

            # Get San Juan row and its index
            san_juan_mask = state_wtshd_df['Basin'] == '21. San Juan'
            san_juan_index = state_wtshd_df[san_juan_mask].index[0]

            # Summations
            sum_vals = subset[['VOL_AF', 'AREA_MI2']].sum()

            # Weighted averages by AREA_MI2
            # weights = subset['AREA_MI2']
            # swe_weighted = (subset['SWE_IN'].mul(weights).sum() / weights.sum())
            # snodas_weighted = (subset['SNODAS'].mul(weights).sum() / weights.sum())
            # pct_weighted = (subset['Avg'].mul(weights).sum() / weights.sum())
            # sca_weighted = (subset['Percent'].mul(weights).sum() / weights.sum())

            weights = subset['AREA_MI2']

            # SWE_IN
            valid_swe = subset['SWE_IN'].notna()
            swe_weighted = (subset.loc[valid_swe, 'SWE_IN'].mul(subset.loc[valid_swe, 'AREA_MI2']).sum() /
                            subset.loc[valid_swe, 'AREA_MI2'].sum()) if valid_swe.any() else np.nan

            # SNODAS
            valid_snodas = subset['SNODAS'].notna()
            snodas_weighted = (subset.loc[valid_snodas, 'SNODAS'].mul(subset.loc[valid_snodas, 'AREA_MI2']).sum() /
                               subset.loc[valid_snodas, 'AREA_MI2'].sum()) if valid_snodas.any() else np.nan

            # Avg (percent of average)
            valid_avg = subset['Avg'].notna()
            pct_weighted = (subset.loc[valid_avg, 'Avg'].mul(subset.loc[valid_avg, 'AREA_MI2']).sum() /
                            subset.loc[valid_avg, 'AREA_MI2'].sum()) if valid_avg.any() else np.nan

            # Percent (SCA)
            valid_pct = subset['Percent'].notna()
            sca_weighted = (subset.loc[valid_pct, 'Percent'].mul(subset.loc[valid_pct, 'AREA_MI2']).sum() /
                            subset.loc[valid_pct, 'AREA_MI2'].sum()) if valid_pct.any() else np.nan

            subset[['SWE_from_sensors', 'Num_sensors']] = subset['sensors'].apply(
                lambda x: pd.Series(parse_sensors(x)))

            # Weighted average SWE for sensors
            if subset['Num_sensors'].sum() > 0:
                weighted_swe = (subset['SWE_from_sensors'] * subset['Num_sensors']).sum() / subset['Num_sensors'].sum()
                total_sensors = subset['Num_sensors'].sum()
                sensors_weighted_str = f"{weighted_swe:.1f} ( {total_sensors:.0f} )"
            else:
                sensors_weighted_str = "NA"

            # Update the San Juan row IN PLACE with combined weighted values
            state_wtshd_df.at[san_juan_index, 'Basin'] = '21. San Juan**'
            state_wtshd_df.at[san_juan_index, 'VOL_AF'] = sum_vals['VOL_AF']
            state_wtshd_df.at[san_juan_index, 'AREA_MI2'] = sum_vals['AREA_MI2']
            state_wtshd_df.at[san_juan_index, 'SWE_IN'] = swe_weighted
            state_wtshd_df.at[san_juan_index, 'Avg'] = pct_weighted
            state_wtshd_df.at[san_juan_index, 'SNODAS'] = snodas_weighted
            state_wtshd_df.at[san_juan_index, 'Percent'] = sca_weighted
            state_wtshd_df.at[san_juan_index, 'sensors'] = sensors_weighted_str

            if surveys_use:
                subset[['SWE_from_surveys', 'Num_surveys']] = subset['surveys'].apply(
                    lambda x: pd.Series(parse_sensors(x)))

                # Weighted average SWE for sensors
                if subset['Num_surveys'].sum() > 0:
                    weighted_swe = (subset['SWE_from_surveys'] * subset['Num_surveys']).sum() / subset[
                        'Num_surveys'].sum()
                    total_surveys = subset['Num_surveys'].sum()
                    surveys_weighted_str = f"{weighted_swe:.1f} ( {total_surveys:.0f} )"
                else:
                    surveys_weighted_str = "NA"

                state_wtshd_df.at[san_juan_index, 'surveys'] = surveys_weighted_str

        state_wtshd_df['VOL_AF'] = state_wtshd_df['VOL_AF'].apply(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)
        state_wtshd_df['AREA_MI2'] = state_wtshd_df['AREA_MI2'].apply(lambda x: f"{x:,.1f}" if isinstance(x, (int, float)) else x)
        state_wtshd_df['SWE_IN'] = state_wtshd_df['SWE_IN'].round(1)
        state_wtshd_df['Percent'] = state_wtshd_df['Percent'].round(1)
        state_wtshd_df['Avg'] = state_wtshd_df['Avg'].fillna("NA")
        state_wtshd_df['Avg'] = state_wtshd_df['Avg'].apply(lambda x: int(round(x)) if x != "NA" else x)
        state_wtshd_df['SNODAS'] = state_wtshd_df['SNODAS'].round(1)

        if difference == "Y":
            difference_cols = ['prev_SWE_IN', 'prev_sensors', 'prev_Avg']
            df_wtshed_prev = pd.read_csv(
                prev_tables_workspace + f"{abbrev}_{prev_rundate}_table{wtshTableIndex[f'{abbrev}']}_raw.csv")
            df_wtshed_prev = df_wtshed_prev.rename(
                columns={"SWE_IN": "prev_SWE_IN", "sensors": "prev_sensors", "Avg": "prev_Avg"})
            df_wtshed_prev = df_wtshed_prev[['Basin', 'prev_SWE_IN', 'prev_sensors', 'prev_Avg']]
            state_wtshd_df = pd.merge(state_wtshd_df, df_wtshed_prev, on='Basin', how='inner')

            # edit and export
            if not surveys_use:
                df_wtshd_tbl = state_wtshd_df[
                    ['Basin', 'SWE_IN', "Percent", "AREA_MI2", 'VOL_AF', "sensors", "Avg", "SNODAS", 'prev_SWE_IN',
                     'prev_sensors', 'prev_Avg']]

                # for export
                df_wtshd_export = state_wtshd_df[
                    ["Basin", 'prev_Avg', "Avg", 'prev_SWE_IN', "SWE_IN", "Percent", "VOL_AF", "AREA_MI2",
                     "prev_sensors", "sensors", "SNODAS"]]
                df_wtshd_export = df_wtshd_export.rename(
                    columns={"prev_Avg": f"%{prev_date_abrev} Avg.", "Avg": f"%{date_abrev} Avg.",
                             "prev_SWE_IN": "SWE (in)", "SWE_IN": "SWE (in)", "Percent": "%SCA",
                             "VOL_AF": "Vol (AF)", "AREA_MI2": "Area (mi2)", "prev_sensors": "Pillows",
                             "sensors": "Pillows", "SNODAS": "SNODAS*"})
                top_header = ["", prev_formatted_date, formatted_date, prev_formatted_date, formatted_date,
                              formatted_date, formatted_date, formatted_date, prev_formatted_date, formatted_date,
                              formatted_date]
                df_wtshd_export.columns = pd.MultiIndex.from_arrays(
                    [top_header, df_wtshd_export.columns]
                )
                df_wtshd_export.to_csv(
                    tables_workspace + f"{abbrev}_{rundate}_table{wtshTableIndex[f'{abbrev}']}_final.csv")

            if surveys_use:
                df_wtshd_tbl = state_wtshd_df[
                    ['Basin', 'SWE_IN', "Percent", "AREA_MI2", 'VOL_AF', "sensors", "Avg", "surveys", "SNODAS",
                     'prev_SWE_IN', 'prev_sensors', 'prev_Avg']]
                # for export
                df_wtshd_export = state_wtshd_df[
                    ["Basin", 'prev_Avg', "Avg", "SWE_IN", "Percent", "VOL_AF", "AREA_MI2", "sensors", "surveys",
                     "SNODAS"]]
                df_wtshd_export = df_wtshd_export.rename(
                    columns={"prev_Avg": f"%{prev_date_abrev} Avg.", "Avg": f"%{date_abrev} Avg.",
                             "prev_SWE_IN": "SWE (in)", "SWE_IN": "SWE (in)", "Percent": "%SCA",
                             "VOL_AF": "Vol (AF)", "AREA_MI2": "Area (mi2)", "prev_sensors": "Pillows",
                             "sensors": "Pillows", "surveys": "Surveys", "SNODAS": "SNODAS*"})
                top_header = ["", prev_formatted_date, formatted_date, prev_formatted_date, formatted_date,
                              formatted_date, formatted_date, formatted_date, prev_formatted_date, formatted_date,
                              survey_date, formatted_date]
                df_wtshd_export.columns = pd.MultiIndex.from_arrays(
                    [top_header, df_wtshd_export.columns]
                )
                df_wtshd_export.to_csv(
                    tables_workspace + f"{abbrev}_{rundate}_table{wtshTableIndex[f'{abbrev}']}_final.csv")

        if difference == "N":
            # edit and export
            # add animas and san juan

            if not surveys_use:
                df_wtshd_tbl = state_wtshd_df[
                    ['Basin', 'SWE_IN', "Percent", "AREA_MI2", 'VOL_AF', "sensors", "Avg", "SNODAS"]]

                # for export
                df_wtshd_export = state_wtshd_df[
                    ["Basin", "Avg", "SWE_IN", "Percent", "VOL_AF", "AREA_MI2", "sensors", "SNODAS"]]
                df_wtshd_export = df_wtshd_export.rename(
                    columns={"Avg": f"%{date_abrev} Avg.", "SWE_IN": "SWE (in)", "Percent": "%SCA",
                             "VOL_AF": "Vol (AF)", "AREA_MI2": "Area (mi2)", "sensors": "Pillows", "SNODAS": "SNODAS*"})
                top_header = ["", formatted_date, formatted_date, formatted_date, formatted_date, formatted_date,
                              formatted_date, formatted_date]
                df_wtshd_export.columns = pd.MultiIndex.from_arrays(
                    [top_header, df_wtshd_export.columns]
                )
                df_wtshd_export.to_csv(
                    tables_workspace + f"{abbrev}_{rundate}_table{wtshTableIndex[f'{abbrev}']}_final.csv")



            if surveys_use:
                df_wtshd_tbl = state_wtshd_df[
                    ['Basin', 'SWE_IN', "Percent", "AREA_MI2", 'VOL_AF', "sensors", "Avg", "surveys", "SNODAS"]]

                # for export
                df_wtshd_export = state_wtshd_df[
                    ["Basin", "Avg", "SWE_IN", "Percent", "VOL_AF", "AREA_MI2", "sensors", "surveys", "SNODAS"]]
                df_wtshd_export = df_wtshd_export.rename(
                    columns={"Avg": f"%{date_abrev} Avg.", "SWE_IN": "SWE (in)", "Percent": "%SCA",
                             "VOL_AF": "Vol (AF)", "AREA_MI2": "Area (mi2)", "sensors": "Pillows", "SNODAS": "SNODAS*"})
                top_header = ["", formatted_date, formatted_date, formatted_date, formatted_date, formatted_date,
                              formatted_date, survey_date, formatted_date]
                df_wtshd_export.columns = pd.MultiIndex.from_arrays(
                    [top_header, df_wtshd_export.columns]
                )
                df_wtshd_export.to_csv(
                    tables_workspace + f"{abbrev}_{rundate}_table{wtshTableIndex[f'{abbrev}']}_final.csv")

        df_wtshd_tbl.to_csv(tables_workspace + f"{abbrev}_{rundate}_table{wtshTableIndex[f'{abbrev}']}_raw.csv", index=False)

# def SNM_tables_for_report(rundate, modelRunName, averageRunName, results_workspace, reports_workspace, difference,
#                          prev_tables_workspace=None, survey_date=None, prev_rundate=None, surveys_use=False):
#
#     # dictionaries
#     elevationBands = {
#         "-1000": "< 0", "00000": "0", "01000": "1,000-2,000'", "02000": "2,000-3,000'", "03000": "3,000-4,000'",
#         "04000": "4,000-5,000'",
#         "05000": "5,000-6,000'", "06000": "6,000-7,000'", "07000": "7,000-8,000'", "08000": "8,000-9,000'",
#         "09000": "9,000-10,000'",
#         "10000": "10,000-11,000'", "11000": "11,000-12,000'", "12000": "12,000-13,000'", "13000": "13,000-14,000'",
#         "14000": "14,000-15,000'",
#         "14000GT": ">14,000'", "13000GT": ">13,000'", "12000GT": ">12,000'", "11000GT": ">11,000'",
#         "10000GT": ">10,000'", "09000GT": ">9,000'", "08000GT": ">8,000'",
#         "07000GT": ">7,000'", "06000GT": ">6,000'", "05000GT": ">5,000'"}
#
#
#     ## set new date structure
#     date_obj = datetime.strptime(rundate, "%Y%m%d")
#     formatted_date = f"{date_obj.month}/{date_obj.day}/{date_obj.year}"
#     date_abrev = f"{date_obj.month}/{date_obj.day}"
#
#     if difference == "Y":
#         prev_date_obj = datetime.strptime(prev_rundate, "%Y%m%d")
#         prev_formatted_date = f"{prev_date_obj.month}/{prev_date_obj.day}/{prev_date_obj.year}"
#         prev_date_abrev = f"{prev_date_obj.month}/{prev_date_obj.day}"
#
#     if surveys_use:
#         surv_date_obj = datetime.strptime(survey_date, "%Y%m%d")
#         surv_date_abrev = f"{surv_date_obj.month}/{surv_date_obj.day}"
#
#     # copy over files
#     shutil.copy(results_workspace + f"{modelRunName}/{rundate}anomBand_table.csv",
#                 reports_workspace + f"{modelRunName}/{rundate}anomBand_table.csv")
#     shutil.copy(results_workspace + f"{modelRunName}/{rundate}anomWtshd_table.csv",
#                 reports_workspace + f"{modelRunName}/{rundate}anomWtshd_table.csv")
#     shutil.copy(results_workspace + f"{modelRunName}/{rundate}anomWtshd_table.csv",
#                 reports_workspace + f"{modelRunName}/{rundate}anomWtshd_table.csv")
#     shutil.copy(results_workspace + f"{modelRunName}/{rundate}band_table.csv",
#                 reports_workspace + f"{modelRunName}/{rundate}band_table.csv")
#     shutil.copy(results_workspace + f"{modelRunName}/{rundate}Wtshd_table.csv",
#                 reports_workspace + f"{modelRunName}/{rundate}Wtshd_table.csv")
#     shutil.copy(results_workspace + f"{averageRunName}/{rundate}anomBand_table.csv",
#                 reports_workspace + f"{averageRunName}/{rundate}anomBand_table.csv")
#     shutil.copy(results_workspace + f"{averageRunName}/{rundate}anomWtshd_table.csv",
#                 reports_workspace + f"{averageRunName}/{rundate}anomWtshd_table.csv")
#     shutil.copy(results_workspace + f"{averageRunName}/{rundate}anomWtshd_table.csv",
#                 reports_workspace + f"{averageRunName}/{rundate}anomWtshd_table.csv")
#     shutil.copy(results_workspace + f"{averageRunName}/{rundate}band_table.csv",
#                 reports_workspace + f"{averageRunName}/{rundate}band_table.csv")
#     shutil.copy(results_workspace + f"{averageRunName}/{rundate}Wtshd_table.csv",
#                 reports_workspace + f"{averageRunName}/{rundate}Wtshd_table.csv")
#     shutil.copy(results_workspace + f"SNODAS/{rundate}_band_SNODAS_swe_table.csv",
#                 reports_workspace + f"{rundate}_band_SNODAS_swe_table.csv")
#     shutil.copy(results_workspace + f"SNODAS/{rundate}_SNODAS_swe_table.csv",
#                 reports_workspace + f"{rundate}_SNODAS_swe_table.csv")
#     shutil.copy(results_workspace + f"{rundate}_sensors_Wtshd_Intersect_stat.csv",
#                 reports_workspace + f"{rundate}_sensors_Wtshd_Intersect_stat.csv")
#     shutil.copy(results_workspace + f"{rundate}_sensors_BandWtshd_Intersect.csv",
#                 reports_workspace + f"{rundate}_sensors_BandWtshd_Intersect.csv")
#
#     if surveys_use:
#         shutil.copy(results_workspace + f"{rundate}_surveys_BandWtshd_Intersect.csv",
#                     reports_workspace + f"{rundate}_surveys_BandWtshd_Intersect.csv")
#         shutil.copy(results_workspace + f"{rundate}_surveys_Wtshd_Intersect.csv",
#                     reports_workspace + f"{rundate}_surveys_Wtshd_Intersect.csv")
#
#     # make tables folder
#     tables_workspace = reports_workspace + f"/{modelRunName}/Tables/"
#     os.makedirs(tables_workspace, exist_ok=True)
#
#     # open band table and sort
#     df_band = pd.read_csv(reports_workspace + f"{modelRunName}/{rundate}band_table.csv")
#     df_band['VOL_AF'] = df_band['VOL_AF'].apply(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)
#     df_band['AREA_MI2'] = df_band['AREA_MI2'].apply(lambda x: f"{x:,.1f}" if isinstance(x, (int, float)) else x)
#     df_band['SWE_IN'] = df_band['SWE_IN'].round(1)
#     df_band['Percent'] = df_band['Percent'].round(1)
#
#     # open and sort the sensors table
#     df_bnd_sens = pd.read_csv(reports_workspace + f"{rundate}_sensors_BandWtshd_Intersect.csv")
#     df_bnd_sens = df_bnd_sens[["SrtNmeBand", "SWE_freq"]]
#     df_bnd_sens = df_bnd_sens.rename(columns={"SWE_freq": "sensors"})
#
#     # open and sort the banded percent of average table
#     df_band_avg = pd.read_csv(reports_workspace + f"{averageRunName}/{rundate}anomBand_table.csv")
#     df_band_avg = df_band_avg[["SrtNmeBand", "Average"]]
#     df_band_avg = df_band_avg.rename(columns={"Average": "Avg"})
#
#     # open and sort SNODAS code
#     df_bnd_snodas = pd.read_csv(reports_workspace + f"{rundate}_band_SNODAS_swe_table.csv")
#     df_bnd_snodas = df_bnd_snodas[["SrtNmeBand", "SWE_IN"]]
#     df_bnd_snodas = df_bnd_snodas.rename(columns={"SWE_IN": "SNODAS"})
#
#     # merge tables together
#     merged_df = pd.merge(df_band, df_band_avg, on="SrtNmeBand", how="left")
#     merged_df = pd.merge(merged_df, df_bnd_sens, on="SrtNmeBand", how="left")
#     merged_df = pd.merge(merged_df, df_bnd_snodas, on="SrtNmeBand", how="left")
#
#     if surveys_use:
#         df_bnd_surv = pd.read_csv(reports_workspace + f"{rundate}_surveys_BandWtshd_Intersect.csv")
#         df_bnd_surv = df_bnd_surv[["SrtNmeBand", "SWE_freq"]]
#         df_bnd_surv = df_bnd_surv.rename(columns={"SWE_freq": "surveys"})
#         merged_df = pd.merge(merged_df, df_bnd_surv, on="SrtNmeBand", how="left")
#         merged_df['surveys'] = merged_df['surveys'].fillna('NA')
#
#     # merge to include NAs -- Check to see if this is done multiple times
#     merged_df['Avg'] = merged_df['Avg'].fillna("NA")
#     merged_df['Avg'] = merged_df['Avg'].apply(lambda x: int(round(x)) if x != "NA" else x)
#     merged_df['sensors'] = merged_df['sensors'].fillna('NA')
#
#     # separate into tables for domain
#     # merged_df['Basin_rw'] = merged_df['SrtNmeBand'].str[2:-5]
#     # state_df['Basin_rw'] = state_df['SrtNmeBand'].apply(lambda x: x[9:-8] if x[-2:] == "GT" else x[9:-6])
#     merged_df['Basin_rw'] = merged_df['SrtNmeBand'].apply(lambda x: x[2:-7] if x[-2:] == "GT" else x[2:-5])
#     merged_df['Num'] = merged_df['SrtNmeBand'].str[:2].astype(int).astype(str) + '.'
#     merged_df['Basin'] = merged_df['Num'] + " " + merged_df['Basin_rw']
#     # merged_df['Elevation Band'] = merged_df['SrtNmeBand'].str[-5:]
#     merged_df['Elevation Band'] = merged_df['SrtNmeBand'].apply(lambda x: x[-7:] if x[-2:] == "GT" else x[-5:])
#     merged_df['Elevation Band'] = merged_df['Elevation Band'].map(elevationBands)
#
#     if difference == "Y":
#         difference_cols = ['prev_SWE_IN', 'prev_sensors', 'prev_Avg']
#         df_band_prev = pd.read_csv(
#             prev_tables_workspace + f"SNM_{prev_rundate}_table10_raw.csv")
#         df_band_prev = df_band_prev.rename(
#             columns={"SWE_IN": "prev_SWE_IN", "sensors": "prev_sensors", "Avg": "prev_Avg"})
#         df_band_prev = df_band_prev[['Basin', 'Elevation Band', 'prev_SWE_IN', 'prev_sensors', 'prev_Avg']]
#         merged_df = pd.merge(merged_df, df_band_prev, on=['Basin', 'Elevation Band'], how='inner')
#
#         # edit and export
#         if not surveys_use:
#             df_band_tbl = merged_df[
#                 ['Basin', 'Elevation Band', 'SWE_IN', "Percent", "AREA_MI2", 'VOL_AF', "sensors", "Avg", "SNODAS",
#                  'prev_SWE_IN', 'prev_sensors', 'prev_Avg']]
#
#             # for export
#             df_band_export = merged_df[
#                 ["Basin", 'Elevation Band', 'prev_Avg', "Avg", "prev_SWE_IN", "SWE_IN", "Percent", "VOL_AF",
#                  "AREA_MI2", 'prev_sensors', "sensors", "SNODAS"]]
#             df_band_export = df_band_export.rename(
#                 columns={"prev_Avg": f"%{prev_date_abrev} Avg.", "Avg": f"%{date_abrev} Avg.",
#                          "prev_SWE_IN": "SWE (in)", "SWE_IN": "SWE (in)", "Percent": "%SCA",
#                          "VOL_AF": "Vol (AF)", "AREA_MI2": "Area (mi2)", "prev_sensors": "Pillows",
#                          "sensors": "Pillows", "SNODAS": "SNODAS*"})
#             top_header = ["", "", prev_formatted_date, formatted_date, prev_formatted_date, formatted_date,
#                           formatted_date, formatted_date, formatted_date, prev_formatted_date, formatted_date,
#                           formatted_date]
#             df_band_export.columns = pd.MultiIndex.from_arrays(
#                 [top_header, df_band_export.columns]
#             )
#             df_band_export.to_csv(
#                 tables_workspace + f"SNM_{rundate}_table10_final.csv")
#
#         if surveys_use:
#             df_band_tbl = merged_df[
#                 ['Basin', 'Elevation Band', 'SWE_IN', "Percent", "AREA_MI2", 'VOL_AF', "sensors", "Avg", "surveys",
#                  "SNODAS", 'prev_SWE_IN', 'prev_sensors', 'prev_Avg']]
#
#             # for export
#             df_band_export = merged_df[
#                 ["Basin", 'Elevation Band', 'prev_Avg', "Avg", "SWE_IN", "Percent", "VOL_AF", "AREA_MI2", "sensors",
#                  "surveys", "SNODAS"]]
#             df_band_export = df_band_export.rename(
#                 columns={"prev_Avg": f"%{prev_date_abrev} Avg.", "Avg": f"%{date_abrev} Avg.",
#                          "prev_SWE_IN": "SWE (in)", "SWE_IN": "SWE (in)", "Percent": "%SCA",
#                          "VOL_AF": "Vol (AF)", "AREA_MI2": "Area (mi2)", "prev_sensors": "Pillows",
#                          "sensors": "Pillows", "surveys": "Surveys", "SNODAS": "SNODAS*"})
#             top_header = ["", "", prev_formatted_date, formatted_date, prev_formatted_date, formatted_date,
#                           formatted_date, formatted_date, formatted_date, prev_formatted_date, formatted_date,
#                           survey_date, formatted_date]
#             df_band_export.columns = pd.MultiIndex.from_arrays(
#                 [top_header, df_band_export.columns]
#             )
#             df_band_export.to_csv(
#                 tables_workspace + f"SNM_{rundate}_table10_final.csv")
#
#     if difference == "N":
#         # edit and export
#         if not surveys_use:
#             df_band_tbl = merged_df[
#                 ['Basin', 'Elevation Band', 'SWE_IN', "Percent", "AREA_MI2", 'VOL_AF', "sensors", "Avg", "SNODAS"]]
#
#             # for export
#             df_band_export = merged_df[
#                 ["Basin", 'Elevation Band', "Avg", "SWE_IN", "Percent", "VOL_AF", "AREA_MI2", "sensors", "SNODAS"]]
#             df_band_export = df_band_export.rename(
#                 columns={"Avg": f"%{date_abrev} Avg.", "SWE_IN": "SWE (in)", "Percent": "%SCA",
#                          "VOL_AF": "Vol (AF)", "AREA_MI2": "Area (mi2)", "sensors": "Pillows", "SNODAS": "SNODAS*"})
#             top_header = ["", "", formatted_date, formatted_date, formatted_date, formatted_date, formatted_date,
#                           formatted_date, formatted_date]
#             df_band_export.columns = pd.MultiIndex.from_arrays(
#                 [top_header, df_band_export.columns]
#             )
#             df_band_export.to_csv(
#                 tables_workspace + f"SNM_{rundate}_table10_final.csv")
#
#         if surveys_use:
#             df_band_tbl = merged_df[
#                 ['Basin', 'Elevation Band', 'SWE_IN', "Percent", "AREA_MI2", 'VOL_AF', "sensors", "Avg", "surveys",
#                  "SNODAS"]]
#
#             # for export
#             df_band_export = merged_df[
#                 ["Basin", 'Elevation Band', "Avg", "SWE_IN", "Percent", "VOL_AF", "AREA_MI2", "sensors", "surveys",
#                  "SNODAS"]]
#             df_band_export = df_band_export.rename(
#                 columns={"Avg": f"%{date_abrev} Avg.", "SWE_IN": "SWE (in)", "Percent": "%SCA",
#                          "VOL_AF": "Vol (AF)", "AREA_MI2": "Area (mi2)", "sensors": "Pillows", "surveys": "Surveys",
#                          "SNODAS": "SNODAS*"})
#             top_header = ["", "", formatted_date, formatted_date, formatted_date, formatted_date, formatted_date,
#                           formatted_date, survey_date, formatted_date]
#             df_band_export.columns = pd.MultiIndex.from_arrays(
#                 [top_header, df_band_export.columns]
#             )
#             df_band_export.to_csv(
#                 tables_workspace + f"SNM_{rundate}_table10_final.csv")
#
#         # check for standard deviation values
#         swe_exceeded = pd.to_numeric(df_band_tbl['SWE_IN'], errors='coerce')
#         avg_exceeded = pd.to_numeric(df_band_tbl['Avg'], errors='coerce')
#         swe_threshold = swe_exceeded.max() + 2 * swe_exceeded.std()
#         avg_threshold = avg_exceeded.max() + 2 * avg_exceeded.std()
#         swe_exceeds = df_band_tbl.loc[swe_exceeded > swe_threshold]
#         avg_exceeds = df_band_tbl.loc[avg_exceeded > avg_threshold]
#         print("SWE rows exceeding threshold:")
#         print(swe_exceeds)
#
#         df_band_tbl.to_csv(tables_workspace + f"SNM_{rundate}_table10_raw.csv",
#                            index=False)
#
#     ###
#     print('Moving on to watershed table')
#     # getting watershed table
#     df_wtshd = pd.read_csv(reports_workspace + f"{modelRunName}/{rundate}Wtshd_table.csv")
#     df_wtshd['VOL_AF'] = df_wtshd['VOL_AF'].apply(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)
#     df_wtshd['AREA_MI2'] = df_wtshd['AREA_MI2'].apply(lambda x: f"{x:,.1f}" if isinstance(x, (int, float)) else x)
#     df_wtshd['SWE_IN'] = df_wtshd['SWE_IN'].round(1)
#     df_wtshd['Percent'] = df_wtshd['Percent'].round(1)
#     df_wtshd['region'] = df_wtshd["SrtName"].str[:5]
#
#     # get and sort the sensors table
#     df_wtshd_sens = pd.read_csv(reports_workspace + f"{rundate}_sensors_Wtshd_Intersect_stat.csv")
#     df_wtshd_sens = df_wtshd_sens[["SrtName", "SWE_freq"]]
#     df_wtshd_sens = df_wtshd_sens.rename(columns={"SWE_freq": "sensors"})
#
#     # get and sort the percent of average table
#     df_wtshd_avg = pd.read_csv(reports_workspace + f"{averageRunName}/{rundate}anomWtshd_table.csv")
#     df_wtshd_avg = df_wtshd_avg[["SrtName", "Average"]]
#     df_wtshd_avg = df_wtshd_avg.rename(columns={"Average": "Avg"})
#
#     # open and sort SNODAS code
#     df_wtshd_snodas = pd.read_csv(reports_workspace + f"{rundate}_SNODAS_swe_table.csv")
#     df_wtshd_snodas = df_wtshd_snodas[["SrtName", "SWE_IN"]]
#     df_wtshd_snodas = df_wtshd_snodas.rename(columns={"SWE_IN": "SNODAS"})
#
#     # merge tables together
#     merged_wtshd_df = pd.merge(df_wtshd, df_wtshd_avg, on="SrtName", how="left")
#     merged_wtshd_df = pd.merge(merged_wtshd_df, df_wtshd_sens, on="SrtName", how="left")
#     merged_wtshd_df = pd.merge(merged_wtshd_df, df_wtshd_snodas, on="SrtName", how="left")
#
#     if surveys_use:
#         df_wtshd_surv = pd.read_csv(reports_workspace + f"{rundate}_surveys_Wtshd_Intersect.csv")
#         df_wtshd_surv = df_wtshd_surv[["SrtName", "SWE_freq"]]
#         df_wtshd_surv = df_wtshd_surv.rename(columns={"SWE_freq": "surveys"})
#         merged_wtshd_df = pd.merge(merged_wtshd_df, df_wtshd_surv, on="SrtName", how="left")
#         merged_wtshd_df['surveys'] = merged_wtshd_df['surveys'].fillna('NA')
#
#     # merge to include NAs -- Check to see if this is done multiple times
#     merged_wtshd_df['Avg'] = merged_wtshd_df['Avg'].fillna("NA")
#     merged_wtshd_df['Avg'] = merged_wtshd_df['Avg'].apply(lambda x: int(round(x)) if x != "NA" else x)
#     merged_wtshd_df['sensors'] = merged_wtshd_df['sensors'].fillna('NA')
#
#     # separate into tables for domain
#     merged_wtshd_df['Basin_rw'] = merged_wtshd_df['SrtName'].str[2:]
#     merged_wtshd_df['Num'] = merged_wtshd_df['SrtName'].str[:2].astype(int).astype(str) + '.'
#     merged_wtshd_df['Basin'] = merged_wtshd_df['Num'] + " " + merged_wtshd_df['Basin_rw']
#
#     if difference == "Y":
#         difference_cols = ['prev_SWE_IN', 'prev_sensors', 'prev_Avg']
#         df_wtshed_prev = pd.read_csv(
#             prev_tables_workspace + f"SNM_{prev_rundate}_table5_raw.csv")
#         df_wtshed_prev = df_wtshed_prev.rename(
#             columns={"SWE_IN": "prev_SWE_IN", "sensors": "prev_sensors", "Avg": "prev_Avg"})
#         df_wtshed_prev = df_wtshed_prev[['Basin', 'prev_SWE_IN', 'prev_sensors', 'prev_Avg']]
#         merged_wtshd_df = pd.merge(merged_wtshd_df, df_wtshed_prev, on='Basin', how='inner')
#
#         # edit and export
#         if not surveys_use:
#             df_wtshd_tbl = merged_wtshd_df[
#                 ['Basin', 'SWE_IN', "Percent", "AREA_MI2", 'VOL_AF', "sensors", "Avg", "SNODAS", 'prev_SWE_IN',
#                  'prev_sensors', 'prev_Avg']]
#
#             # for export
#             df_wtshd_export = merged_wtshd_df[
#                 ["Basin", 'prev_Avg', "Avg", 'prev_SWE_IN', "SWE_IN", "Percent", "VOL_AF", "AREA_MI2",
#                  "prev_sensors", "sensors", "SNODAS"]]
#             df_wtshd_export = df_wtshd_export.rename(
#                 columns={"prev_Avg": f"%{prev_date_abrev} Avg.", "Avg": f"%{date_abrev} Avg.",
#                          "prev_SWE_IN": "SWE (in)", "SWE_IN": "SWE (in)", "Percent": "%SCA",
#                          "VOL_AF": "Vol (AF)", "AREA_MI2": "Area (mi2)", "prev_sensors": "Pillows",
#                          "sensors": "Pillows", "SNODAS": "SNODAS*"})
#             top_header = ["", prev_formatted_date, formatted_date, prev_formatted_date, formatted_date,
#                           formatted_date, formatted_date, formatted_date, prev_formatted_date, formatted_date,
#                           formatted_date]
#             df_wtshd_export.columns = pd.MultiIndex.from_arrays(
#                 [top_header, df_wtshd_export.columns]
#             )
#             df_wtshd_export.to_csv(
#                 tables_workspace + f"SNM_{rundate}_table5_final.csv")
#
#         if surveys_use:
#             df_wtshd_tbl = merged_wtshd_df[
#                 ['Basin', 'SWE_IN', "Percent", "AREA_MI2", 'VOL_AF', "sensors", "Avg", "surveys", "SNODAS",
#                  'prev_SWE_IN', 'prev_sensors', 'prev_Avg']]
#             # for export
#             df_wtshd_export = merged_wtshd_df[
#                 ["Basin", 'prev_Avg', "Avg", "SWE_IN", "Percent", "VOL_AF", "AREA_MI2", "sensors", "surveys",
#                  "SNODAS"]]
#             df_wtshd_export = df_wtshd_export.rename(
#                 columns={"prev_Avg": f"%{prev_date_abrev} Avg.", "Avg": f"%{date_abrev} Avg.",
#                          "prev_SWE_IN": "SWE (in)", "SWE_IN": "SWE (in)", "Percent": "%SCA",
#                          "VOL_AF": "Vol (AF)", "AREA_MI2": "Area (mi2)", "prev_sensors": "Pillows",
#                          "sensors": "Pillows", "surveys": "Surveys", "SNODAS": "SNODAS*"})
#             top_header = ["", prev_formatted_date, formatted_date, prev_formatted_date, formatted_date,
#                           formatted_date, formatted_date, formatted_date, prev_formatted_date, formatted_date,
#                           survey_date, formatted_date]
#             df_wtshd_export.columns = pd.MultiIndex.from_arrays(
#                 [top_header, df_wtshd_export.columns]
#             )
#             df_wtshd_export.to_csv(
#                 tables_workspace + f"SNM_{rundate}_table5_final.csv")
#
#     if difference == "N":
#         # edit and export
#         if not surveys_use:
#             df_wtshd_tbl = merged_wtshd_df[
#                 ['Basin', 'SWE_IN', "Percent", "AREA_MI2", 'VOL_AF', "sensors", "Avg", "SNODAS"]]
#
#             # for export
#             df_wtshd_export = merged_wtshd_df[
#                 ["Basin", "Avg", "SWE_IN", "Percent", "VOL_AF", "AREA_MI2", "sensors", "SNODAS"]]
#             df_wtshd_export = df_wtshd_export.rename(
#                 columns={"Avg": f"%{date_abrev} Avg.", "SWE_IN": "SWE (in)", "Percent": "%SCA",
#                          "VOL_AF": "Vol (AF)", "AREA_MI2": "Area (mi2)", "sensors": "Pillows", "SNODAS": "SNODAS*"})
#             top_header = ["", formatted_date, formatted_date, formatted_date, formatted_date, formatted_date,
#                           formatted_date, formatted_date]
#             df_wtshd_export.columns = pd.MultiIndex.from_arrays(
#                 [top_header, df_wtshd_export.columns]
#             )
#             df_wtshd_export.to_csv(
#                 tables_workspace + f"SNM_{rundate}_table5_final.csv")
#
#         if surveys_use:
#             df_wtshd_tbl = merged_wtshd_df[
#                 ['Basin', 'SWE_IN', "Percent", "AREA_MI2", 'VOL_AF', "sensors", "Avg", "surveys", "SNODAS"]]
#
#             # for export
#             df_wtshd_export = merged_wtshd_df[
#                 ["Basin", "Avg", "SWE_IN", "Percent", "VOL_AF", "AREA_MI2", "sensors", "surveys", "SNODAS"]]
#             df_wtshd_export = df_wtshd_export.rename(
#                 columns={"Avg": f"%{date_abrev} Avg.", "SWE_IN": "SWE (in)", "Percent": "%SCA",
#                          "VOL_AF": "Vol (AF)", "AREA_MI2": "Area (mi2)", "sensors": "Pillows", "SNODAS": "SNODAS*"})
#             top_header = ["", formatted_date, formatted_date, formatted_date, formatted_date, formatted_date,
#                           formatted_date, survey_date, formatted_date]
#             df_wtshd_export.columns = pd.MultiIndex.from_arrays(
#                 [top_header, df_wtshd_export.columns]
#             )
#             df_wtshd_export.to_csv(
#                 tables_workspace + f"SNM_{rundate}_table5_final.csv")
#
#     df_wtshd_tbl.to_csv(tables_workspace + f"SNM_{rundate}_table5_raw.csv",
#                         index=False)
def SNM_tables_for_report(rundate, modelRunName, averageRunName, results_workspace, reports_workspace, difference,
                          aso_bc_basins, aso_symbol, prev_tables_workspace=None, survey_date=None, prev_rundate=None,
                          surveys_use=False):
    # dictionaries
    elevationBands = {
        "-1000": "< 0", "00000": "0", "01000": "1,000-2,000'", "02000": "2,000-3,000'", "03000": "3,000-4,000'",
        "04000": "4,000-5,000'",
        "05000": "5,000-6,000'", "06000": "6,000-7,000'", "07000": "7,000-8,000'", "08000": "8,000-9,000'",
        "09000": "9,000-10,000'",
        "10000": "10,000-11,000'", "11000": "11,000-12,000'", "12000": "12,000-13,000'", "13000": "13,000-14,000'",
        "14000": "14,000-15,000'",
        "14000GT": ">14,000'", "13000GT": ">13,000'", "12000GT": ">12,000'", "11000GT": ">11,000'",
        "10000GT": ">10,000'", "09000GT": ">9,000'", "08000GT": ">8,000'",
        "07000GT": ">7,000'", "06000GT": ">6,000'", "05000GT": ">5,000'"}

    # Add a helper function to append the symbol
    def add_special_symbol(basin_name, aso_bc_basins, aso_symbol):
        """Add symbol to basin name if it's in the special list"""
        if aso_bc_basins is None or len(aso_bc_basins) == 0:
            return basin_name
        if basin_name in aso_bc_basins:
            return f"{basin_name}{aso_symbol}"
        return basin_name

    ## set new date structure
    date_obj = datetime.strptime(rundate, "%Y%m%d")
    formatted_date = f"{date_obj.month}/{date_obj.day}/{date_obj.year}"
    date_abrev = f"{date_obj.month}/{date_obj.day}"

    if difference == "Y":
        prev_date_obj = datetime.strptime(prev_rundate, "%Y%m%d")
        prev_formatted_date = f"{prev_date_obj.month}/{prev_date_obj.day}/{prev_date_obj.year}"
        prev_date_abrev = f"{prev_date_obj.month}/{prev_date_obj.day}"

    if surveys_use:
        surv_date_obj = datetime.strptime(survey_date, "%Y%m%d")
        surv_date_abrev = f"{surv_date_obj.month}/{surv_date_obj.day}"

    # copy over files
    shutil.copy(results_workspace + f"{modelRunName}/{rundate}anomBand_table.csv",
                reports_workspace + f"{modelRunName}/{rundate}anomBand_table.csv")
    shutil.copy(results_workspace + f"{modelRunName}/{rundate}anomWtshd_table.csv",
                reports_workspace + f"{modelRunName}/{rundate}anomWtshd_table.csv")
    shutil.copy(results_workspace + f"{modelRunName}/{rundate}anomWtshd_table.csv",
                reports_workspace + f"{modelRunName}/{rundate}anomWtshd_table.csv")
    shutil.copy(results_workspace + f"{modelRunName}/{rundate}band_table.csv",
                reports_workspace + f"{modelRunName}/{rundate}band_table.csv")
    shutil.copy(results_workspace + f"{modelRunName}/{rundate}Wtshd_table.csv",
                reports_workspace + f"{modelRunName}/{rundate}Wtshd_table.csv")
    shutil.copy(results_workspace + f"{averageRunName}/{rundate}anomBand_table.csv",
                reports_workspace + f"{averageRunName}/{rundate}anomBand_table.csv")
    shutil.copy(results_workspace + f"{averageRunName}/{rundate}anomWtshd_table.csv",
                reports_workspace + f"{averageRunName}/{rundate}anomWtshd_table.csv")
    shutil.copy(results_workspace + f"{averageRunName}/{rundate}anomWtshd_table.csv",
                reports_workspace + f"{averageRunName}/{rundate}anomWtshd_table.csv")
    shutil.copy(results_workspace + f"{averageRunName}/{rundate}band_table.csv",
                reports_workspace + f"{averageRunName}/{rundate}band_table.csv")
    shutil.copy(results_workspace + f"{averageRunName}/{rundate}Wtshd_table.csv",
                reports_workspace + f"{averageRunName}/{rundate}Wtshd_table.csv")
    shutil.copy(results_workspace + f"SNODAS/{rundate}_band_SNODAS_swe_table.csv",
                reports_workspace + f"{rundate}_band_SNODAS_swe_table.csv")
    shutil.copy(results_workspace + f"SNODAS/{rundate}_SNODAS_swe_table.csv",
                reports_workspace + f"{rundate}_SNODAS_swe_table.csv")
    shutil.copy(results_workspace + f"{rundate}_sensors_Wtshd_Intersect_stat.csv",
                reports_workspace + f"{rundate}_sensors_Wtshd_Intersect_stat.csv")
    shutil.copy(results_workspace + f"{rundate}_sensors_BandWtshd_Intersect.csv",
                reports_workspace + f"{rundate}_sensors_BandWtshd_Intersect.csv")

    if surveys_use:
        shutil.copy(results_workspace + f"{rundate}_surveys_BandWtshd_Intersect.csv",
                    reports_workspace + f"{rundate}_surveys_BandWtshd_Intersect.csv")
        shutil.copy(results_workspace + f"{rundate}_surveys_Wtshd_Intersect.csv",
                    reports_workspace + f"{rundate}_surveys_Wtshd_Intersect.csv")

    # make tables folder
    tables_workspace = reports_workspace + f"/{modelRunName}/Tables/"
    os.makedirs(tables_workspace, exist_ok=True)

    # open band table and sort
    df_band = pd.read_csv(reports_workspace + f"{modelRunName}/{rundate}band_table.csv")
    df_band['VOL_AF'] = df_band['VOL_AF'].apply(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)
    df_band['AREA_MI2'] = df_band['AREA_MI2'].apply(lambda x: f"{x:,.1f}" if isinstance(x, (int, float)) else x)
    df_band['SWE_IN'] = df_band['SWE_IN'].round(1)
    df_band['Percent'] = df_band['Percent'].round(1)

    # open and sort the sensors table
    df_bnd_sens = pd.read_csv(reports_workspace + f"{rundate}_sensors_BandWtshd_Intersect.csv")
    df_bnd_sens = df_bnd_sens[["SrtNmeBand", "SWE_freq"]]
    df_bnd_sens = df_bnd_sens.rename(columns={"SWE_freq": "sensors"})

    # open and sort the banded percent of average table
    df_band_avg = pd.read_csv(reports_workspace + f"{averageRunName}/{rundate}anomBand_table.csv")
    df_band_avg = df_band_avg[["SrtNmeBand", "Average"]]
    df_band_avg = df_band_avg.rename(columns={"Average": "Avg"})

    # open and sort SNODAS code
    df_bnd_snodas = pd.read_csv(reports_workspace + f"{rundate}_band_SNODAS_swe_table.csv")
    df_bnd_snodas = df_bnd_snodas[["SrtNmeBand", "SWE_IN"]]
    df_bnd_snodas = df_bnd_snodas.rename(columns={"SWE_IN": "SNODAS"})

    # merge tables together
    merged_df = pd.merge(df_band, df_band_avg, on="SrtNmeBand", how="left")
    merged_df = pd.merge(merged_df, df_bnd_sens, on="SrtNmeBand", how="left")
    merged_df = pd.merge(merged_df, df_bnd_snodas, on="SrtNmeBand", how="left")

    if surveys_use:
        df_bnd_surv = pd.read_csv(reports_workspace + f"{rundate}_surveys_BandWtshd_Intersect.csv")
        df_bnd_surv = df_bnd_surv[["SrtNmeBand", "SWE_freq"]]
        df_bnd_surv = df_bnd_surv.rename(columns={"SWE_freq": "surveys"})
        merged_df = pd.merge(merged_df, df_bnd_surv, on="SrtNmeBand", how="left")
        merged_df['surveys'] = merged_df['surveys'].fillna('NA')

    # merge to include NAs -- Check to see if this is done multiple times
    merged_df['Avg'] = merged_df['Avg'].fillna("NA")
    merged_df['Avg'] = merged_df['Avg'].apply(lambda x: int(round(x)) if x != "NA" else x)
    merged_df['sensors'] = merged_df['sensors'].fillna('NA')

    # separate into tables for domain
    merged_df['Basin_rw'] = merged_df['SrtNmeBand'].apply(lambda x: x[2:-7] if x[-2:] == "GT" else x[2:-5])
    merged_df['Basin_rw'] = merged_df['Basin_rw'].apply(lambda x: add_special_symbol(x, aso_bc_basins, aso_symbol))
    merged_df['Num'] = merged_df['SrtNmeBand'].str[:2].astype(int).astype(str) + '.'
    merged_df['Basin'] = merged_df['Num'] + " " + merged_df['Basin_rw']
    merged_df['Elevation Band'] = merged_df['SrtNmeBand'].apply(lambda x: x[-7:] if x[-2:] == "GT" else x[-5:])
    merged_df['Elevation Band'] = merged_df['Elevation Band'].map(elevationBands)

    if difference == "Y":
        difference_cols = ['prev_SWE_IN', 'prev_sensors', 'prev_Avg']
        df_band_prev = pd.read_csv(
            prev_tables_workspace + f"SNM_{prev_rundate}_table10_raw.csv")
        df_band_prev = df_band_prev.rename(
            columns={"SWE_IN": "prev_SWE_IN", "sensors": "prev_sensors", "Avg": "prev_Avg"})
        df_band_prev = df_band_prev[['Basin', 'Elevation Band', 'prev_SWE_IN', 'prev_sensors', 'prev_Avg']]
        merged_df = pd.merge(merged_df, df_band_prev, on=['Basin', 'Elevation Band'], how='inner')

        # edit and export
        if not surveys_use:
            df_band_tbl = merged_df[
                ['Basin', 'Elevation Band', 'SWE_IN', "Percent", "AREA_MI2", 'VOL_AF', "sensors", "Avg", "SNODAS",
                 'prev_SWE_IN', 'prev_sensors', 'prev_Avg']]

            # for export
            df_band_export = merged_df[
                ["Basin", 'Elevation Band', 'prev_Avg', "Avg", "prev_SWE_IN", "SWE_IN", "Percent", "VOL_AF",
                 "AREA_MI2", 'prev_sensors', "sensors", "SNODAS"]]
            df_band_export = df_band_export.rename(
                columns={"prev_Avg": f"%{prev_date_abrev} Avg.", "Avg": f"%{date_abrev} Avg.",
                         "prev_SWE_IN": "SWE (in)", "SWE_IN": "SWE (in)", "Percent": "%SCA",
                         "VOL_AF": "Vol (AF)", "AREA_MI2": "Area (mi2)", "prev_sensors": "Pillows",
                         "sensors": "Pillows", "SNODAS": "SNODAS*"})
            top_header = ["", "", prev_formatted_date, formatted_date, prev_formatted_date, formatted_date,
                          formatted_date, formatted_date, formatted_date, prev_formatted_date, formatted_date,
                          formatted_date]
            df_band_export.columns = pd.MultiIndex.from_arrays(
                [top_header, df_band_export.columns]
            )
            df_band_export.to_csv(
                tables_workspace + f"SNM_{rundate}_table10_final.csv")

        if surveys_use:
            df_band_tbl = merged_df[
                ['Basin', 'Elevation Band', 'SWE_IN', "Percent", "AREA_MI2", 'VOL_AF', "sensors", "Avg", "surveys",
                 "SNODAS", 'prev_SWE_IN', 'prev_sensors', 'prev_Avg']]

            # for export
            df_band_export = merged_df[
                ["Basin", 'Elevation Band', 'prev_Avg', "Avg", "SWE_IN", "Percent", "VOL_AF", "AREA_MI2", "sensors",
                 "surveys", "SNODAS"]]
            df_band_export = df_band_export.rename(
                columns={"prev_Avg": f"%{prev_date_abrev} Avg.", "Avg": f"%{date_abrev} Avg.",
                         "prev_SWE_IN": "SWE (in)", "SWE_IN": "SWE (in)", "Percent": "%SCA",
                         "VOL_AF": "Vol (AF)", "AREA_MI2": "Area (mi2)", "prev_sensors": "Pillows",
                         "sensors": "Pillows", "surveys": "Surveys", "SNODAS": "SNODAS*"})
            top_header = ["", "", prev_formatted_date, formatted_date, prev_formatted_date, formatted_date,
                          formatted_date, formatted_date, formatted_date, prev_formatted_date, formatted_date,
                          survey_date, formatted_date]
            df_band_export.columns = pd.MultiIndex.from_arrays(
                [top_header, df_band_export.columns]
            )
            df_band_export.to_csv(
                tables_workspace + f"SNM_{rundate}_table10_final.csv")

    if difference == "N":
        # edit and export
        if not surveys_use:
            df_band_tbl = merged_df[
                ['Basin', 'Elevation Band', 'SWE_IN', "Percent", "AREA_MI2", 'VOL_AF', "sensors", "Avg", "SNODAS"]]

            # for export
            df_band_export = merged_df[
                ["Basin", 'Elevation Band', "Avg", "SWE_IN", "Percent", "VOL_AF", "AREA_MI2", "sensors", "SNODAS"]]
            df_band_export = df_band_export.rename(
                columns={"Avg": f"%{date_abrev} Avg.", "SWE_IN": "SWE (in)", "Percent": "%SCA",
                         "VOL_AF": "Vol (AF)", "AREA_MI2": "Area (mi2)", "sensors": "Pillows", "SNODAS": "SNODAS*"})
            top_header = ["", "", formatted_date, formatted_date, formatted_date, formatted_date, formatted_date,
                          formatted_date, formatted_date]
            df_band_export.columns = pd.MultiIndex.from_arrays(
                [top_header, df_band_export.columns]
            )
            df_band_export.to_csv(
                tables_workspace + f"SNM_{rundate}_table10_final.csv")

        if surveys_use:
            df_band_tbl = merged_df[
                ['Basin', 'Elevation Band', 'SWE_IN', "Percent", "AREA_MI2", 'VOL_AF', "sensors", "Avg", "surveys",
                 "SNODAS"]]

            # for export
            df_band_export = merged_df[
                ["Basin", 'Elevation Band', "Avg", "SWE_IN", "Percent", "VOL_AF", "AREA_MI2", "sensors", "surveys",
                 "SNODAS"]]
            df_band_export = df_band_export.rename(
                columns={"Avg": f"%{date_abrev} Avg.", "SWE_IN": "SWE (in)", "Percent": "%SCA",
                         "VOL_AF": "Vol (AF)", "AREA_MI2": "Area (mi2)", "sensors": "Pillows", "surveys": "Surveys",
                         "SNODAS": "SNODAS*"})
            top_header = ["", "", formatted_date, formatted_date, formatted_date, formatted_date, formatted_date,
                          formatted_date, survey_date, formatted_date]
            df_band_export.columns = pd.MultiIndex.from_arrays(
                [top_header, df_band_export.columns]
            )
            df_band_export.to_csv(
                tables_workspace + f"SNM_{rundate}_table10_final.csv")

        # check for standard deviation values
        swe_exceeded = pd.to_numeric(df_band_tbl['SWE_IN'], errors='coerce')
        avg_exceeded = pd.to_numeric(df_band_tbl['Avg'], errors='coerce')
        swe_threshold = swe_exceeded.max() + 2 * swe_exceeded.std()
        avg_threshold = avg_exceeded.max() + 2 * avg_exceeded.std()
        swe_exceeds = df_band_tbl.loc[swe_exceeded > swe_threshold]
        avg_exceeds = df_band_tbl.loc[avg_exceeded > avg_threshold]
        print("SWE rows exceeding threshold:")
        print(swe_exceeds)

        df_band_tbl.to_csv(tables_workspace + f"SNM_{rundate}_table10_raw.csv",
                           index=False)

    ###
    print('Moving on to watershed table')
    # getting watershed table
    df_wtshd = pd.read_csv(reports_workspace + f"{modelRunName}/{rundate}Wtshd_table.csv")
    df_wtshd['VOL_AF'] = df_wtshd['VOL_AF'].apply(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)
    df_wtshd['AREA_MI2'] = df_wtshd['AREA_MI2'].apply(lambda x: f"{x:,.1f}" if isinstance(x, (int, float)) else x)
    df_wtshd['SWE_IN'] = df_wtshd['SWE_IN'].round(1)
    df_wtshd['Percent'] = df_wtshd['Percent'].round(1)
    df_wtshd['region'] = df_wtshd["SrtName"].str[:5]

    # get and sort the sensors table
    df_wtshd_sens = pd.read_csv(reports_workspace + f"{rundate}_sensors_Wtshd_Intersect_stat.csv")
    df_wtshd_sens = df_wtshd_sens[["SrtName", "SWE_freq"]]
    df_wtshd_sens = df_wtshd_sens.rename(columns={"SWE_freq": "sensors"})

    # get and sort the percent of average table
    df_wtshd_avg = pd.read_csv(reports_workspace + f"{averageRunName}/{rundate}anomWtshd_table.csv")
    df_wtshd_avg = df_wtshd_avg[["SrtName", "Average"]]
    df_wtshd_avg = df_wtshd_avg.rename(columns={"Average": "Avg"})

    # open and sort SNODAS code
    df_wtshd_snodas = pd.read_csv(reports_workspace + f"{rundate}_SNODAS_swe_table.csv")
    df_wtshd_snodas = df_wtshd_snodas[["SrtName", "SWE_IN"]]
    df_wtshd_snodas = df_wtshd_snodas.rename(columns={"SWE_IN": "SNODAS"})

    # merge tables together
    merged_wtshd_df = pd.merge(df_wtshd, df_wtshd_avg, on="SrtName", how="left")
    merged_wtshd_df = pd.merge(merged_wtshd_df, df_wtshd_sens, on="SrtName", how="left")
    merged_wtshd_df = pd.merge(merged_wtshd_df, df_wtshd_snodas, on="SrtName", how="left")

    if surveys_use:
        df_wtshd_surv = pd.read_csv(reports_workspace + f"{rundate}_surveys_Wtshd_Intersect.csv")
        df_wtshd_surv = df_wtshd_surv[["SrtName", "SWE_freq"]]
        df_wtshd_surv = df_wtshd_surv.rename(columns={"SWE_freq": "surveys"})
        merged_wtshd_df = pd.merge(merged_wtshd_df, df_wtshd_surv, on="SrtName", how="left")
        merged_wtshd_df['surveys'] = merged_wtshd_df['surveys'].fillna('NA')

    # merge to include NAs -- Check to see if this is done multiple times
    merged_wtshd_df['Avg'] = merged_wtshd_df['Avg'].fillna("NA")
    merged_wtshd_df['Avg'] = merged_wtshd_df['Avg'].apply(lambda x: int(round(x)) if x != "NA" else x)
    merged_wtshd_df['sensors'] = merged_wtshd_df['sensors'].fillna('NA')

    # separate into tables for domain
    merged_wtshd_df['Basin_rw'] = merged_wtshd_df['SrtName'].str[2:]
    merged_wtshd_df['Basin_rw'] = merged_wtshd_df['Basin_rw'].apply(
        lambda x: add_special_symbol(x, aso_bc_basins, aso_symbol))
    merged_wtshd_df['Num'] = merged_wtshd_df['SrtName'].str[:2].astype(int).astype(str) + '.'
    merged_wtshd_df['Basin'] = merged_wtshd_df['Num'] + " " + merged_wtshd_df['Basin_rw']

    if difference == "Y":
        difference_cols = ['prev_SWE_IN', 'prev_sensors', 'prev_Avg']
        df_wtshed_prev = pd.read_csv(
            prev_tables_workspace + f"SNM_{prev_rundate}_table5_raw.csv")
        df_wtshed_prev = df_wtshed_prev.rename(
            columns={"SWE_IN": "prev_SWE_IN", "sensors": "prev_sensors", "Avg": "prev_Avg"})
        df_wtshed_prev = df_wtshed_prev[['Basin', 'prev_SWE_IN', 'prev_sensors', 'prev_Avg']]
        merged_wtshd_df = pd.merge(merged_wtshd_df, df_wtshed_prev, on='Basin', how='inner')

        # edit and export
        if not surveys_use:
            df_wtshd_tbl = merged_wtshd_df[
                ['Basin', 'SWE_IN', "Percent", "AREA_MI2", 'VOL_AF', "sensors", "Avg", "SNODAS", 'prev_SWE_IN',
                 'prev_sensors', 'prev_Avg']]

            # for export
            df_wtshd_export = merged_wtshd_df[
                ["Basin", 'prev_Avg', "Avg", 'prev_SWE_IN', "SWE_IN", "Percent", "VOL_AF", "AREA_MI2",
                 "prev_sensors", "sensors", "SNODAS"]]
            df_wtshd_export = df_wtshd_export.rename(
                columns={"prev_Avg": f"%{prev_date_abrev} Avg.", "Avg": f"%{date_abrev} Avg.",
                         "prev_SWE_IN": "SWE (in)", "SWE_IN": "SWE (in)", "Percent": "%SCA",
                         "VOL_AF": "Vol (AF)", "AREA_MI2": "Area (mi2)", "prev_sensors": "Pillows",
                         "sensors": "Pillows", "SNODAS": "SNODAS*"})
            top_header = ["", prev_formatted_date, formatted_date, prev_formatted_date, formatted_date,
                          formatted_date, formatted_date, formatted_date, prev_formatted_date, formatted_date,
                          formatted_date]
            df_wtshd_export.columns = pd.MultiIndex.from_arrays(
                [top_header, df_wtshd_export.columns]
            )
            df_wtshd_export.to_csv(
                tables_workspace + f"SNM_{rundate}_table5_final.csv")

        if surveys_use:
            df_wtshd_tbl = merged_wtshd_df[
                ['Basin', 'SWE_IN', "Percent", "AREA_MI2", 'VOL_AF', "sensors", "Avg", "surveys", "SNODAS",
                 'prev_SWE_IN', 'prev_sensors', 'prev_Avg']]
            # for export
            df_wtshd_export = merged_wtshd_df[
                ["Basin", 'prev_Avg', "Avg", "SWE_IN", "Percent", "VOL_AF", "AREA_MI2", "sensors", "surveys",
                 "SNODAS"]]
            df_wtshd_export = df_wtshd_export.rename(
                columns={"prev_Avg": f"%{prev_date_abrev} Avg.", "Avg": f"%{date_abrev} Avg.",
                         "prev_SWE_IN": "SWE (in)", "SWE_IN": "SWE (in)", "Percent": "%SCA",
                         "VOL_AF": "Vol (AF)", "AREA_MI2": "Area (mi2)", "prev_sensors": "Pillows",
                         "sensors": "Pillows", "surveys": "Surveys", "SNODAS": "SNODAS*"})
            top_header = ["", prev_formatted_date, formatted_date, prev_formatted_date, formatted_date,
                          formatted_date, formatted_date, formatted_date, prev_formatted_date, formatted_date,
                          survey_date, formatted_date]
            df_wtshd_export.columns = pd.MultiIndex.from_arrays(
                [top_header, df_wtshd_export.columns]
            )
            df_wtshd_export.to_csv(
                tables_workspace + f"SNM_{rundate}_table5_final.csv")

    if difference == "N":
        # edit and export
        if not surveys_use:
            df_wtshd_tbl = merged_wtshd_df[
                ['Basin', 'SWE_IN', "Percent", "AREA_MI2", 'VOL_AF', "sensors", "Avg", "SNODAS"]]

            # for export
            df_wtshd_export = merged_wtshd_df[
                ["Basin", "Avg", "SWE_IN", "Percent", "VOL_AF", "AREA_MI2", "sensors", "SNODAS"]]
            df_wtshd_export = df_wtshd_export.rename(
                columns={"Avg": f"%{date_abrev} Avg.", "SWE_IN": "SWE (in)", "Percent": "%SCA",
                         "VOL_AF": "Vol (AF)", "AREA_MI2": "Area (mi2)", "sensors": "Pillows", "SNODAS": "SNODAS*"})
            top_header = ["", formatted_date, formatted_date, formatted_date, formatted_date, formatted_date,
                          formatted_date, formatted_date]
            df_wtshd_export.columns = pd.MultiIndex.from_arrays(
                [top_header, df_wtshd_export.columns]
            )
            df_wtshd_export.to_csv(
                tables_workspace + f"SNM_{rundate}_table5_final.csv")

        if surveys_use:
            df_wtshd_tbl = merged_wtshd_df[
                ['Basin', 'SWE_IN', "Percent", "AREA_MI2", 'VOL_AF', "sensors", "Avg", "surveys", "SNODAS"]]

            # for export
            df_wtshd_export = merged_wtshd_df[
                ["Basin", "Avg", "SWE_IN", "Percent", "VOL_AF", "AREA_MI2", "sensors", "surveys", "SNODAS"]]
            df_wtshd_export = df_wtshd_export.rename(
                columns={"Avg": f"%{date_abrev} Avg.", "SWE_IN": "SWE (in)", "Percent": "%SCA",
                         "VOL_AF": "Vol (AF)", "AREA_MI2": "Area (mi2)", "sensors": "Pillows", "SNODAS": "SNODAS*"})
            top_header = ["", formatted_date, formatted_date, formatted_date, formatted_date, formatted_date,
                          formatted_date, survey_date, formatted_date]
            df_wtshd_export.columns = pd.MultiIndex.from_arrays(
                [top_header, df_wtshd_export.columns]
            )
            df_wtshd_export.to_csv(
                tables_workspace + f"SNM_{rundate}_table5_final.csv")

    df_wtshd_tbl.to_csv(tables_workspace + f"SNM_{rundate}_table5_raw.csv",
                        index=False)