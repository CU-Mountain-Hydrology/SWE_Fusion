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
                        f"SPIRES_NRT_{tile}_MOD09GA061_{yyyymmddd_str}_V1.0.nc")
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

import rasterio
import numpy as np
import os
from datetime import datetime, timedelta

def calculate_dmfsca(
    fSCA_folder,
    output_folder,
    wateryear_start,
    process_start_date,
    process_end_date
):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Sort .tif files
    raster_files = sorted([f for f in os.listdir(fSCA_folder) if f.endswith(".tif")])

    # Build dictionary mapping date -> filepath
    raster_dict = {}
    for f in raster_files:
        try:
            date_str = os.path.splitext(f)[0]
            file_date = datetime.strptime(date_str, "%Y%m%d")
            raster_dict[file_date] = os.path.join(fSCA_folder, f)
        except ValueError:
            continue

    sorted_dates = sorted(raster_dict.keys())

    for current_date in sorted_dates:
        if current_date < process_start_date or current_date > process_end_date:
            continue

        # Get all dates from wateryear start to current date
        date_subset = [d for d in sorted_dates if wateryear_start <= d <= current_date]

        sum_array = None
        count = 0

        for d in date_subset:
            with rasterio.open(raster_dict[d]) as src:
                data = src.read(1).astype(np.float32)
                mask = data == src.nodata
                data[mask] = 0
                if sum_array is None:
                    sum_array = np.zeros_like(data)
                    valid_mask = np.zeros_like(data, dtype=np.int32)
                sum_array += data
                valid_mask += ~mask
                profile = src.profile

        # Calculate average
        with np.errstate(invalid='ignore'):
            avg_array = np.divide(sum_array, valid_mask, where=valid_mask != 0)
            avg_array[valid_mask == 0] = profile['nodata']

        out_filename = os.path.join(output_folder, f"{current_date.strftime('%Y%m%d')}_dmfsca.tif")
        with rasterio.open(out_filename, "w", **profile) as dst:
            dst.write(avg_array, 1)

# downloading snow surveys
import os
import requests
import pandas as pd
import glob
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime

def download_snow_surveys(report_date, survey_workspace, results_workspace, WW_url_file, NRCS_shp, WW_state_list):
    """
    Downloads, cleans, merges, and georeferences NRCS survey data for a given rundate.

    Parameters:
        rundate (str): e.g. "20250401"
        surveyWorkspace (str): Base directory for output folder
        resultsWorkspace (str): Directory to save final results
        url_file (str): Path to .txt file with URLs formatted as "STATE_ABBR|URL"
        NRCS_shp (str): Path to NRCS course shapefile
        state_list (list): List of 2-letter state abbreviations (e.g. ["CO", "UT", ...])
    """

    # ---- Set up workspace ----
    path = os.path.join(survey_workspace, report_date)
    os.makedirs(path, exist_ok=True)
    snowCourseWorkspace = os.path.join(survey_workspace, report_date)
    date_obj = datetime.strptime(report_date, "%Y%m%d")
    month = date_obj.strftime("%B")
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

    for url, text, edit, state in zip(state_url_list, state_text_list, state_edit_list, WW_state_list):
        # Download text data
        state_data = requests.get(url)
        with open(text, 'w') as out_f:
            out_f.write(state_data.text)

        # Remove headers and blank lines
        with open(text, "r") as file:
            content = file.readlines()

        marker = [i for i, line in enumerate(content) if line.startswith("#") or line == "\n"]

        with open(edit, "w") as file:
            for i, line in enumerate(content):
                if i not in marker:
                    file.write(line)

        # Convert cleaned text to CSV
        df = pd.read_csv(edit, sep=",")
        df.to_csv(f"{edit[:-4]}.csv", index=False)

        # Clean and structure CSV
        df = pd.read_csv(f"{edit[:-4]}.csv")
        df = df[~df[month].astype(str).str.contains('Snow Water', na=False)]
        df.drop(columns=[col for col in df.columns if "Unnamed" in col], inplace=True)
        df['State'] = state
        df = df[['Station Id', 'Station Name', 'Water Year', month, 'State']]
        df.to_csv(f"{edit[:-4]}_update.csv", index=False)

    # Merge all updated CSVs
    all_update_csvs = glob.glob(os.path.join(snowCourseWorkspace, "*_update.csv"))
    df_list = [pd.read_csv(csv) for csv in all_update_csvs]
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df["SWE_in"] = merged_df[month]
    merged_df["SWE_m"] = merged_df["SWE_in"] * 0.0254
    merged_df.to_csv(os.path.join(snowCourseWorkspace, f"{report_date}_WestWide_surveys.csv"), index=False)

    # Merge with shapefile
    gdf = gpd.read_file(NRCS_shp)
    df = pd.read_csv(os.path.join(snowCourseWorkspace, f"{report_date}_WestWide_surveys.csv"))

    df = df[["Station Name", "Station Id", month, "SWE_in", "SWE_m"]]
    gdf = gdf[["Station_Na", "Station_Id", "State_Code", "Network_Co", "Elevation", "Latitude", "Longitude", "geometry"]]

    merged_df = pd.merge(df, gdf, left_on="Station Name", right_on="Station_Na", how="right")
    merged_df = merged_df.dropna(subset=[month]).drop_duplicates(subset=["Station Id"])

    # Export as shapefile
    geometry = [Point(xy) for xy in zip(merged_df["Longitude"], merged_df["Latitude"])]
    gdf_stateSurvey = gpd.GeoDataFrame(merged_df, geometry=geometry, crs="EPSG:4326")

    results_dir = os.path.join(results_workspace, f"{report_date}_results")
    os.makedirs(results_dir, exist_ok=True)

    gdf_stateSurvey.to_file(os.path.join(results_dir, f"{report_date}_surveys.shp"), driver="ESRI Shapefile")

    print(f"Snow Courses Downloaded")


# code for downloading CDEC sensors
import requests
import pandas as pd
import os
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime

def download_cdec_snow_surveys(report_date, survey_workspace, SNM_results_workspace, cdec_shapefile, basin_list):
    print("Starting CDEC snow survey download...")

    # Parse date
    date_obj = datetime.strptime(report_date, "%Y%m%d")
    month = date_obj.strftime("%B")
    year = date_obj.year
    other_date = report_date[:-2]

    # Set paths
    snow_course_workspace = os.path.join(survey_workspace, report_date)
    os.makedirs(snow_course_workspace, exist_ok=True)

    cdec_url = f"https://cdec.water.ca.gov/reportapp/javareports?name=COURSES.{other_date}"
    original_csv = os.path.join(survey_workspace, f"{report_date}_SnowCourseMeasurements_original.csv")
    v1_csv = os.path.join(survey_workspace, f"{report_date}_SnowCourseMeasurements_v1.csv")
    clean_csv = os.path.join(survey_workspace, f"{report_date}_SnowCourseMeasurements.csv")
    shapefile_out = os.path.join(snow_course_workspace, f"{report_date}_surveys_cdec.shp")
    merged_csv = os.path.join(survey_workspace, report_date, f"{report_date}_surveys_cdec.csv")
    final_shapefile = os.path.join(SNM_results_workspace, f"{report_date}_results", f"{report_date}_surveys.shp")

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
    header_names = ["ID", "Num", "Station_Na", "Elev_Sur", "Date", "SWE_in"]
    file = pd.read_csv(v1_csv, header=None, names=header_names)
    file = file[~file["SWE_in"].astype(str).str.contains("Water Content", na=False)]
    file.drop(columns=["ID", "Num"], inplace=True, errors="ignore")

    # Convert SWE
    file["SWE_in"] = pd.to_numeric(file["SWE_in"], errors="coerce")
    file["SWE_m"] = file["SWE_in"] * 0.0254
    file.to_csv(clean_csv, index=False)
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

def safe_read_shapefile(path):
        with fiona.open(path, 'r') as src:
            return gpd.GeoDataFrame.from_features(src, crs=src.crs)

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
    site_id_list = gdf_final[id_column].unique().tolist()

    return gdf_final, site_id_list


def tables_and_layers(user, year, report_date, mean_date, meanWorkspace, model_run, masking, watershed_zones,
                      band_zones, HUC6_zones, region_zones, case_field_wtrshd, case_field_band, watermask, glacierMask, snapRaster_geon83,
                      snapRaster_albn83, projGEO, projALB, ProjOut_UTM, bias, prev_report_date=None, prev_model_run=None):

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

    # create directory
    if bias == "N":
        arcpy.CreateFolder_management(resultsWorkspace, RunNameMod)
        outWorkspace = resultsWorkspace + RunNameMod + "/"
        print("model run workspace created")

    if bias == "Y":
        outWorkspace = resultsWorkspace + RunNameMod + "/"

    # meanWorkspace = workspaceBase + "mean_2001_2021_Nodmfsca/"
    prevRepWorkspace = workspaceBase + f"RT_report_data/{prev_report_date}_results/{prev_model_run}/"

    meanMask = outWorkspace + f"{mean_date}_mean_msk.tif"
    MODSCAG_tif_plus = f"H:/WestUS_Data/Rittger_data/fsca_v2024.0d/NRT_FSCA_WW_N83/{year}/{report_date}.tif"
    MODSCAG_tif_plus_proj = outWorkspace + f"fSCA_{report_date}_albn83.tif"

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
    meanMap = meanWorkspace + f"WW_{mean_date}_mean_geon83.tif"
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
    if bias == "N":
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
    arcpy.CopyRaster_management(meanMap, meanMap_copy)

    print("Project both fSCA and phvRaster...")
    # project fSCA image
    arcpy.env.snapRaster = snapRaster_albn83
    arcpy.env.cellSize = snapRaster_albn83
    arcpy.env.extent = snapRaster_albn83
    arcpy.ProjectRaster_management(meanMap_copy, meanMap_proj, projALB,
                                   "NEAREST", "500 500",
                                   "", "", projGEO)

    if bias == "N":
        arcpy.ProjectRaster_management(rcn_glacMask, rcn_raw_proj, projALB,
                                       "NEAREST", "500 500",
                                       "", "")
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

        rcn_final = Raster(rcn_raw_proj) * Raster(watermask)
        rcn_final_wtshd = (Con((IsNull(rcn_final)) & (Raster(modscag_per_msk) >= 0), 0, rcn_final))
        rcn_final_wtshd.save(rcnFinal)
        print("rcn final created")

    # ASK LEANNE, WHAT'S THE DIFFERENCE BETWEEN TIF MASK AND WATER MASK
    print(f"Creating snowline shapefile: {snowPolyElim}")
    mod_01_mask = Raster(modscag_0_1) * Raster(watermask)
    mod_01_mask_glacier = Raster(modscag_0_1) * Raster(glacierMask)
    mod_01_msk_null = SetNull(mod_01_mask_glacier == 0, mod_01_mask_glacier)
    mod_01_msk_null.save(mod_null)

    # Convert raster to polygon
    arcpy.RasterToPolygon_conversion(mod_null, mod_poly, "NO_SIMPLIFY", "Value")
    arcpy.Project_management(mod_poly, mod_poly_utm, ProjOut_UTM)
    arcpy.AddGeometryAttributes_management(mod_poly_utm, "AREA", "", "SQUARE_KILOMETERS", "")
    arcpy.Select_analysis(mod_poly_utm, snowPolySel, where_clause)
    arcpy.EliminatePolygonPart_management(snowPolySel, snowPolyElim, "AREA", part_area, "0", "CONTAINED_ONLY")

    print(f"creating masked SWE product")
    if bias == "N":
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

    print("creating mean mask")
    MeanMapMsk = Raster(meanMap_proj) * Raster(watermask)
    MeanMapALlMsk = MeanMapMsk * Raster(glacierMask)
    MeanMapALlMsk.save(meanMapMask)

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

    ZonalStatisticsAsTable(watershed_zones, case_field_wtrshd, modscag_per, scatable, "DATA",
                                         "ALL")
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
    anombt = ZonalStatisticsAsTable(band_zones, case_field_band, product10, anombandTable, "DATA", "MEAN")
    anomh6 = ZonalStatisticsAsTable(HUC6_zones, "name", product11, anomHuc6Table, "DATA", "MEAN")
    anomreg = ZonalStatisticsAsTable(region_zones, "RegionAll", product12, anomRegionTable, "DATA", "MEAN")

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
    prevRepWorkspace = SNM_results_workspace + f"{prev_report_date}_results/{previous_model_run}/"
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
    MODSCAG_tif_plus = f"H:/WestUS_Data/Rittger_data/fsca_v2024.0d/NRT_FSCA_WW_N83/{year}/{rundate}.tif"
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
    meanMap_proj_WW = WW_results_workspace + f"{rundate}_results_ET/{WW_model_run}/" + f"WW_{mean_date}_mean_albn83.tif"
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
    MeanMapMsk = Raster(meanMap_proj) * Raster(watermask)
    MeanMapALlMsk = MeanMapMsk * Raster(glacier_mask)
    MeanMapALlMsk_2 = MeanMapALlMsk * Raster(domain_mask)
    MeanMapALlMsk_2.save(meanMapMask)

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

    arcpy.env.workspace = None
    arcpy.ClearEnvironment("workspace")
    arcpy.ClearEnvironment("extent")

    import gc
    gc.collect()
    import time
    time.sleep(2)

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

    # Verify source file exists
    print(f"Checking source file: {OutSNODASplus}")
    if not arcpy.Exists(OutSNODASplus):
        raise FileNotFoundError(f"ERROR: Source SNODAS file not found: {OutSNODASplus}")



    ## Copy to floating point raster
    # print(FloatSNODAS)
    # # arcpy.CopyRaster_management(OutSNODASplus, FloatSNODAS, "", "", "-2147483648", "NONE", "NONE", "32_BIT_FLOAT",
    # #                             "NONE", "NONE")
    # arcpy.CopyRaster_management(OutSNODASplus, FloatSNODAS)

    print("Creating SWE in meters ...")

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

    ## If test run previously then SCA_SNODAS will exist, delete and then create
    if arcpy.Exists(SCA_SNODAS):
        arcpy.Delete_management(SCA_SNODAS, "#")
        SNODASfSCA = Con(SNODASallMsk > .001, 100, 0)
        SNODASfSCA.save(SCA_SNODAS)
    ## Else if this is a test run create it
    else:
        SNODASfSCA = Con(SNODASallMsk > .001, 100, 0)
        SNODASfSCA.save(SCA_SNODAS)


    # Do zonal stats for real time swe layer table
    print("creating zonal stats for SNODAS swe = " + SWEtable)
    ZonalStatisticsAsTable(band_zones, "SrtNmeBand", ClipSNODAS, SWEbandtable, "DATA", "MEAN")
    ZonalStatisticsAsTable(watershed_zones, "SrtName", ClipSNODAS, SWEtable, "DATA", "MEAN")
    arcpy.Delete_management("in-memory")
    gc.collect()

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
                               band_shapefile, case_field_band, merge, projOut, projIn=None, domainList=None, domain_shapefile=None,
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



