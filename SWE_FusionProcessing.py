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

def download_snow_surveys(rundate, surveyWorkspace, resultsWorkspace, url_file, NRCS_shp, state_list):
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
    path = os.path.join(surveyWorkspace, rundate)
    os.makedirs(path, exist_ok=True)
    snowCourseWorkspace = os.path.join(surveyWorkspace, rundate)
    date_obj = datetime.strptime(rundate, "%Y%m%d")
    month = date_obj.strftime("%B")
    year = date_obj.year

    # ---- Read URLs from file ----
    state_url_dict = {}
    with open(url_file, "r") as f:
        for line in f:
            if "|" in line:
                state, url = line.strip().split("|", 1)
                state_url_dict[state] = url

    state_url_list = [state_url_dict[state] for state in state_list]
    state_text_list = [os.path.join(snowCourseWorkspace, f"{state}_original.txt") for state in state_list]
    state_edit_list = [os.path.join(snowCourseWorkspace, f"{state}.txt") for state in state_list]

    for url, text, edit, state in zip(state_url_list, state_text_list, state_edit_list, state_list):
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
    merged_df.to_csv(os.path.join(snowCourseWorkspace, f"{rundate}_WestWide_surveys.csv"), index=False)

    # Merge with shapefile
    gdf = gpd.read_file(NRCS_shp)
    df = pd.read_csv(os.path.join(snowCourseWorkspace, f"{rundate}_WestWide_surveys.csv"))

    df = df[["Station Name", "Station Id", month, "SWE_in", "SWE_m"]]
    gdf = gdf[["Station_Na", "Station_Id", "State_Code", "Network_Co", "Elevation", "Latitude", "Longitude", "geometry"]]

    merged_df = pd.merge(df, gdf, left_on="Station Name", right_on="Station_Na", how="right")
    merged_df = merged_df.dropna(subset=[month]).drop_duplicates(subset=["Station Id"])

    # Export as shapefile
    geometry = [Point(xy) for xy in zip(merged_df["Longitude"], merged_df["Latitude"])]
    gdf_stateSurvey = gpd.GeoDataFrame(merged_df, geometry=geometry, crs="EPSG:4326")

    results_dir = os.path.join(resultsWorkspace, f"{rundate}_results")
    os.makedirs(results_dir, exist_ok=True)

    gdf_stateSurvey.to_file(os.path.join(results_dir, f"{rundate}_surveys.shp"), driver="ESRI Shapefile")

    print(f"Snow Courses Downloaded")


# code for downloading CDEC sensors
def download_cdec_snow_surveys(rundate, base_workspace, results_workspace, cdec_shapefile, basin_list):
    import requests
    import pandas as pd
    import os
    import geopandas as gpd
    from shapely.geometry import Point
    from datetime import datetime

    print("Starting CDEC snow survey download...")

    # Parse date
    date_obj = datetime.strptime(rundate, "%Y%m%d")
    month = date_obj.strftime("%B")
    year = date_obj.year
    other_date = rundate[:-2]

    # Set paths
    snow_course_workspace = os.path.join(base_workspace, rundate)
    os.makedirs(snow_course_workspace, exist_ok=True)

    cdec_url = f"https://cdec.water.ca.gov/reportapp/javareports?name=COURSES.{other_date}"
    original_csv = os.path.join(base_workspace, f"{rundate}_SnowCourseMeasurements_original.csv")
    v1_csv = os.path.join(base_workspace, f"{rundate}_SnowCourseMeasurements_v1.csv")
    clean_csv = os.path.join(base_workspace, f"{rundate}_SnowCourseMeasurements.csv")
    shapefile_out = os.path.join(snow_course_workspace, f"{rundate}_surveys_cdec.shp")
    merged_csv = os.path.join(base_workspace, rundate, f"{rundate}_surveys_cdec.csv")
    final_shapefile = os.path.join(results_workspace, f"{rundate}_results", f"{rundate}_surveys.shp")

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
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.merge import merge
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import TwoSlopeNorm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import os 
import io
import re
import math
from rasterio.transform import rowcol
import requests
import matplotlib.patches as mpatches
import numpy as np
from shapely.geometry import box
import geopandas as gpd
# import fiona
# from matplotlib_scalebar.scalebar import ScaleBar
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
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
        print(url)
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

import shutil
import os
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
    shutil.rmtree(temp_extract_path)

# safely read in a shapefile
import geopandas as gpd
import fiona

def safe_read_shapefile(path):
        with fiona.open(path, 'r') as src:
            return gpd.GeoDataFrame.from_features(src, crs=src.crs)

import geopandas
import rasterio
import numpy
import os


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