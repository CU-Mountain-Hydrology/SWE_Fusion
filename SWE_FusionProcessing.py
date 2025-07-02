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
            raster_dict[file_date] = os.path.join(original_folder, f)
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
