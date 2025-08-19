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
