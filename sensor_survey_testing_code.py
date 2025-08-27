# import modules
import os
import shutil
import geopandas as gpd

print('modules imported')

user = "Emma"
report_date = "20250315"
pillow_date = "15Mar2025"
model_run = "RT_CanAdj_rcn_noSW_woCCR"
domainList = ["NOCN", "PNW", "SNM", "SOCN", "INMT"]
workspaceBase = fr"M:/SWE/WestWide/Spatial_SWE/WW_regression/"
results_workspace = workspaceBase + f"RT_report_data/{report_date}_results_ET/"
model_workspace = fr"W:/Spatial_SWE/WW_regression/RT_report_data/ModelWS_Test/"

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

print("\nProcessing GeoPackage")
geopackage_to_shapefile(report_date=report_date, pillow_date=pillow_date, model_run=model_run,
                        user=user, domainList=domainList, model_workspace=model_workspace,
                        results_workspace=results_workspace)