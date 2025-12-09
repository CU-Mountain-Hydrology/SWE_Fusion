# import modules
import arcpy
import os
from SWE_Fusion_functions import *
print("modules")

# paths
rundate = "20250503"
resultsWorkspace = f"W:/Spatial_SWE/WW_regression/RT_report_data/{rundate}_results_ET/"
sensors = resultsWorkspace + f"{rundate}_sensors_albn83.shp"
surveys = resultsWorkspace + f"{rundate}_surveys_albn83.shp"
raster = r"W:\Spatial_SWE\WW_regression\RT_report_data\20250503_results_ET\ASO_BiasCorrect_fSCA_RT_CanAdj_rcn_noSW_woCCR\RECENT\20250503_SouthPlatte_RECENT_BC_fix_albn83.tif"
# load raster
gdf_pts, site_ids = get_points_within_raster(sensors, raster, id_column="Site_ID")
print(gdf_pts.head(10))

# get percent error
with rasterio.open(raster) as src:
    raster_vals = []

    for _, row in gdf_pts.iterrows():
        x, y = row.geometry.x, row.geometry.y
        raster_val = list(src.sample([(x, y)]))[0][0]
        raster_vals.append(raster_val)

gdf_pts["raster_value"] = raster_vals
gdf_pts["percent_error"] = ((gdf_pts["raster_value"] - gdf_pts["pillowswe"]) /
                             gdf_pts["pillowswe"]) * 100

gdf_pts["abs_percent_error"] = gdf_pts["percent_error"].abs()
gdf_pts["error"] = gdf_pts["raster_value"] - gdf_pts["pillowswe"]
gdf_pts["abs_error"] = gdf_pts["error"].abs()

avg_percent_error = gdf_pts["percent_error"].mean()
avg_abs_percent_error = gdf_pts["abs_percent_error"].mean()
avg_error = gdf_pts["error"].mean()
mae = gdf_pts["abs_error"].mean()

print("Average Percent Error:", avg_percent_error)
print("Average Absolute Percent Error:", avg_abs_percent_error)
print("Average Error (bias):", avg_error)
print("Mean Absolute Error (MAE):", mae)

## SURVEYS
gdf_pts, site_ids = get_points_within_raster(surveys, raster, id_column="Station_Id")
print(gdf_pts.head(10))

# get percent error
with rasterio.open(raster) as src:
    raster_vals = []

    for _, row in gdf_pts.iterrows():
        x, y = row.geometry.x, row.geometry.y
        raster_val = list(src.sample([(x, y)]))[0][0]
        raster_vals.append(raster_val)

gdf_pts["raster_value"] = raster_vals
gdf_pts["percent_error"] = ((gdf_pts["raster_value"] - gdf_pts["SWE_m"]) /
                             gdf_pts["SWE_m"]) * 100

gdf_pts["abs_percent_error"] = gdf_pts["percent_error"].abs()
gdf_pts["error"] = gdf_pts["raster_value"] - gdf_pts["SWE_m"]
gdf_pts["abs_error"] = gdf_pts["error"].abs()

avg_percent_error = gdf_pts["percent_error"].mean()
avg_abs_percent_error = gdf_pts["abs_percent_error"].mean()
avg_error = gdf_pts["error"].mean()
mae = gdf_pts["abs_error"].mean()

print("Average Percent Error:", avg_percent_error)
print("Average Absolute Percent Error:", avg_abs_percent_error)
print("Average Error (bias):", avg_error)
print("Mean Absolute Error (MAE):", mae)
## if there is an ASO data within 2 days
## get percent error be elevation and

## compare SWE volume be basin from previous report
## figure out if it's increasing or decreasing
## determin if the pillows are increasing
## flag if there is a mismatch

### FOR WW
## get really high values -- within x standard deviations of the max value
## give a shapefile adn find out which basins are there