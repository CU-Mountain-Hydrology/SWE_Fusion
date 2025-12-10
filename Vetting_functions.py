import rasterio
import os
from SWE_Fusion_functions import *
import geopandas as gpd

# get sensor data for extent of raster
# get raster value
# get % error per sensor and average
# output error per sensor in one csv
# output average error basins on zone
    ## could be by domain, or basin

# check to see if there is an ASO flight within 3 days
## calcuate % error and % error by elevation band

# check file size
    ## if it's less than the lowest one from last year, have a flag
    ## plot it and prompt you on whether to continue

# have a flag for high values
    # print out the basin and the values per elevation band on if we should edit them

def point_vetting_raster(point_dataset, raster, id_column, swe_column, output_csv, csv_name=None, output_shp=None):
    import rasterio
    import os
    from SWE_Fusion_functions import *
    import geopandas as gpd

    gdf_pts, site_ids = get_points_within_raster(point_dataset, raster, id_column=id_column)

    # get percent error
    with rasterio.open(raster) as src:
        raster_vals = []

        for _, row in gdf_pts.iterrows():
            x, y = row.geometry.x, row.geometry.y
            raster_val = list(src.sample([(x, y)]))[0][0]
            raster_vals.append(raster_val)

    gdf_pts["raster_value"] = raster_vals
    gdf_pts["percent_error"] = ((gdf_pts["raster_value"] - gdf_pts[swe_column]) /
                                gdf_pts[swe_column]) * 100

    gdf_pts["abs_percent_error"] = gdf_pts["percent_error"].abs()
    gdf_pts["error"] = gdf_pts["raster_value"] - gdf_pts[swe_column]
    gdf_pts["abs_error"] = gdf_pts["error"].abs()

    avg_percent_error = gdf_pts["percent_error"].mean()
    avg_abs_percent_error = gdf_pts["abs_percent_error"].mean()
    avg_error = gdf_pts["error"].mean()
    mae = gdf_pts["abs_error"].mean()

    ## add acre feet

    print(f"\nRaster: {os.path.basename(raster)}")
    print(f"Shapefile: {os.path.basename(point_dataset)}")
    print("Average Percent Error:", avg_percent_error)
    print("Average Absolute Percent Error:", avg_abs_percent_error)
    print("Average Error (bias):", avg_error)
    print("Mean Absolute Error (MAE):", mae)

    error_stats = {
        'raster': os.path.basename(raster),
        "avg_percent_error": avg_percent_error,
        "avg_abs_percent_error": avg_abs_percent_error,
        "avg_error": avg_error,
        "mae": mae,
    }

    if output_csv == "Y":
        error_stats.to_csv(csv_name)

    if output_shp == "Y":
        gdf_pts.to_file(output_shp, driver="ESRI Shapefile")

    error_stats_df = pd.DataFrame(error_stats)

    return error_stats_df