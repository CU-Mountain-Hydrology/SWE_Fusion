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
    # from SWE_Fusion_functions import *
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

import os
import pandas as pd
import rasterio
import arcpy
from arcpy.sa import ExtractByMask

def bias_correction_vetting(raster, point, swe_col, id_col, rundate, name, method, out_csv, folder, control_raster=None):
    """
    Process a raster, compute SWE statistics, and optionally update CSV.

    Parameters:
        raster (str): Path to raster to process.
        point (str): Path to point shapefile with SWE measurements.
        swe_col (str): Column name in point shapefile for observed SWE.
        id_col (str): Column name in point shapefile for site ID.
        rundate (str): Run date string for output.
        name (str): Basin name or unique identifier.
        method (str): Method name (e.g., "RECENT").
        out_csv (str): Path to output CSV.
        folder (str): Folder where clipped rasters can be stored.
        control_raster (str, optional): Path to control raster for comparison.

    Returns:
        None. Updates CSV in place.
    """

    if not os.path.exists(raster):
        print(f"Raster not found: {raster}")
        return

    # Load points within raster
    from SWE_Fusion_functions import get_points_within_raster
    gdf_pts, site_ids = get_points_within_raster(point, raster, id_column=id_col)

    # Sample raster values
    raster_vals = []
    with rasterio.open(raster) as src:
        for _, row in gdf_pts.iterrows():
            x, y = row.geometry.x, row.geometry.y
            val = list(src.sample([(x, y)]))[0][0]
            raster_vals.append(val)

    # Filter out zero SWE points
    zero_swe = gdf_pts[gdf_pts[swe_col] == 0]
    if len(zero_swe) > 0:
        print("Station IDs with zero SWE:", zero_swe[id_col].tolist())
    gdf_pts_filtered = gdf_pts[gdf_pts[swe_col] != 0].copy()

    # Align indexes
    raster_vals_series = pd.Series(raster_vals, index=gdf_pts.index)
    gdf_pts_filtered["raster_value"] = raster_vals_series.loc[gdf_pts_filtered.index]

    # Compute error metrics
    gdf_pts_filtered["percent_error"] = ((gdf_pts_filtered["raster_value"] - gdf_pts_filtered[swe_col]) /
                                         gdf_pts_filtered[swe_col]) * 100
    gdf_pts_filtered["abs_percent_error"] = gdf_pts_filtered["percent_error"].abs()
    gdf_pts_filtered["error"] = gdf_pts_filtered["raster_value"] - gdf_pts_filtered[swe_col]
    gdf_pts_filtered["abs_error"] = gdf_pts_filtered["error"].abs()

    avg_percent_error = gdf_pts_filtered["percent_error"].mean()
    avg_abs_percent_error = gdf_pts_filtered["abs_percent_error"].mean()
    avg_error = gdf_pts_filtered["error"].mean()
    mae = gdf_pts_filtered["abs_error"].mean()
    max_val = float(arcpy.management.GetRasterProperties(raster, "MAXIMUM").getOutput(0))

    # Compute SWE volume
    mean_swe = float(arcpy.management.GetRasterProperties(raster, "MEAN").getOutput(0))
    rows = int(arcpy.management.GetRasterProperties(raster, "ROWCOUNT").getOutput(0))
    cols = int(arcpy.management.GetRasterProperties(raster, "COLUMNCOUNT").getOutput(0))
    total_area = rows * cols * (500 ** 2)  # m²
    total_swe_m3 = mean_swe * total_area
    total_swe_af = total_swe_m3 * 0.000810714  # acre-feet

    print(f"\nRaster: {os.path.basename(raster)}")
    print(f"Average Percent Error: {avg_percent_error}")
    print(f"Average Absolute Percent Error: {avg_abs_percent_error}")
    print(f"Average Error (bias): {avg_error}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"SWE_AF: {total_swe_af}")
    print(f"Max Value: {max_val}")

    # Create error frame
    error_frame = {
        'rundate': rundate,
        'Basin': name.split("_")[1],
        'Method': method,
        'Avg.Perc.Error': avg_percent_error,
        'Avg.Abs.Perc.Error': avg_abs_percent_error,
        'Avg.Error': avg_error,
        'MAE': mae,
        'SWE_AF': total_swe_af,
        'Max Value': max_val,
    }

    # Update CSV
    if os.path.isfile(out_csv):
        df = pd.read_csv(out_csv)
        new_df = pd.DataFrame([error_frame])
        df = pd.concat([df, new_df], ignore_index=True)
    else:
        df = pd.DataFrame([error_frame])
    df.to_csv(out_csv, index=False)

    # Process control raster if provided
    if control_raster:
        control_raster_clp = os.path.join(folder, f"{name}_p8_Control_clp.tif")
        if not os.path.exists(control_raster_clp):
            outControlClip = ExtractByMask(control_raster, raster, 'INSIDE')
            outControlClip.save(control_raster_clp)

            raster_vals = []
            with rasterio.open(control_raster_clp) as src:
                for _, row in gdf_pts.iterrows():
                    x, y = row.geometry.x, row.geometry.y
                    val = list(src.sample([(x, y)]))[0][0]
                    raster_vals.append(val)

            gdf_pts_filtered = gdf_pts[gdf_pts[swe_col] != 0].copy()
            raster_vals_series = pd.Series(raster_vals, index=gdf_pts.index)
            gdf_pts_filtered["control_value"] = raster_vals_series.loc[gdf_pts_filtered.index]

            # Compute control metrics
            gdf_pts_filtered["percent_error"] = ((gdf_pts_filtered["control_value"] - gdf_pts_filtered[swe_col]) /
                                                 gdf_pts_filtered[swe_col]) * 100
            gdf_pts_filtered["abs_percent_error"] = gdf_pts_filtered["percent_error"].abs()
            gdf_pts_filtered["error"] = gdf_pts_filtered["control_value"] - gdf_pts_filtered[swe_col]
            gdf_pts_filtered["abs_error"] = gdf_pts_filtered["error"].abs()

            avg_percent_error = gdf_pts_filtered["percent_error"].mean()
            avg_abs_percent_error = gdf_pts_filtered["abs_percent_error"].mean()
            avg_error = gdf_pts_filtered["error"].mean()
            mae = gdf_pts_filtered["abs_error"].mean()
            mean_swe = float(arcpy.management.GetRasterProperties(control_raster_clp, "MEAN").getOutput(0))
            rows = int(arcpy.management.GetRasterProperties(control_raster_clp, "ROWCOUNT").getOutput(0))
            cols = int(arcpy.management.GetRasterProperties(control_raster_clp, "COLUMNCOUNT").getOutput(0))
            total_area = rows * cols * (500 ** 2)
            total_swe_m3 = mean_swe * total_area
            total_swe_af = total_swe_m3 * 0.000810714

            error_frame = {
                'rundate': rundate,
                'Basin': name.split("_")[1],
                'Method': "CONTROL",
                'Avg.Perc.Error': avg_percent_error,
                'Avg.Abs.Perc.Error': avg_abs_percent_error,
                'Avg.Error': avg_error,
                'MAE': mae,
                'SWE_AF': total_swe_af,
                'Max Value': float(arcpy.management.GetRasterProperties(raster, "MAXIMUM").getOutput(0)),
            }

            if os.path.isfile(out_csv):
                df = pd.read_csv(out_csv)
                new_df = pd.DataFrame([error_frame])
                df = pd.concat([df, new_df], ignore_index=True)
            else:
                df = pd.DataFrame([error_frame])
            df.to_csv(out_csv, index=False)
        else:
            print("CONTROL exists")