# import modules
import arcpy
from arcpy.sa import *
import os
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from SWE_Fusion_functions import get_points_within_raster
print("modules")

# paths
rundate = "20250503"
resultsWorkspace = f"W:/Spatial_SWE/WW_regression/RT_report_data/{rundate}_results_ET/"
sensors = resultsWorkspace + f"{rundate}_sensors_albn83.shp"
surveys = resultsWorkspace + f"{rundate}_surveys_albn83.shp"
raster = r"W:\Spatial_SWE\WW_regression\RT_report_data\20250503_results_ET\ASO_BiasCorrect_fSCA_RT_CanAdj_rcn_noSW_woCCR\RECENT\20250503_SouthPlatte_RECENT_BC_fix_albn83.tif"

snapRaster_albn83 = "M:/SWE/WestWide/data/boundaries/SnapRaster_albn83.tif"
arcpy.env.snapRaster = snapRaster_albn83
# load raster
# parameters
# raster =
# control = True
# output_folder =
out_csv = r"W:/Spatial_SWE/WW_regression/RT_report_data/20250503_results_ET/ASO_BiasCorrect_fSCA_RT_CanAdj_rcn_noSW_woCCR/BC_error_surveys.csv"
# point = sensors or surveys
# swe_column = "pillowswe"
# site_id_column = "Site_ID"

folder = r"W:\Spatial_SWE\WW_regression\RT_report_data\20250503_results_ET\ASO_BiasCorrect_fSCA_RT_CanAdj_rcn_noSW_woCCR/"
prefix = "20250503"

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

unique_names = set()  # use a set to keep unique values
file_mapping = {}
for root, dirs, files in os.walk(folder):
    for file in files:
        if file.startswith(prefix):
            # Split by "_" and take the first two parts
            parts = file.split("_")
            if len(parts) >= 2:
                name = "_".join(parts[:2])
                unique_names.add(name)

# Convert to list if needed
unique_names = list(unique_names)
print(unique_names)

print("\nFull file names by prefix:")
for name, files in file_mapping.items():
    print(f"{name}:")
    for f in files:
        print(f"  {f}")

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
control_raster = r"M:\SWE\WestWide\Spatial_SWE\WW_regression\RT_report_data\20250503_results_ET\ASO_BiasCorrect_fSCA_RT_CanAdj_rcn_noSW_woCCR\p8_20250503_noneg.tif"
# start loops
swe_col = 'SWE_m'
id_col = 'Station_ID'

methods = ["RECENT", "GRADE", "SENSOR_PATTERN", "GRADES_SPECF"]
for method in methods:
    print(f"\nMethod: {method}"'')
    BC_path = folder + f"/{method}/"
    for name in unique_names:
        print(f"Name: {name}")
        raster = BC_path + f"{name}_{method}_BC_fix_albn83.tif"

        if os.path.exists(raster):

            # ## SURVEYS
            # gdf_pts, site_ids = get_points_within_raster(sensors, raster, id_column="Site_ID")
            # # print(gdf_pts.head(10))
            #
            # # get percent error
            # with rasterio.open(raster) as src:
            #     raster_vals = []
            #
            #     for _, row in gdf_pts.iterrows():
            #         x, y = row.geometry.x, row.geometry.y
            #         raster_val = list(src.sample([(x, y)]))[0][0]
            #         raster_vals.append(raster_val)
            #
            # # get zero SWE points
            # zero_swe = gdf_pts[gdf_pts[swe_col] == 0]
            # if len(zero_swe) > 0:
            #     print("Station IDs with zero SWE:")
            #     print(zero_swe[id_col].tolist())
            #
            # gdf_pts_filtered = gdf_pts[gdf_pts[swe_col] != 0].copy()
            #
            # # align indexes
            # raster_vals_series = pd.Series(raster_vals, index=gdf_pts.index)
            # gdf_pts_filtered["raster_value"] = raster_vals_series.loc[gdf_pts_filtered.index]
            #
            # # gdf_pts_filtered["raster_value"] = raster_vals
            # gdf_pts_filtered["percent_error"] = ((gdf_pts_filtered["raster_value"] - gdf_pts_filtered[swe_col]) /
            #                              gdf_pts_filtered[swe_col]) * 100
            #
            # gdf_pts_filtered["abs_percent_error"] = gdf_pts_filtered["percent_error"].abs()
            # gdf_pts_filtered["error"] = gdf_pts_filtered["raster_value"] - gdf_pts_filtered[swe_col]
            # gdf_pts_filtered["abs_error"] = gdf_pts_filtered["error"].abs()
            #
            # avg_percent_error = gdf_pts_filtered["percent_error"].mean()
            # avg_abs_percent_error = gdf_pts_filtered["abs_percent_error"].mean()
            # avg_error = gdf_pts_filtered["error"].mean()
            # mae = gdf_pts_filtered["abs_error"].mean()
            # max_val = float(arcpy.management.GetRasterProperties(raster, "MAXIMUM").getOutput(0))
            #
            # # get SWE volume
            # mean_swe = float(arcpy.management.GetRasterProperties(raster, "MEAN").getOutput(0))
            #
            # # Compute total area
            # rows = int(arcpy.management.GetRasterProperties(raster, "ROWCOUNT").getOutput(0))
            # cols = int(arcpy.management.GetRasterProperties(raster, "COLUMNCOUNT").getOutput(0))
            # total_area = rows * cols * (500 ** 2)  # m²
            #
            # # SWE volume in m³
            # total_swe_m3 = mean_swe * total_area
            #
            # # Optional: convert to acre-feet
            # total_swe_af = total_swe_m3 * 0.000810714
            #
            # print("Average Percent Error:", avg_percent_error)
            # print("Average Absolute Percent Error:", avg_abs_percent_error)
            # print("Average Error (bias):", avg_error)
            # print("Mean Absolute Error (MAE):", mae)
            # print('SWE_AF', total_swe_af)
            # print('Max Value:', max_val)
            #
            # error_frame = {'rundate': rundate,
            #                'Basin' : name.split("_")[1] ,
            #                'Method' : method,
            #                'Avg.Perc.Error' : avg_percent_error,
            #                'Avg.Abs.Perc.Error' : avg_abs_percent_error,
            #                'Avg.Error' : avg_error,
            #                'MAE' : mae,
            #                'SWE_AF' : total_swe_af,
            #                'Max Value' : max_val,
            #
            # }
            #
            # if os.path.isfile(out_csv):
            #     df = pd.read_csv(out_csv)
            #     new_df = pd.DataFrame([error_frame])
            #     df = pd.concat([df, new_df], ignore_index=True)
            # else:
            #     df = pd.DataFrame([error_frame])
            #
            # df.to_csv(out_csv, index=False)
            #
            # # Control
            # control_raster = r"M:\SWE\WestWide\Spatial_SWE\WW_regression\RT_report_data\20250503_results_ET\ASO_BiasCorrect_fSCA_RT_CanAdj_rcn_noSW_woCCR\p8_20250503_noneg.tif"
            # control_raster_clp = folder + f"{name}_p8_Control_clp.tif"
            # if not os.path.exists(control_raster_clp):
            #
            #     outControlClip = ExtractByMask(control_raster, raster, 'INSIDE')
            #     outControlClip.save(control_raster_clp)
            #
            #
            #     with rasterio.open(control_raster_clp) as src:
            #         raster_vals = []
            #
            #         for _, row in gdf_pts.iterrows():
            #             x, y = row.geometry.x, row.geometry.y
            #             raster_val = list(src.sample([(x, y)]))[0][0]
            #             raster_vals.append(raster_val)
            #
            #     # get zero SWE points
            #     zero_swe = gdf_pts[gdf_pts["pillowswe"] == 0]
            #     if len(zero_swe) > 0:
            #         print("Station IDs with zero SWE:")
            #         print(zero_swe[id_col].tolist())
            #
            #     # only calculate non-zeros
            #     gdf_pts_filtered = gdf_pts[gdf_pts["pillowswe"] != 0].copy()
            #
            #     # align indexes
            #     raster_vals_series = pd.Series(raster_vals, index=gdf_pts.index)
            #     gdf_pts_filtered["control_value"] = raster_vals_series.loc[gdf_pts_filtered.index]
            #
            #     # gdf_pts_filtered["control_value"] = raster_vals
            #     gdf_pts_filtered["percent_error"] = ((gdf_pts_filtered["control_value"] - gdf_pts_filtered["pillowswe"]) /
            #                                  gdf_pts_filtered["pillowswe"]) * 100
            #
            #     gdf_pts_filtered["abs_percent_error"] = gdf_pts_filtered["percent_error"].abs()
            #     gdf_pts_filtered["error"] = gdf_pts_filtered["control_value"] - gdf_pts_filtered["pillowswe"]
            #     gdf_pts_filtered["abs_error"] = gdf_pts_filtered["error"].abs()
            #
            #     avg_percent_error = gdf_pts_filtered["percent_error"].mean()
            #     avg_abs_percent_error = gdf_pts_filtered["abs_percent_error"].mean()
            #     avg_error = gdf_pts_filtered["error"].mean()
            #     mae = gdf_pts_filtered["abs_error"].mean()
            #     max_val = float(arcpy.management.GetRasterProperties(raster, "MAXIMUM").getOutput(0))
            #
            #     # get SWE volume
            #     mean_swe = float(arcpy.management.GetRasterProperties(control_raster_clp, "MEAN").getOutput(0))
            #
            #     # Compute total area
            #     rows = int(arcpy.management.GetRasterProperties(control_raster_clp, "ROWCOUNT").getOutput(0))
            #     cols = int(arcpy.management.GetRasterProperties(control_raster_clp, "COLUMNCOUNT").getOutput(0))
            #     total_area = rows * cols * (500 ** 2)  # m²
            #
            #     # SWE volume in m³
            #     total_swe_m3 = mean_swe * total_area
            #
            #     # Optional: convert to acre-feet
            #     total_swe_af = total_swe_m3 * 0.000810714
            #
            #     print("\n CONTROL")
            #     print("Control Average Percent Error:", avg_percent_error)
            #     print("Control Average Absolute Percent Error:", avg_abs_percent_error)
            #     print("Control Average Error (bias):", avg_error)
            #     print("Control Mean Absolute Error (MAE):", mae)
            #     print('Control SWE_AF', total_swe_af)
            #     print('Max Value:', max_val)
            #
            #     error_frame = {'rundate': rundate,
            #                    'Basin': name.split("_")[1],
            #                    'Method': "CONTROL",
            #                    'Avg.Perc.Error': avg_percent_error,
            #                    'Avg.Abs.Perc.Error': avg_abs_percent_error,
            #                    'Avg.Error': avg_error,
            #                    'MAE': mae,
            #                    'SWE_AF': total_swe_af,
            #                    'Max Value': max_val,
            #
            #                    }
            #
            #     if os.path.isfile(out_csv):
            #         df = pd.read_csv(out_csv)
            #         new_df = pd.DataFrame([error_frame])
            #         # Concatenate
            #         df = pd.concat([df, new_df], ignore_index=True)
            #     else:
            #         df = pd.DataFrame([error_frame])
            #
            #     df.to_csv(out_csv, index=False)
            #
            # else:
            #     print('CONTROL exists')
            bias_correction_vetting(
                raster=raster,
                point=surveys,
                swe_col="SWE_m",
                id_col="Station_Id",
                rundate="20250503",
                name=name,
                method=method,
                out_csv=out_csv,
                folder=folder,
                control_raster=control_raster
            )


