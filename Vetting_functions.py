import rasterio
import os
from SWE_Fusion_functions import *
import geopandas as gpd
import matplotlib.pyplot as plt

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

def bias_correction_vetting(raster, point, domain, swe_col, id_col, rundate, name, method, out_csv, folder, control_raster=None):
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
        'Domain': domain,
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
                'Domain': domain,
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


import os
import pandas as pd
import rasterio
import arcpy
from arcpy.sa import ExtractByMask
####################################
#FUNCTIONS
####################################
def model_domain_vetting(raster, point, swe_col, id_col, rundate, domain, modelRun, out_csv):
    """
    Process a raster, compute SWE statistics, and optionally update CSV.

    Parameters:
        raster (str): Path to raster to process.
        point (str): Path to point shapefile with SWE measurements.
        swe_col (str): Column name in point shapefile for observed SWE.
        id_col (str): Column name in point shapefile for site ID.
        rundate (str): Run date string for output.
        domain (str): Basin name or unique identifier.
        method (str): Method name (e.g., "RECENT").
        out_csv (str): Path to output CSV.

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
        print("Number of Station IDs with zero SWE:", len(zero_swe[id_col].tolist()))
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

    # Create error frame
    error_frame = {
        'rundate': rundate,
        'Domain': domain,
        'ModelRun': modelRun,
        'Avg.Perc.Error': avg_percent_error,
        'Avg.Abs.Perc.Error': avg_abs_percent_error,
        'Avg.Error': avg_error,
        'MAE': mae,
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

def pillow_date_comparison(rundate, prev_model_date, raster, point_dataset, prev_pointDataset, id_column, swe_col,
                           elev_col, output_png, convert_meters_feet=None, convert_feet_meters=None):
    # get points within a raster
    gdf_pts, site_ids = get_points_within_raster(point_dataset, raster, id_column=id_column)

    print(gdf_pts.columns)
    print(site_ids)

    # open previous shapefile
    prev_gdf = gpd.read_file(prev_pointDataset)
    prev_gdf = prev_gdf[prev_gdf[id_column].isin(site_ids)]

    # rename and reorganize columns
    prev_gdf = prev_gdf.rename(columns={swe_col: f"prev_{swe_col}"})
    gdf_pts = gdf_pts[[id_column, swe_col, elev_col]]
    prev_gdf = prev_gdf[[id_column, f"prev_{swe_col}"]]

    # merge
    merged_points = gdf_pts.merge(prev_gdf, on=id_column, how='inner')
    print(merged_points.columns)
    print(merged_points.head(4))

    # convert dem from meters to feet
    if convert_meters_feet == "Y":
        merged_points[elev_col] = merged_points[elev_col] * 3.28084
        print(merged_points.head(4))

    if convert_feet_meters == "Y":
        merged_points[elev_col] = merged_points[elev_col] * 0.3048
        print(merged_points.head(4))

    # ensure consistent x ordering
    df = merged_points.sort_values(elev_col)
    x = df[elev_col]

    # --- create stacked plots ---
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(15, 8), sharex=True)

    # --- Top plot: paired SWE with color-coded markers ---
    for _, row in df.iterrows():
        # Determine color based on change
        if row[swe_col] > row[f"prev_{swe_col}"]:
            point_color = "blue"  # increase
        elif row[swe_col] < row[f"prev_{swe_col}"]:
            point_color = "red"   # decline
        else:
            point_color = "black" # no change

        # Plot both markers with the same color
        ax1.scatter(row[elev_col], row[swe_col], marker="s", color=point_color)
        ax1.scatter(row[elev_col], row[f"prev_{swe_col}"], marker="^", color=point_color)

        # Draw connecting line
        ax1.plot(
            [row[elev_col], row[elev_col]],
            [row[swe_col], row[f"prev_{swe_col}"]],
            color=point_color,
            linewidth=1
        )

    # Add legend with dummy artists
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=8, label=f'{rundate} (increase)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', markersize=8,
               label=f'{prev_model_date} (increase)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=8, label=f'{rundate} (decline)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=8,
               label=f'{prev_model_date} (decline)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=8, label=f'{rundate} (no change)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=8,
               label=f'{prev_model_date} (no change)')
    ]
    ax1.legend(handles=legend_elements)

    ax1.set_ylabel("SWE (m)")
    ax1.set_title(f"{os.path.basename(raster)[:-4]} Pillow Difference | {prev_model_date} to {rundate}")

    # --- Bottom plot: single SWE column (current rundate) ---
    ax2.scatter(x, df[swe_col], color="green", marker="o", label=f"{rundate} {swe_col}")
    ax2.set_xlabel("Elevation (ft)")
    ax2.set_ylabel("SWE (m)")
    ax2.set_title(f"{swe_col} vs Elevation")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_png)
    plt.show()

## create box and whiskers plots of the rasters before and after
def raster_box_whisker_plot(rundate, prev_model_date, raster, prev_raster, domain, variable, unit, output_png):
    """
    Create box plot with statistical annotations.
    Handles extreme outliers to prevent image size errors.
    """

    arr1 = read_raster_values(prev_raster)
    arr2 = read_raster_values(raster)

    # Remove extreme outliers and invalid values
    def clean_array(arr):
        """Remove NaN, inf, and extreme outliers"""
        arr_clean = arr[~np.isnan(arr) & ~np.isinf(arr)]
        arr_clean = arr_clean[arr_clean >= 0]  # Remove negative values

        # Remove values beyond 99.9th percentile (extreme outliers)
        p999 = np.percentile(arr_clean, 99.9)
        arr_clean = arr_clean[arr_clean <= p999]

        return arr_clean

    arr1_clean = clean_array(arr1)
    arr2_clean = clean_array(arr2)

    print(f"\nData cleaning summary:")
    print(
        f"  {prev_model_date}: {len(arr1)} → {len(arr1_clean)} values (removed {len(arr1) - len(arr1_clean)} outliers)")
    print(f"  {rundate}: {len(arr2)} → {len(arr2_clean)} values (removed {len(arr2) - len(arr2_clean)} outliers)")

    # Calculate statistics
    stats1 = {
        'min': np.min(arr1_clean),
        'q25': np.percentile(arr1_clean, 25),
        'median': np.median(arr1_clean),
        'q75': np.percentile(arr1_clean, 75),
        'max': np.max(arr1_clean),
        'mean': np.mean(arr1_clean)
    }

    stats2 = {
        'min': np.min(arr2_clean),
        'q25': np.percentile(arr2_clean, 25),
        'median': np.median(arr2_clean),
        'q75': np.percentile(arr2_clean, 75),
        'max': np.max(arr2_clean),
        'mean': np.mean(arr2_clean)
    }

    # Set reasonable y-axis limit (99th percentile)
    upper = max(np.percentile(arr1_clean, 99), np.percentile(arr2_clean, 99))

    # Safety check: ensure upper limit is reasonable
    if upper > 100:  # Adjust this threshold based on your expected SWE range
        print(f"Warning: Upper limit ({upper:.2f}) seems high. Data may have outliers.")

    # Limit figure height to prevent memory errors
    fig_height = min(7, max(5, upper / 100))  # Dynamically adjust but cap at 7

    # Create plot
    fig, ax = plt.subplots(figsize=(10, fig_height))

    # Create box plot (hide fliers to avoid extreme outliers)
    bp = ax.boxplot([arr1_clean, arr2_clean],
                    labels=[prev_model_date, rundate],
                    patch_artist=True,
                    showfliers=False,  # HIDE OUTLIERS to prevent huge plots
                    widths=0.6,
                    showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='green',
                                   markeredgecolor='black', markersize=8))

    # Color boxes
    colors = ['steelblue', 'coral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Add median and mean values
    for i, (stat, x_pos) in enumerate(zip([stats1, stats2], [1, 2])):
        # Median (on box)
        ax.text(x_pos, stat['median'], f"{stat['median']:.2f}",
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor='black', linewidth=1.5))

        # Mean (to the side)
        ax.text(x_pos - 0.45, stat['mean'], f"μ={stat['mean']:.2f}",
                ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))

    # Labels and formatting
    ax.set_ylabel(f"{variable} ({unit})", fontsize=12, fontweight='bold')
    ax.set_title(f"{variable} Distribution: {domain}\n{prev_model_date} → {rundate}",
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, upper * 1.05)  # 5% padding above 99th percentile
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Remove the problematic tight_layout line or wrap it in try/except
    try:
        plt.tight_layout()
    except:
        pass  # Skip if layout can't be adjusted

    # Save with error handling
    try:
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_png}")
    except Exception as e:
        print(f"Warning: Could not save with tight bbox. Trying standard save...")
        plt.savefig(output_png, dpi=300)
        print(f"Saved: {output_png}")

    plt.show()
    plt.close()

from rasterio.warp import reproject, Resampling
def swe_vs_elevation(swe_raster, elev_raster, elev_bins):
    elev_bins = np.asarray(elev_bins, dtype=float)

    with rasterio.open(swe_raster) as swe_src:
        swe = swe_src.read(1).astype(float)
        swe[swe == swe_src.nodata] = np.nan
        swe_transform = swe_src.transform
        swe_crs = swe_src.crs
        swe_shape = swe.shape

    elev_on_swe = np.full(swe_shape, np.nan, dtype=float)

    with rasterio.open(elev_raster) as elev_src:
        reproject(
            source=rasterio.band(elev_src, 1),
            destination=elev_on_swe,
            src_transform=elev_src.transform,
            src_crs=elev_src.crs,
            dst_transform=swe_transform,
            dst_crs=swe_crs,
            resampling=Resampling.bilinear
        )

    # convert elevation to meters if needed
    # elev_on_swe = elev_on_swe * 0.3048  # if DEM is in feet

    # mask SWE and elevation
    mask = (~np.isnan(swe)) & (~np.isnan(elev_on_swe))
    swe = swe[mask]
    elev = elev_on_swe[mask]

    if swe.size == 0:
        return [], []

    # compute mean SWE per elevation bin
    bin_idx = np.digitize(elev, elev_bins)
    mean_swe = []
    bin_centers = []

    for i in range(1, len(elev_bins)):
        vals = swe[bin_idx == i]
        mean_swe.append(np.mean(vals) if vals.size > 0 else 0)
        bin_centers.append((elev_bins[i-1] + elev_bins[i]) / 2)

    return bin_centers, mean_swe

def swe_elevation_step_plot(rundate, prev_model_date, domain, raster, prev_raster,
                            variable, unit, elevation_tif, elev_bins, output_png):
    # After collecting your data, create the elevation plot
    file_list = [prev_raster, raster]
    file_labels = [prev_model_date, rundate]
    plt.figure(figsize=(10, 6))

    colors = ['blue', 'red', 'green', 'orange', 'purple']  # Add more if needed

    for i, (rf, label) in enumerate(zip(file_list, file_labels)):
        elev_centers, mean_swe = swe_vs_elevation(rf, elevation_tif, elev_bins)

        if len(elev_centers) > 0:  # only plot if data exists
            plt.step(elev_centers, mean_swe, where='mid',
                     label=label, color=colors[i], linewidth=2)

    plt.xlabel("Elevation (ft)")
    plt.ylabel(f"Mean {variable} ({unit})")
    plt.title(f"{variable} vs Elevation for {domain} — {prev_model_date} to {rundate}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.show()

    # Elevation difference plot (for exactly 2 rasters)
    # Elevation difference plot (for exactly 2 rasters)
    if len(file_list) == 2:
        plt.figure(figsize=(10, 6))

        # Get data for both rasters
        elev_centers_1, mean_swe_1 = swe_vs_elevation(file_list[0], elevation_tif, elev_bins)
        elev_centers_2, mean_swe_2 = swe_vs_elevation(file_list[1], elevation_tif, elev_bins)

        if len(elev_centers_1) > 0 and len(elev_centers_2) > 0:
            # Calculate difference (raster2 - raster1)
            elev_centers = np.array(elev_centers_1)
            swe_diff = np.array(mean_swe_2) - np.array(mean_swe_1)

            # Create step plot
            plt.step(elev_centers, swe_diff, where='mid',
                     color='black', linewidth=2,
                     label=f'{file_labels[1]} - {file_labels[0]}')

            # Fill between zero and the difference line
            # Blue where positive, red where negative
            plt.fill_between(elev_centers, 0, swe_diff,
                             where=(swe_diff >= 0),
                             step='mid',
                             color='blue',
                             alpha=0.3,
                             label='Positive difference')

            plt.fill_between(elev_centers, 0, swe_diff,
                             where=(swe_diff < 0),
                             step='mid',
                             color='red',
                             alpha=0.3,
                             label='Negative difference')

            # Add a horizontal line at zero for reference
            plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

            plt.xlabel("Elevation (ft)")
            plt.ylabel(f"{variable} Difference ({unit})")
            plt.title(f"{variable} Difference vs Elevation for {domain} — {prev_model_date} to {rundate}")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_png[:-4]}_diff.png", dpi=300)
            plt.show()


def filter_data_by_date(df, rundate, max_days_back=2):
    """
    For each station, select the most recent available data
    on or before rundate, up to max_days_back.
    """

    rundate = pd.to_datetime(rundate)
    df = df.copy()

    # Only keep dates within allowed window
    df = df[
        (df["DATE"] <= rundate) &
        (df["DATE"] >= rundate - pd.Timedelta(days=max_days_back))
    ]

    if df.empty:
        return df

    # Sort so newest date is first
    df = df.sort_values(["STATION_NAME", "DATE"], ascending=[True, False])

    # Keep the most recent row per station
    df_latest = df.groupby("STATION_NAME", as_index=False).first()
    df_latest["DAYS_BACK"] = (rundate - df_latest["DATE"]).dt.days

    return df_latest

def snowtrax_comparision(rundate, snowTrax_csv, results_WS, output_csv, model_list, model_labels, reference_col,
                         output_png):
    # load csv and prepare
    df = pd.read_csv(snowTrax_csv)
    df = df[['DATE', 'STATION_NAME', 'ASO_SWE_AF', 'ISNOBAL_DWR_SWE_AF', 'ISNOBAL_M3WORKS_SWE_AF',
             'SNODAS_SWE_AF', 'SNOW17_SWE_AF', 'SWANN_UA_SWE_AF']]
    df['DATE'] = df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df['DATE_formatted'] = df['DATE'].dt.strftime('%Y%m%d')
    print(df.head(5))
    print(df.columns)

    basin_dictionary = {"American River Basin": "07American", "Cosumnes River Basin": "08Cosumnes",
                        "East Carson River River Basin":
                            "21E Carson", "East Walker River Basin": "23E Walker", "Feather River Basin": "05Feather",
                        "KingsRiver  Basin River Basin": "14Kings", "Kern River Basin": "17Kern", "Kaweah River Basin":
                            '15Kaweah', "Mokelumne River Basin": "09Mokelumne", "Merced River Basin": "12Merced",
                        "McCloud River Basin": "02McCloud", "Pit at Shasta Lake River Basin": "03Pit",
                        "Sacramento at Bend Bridge River Basin": "04Sacramento at Bend Bridge",
                        "San Joaquin River Basin":
                            "13San Joaquin", "Tule River Basin": "16Tule",
                        "Sacramento at Delta River Basin": "01Upper Sacramento",
                        "Stanislaus River Basin": "10Stanislaus", "Lake Tahoe Rise": "19Tahoe",
                        "Tuolumne River Basin": "11Tuolomne",
                        "Trinity River Basin": "00Trinity", "Truckee River Basin": "18Truckee",
                        "West Carson River Basin": "20W Carson",
                        "West Walker River Basin": "22W Walker", "Yuba River Basin": "06Yuba"}

    # filter data for the rundate
    filtered_df = filter_data_by_date(df, rundate, max_days_back=2)

    if not filtered_df.empty:
        print("\nFiltered Data Sample:")
        print(filtered_df[['DATE', 'STATION_NAME', 'SNODAS_SWE_AF', 'DATE_formatted']].head())

        # save filtered data
        filtered_df.to_csv(output_csv, index=False)
        print(f"\nSaved filtered data to: {output_csv}")

        # check which models have data
        model_cols = ['ASO_SWE_AF', 'ISNOBAL_DWR_SWE_AF', 'ISNOBAL_M3WORKS_SWE_AF',
                      'SNODAS_SWE_AF', 'SNOW17_SWE_AF', 'SWANN_UA_SWE_AF']

        print(f"\nModel Data Availability:")
        for col in model_cols:
            non_null = filtered_df[col].notna().sum()
            print(f"  {col:30} {non_null}/{len(filtered_df)} stations")

    filtered_df['BASIN_CODE'] = filtered_df['STATION_NAME'].map(basin_dictionary)
    print(filtered_df[['STATION_NAME', 'BASIN_CODE']].head())

    # merge with model runs
    for model, label in zip(model_list, model_labels):
        watershed_csv = results_WS + f"/{rundate}_results_ET/{model}/{rundate}Wtshd_table.csv"

        # merge
        df_model = pd.read_csv(watershed_csv)
        df_model_filter = df_model[['SrtName', 'VOL_AF']]
        df_model_filter = df_model_filter.rename(columns={'VOL_AF': f'{label}_VOL_AF'})
        filtered_df = filtered_df.merge(df_model_filter, left_on='BASIN_CODE', right_on='SrtName', how='inner')
        filtered_df = filtered_df.drop('SrtName', axis=1)
        print(filtered_df.head(5))

        filtered_df.to_csv(f"{output_csv[:-4]}_edit.csv", index=False)
        print(f"\nSaved filtered data to: {output_csv}")

    # calculate % error
    # Calculate percent error: ((model - reference) / reference) × 100
    filtered_df['woCCR_pct_error'] = ((filtered_df['woCCR_VOL_AF'] - filtered_df[reference_col]) /
                                      filtered_df[reference_col]) * 100

    filtered_df['wCCR_pct_error'] = ((filtered_df['wCCR_VOL_AF'] - filtered_df[reference_col]) /
                                     filtered_df[reference_col]) * 100

    # Calculate mean absolute percent error for each model
    woCCR_mean_error = filtered_df['woCCR_pct_error'].abs().mean()
    wCCR_mean_error = filtered_df['wCCR_pct_error'].abs().mean()

    print(f"\nMean Absolute Percent Error vs {reference_col.replace('_SWE_AF', '')}:")
    print(f"  woCCR: {woCCR_mean_error:6.2f}%")
    print(f"  wCCR:  {wCCR_mean_error:6.2f}%")

    # make bar graph
    # Method 2: Grouped bar chart for multiple models
    models = ['ASO_SWE_AF', 'ISNOBAL_DWR_SWE_AF', 'ISNOBAL_M3WORKS_SWE_AF',
              'SNODAS_SWE_AF', 'SNOW17_SWE_AF', 'SWANN_UA_SWE_AF', f'woCCR_VOL_AF', f'wCCR_VOL_AF']

    # Filter to only models with data
    filtered_df = filtered_df.sort_values('BASIN_CODE', ascending=True)
    models_with_data = [col for col in models if filtered_df[col].notna().any()]

    # Create figure with 2 subplots (stacked vertically)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12),
                                   gridspec_kw={'height_ratios': [2, 1]})

    # ============================================================================
    # TOP SUBPLOT: SWE Comparison Bar Chart
    # ============================================================================

    x = range(len(filtered_df))
    width = 0.15
    positions = [i - (len(models_with_data) - 1) * width / 2 for i in x]

    # Add alternating gray background shading for each basin
    for i in range(len(filtered_df)):
        if i % 2 == 0:  # Even indices get gray shading
            ax1.axvspan(i - 0.5, i + 0.5, facecolor='lightgray', alpha=0.3, zorder=0)

    # Plot bars for each model
    for i, model in enumerate(models_with_data):
        offset = [p + i * width for p in positions]
        label = model.replace('_SWE_AF', '').replace('_VOL_AF', '')
        ax1.bar(offset, filtered_df[model], width, label=label)

    ax1.set_xlabel('Basin', fontsize=12, fontweight='bold')
    ax1.set_ylabel('SWE (Acre-Feet)', fontsize=12, fontweight='bold')
    ax1.set_title(f'SWE Comparison by Basin and Model - {rundate[:4]}-{rundate[4:6]}-{rundate[6:]}',
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(filtered_df['BASIN_CODE'], rotation=90, fontsize=10)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # ============================================================================
    # BOTTOM SUBPLOT: Percent Error Bar Chart
    # ============================================================================

    width_error = 0.35
    x_error = np.arange(len(filtered_df))

    # Add alternating gray background shading for each basin
    for i in range(len(filtered_df)):
        if i % 2 == 0:  # Even indices get gray shading
            ax2.axvspan(i - 0.5, i + 0.5, facecolor='lightgray', alpha=0.3, zorder=0)

    # Plot percent error bars
    bars1 = ax2.bar(x_error - width_error / 2, filtered_df['woCCR_pct_error'],
                    width_error, label='woCCR', color='steelblue', alpha=0.8)
    bars2 = ax2.bar(x_error + width_error / 2, filtered_df['wCCR_pct_error'],
                    width_error, label='wCCR', color='coral', alpha=0.8)

    # Add zero reference line
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)

    # Add ±10% reference lines
    ax2.axhline(y=10, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(y=-10, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax2.set_xlabel('Basin', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Percent Error (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Percent Error vs {reference_col}', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_error)
    ax2.set_xticklabels(filtered_df['BASIN_CODE'], rotation=90, fontsize=10)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import rasterio
import matplotlib.colors as mcolors


def create_aspect_comparison(aspect_path, raster, prev_raster,
                                 label_1, label_2, title, variable, unit,
                                 output_path=None, num_bins=16):
    """
    Create three-panel compass rose comparison of SWE by aspect.
    Order: Previous date | Current date | Difference

    Parameters:
    -----------
    aspect_path : str
        Path to aspect raster (0-360°, larger extent is OK)
    raster : str
        Path to CURRENT SWE raster
    prev_raster : str
        Path to PREVIOUS SWE raster
    label_1 : str
        Label for CURRENT date (displayed in middle)
    label_2 : str
        Label for PREVIOUS date (displayed on left)
    output_path : str, optional
        Path to save figure (e.g., 'comparison.png')
    num_bins : int
        Number of aspect bins (8=45° per bin, 16=22.5° per bin)
    """


    # =========================================================================
    # INNER FUNCTION: Calculate SWE by aspect
    # =========================================================================
    def calc_swe_by_aspect(aspect_path, swe_path):
        """Calculate mean SWE for each aspect bin."""

        # Read aspect
        with rasterio.open(aspect_path) as src:
            aspect = src.read(1).astype(float)
            aspect_nodata = src.nodata
            aspect_transform = src.transform
            aspect_bounds = src.bounds

            if aspect_nodata is not None:
                aspect[aspect == aspect_nodata] = np.nan
            aspect[aspect < 0] = np.nan  # Handle flat areas

        # Read SWE (in METERS)
        with rasterio.open(swe_path) as src:
            swe = src.read(1).astype(float)
            swe_nodata = src.nodata
            swe_bounds = src.bounds

            if swe_nodata is not None:
                swe[swe == swe_nodata] = np.nan

        # Handle extent mismatch
        if aspect.shape != swe.shape:
            col_off = int((swe_bounds.left - aspect_bounds.left) / aspect_transform.a)
            row_off = int((aspect_bounds.top - swe_bounds.top) / abs(aspect_transform.e))
            aspect = aspect[row_off:row_off + swe.shape[0], col_off:col_off + swe.shape[1]]

        # Define aspect bins
        bin_edges = np.linspace(0, 360, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Calculate mean SWE per bin
        aspect_means = np.zeros(num_bins)
        pixel_counts = np.zeros(num_bins, dtype=int)

        for i in range(num_bins):
            if i == num_bins - 1:
                mask = (((aspect >= bin_edges[i]) | (aspect < bin_edges[0])) &
                        ~np.isnan(aspect) & ~np.isnan(swe))
            else:
                mask = ((aspect >= bin_edges[i]) & (aspect < bin_edges[i + 1]) &
                        ~np.isnan(aspect) & ~np.isnan(swe))

            pixel_counts[i] = mask.sum()
            aspect_means[i] = np.nanmean(swe[mask]) if pixel_counts[i] > 0 else np.nan

        valid_pixels = (~np.isnan(swe) & ~np.isnan(aspect)).sum()
        swe_range = (np.nanmin(swe), np.nanmax(swe))
        mean_swe = np.nanmean(swe[~np.isnan(aspect)])

        return aspect_means, bin_centers, pixel_counts, valid_pixels, swe_range, mean_swe

    # =========================================================================
    # CALCULATE DISTRIBUTIONS (Note: order matters for labels)
    # =========================================================================
    values_prev, bin_centers, counts_prev, valid_prev, range_prev, mean_prev = calc_swe_by_aspect(aspect_path,
                                                                                                  prev_raster)

    values_curr, _, counts_curr, valid_curr, range_curr, mean_curr = calc_swe_by_aspect(aspect_path, raster)

    # =========================================================================
    # CREATE FIGURE
    # =========================================================================
    fig = plt.figure(figsize=(18, 6))

    # Common scale for first two plots
    vmin_common = np.nanmin([values_prev, values_curr])
    vmax_common = np.nanmax([values_prev, values_curr])

    # Convert to radians
    theta = np.deg2rad(bin_centers)
    width = 2 * np.pi / len(values_prev)

    # Create BLUE colormap for SWE (white -> light blue -> dark blue)
    colors_blue = ['white', 'lightblue', 'deepskyblue', 'dodgerblue', 'blue', 'darkblue']
    cmap_blue = mcolors.LinearSegmentedColormap.from_list('white_blue', colors_blue, N=256)

    # -------------------------------------------------------------------------
    # SUBPLOT 1: PREVIOUS Date (LEFT)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(131, projection='polar')

    norm1 = mcolors.Normalize(vmin=vmin_common, vmax=vmax_common)

    bars1 = ax1.bar(theta, values_prev, width=width, bottom=0,
                    edgecolor='black', linewidth=0.5)

    for bar, val in zip(bars1, values_prev):
        if not np.isnan(val):
            bar.set_facecolor(cmap_blue(norm1(val)))
            bar.set_alpha(0.8)

    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    ax1.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax1.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],
                        fontsize=11, fontweight='bold')
    ax1.set_ylim(0, vmax_common * 1.1)
    ax1.set_title(f'{label_2}\nMean {variable} by Aspect', fontsize=13, fontweight='bold', pad=20)

    sm1 = plt.cm.ScalarMappable(cmap=cmap_blue, norm=norm1)
    sm1.set_array([])
    cbar1 = plt.colorbar(sm1, ax=ax1, pad=0.1, shrink=0.8)
    cbar1.set_label(f'Mean {variable} ({unit})', fontsize=11, fontweight='bold')

    # -------------------------------------------------------------------------
    # SUBPLOT 2: CURRENT Date (MIDDLE)
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(132, projection='polar')

    bars2 = ax2.bar(theta, values_curr, width=width, bottom=0,
                    edgecolor='black', linewidth=0.5)

    for bar, val in zip(bars2, values_curr):
        if not np.isnan(val):
            bar.set_facecolor(cmap_blue(norm1(val)))
            bar.set_alpha(0.8)

    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    ax2.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax2.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],
                        fontsize=11, fontweight='bold')
    ax2.set_ylim(0, vmax_common * 1.1)
    ax2.set_title(f'{label_1}\nMean {variable} by Aspect', fontsize=13, fontweight='bold', pad=20)

    cbar2 = plt.colorbar(sm1, ax=ax2, pad=0.1, shrink=0.8)
    cbar2.set_label(f'Mean {variable} ({unit})', fontsize=11, fontweight='bold')

    # -------------------------------------------------------------------------
    # SUBPLOT 3: Difference (Current - Previous) (RIGHT)
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(133, projection='polar')

    diff = values_curr - values_prev  # Current - Previous
    vmax_diff = np.nanmax(np.abs(diff))

    # Create colormap: Red (decrease) -> White (no change) -> Blue (increase)
    norm3 = mcolors.TwoSlopeNorm(vmin=-vmax_diff, vcenter=0, vmax=vmax_diff)
    cmap3 = mcolors.LinearSegmentedColormap.from_list(
        'red_white_blue',
        ['darkred', 'red', 'lightcoral', 'white', 'lightblue', 'blue', 'darkblue'],
        N=256
    )

    bars3 = ax3.bar(theta, np.abs(diff), width=width, bottom=0,
                    edgecolor='black', linewidth=0.5)

    for bar, d in zip(bars3, diff):
        if not np.isnan(d):
            bar.set_facecolor(cmap3(norm3(d)))
            bar.set_alpha(0.8)

    ax3.set_theta_zero_location('N')
    ax3.set_theta_direction(-1)
    ax3.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax3.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],
                        fontsize=11, fontweight='bold')
    ax3.set_ylim(0, vmax_diff * 1.1)
    ax3.set_title(f'Difference ({label_1} - {label_2})\nBlue=Increase  Red=Decrease',
                  fontsize=13, fontweight='bold', pad=20)

    sm3 = plt.cm.ScalarMappable(cmap=cmap3, norm=norm3)
    sm3.set_array([])
    cbar3 = plt.colorbar(sm3, ax=ax3, pad=0.1, shrink=0.8)
    cbar3.set_label(f'Difference {variable}', fontsize=11, fontweight='bold')

    plt.suptitle(f'{title}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.show()

def raster_box_whisker_plot_multi(
    rundate,
    raster_paths,
    labels,
    domain,
    variable,
    unit,
    output_png
):
    import numpy as np
    import matplotlib.pyplot as plt

    arrays_clean = []
    stats_list = []

    def clean_array(arr):
        arr = arr[~np.isnan(arr) & ~np.isinf(arr)]
        arr = arr[arr >= 0]
        p999 = np.percentile(arr, 99.9)
        return arr[arr <= p999]

    # Read, clean, and compute stats
    for r in raster_paths:
        arr = read_raster_values(r)
        arr = clean_array(arr)

        arrays_clean.append(arr)

        stats_list.append({
            "mean": np.mean(arr),
            "q25": np.percentile(arr, 25),
            "q75": np.percentile(arr, 75)
        })

    upper = max(np.percentile(a, 99) for a in arrays_clean)

    fig_height = min(7, max(5, upper / 100))
    fig, ax = plt.subplots(figsize=(12, fig_height))

    bp = ax.boxplot(
        arrays_clean,
        labels=labels,
        patch_artist=True,
        showfliers=False,
        widths=0.6,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='green',
                       markeredgecolor='black', markersize=7)
    )

    # Color boxes
    colors = plt.cm.tab10.colors
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # ---- STAT ANNOTATIONS ----
    for i, stats in enumerate(stats_list, start=1):
        # Mean (slightly left)
        ax.text(
            i - 0.25,
            stats["mean"],
            f"μ={stats['mean']:.1f}",
            fontsize=9,
            ha="right",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor="lightgreen",
                      alpha=0.8)
        )

        # Q25
        ax.text(
            i + 0.28,
            stats["q25"],
            f"Q1={stats['q25']:.1f}",
            fontsize=8,
            ha="left",
            va="top",
            color="dimgray"
        )

        # Q75
        ax.text(
            i + 0.28,
            stats["q75"],
            f"Q3={stats['q75']:.1f}",
            fontsize=8,
            ha="left",
            va="bottom",
            color="dimgray"
        )

    ax.set_ylabel(f"{variable} ({unit})", fontweight="bold")
    ax.set_title(f"{rundate} {variable} Distribution – {domain}", fontweight="bold")
    ax.set_ylim(0, upper * 1.05)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.show()
    plt.close()

    print(f"Saved: {output_png}")

def plot_rasters_side_by_side(
    rundate,
    basin,
    raster_paths,
    titles,
    variable,
    unit,
    output_png,
    cmap="Blues",
    vmin_percentile=2,
    vmax_percentile=98
):
    import rasterio
    import numpy as np
    import matplotlib.pyplot as plt

    arrays = []
    profiles = []

    # Read rasters
    for r in raster_paths:
        with rasterio.open(r) as src:
            arr = src.read(1).astype(float)
            arr[arr == src.nodata] = np.nan
            arrays.append(arr)
            profiles.append(src.profile)

    # Shared color scale
    stacked = np.hstack([a.flatten() for a in arrays])
    stacked = stacked[~np.isnan(stacked)]

    vmin = np.percentile(stacked, vmin_percentile)
    vmax = np.percentile(stacked, vmax_percentile)

    n = len(arrays)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), constrained_layout=True)

    fig.suptitle(
        f"SWE – SNM – {basin} – {rundate}",
        fontsize=14,
        fontweight="bold"
    )

    fig.subplots_adjust(top=0.8)

    if n == 1:
        axes = [axes]

    for ax, arr, title in zip(axes, arrays, titles):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    cbar = fig.colorbar(
        im,
        ax=axes,
        orientation="vertical",
        fraction=0.025,
        pad=0.02
    )
    cbar.set_label(f"{variable} ({unit})")

    plt.savefig(output_png, dpi=300)
    plt.show()
    plt.close()

    print(f"Saved raster visualization: {output_png}")