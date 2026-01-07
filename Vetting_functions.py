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
        point_color = "blue" if row[swe_col] > row[f"prev_{swe_col}"] else "red"

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
               label=f'{prev_model_date} (decline)')
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
##############3
#def
# open raster

def raster_box_whisker_plot(rundate, prev_model_date, raster, prev_raster, domain, output_png):
    arr1 = read_raster_values(prev_raster)
    arr2 = read_raster_values(raster)

    # optional: limit y-axis based on 99th percentile across both rasters
    upper = max(np.percentile(arr1, 99), np.percentile(arr2, 99))

    # plot side-by-side
    plt.figure(figsize=(8, 6))
    plt.boxplot([arr1, arr2],
                labels=[os.path.basename(prev_raster), os.path.basename(raster)],
                patch_artist=True,
                showfliers=True)
    plt.ylabel("SWE (m)")
    plt.title(f"Comparison of SWE Distributions {domain} | {prev_model_date} to {rundate}")
    plt.ylim(0, upper)
    plt.savefig(output_png)
    plt.show()

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

def swe_elevation_step_plot(rundate, prev_model_date, domain, raster, prev_raster, elevation_tif, elev_bins, output_png):
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

    plt.xlabel("Elevation (m)")
    plt.ylabel("Mean SWE (m)")
    plt.title(f"SWE vs Elevation for {domain} — {prev_model_date} to {rundate}")
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

            plt.xlabel("Elevation (m)")
            plt.ylabel("SWE Difference (m)")
            plt.title(f"SWE Difference vs Elevation for {domain} — {prev_model_date} to {rundate}")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_png[:-4]}_diff.png", dpi=300)
            plt.show()


def filter_data_by_date(df, target_date, max_days_back=2):
    """
    Filter data for a specific date, using fallback dates if needed.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with DATE column already parsed
    target_date : str
        Target date in format 'YYYYMMDD'
    max_days_back : int
        Maximum number of days to look back (default: 2)

    Returns:
    --------
    pd.DataFrame
        Filtered dataframe with one row per station/basin
    """

    # Convert target_date to datetime
    target_dt = pd.to_datetime(target_date, format='%Y%m%d')

    print(f"\n{'=' * 70}")
    print(f"TARGET DATE: {target_dt.strftime('%Y-%m-%d')} ({target_date})")
    print(f"LOOKBACK WINDOW: {max_days_back} days")
    print(f"{'=' * 70}\n")

    # Get unique stations/basins
    stations = df['STATION_NAME'].unique()
    print(f"Processing {len(stations)} stations/basins\n")

    # List of model columns
    model_cols = ['ASO_SWE_AF', 'ISNOBAL_DWR_SWE_AF', 'ISNOBAL_M3WORKS_SWE_AF',
                  'SNODAS_SWE_AF', 'SNOW17_SWE_AF', 'SWANN_UA_SWE_AF']

    result_rows = []

    # For each station, find the best matching date
    for station in stations:
        station_df = df[df['STATION_NAME'] == station].copy()

        # Try exact date match first
        exact_match = station_df[station_df['DATE'] == target_dt]

        if len(exact_match) > 0:
            result_rows.append(exact_match.iloc[0])
            print(f"✓ {station:40} Exact date ({target_dt.strftime('%Y-%m-%d')})")
        else:
            # Look for fallback within window
            date_range_start = target_dt - timedelta(days=max_days_back)

            fallback_matches = station_df[
                (station_df['DATE'] >= date_range_start) &
                (station_df['DATE'] <= target_dt)
                ].sort_values('DATE', ascending=False)

            if len(fallback_matches) > 0:
                best_match = fallback_matches.iloc[0]
                result_rows.append(best_match)
                days_diff = (target_dt - best_match['DATE']).days
                print(f"{station:40} Fallback ({best_match['DATE'].strftime('%Y-%m-%d')}, -{days_diff}d)")
            else:
                print(f"{station:40} EXCLUDED (no data within {max_days_back} days)")

    # Create result dataframe
    if result_rows:
        result_df = pd.DataFrame(result_rows)

        print(f"\n{'=' * 70}")
        print(f"SUMMARY:")
        print(f"  Total stations processed: {len(stations)}")
        print(f"  Stations included:        {len(result_df)}")
        print(f"  Stations excluded:        {len(stations) - len(result_df)}")
        print(f"{'=' * 70}\n")

        return result_df
    else:
        print("\nNo stations found with data in the specified window")
        return pd.DataFrame()


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