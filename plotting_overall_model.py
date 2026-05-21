# import modules
import matplotlib.pyplot as plt
import os
import pandas as pd
import rasterio
import arcpy
from SWE_Fusion_functions import *
import matplotlib.pyplot as plt
import numpy as np

print('modules imported')

# parameters
prev_model = "Y"
# prev_model_date = ""
prev_model_run = ""

## plot side-by-side histograms for each domain over elevation
## plot side by side box and whiskers plots for each domain


## plot all the sensors for SWE with a line on if they are increasing or decreasing and blue for increasing and red for
## decreasing

## SENSOR DATA BY ELEVATION
# get points within raster
# rundate = "20260101"
# prev_model_date = "20251227"
#
# raster = r"M:\SWE\WestWide\documents\2025_RT_Reports\20250406_RT_Report\RT_CanAdj_rcn_noSW_wCCR\p8_20250406_noneg.tif"
# prev_raster = r"M:\SWE\WestWide\documents\2025_RT_Reports\20250331_RT_Report\RT_CanAdj_rcn_noSW_wCCR\p8_20250331_noneg.tif"
# point_dataset = fr"M:\SWE\WestWide\Spatial_SWE\WW_regression\RT_report_data\{rundate}_results_ET\{rundate}_sensors_albn83.shp"
# prev_pointDataset = fr"M:\SWE\WestWide\Spatial_SWE\WW_regression\RT_report_data\{prev_model_date}_results\{prev_model_date}_sensors_albn83.shp"
# id_column = "Site_ID"
# swe_col = 'pillowswe'
# convert_meters_feet = "Y"
# convert_feet_meters = "N"
# elev_col = 'dem'
# elevation_tif = r"M:\SWE\WestWide\data\topo\ww_DEM_albn83_feet_banded_100.tif"
# elev_bins = np.arange(1500, 15000, 100, dtype=float)

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
def raster_box_whisker_plot(raster, prev_raster, domain, output_png):
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

def swe_elevation_step_plot(raster, prev_raster, elevation_tif, elev_bins, output_png):
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
            plt.savefig(output_png, dpi=300)
            plt.show()

######################
rundate = '20260101'
prev_model_date = '20251227'
current_model_run = "RT_CanAdj_rcn_wCCR_nofscamskSens_testReport"
prev_model_run = "RT_CanAdj_rcn_wCCR_nofscamskSens"
domains = ['INMT', 'PNW', 'SOCN', 'NOCN', 'SNM']
prev_vetting_WS = f"W:/documents/2025_RT_Reports/{prev_model_date}_RT_report_ET/{prev_model_run}/vetting_domains/"
vetting_WS = f"W:/documents/2026_RT_Reports/{rundate}_RT_report_ET/{current_model_run}/vetting_domains/"
elevation_tif = r"M:\SWE\WestWide\data\topo\ww_DEM_albn83_feet_banded_100.tif"
elev_bins = np.arange(1500, 15000, 100, dtype=float)

for domain in domains:

    if domain == "SNM":
        print('analyzing Sierras')
        prev_vetting_WS = f"J:/paperwork/0_UCSB_DWR_Project/2025_RT_Reports/{prev_model_date}_RT_report_ET/{prev_model_run}/vetting_domains/"
        vetting_WS = f"J:/paperwork/0_UCSB_DWR_Project/2026_RT_Reports/{rundate}_RT_report_ET/{current_model_run}/vetting_domains/"
        raster = vetting_WS + f"p8_{rundate}_noneg.tif"
        prev_raster = prev_vetting_WS + f"p8_{prev_model_date}_noneg.tif"
        point_dataset = fr"M:\SWE\Sierras\Spatial_SWE\SNM_regression\RT_report_data\{rundate}_results_ET\SNM_{rundate}_sensors_albn83.shp"
        prev_pointDataset = fr"M:\SWE\Sierras\Spatial_SWE\SNM_regression\RT_report_data\{prev_model_date}_results_ET\SNM_{prev_model_date}_sensors_albn83.shp"

    else:
        print(f"analyzing {domain}")
        prev_vetting_WS = f"W:/documents/2025_RT_Reports/{prev_model_date}_RT_report_ET/{prev_model_run}/vetting_domains/"
        vetting_WS = f"W:/documents/2026_RT_Reports/{rundate}_RT_report_ET/{current_model_run}/vetting_domains/"
        raster = vetting_WS + f"p8_{rundate}_noneg_{domain}_clp.tif"
        prev_raster = prev_vetting_WS + f"p8_{prev_model_date}_noneg_{domain}_clp.tif"
        point_dataset = fr"M:\SWE\WestWide\Spatial_SWE\WW_regression\RT_report_data\{rundate}_results_ET\{rundate}_sensors_albn83.shp"
        prev_pointDataset = fr"M:\SWE\WestWide\Spatial_SWE\WW_regression\RT_report_data\{prev_model_date}_results_ET\{prev_model_date}_sensors_albn83.shp"

    ## engage plots
    print("Plotting pillow change comparison...")
    pillow_date_comparison(rundate=rundate, prev_model_date=prev_model_date, raster=raster, point_dataset=point_dataset,
                           prev_pointDataset=prev_pointDataset, id_column="Site_ID", swe_col="pillowswe", elev_col="dem",
                           output_png= vetting_WS + f"{domain}_sensor_difference.png", convert_meters_feet="Y")

    print('Creating box and whiskers plot...')
    raster_box_whisker_plot(raster=raster, prev_raster=prev_raster,
                            domain=domain, output_png=vetting_WS + f"{domain}_{rundate}_box_whisker.png")

    print('Creating elevation step plot...')
    swe_elevation_step_plot(raster=raster, prev_raster=prev_raster,
                            output_png=vetting_WS + f"{domain}_{rundate}_elevation_step.png", elevation_tif=elevation_tif,
                            elev_bins=elev_bins)