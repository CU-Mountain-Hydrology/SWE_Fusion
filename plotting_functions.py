# import modules
import arcpy
from arcpy.sa import *
import os
import pandas as pd
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from SWE_Fusion_functions import get_points_within_raster
print("modules")

folder = r"W:\Spatial_SWE\WW_regression\RT_report_data\20250503_results_ET\ASO_BiasCorrect_fSCA_RT_CanAdj_rcn_noSW_woCCR/"
vetting_output = folder + "vetting/"
elevation_tif = r"M:\SWE\WestWide\data\topo\ww_DEM_albn83_feet_banded_100.tif"
prefix = "20250503"
elev_bins = np.arange(1500, 15000, 100, dtype=float)

os.makedirs(vetting_output, exist_ok=True)

unique_names = set()  # use a set to keep unique values
file_mapping = {}

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


for root, dirs, files in os.walk(folder):
    for file in files:
        if file.startswith(prefix):
            parts = file.split("_")
            if len(parts) >= 2:
                name = "_".join(parts[:2])
                unique_names.add(name)

                # populate file_mapping
                if name not in file_mapping:
                    file_mapping[name] = []
                file_mapping[name].append(os.path.join(root, file))  # full path

# Convert unique names to list if needed
unique_names = list(unique_names)
print("Unique names:", unique_names)

print("\nFull file names by prefix:")
for name, files in file_mapping.items():
    print(f"\n{name}:")
    file_list = []
    labels = []
    for f in files:
        if f.endswith("_clp.tif") or f.endswith("fix_albn83.tif"):
            print(f"  {f}")
            file_list.append(f)
            splits = os.path.basename(f).split("_")
            labels.append("_".join(splits[2:4]))

    all_values = []
    final_labels = []

    # read raster values
    for rf, label in zip(file_list, labels):
        with rasterio.open(rf) as src:
            arr = src.read(1).astype(float)

            if src.nodata is not None:
                arr = arr[arr != src.nodata]
            else:
                arr = arr[~np.isnan(arr)]

            # skip empty rasters
            if arr.size == 0:
                print(f"Skipping empty raster: {rf}")
                continue

            all_values.append(arr)
            final_labels.append(label)

    # create boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(all_values, labels=final_labels, showfliers=False)
    plt.ylabel("SWE (m)")
    plt.title(f"SWE Distribution — {name}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(vetting_output + f"{name}_biasCorrection_boxplot.png", dpi=300)
    plt.show()

    # elevation plots
    plt.figure(figsize=(10, 6))
    bar_width = 100  # match your elevation bin width

    # Store data for all models
    all_data = {}
    for rf, label in zip(file_list, labels):
        elev_centers, mean_swe = swe_vs_elevation(rf, elevation_tif, elev_bins)
        if len(elev_centers) > 0:  # only add if data exists
            all_data[label] = (elev_centers, mean_swe)
            # Debug: print summary stats
            mask = np.array(elev_centers) >= 5000
            swe_filtered = np.array(mean_swe)[mask]
            print(f"{label}: mean SWE = {np.mean(swe_filtered):.4f}, max = {np.max(swe_filtered):.4f}")

    # Create overlapping bars with more transparency
    colors = plt.cm.tab10(range(len(all_data)))
    linestyles = ['-', '--', '-.', ':', '-']  # Different line styles

    for i, (label, (elev_centers, mean_swe)) in enumerate(all_data.items()):
        # Filter to only show >= 5000m
        mask = np.array(elev_centers) >= 5000
        elev_filtered = np.array(elev_centers)[mask]
        swe_filtered = np.array(mean_swe)[mask]

        # Plot bars with high transparency
        plt.bar(elev_filtered, swe_filtered, width=bar_width,
                alpha=0.15, color=colors[i], edgecolor='none')

        # Plot step line with different styles to distinguish overlapping lines
        plt.step(elev_filtered, swe_filtered, where='mid', color=colors[i],
                 linewidth=2.5, label=label, linestyle=linestyles[i])

    plt.xlabel("Elevation (m)")
    plt.ylabel("Mean SWE (m)")
    plt.title(f"SWE vs Elevation — {name}")
    plt.legend(loc='upper left')
    plt.grid(alpha=0.3, axis='y')
    plt.xlim(left=5000)  # Start x-axis at 5000m
    plt.tight_layout()
    plt.savefig(vetting_output + f"{name}_biasCorrection_elevation.png", dpi=300)
    plt.show()


