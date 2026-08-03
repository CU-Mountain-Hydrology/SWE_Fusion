"""
Functions to create the following plots for the Sierra Nevada domain:
 * Week-long daily box plot of SWE (m) with dots per basin
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import datetime as dt

from config import Config


def weekly_swe_trend_snm(date: int, cfg: Config, basin_means: bool = False, volume_total: bool = False, show_plot: bool = False):
    """
    Creates a plot comparing the trend distribution of daily SWE throughout one week.
    Each day has a box and whiskers plot for the whole SNM domain, with the mean of each basin marked with an X.

    :param date: (YYYYMMDD) Date of the first day of the week
    :param cfg: Configuration object containing environment variables set in the .env
    :param basin_means: (bool) If true, plot an x for each basin mean on each day. Default: False
    :param volume_total: (bool) If true, plot a second axis and line containing daily volume totals in acre-feet. Default: False
    :param show_plot: (bool) If true, show the plot when it's created

    """
    import arcpy
    from arcpy.sa import ExtractByMask

    # TODO: move raster_dir to .env once the daily files are from operational daily running
    raster_dir = r"H:/WestUS_Data/Regress_SWE/WY2026_Daily/2026"
    # TODO: move to .env
    snm_basins = r"M:/SWE/WestWide/data/hydro/SNM/dwr_basins_albn83.shp"

    ACRE_FEET_PER_CUBIC_METER = 1 / 1233.48184

    # Check that rasters exist for each day
    # TODO: handle one or more days missing by excluding them from the plot. What about trend lines (discontinuity or go to next)?
    start_date = dt.datetime.strptime(str(date), "%Y%m%d")
    raster_paths = []
    day_labels = []
    for i in range(7):
        current_date = start_date + timedelta(days=i)
        date_str = current_date.strftime("%Y%m%d")
        # TODO: change filepath when running operationally, move to .env?
        current_path = f"{raster_dir}/{date_str}/SNM_phvrcn_{date_str}_fscamsk_clp.tif"
        if os.path.exists(current_path):
            raster_paths.append(current_path)
            day_labels.append(current_date.strftime("%b %e")) # Formatted as "Apr 9"
    raster_paths.sort()

    if not raster_paths:
        raise FileNotFoundError(f"No rasters found for week starting on {date}!")

    print(f"Found {len(raster_paths)} rasters:")
    for p in raster_paths:
        print(f"  {os.path.basename(p)}")


    arcpy.CheckOutExtension("Spatial")

    # Extract and mask pixel values for each day
    day_values = []
    day_volumes_af = []
    for path in raster_paths:
        # masked_raster = ExtractByMask(path, BASIN_SHP)

        # TODO: make sure rasters are clipped to the SNM domain when running operationally

        # Get NoData value and set to NaN
        nodata_vals = [arcpy.Describe(path).noDataValue, -9999]
        arr = arcpy.RasterToNumPyArray(path, nodata_to_value=np.nan).astype(float)
        for nodata_val in nodata_vals:
            arr[arr == nodata_val] = np.nan

        # Flatten and drop NoData and zero-valued cells
        flat = arr.flatten()
        flat = flat[~np.isnan(flat)]
        flat = flat[flat != 0]

        day_values.append(flat)
        print(f"{os.path.basename(path)}: {flat.size} valid non-zero pixels, "
              f"mean={flat.mean():.4f} m, max={flat.max():.4f} m")

        # Calculate volume in acre feet
        if volume_total:
            cell_area_m2 = 250000 # TODO: Get this from the raster instead of hardcoding it
            print(f"cell_area_m2: {cell_area_m2:.4f} m2")
            volume_m3 = flat.sum() * cell_area_m2
            print(f"sum: {flat.sum():.4f}")
            volume_af = volume_m3 * ACRE_FEET_PER_CUBIC_METER
            day_volumes_af.append(volume_af)
            print(f"{os.path.basename(path)}: volume={volume_af:,.0f} acre-feet")

    arcpy.CheckInExtension("Spatial")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.boxplot(
        day_values,
        tick_labels=day_labels,
        vert=True,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="#7fb3d5", edgecolor="#1b4f72"),
        medianprops=dict(color="#c0392b", linewidth=2),
    )

    ax.set_ylabel("SWE (m)")
    ax.set_xlabel(f"Date ({str(date)[:4]})")
    ax.set_title("Weekly SWE Trend for SNM")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=30, ha="right")

    if volume_total:
        # Boxplot x-positions are 1-indexed (1, 2, 3, ...)
        x_positions = range(1, len(day_labels) + 1)

        ax2 = ax.twinx()
        ax2.plot(
            x_positions,
            day_volumes_af,
            color="#27ae60",
            marker="o",
            linewidth=2,
            label="Volume (acre-feet)",
        )
        ax2.set_ylabel("SWE Volume (acre-feet)", color="#27ae60")
        ax2.tick_params(axis="y", labelcolor="#27ae60")
        ax2.set_ylim(bottom=0)  # volume shouldn't go negative; keeps the line readable

    plt.tight_layout()

    # TODO: move output path to .env
    output_png = f"./output/weekly_swe_trend_snm_{date}_vt.png"

    plt.savefig(output_png, dpi=300)
    print(f"\nSaved plot to {output_png}")
    if show_plot:
        plt.show()

if __name__ == "__main__":
    # Development tests
    config = Config()
    weekly_swe_trend_snm(20260409, config, basin_means=False, volume_total=True, show_plot=True)