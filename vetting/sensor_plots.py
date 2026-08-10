# vetting/snotel_plots.py
"""
Functions to create the following Snotel and CDEC sensor plots:
 * TODO: Sensor difference plot
 * Scatter plot of model swe vs sensor swe, for both single cell and 3x3 averaged window, within a given area
 * TODO: Plot of model vs sensor error over time (week to months)
"""

import os
import numpy as np
import arcpy
import matplotlib.pyplot as plt
from arcpy.sa import ExtractMultiValuesToPoints, FocalStatistics, NbrRectangle

from config import Config

def _filter_extent(sensor_shapefile: str, extent_shapefile: str):
    """
    Clips the sensor shapefile to only contain those that are inside extent_shapefile. Returns the filtered shapefile.

    :param sensor_shapefile: Path to the SNOTEL/CDEC sensor point shapefile
    :param extent_shapefile: Path to a polygon shapefile defining the area of interest

    :return: Path to an in-memory feature class containing only the points within extent_shp
    """
    if not arcpy.Exists(extent_shapefile):
        raise FileNotFoundError(f"Extent shapefile not found: {extent_shapefile}")

    sensor_lyr = "sensor_lyr"
    if arcpy.Exists(sensor_lyr):
        arcpy.management.Delete(sensor_lyr)
    arcpy.management.MakeFeatureLayer(sensor_shapefile, sensor_lyr)

    arcpy.management.SelectLayerByLocation(sensor_lyr, "WITHIN", extent_shapefile)

    matched = int(arcpy.management.GetCount(sensor_lyr)[0])
    if matched == 0:
        raise ValueError("No sensor points fall within the provided extent shapefile.")

    filtered_fc = "in_memory/sensor_filtered"
    if arcpy.Exists(filtered_fc):
        arcpy.management.Delete(filtered_fc)
    arcpy.management.CopyFeatures(sensor_lyr, filtered_fc)
    arcpy.management.Delete(sensor_lyr)

    print(f"{matched} sensor points fall within the extent shapefile.")
    return filtered_fc


def _extract_values(sensor_fc: str, raster_path: str, cfg: Config, focal_window: int = None) -> dict:
    """
    Runs ExtractMultiValuesToPoints against a raster (optionally pre-smoothed with FocalStatistics) and returns
    paired model/sensor values.

    :param sensor_fc: Path to the sensor point shapefile
    :param raster_path: Path to the model SWE raster (.tif)
    :param cfg: Configuration object containing environment variables set in the .env
    :param focal_window: If set, size of the square focal mean window (e.g. 3 for 3x3). If None, the raster is sampled
        at the single cell containing each point.

    :return: dict with "model" and "sensor" numpy arrays of paired, NoData-filtered values
    """
    raster = raster_path
    if focal_window is not None:
        neighborhood = NbrRectangle(focal_window, focal_window, "CELL")
        raster = FocalStatistics(raster_path, neighborhood, "MEAN", "DATA")

    # Work on a scratch copy so repeated calls (single-cell vs focal) don't collide on field names
    scratch_fc = "in_memory/sensor_extract"
    if arcpy.Exists(scratch_fc):
        arcpy.management.Delete(scratch_fc)
    arcpy.management.CopyFeatures(sensor_fc, scratch_fc)

    model_field = "model_val"
    ExtractMultiValuesToPoints(scratch_fc, [[raster, model_field]])

    model_vals, sensor_vals = [], []
    with arcpy.da.SearchCursor(scratch_fc, [model_field, cfg.sensor_swe_field]) as cursor:
        for model_val, sensor_val in cursor:
            if model_val is None or sensor_val is None:
                continue
            if model_val <= 0 or sensor_val <= 0:
                continue
            model_vals.append(model_val)
            sensor_vals.append(sensor_val)

    arcpy.management.Delete(scratch_fc)

    return {"model": np.array(model_vals), "sensor": np.array(sensor_vals)}


def _plot_scatter(single_cell: dict, focal: dict, date: int, out_path: str = None):
    """
    Plots single-cell vs 3x3-averaged extraction on the same axes for comparison, with a 1:1 line.
    TODO: remove 3x3 from legend if focal is None
    TODO: docs
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(single_cell["sensor"], single_cell["model"], s=25, alpha=0.7,
               label="Single cell", color="tab:blue")
    ax.scatter(focal["sensor"], focal["model"], s=25, alpha=0.7,
               label="3x3 average", color="tab:orange")

    all_vals = np.concatenate([single_cell["sensor"], single_cell["model"],
                                focal["sensor"], focal["model"]])
    if all_vals.size:
        lims = [0, np.nanmax(all_vals) * 1.05]
        ax.plot(lims, lims, "k--", linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    def _stats(d):
        if d["sensor"].size < 2:
            return None
        rmse = np.sqrt(np.mean((d["model"] - d["sensor"]) ** 2)) # TODO: this probably isn't that useful, use abs mae
        bias = np.mean(d["model"] - d["sensor"])
        r2 = np.corrcoef(d["sensor"], d["model"])[0, 1] ** 2
        return rmse, bias, r2

    stats_lines = []
    for label, d in [("Single cell", single_cell), ("3x3 avg", focal)]:
        s = _stats(d)
        if s:
            rmse, bias, r2 = s
            stats_lines.append(f"{label}: RMSE={rmse:.2f}  Bias={bias:.2f}  R2={r2:.2f}")
    if stats_lines:
        ax.text(0.03, 0.97, "\n".join(stats_lines), transform=ax.transAxes,
                va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel("Sensor SWE (m)")
    ax.set_ylabel("Model SWE (m)")
    ax.set_title(f"Model vs Sensor SWE — {date}")
    ax.legend(loc="lower right")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    if out_path:
        # Confirm directory exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # Save plot
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")
    else:
        plt.show()

    plt.close(fig)


def model_vs_sensor(date: int, cfg: Config, focal_sampling: bool = False, extent_shapefile: str = None, out_path: str = None):
    """
    Creates a scatter plot of model SWE vs sensor SWE.
    TODO: print statements
    TODO: pass plot title as parameter?
    TODO: save error stats to a csv so it can be plotted over time

    :param date: (YYYYMMDD) Date to compare the two products on
    :param cfg: Configuration object containing environment variables set in the .env
    :param focal_sampling: (bool) If true, compare sensors to a 3x3 focal-mean sample instead of a single cell value
    :param extent_shapefile: Path to a shapefile defining the extent for comparison. Default = None (full extent)
    :param out_path: Path to where the png should be saved. Default = None (show plot but don't save)
    """

    # Select model SWE raster
    # TODO: how to know which tif should be used? What about after ASO bias correction?
    # TODO: remove hardcoded path
    model_raster = "W:/documents/2026_RT_Reports/20260412_RT_report/ASO_BiasCorrect_RT_CanAdj_rcn_woCCR_nofscamskSens_noMdlFsca_UseThis/p8_20260412_noneg.tif"
    if not os.path.exists(model_raster):
        raise FileNotFoundError(f"Model SWE raster not found at {model_raster}")

    # Select sensor shapefile
    sensor_shapefile = f"{cfg.ww_results_workspace}/{date}_results/{date}_sensors_albn83.shp"
    if not arcpy.Exists(sensor_shapefile):
        raise FileNotFoundError(f"Sensor shapefile not found at {sensor_shapefile}")

    # Check that the extent shapefile is valid
    if extent_shapefile and not arcpy.Exists(extent_shapefile):
        raise FileNotFoundError(f"Extent shapefile not found at {extent_shapefile}")

    # Clip sensors to the extent shapefile
    points_fc = sensor_shapefile
    if extent_shapefile:
        print(f"Filtering sensor points to extent: {extent_shapefile}")
        points_fc = _filter_extent(sensor_shapefile, extent_shapefile)

    # Compute difference for a single cell (no focal window)
    arcpy.CheckOutExtension("Spatial")
    single_cell = _extract_values(points_fc, model_raster, cfg, focal_window=None)

    # Compute difference for a 3x3 focal window
    focal = {"model": np.array([]), "sensor": np.array([])}
    if focal_sampling:
        print("Running 3x3 focal-mean extraction...")
        focal = _extract_values(points_fc, model_raster, cfg, focal_window=3)

    # Clear filtered points from local memory
    if extent_shapefile and arcpy.Exists(points_fc):
        arcpy.management.Delete(points_fc)

    print(f"{len(single_cell['sensor'])} valid sensor points matched (single cell), "
          f"{len(focal['sensor'])} matched (3x3 average).")

    arcpy.CheckInExtension("Spatial")

    # Create scatter plot
    _plot_scatter(single_cell, focal, date, out_path=out_path)


def sensor_error_trend(date: int, n_days: int, cfg: Config, focal_sampling: bool = False, extent_shapefile: str = None, out_path: str = None):
    """
    Plots the daily error between the model and sensors over some period of time.

    :param date: (YYYYMMDD) The date the comparison is run on. This is the last day on the plot.
    :param n_days: The number of days to plot
    :param cfg: Configuration object containing environment variables set in the .env
    :param focal_sampling: (bool) If true, compare sensors to a 3x3 focal-mean sample instead of a single cell value
    :param extent_shapefile: Path to a shapefile defining the extent for comparison. Default = None (full extent)
    :param out_path: Path to where the png should be saved. Default = None (show plot but don't save)
    """
    # TODO: Load csv with error stats generated when the model_vs_sensor function was called previously.
    # TODO: If any dates are missing for the focal method chosen, run _extract_values
    # TODO: Plot
    pass

if __name__ == "__main__":
    config = Config()
    socn_extent = r"W:/data/hydro/SOCN_Region_albn83.shp"
    model_vs_sensor(20260412, cfg=config, focal_sampling=True, extent_shapefile=socn_extent, out_path="./output/model_vs_sensor_20260412_focal_socn.png")