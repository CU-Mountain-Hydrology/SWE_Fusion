# vetting/snotel_plots.py
"""
Functions to create the following Snotel and CDEC sensor plots:
 * TODO: Sensor difference plot
 * Scatter plot of model swe vs sensor swe, for both single cell and 3x3 averaged window, within a given area
"""

import os
import numpy as np
import arcpy
import matplotlib.pyplot as plt
from arcpy.sa import ExtractMultiValuesToPoints, FocalStatistics, NbrRectangle

from config import Config

def _extract_values(sensor_fc: str, raster_path: str, cfg: Config, focal_window: int = None) -> dict:
    """
    Runs ExtractMultiValuesToPoints against a raster (optionally pre-smoothed with FocalStatistics) and returns
    paired model/sensor values.
    TODO: pass region shapefile to limit extraction extent

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
        rmse = np.sqrt(np.mean((d["model"] - d["sensor"]) ** 2))
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
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")
    else:
        plt.show()

    plt.close(fig)



def model_vs_sensor(date: int, cfg: Config, focal_sampling: bool = False, extent_shp: str = None):
    """
    Creates a scatter plot of model SWE vs sensor SWE.
    TODO: print statements
    TODO: pass region shapefile to clip extent of plot for. Maybe clip difference computation too if its slow
            or have some way of storing the computed difference to then make many plots

    :param date: (YYYYMMDD) Date to compare the two products on
    :param cfg: Configuration object containing environment variables set in the .env
    :param focal_sampling: (bool) If true, compare sensors to a 3x3 focal-mean sample instead of a single cell value
    :param extent_shp: Path to a shapefile defining the extent for comparison. Default = None (full extent)
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
    # TODO

    # Compute difference for a single cell (no focal window)
    arcpy.CheckOutExtension("Spatial")
    single_cell = _extract_values(sensor_shapefile, model_raster, cfg, focal_window=None)

    # Compute difference for a 3x3 focal window
    focal = {"model": np.array([]), "sensor": np.array([])}
    if focal_sampling:
        print("Running 3x3 focal-mean extraction...")
        focal = _extract_values(sensor_shapefile, model_raster, cfg, focal_window=3)

    print(f"{len(single_cell['sensor'])} valid sensor points matched (single cell), "
          f"{len(focal['sensor'])} matched (3x3 average).")

    arcpy.CheckInExtension("Spatial")

    # Create scatter plot
    _plot_scatter(single_cell, focal, date, "./output/model_vs_sensor_20260412_focal.png")



if __name__ == "__main__":
    config = Config()
    model_vs_sensor(20260412, cfg=config, focal_sampling=True)