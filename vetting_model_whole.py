# import modules
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

# parameters
domains = ["SNM", "PNW", "INMT", "SOCN", "NOCN"]
clipbox_WS = "M:/SWE/WestWide/data/boundaries/Domains/DomainShapefiles/"
rundate = "20250503"
modelRun = "fSCA_RT_CanAdj_rcn_noSW_woCCR"
surveys_use ="Y"
resultsWorkspace = f"W:/Spatial_SWE/WW_regression/RT_report_data/"
raster = f"{resultsWorkspace}/{rundate}_results_ET/{modelRun}/p8_{rundate}_noneg.tif"
snapRaster_albn83 = "M:/SWE/WestWide/data/boundaries/SnapRaster_albn83.tif"
sensors = resultsWorkspace + f"{rundate}_results_ET/{rundate}_sensors_albn83.shp"
surveys = resultsWorkspace + f"{rundate}_results_ET/{rundate}_surveys_albn83.shp"

# set environment settings
arcpy.env.snapRaster = snapRaster_albn83
arcpy.env.cellSize = snapRaster_albn83

## make vetting folder
outVettingWS = f"{resultsWorkspace}/{rundate}_results_ET/{modelRun}/vetting_domains/"
os.makedirs(outVettingWS, exist_ok=True)
print('folder created')

# loop through domains
for domain in domains:
    # extract by mask
    outMask = ExtractByMask(raster, clipbox_WS + f"WW_{domain}_Clipbox_albn83.shp")
    outMask.save(outVettingWS + f"p8_{rundate}_noneg_{domain}_clp.tif")
    print(f"{domain} clipped and saved")

for domain in domains:
    raster = outVettingWS + f"p8_{rundate}_noneg_{domain}_clp.tif"
    if surveys_use == "Y":
        swe_col_surv = 'SWE_m'
        id_col_surv = 'Station_Id'

        model_domain_vetting(raster=raster, point=surveys, swe_col=swe_col_surv, id_col=id_col_surv, rundate=rundate, domain=domain, modelRun=modelRun, out_csv=outVettingWS + f"{rundate}_surveys_error.csv")

    swe_col_sens = 'pillowswe'
    id_col_sens = 'Site_ID'
    model_domain_vetting(raster=raster, point=sensors, swe_col=swe_col_sens, id_col=id_col_sens, rundate=rundate, domain=domain,
                         modelRun=modelRun, out_csv=outVettingWS + f"{rundate}_sensors_error.csv")

## get error by surveys

## get error by sensors
## if there was a previous report
    ## get two overlapping SWE by elevation plots
## if else
    ## just get one of elevation banded swe
## do elevation by SWE distribution by box and whiskers plot

## get difference of snow surveys from one report to another




