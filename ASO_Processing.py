# function for getting files out of a zip file
import zipfile
import os
import shutil

def extract_zip(zip_path, ext, output_folder):
    """
    Extracts files with a specific tag from a zip archive and moves them to a new folder.

    Parameters:
        zip_path (str): Path to the .zip file.
        tag (str): Substring to match in filenames (e.g., "maps_aso_bestpred2014").
        destination_folder (str): Directory to move matching files to.
    """
    # Create the destination folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Create a temporary extraction folder
    temp_extract_path = os.path.join(os.path.dirname(zip_path), "temp_extract")
    os.makedirs(temp_extract_path, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract all to temporary location
        zip_ref.extractall(temp_extract_path)

        # Iterate through extracted files
        for root, _, files in os.walk(temp_extract_path):
            for file in files:
                if ext in file:
                    src_file = os.path.join(root, file)
                    dst_file = os.path.join(output_folder, file)

                    # Move file to destination
                    shutil.move(src_file, dst_file)
                    print(f"Moved: {src_file} → {dst_file}")

    # Clean up temp folder
    shutil.rmtree(temp_extract_path)


import os
import arcpy
from arcpy.sa import *

def process_aso_comparison(file, rundate, modelRun, data_folder, modelRunWorkspace, compareWS, snapRaster, projIn,
                           zonalRaster):
    """
    Process ASO and model comparison including projection, masking, difference, percent difference, and zonal stats.

    Parameters:
        file (str): filename of the ASO raster (e.g., 'ASO_Basin_20250401.tif')
        rundate (str): run date string (e.g., '20250401')
        modelRun (str): name of the model run (e.g., 'v1')
        data_folder (str): path to ASO rasters
        modelRunWorkspace (str): path to model raster (P8) workspace
        compareWS (str): base output workspace
        snapRaster (str): path to raster to snap to
        projIn (arcpy.SpatialReference): input projection for ProjectRaster
        zonalRaster (str): raster with zonal units and 'SrtNmeBand' field
    """
    arcpy.env.snapRaster = snapRaster
    arcpy.CheckOutExtension("Spatial")

    basinName = file.split("_")[1]
    output_dir = os.path.join(compareWS, f"{rundate}_{modelRun}")
    os.makedirs(output_dir, exist_ok=True)

    aso_raster = os.path.join(data_folder, file)
    projected_aso = os.path.join(output_dir, f"{file[:-4]}_albn83_50.tif")

    # Project
    arcpy.ProjectRaster_management(
        aso_raster, projected_aso, snapRaster,
        "NEAREST", "50 50", "", "", projIn
    )

    # Create mask where ASO >= 0
    ASOmask = Con(Raster(aso_raster) >= 0, 1, -9999)
    mask_path = os.path.join(output_dir, f"{file[:-4]}_albn83_msk.tif")
    ASOmask.save(mask_path)

    # Resample P8 model to 50m
    p8_input = os.path.join(modelRunWorkspace, f"p8_{rundate}_noneg.tif")
    p8_resampled = os.path.join(output_dir, f"p8_{rundate}_50m.tif")
    arcpy.Resample_management(p8_input, p8_resampled)

    # Apply mask
    masked_p8 = Raster(p8_resampled) * Raster(mask_path)
    masked_p8_path = os.path.join(output_dir, f"p8_{rundate}_50m_msk.tif")
    masked_p8.save(masked_p8_path)

    # Difference
    diff = Raster(masked_p8_path) - Raster(projected_aso)
    diff_path = os.path.join(output_dir, f"DIFF_LRM-ASO_{rundate}_{modelRun}_{basinName}.tif")
    diff.save(diff_path)

    # Percent difference
    perc_diff = ((Raster(masked_p8_path) - Raster(projected_aso)) / Raster(projected_aso)) * 100
    perc_diff_path = os.path.join(output_dir, f"PercDIFF_LRM-ASO_{rundate}_{modelRun}_{basinName}.tif")
    perc_diff.save(perc_diff_path)

    # Zonal stats for % difference
    zonal_table_perc = os.path.join(output_dir, f"PercDIFF_LRM-ASO_{rundate}_{modelRun}_{basinName}_byBands.dbf")
    ZonalStatisticsAsTable(zonalRaster, "SrtNmeBand", perc_diff_path, zonal_table_perc, "", "ALL")
    arcpy.ExportTable_conversion(zonal_table_perc, zonal_table_perc.replace(".dbf", ".csv"))

    # Zonal stats for ASO mask
    zonal_table_aso = os.path.join(output_dir, f"{file[:-4]}_albn83_byBands.dbf")
    ZonalStatisticsAsTable(zonalRaster, "SrtNmeBand", mask_path, zonal_table_aso, "", "ALL")
    arcpy.ExportTable_conversion(zonal_table_aso, zonal_table_aso.replace(".dbf", ".csv"))

    # Zonal stats for masked P8
    zonal_table_p8 = os.path.join(output_dir, f"p8_{rundate}_50m_msk_byBands.dbf")
    ZonalStatisticsAsTable(zonalRaster, "SrtNmeBand", masked_p8_path, zonal_table_p8, "", "ALL")
    arcpy.ExportTable_conversion(zonal_table_p8, zonal_table_p8.replace(".dbf", ".csv"))

    print("All ASO comparison outputs created.")


def fractional_error(filename, input_folder, output_folder, snapRaster, projIn, modelRunWorkspace, rundate, delete=None):
    """
    Process a raster file by projecting, aggregating, and calculating fractional error.

    Parameters:
    file (str): Input raster filename
    data_folder (str): Path to input data folder
    compareWS (str): Path to comparison workspace
    snapRaster: Snap raster for projection
    projIn: Input projection
    modelRunWorkspace (str): Path to model run workspace
    rundate (str): Run date for p8 file naming
    """
    # Set snap raster
    arcpy.env.snapRaster = snapRaster

    # Create file paths
    input_path = input_folder + filename
    proj_output = output_folder + f"{filename[:-4]}_proj_50.tif"
    agg_output = output_folder + f"{filename[:-4]}_proj_500Agg.tif"
    error_output = output_folder + f"{filename[:-4]}_fraErr.tif"

    # Project raster
    arcpy.ProjectRaster_management(
        input_path, proj_output, snapRaster,
        "NEAREST", "50 50", "", "", projIn
    )

    # Aggregate raster
    outAgg = Aggregate(proj_output, 10, "MEAN", "TRUNCATE", "DATA")
    outAgg.save(agg_output)

    # Calculate fractional error
    p8_input = os.path.join(modelRunWorkspace, f"p8_{rundate}_noneg.tif")
    FracError = Raster(p8_input) / (1 + Raster(agg_output))
    FracError.save(error_output)

    if delete is True or delete == "True":
        arcpy.DeleteRaster_management([proj_output, agg_output])
    else:
        print("intermediary files not deleted")

    return error_output  # Return the final output path