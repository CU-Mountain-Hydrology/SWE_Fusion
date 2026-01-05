# import modules
import arcpy
import pandas as pd
import os
import shutil
from arcpy.sa import *
from arcpy import *
from datetime import datetime
# from Vetting_functions import *
print('modules imported')

# parameters
rundate = "20250503"
ModelRun = "fSCA_RT_CanAdj_rcn_noSW_woCCR"
currentYear = True
current_year = datetime.now().year
method_list = ["RECENT", "GRADE", "SENSOR_PATTERN", "GRADES_SPECF"]
grade = "positive"
grade_range = False
grade_amount = -10
sensorTrend = "Mixed"
SNOTEL = "Decreasing"
domains = ['SNM', 'SOCN']
basinList = ["SouthPlatte", "Uinta", "Kings"]
output_csv = "Y"
csv_outFile = r"W:/Spatial_SWE/ASO/2025/data_testing/FracError_data_test.csv"
# asoCatalog = r"W:/Spatial_SWE/ASO/2025/data_testing/ASO_SNOTEL_DifferenceStats.csv"
basin_List = r"W:/Spatial_SWE/ASO/ASO_Metadata/ASO_in_Basin.txt"
fracErrorWorkspace = "W:/Spatial_SWE/ASO/2025/data_testing/"
domainList = r"M:\SWE\WestWide\Spatial_SWE\ASO\ASO_Metadata\State_Basin.txt"
results_workspace = f"M:/SWE/WestWide/Spatial_SWE/WW_regression/RT_report_data/{rundate}_results_ET/"

## function for fix
## function for validation

def bias_correction_selection(rundate, aso_snotel_data, basin_List, domainList, method_list, fracErrorWorkspace,
                              output_csv, csv_outFile=None, currentYear=None, year=None, grade_amount=None,
                              sensorTrend=None, SNOTEL=None, grade=None, grade_range=None):

    """
    :param rundate: the rundate of the report
    :param basin_List:
    :param domainList: 
    :param method_list: ["RECENT", "GRADE", "SENSOR_PATTERN", "GRADES_SPECF"]
    :param fracErrorWorkspace: 
    :param output_csv: Whether you want the output csv file or not
    :param csv_outFile: The file path of the output csv file
    :param currentYear: True/False -- whether you want the current year or not
    :param grade_amount: -10
    :param sensorTrend: "Mixed"
    :param SNOTEL: "Decreasing"
    :param grade: "positive"
    :param grade_range: True/False
    :return: 
    """
        
    # current_year = datetime.now().year
        
    mapping = {}
    with open(basin_List) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            before, after = line.split("|")
            # clean up
            before = before.strip().strip('"')
            after_items = [item.strip().strip('"') for item in after.split(",")]
            # store in dictionary
            mapping[before] = after_items

    state_mapping = {}
    with open(domainList, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            basin_name, state = line.split("|")
            basin_name = basin_name.strip().strip('"')
            state = state.strip().strip('"')
            state_mapping[basin_name] = state

    # set the list
    all_results = []
    for main_group, sub_items in mapping.items():
        for item in sub_items:
            print(f"{item}")

            state = state_mapping.get(item, "Unknown")
            print(f"Basin: {item}, State: {state}")
#
            #establish domain
            if state == "CA":
                domain = "SNM"
            else:
                domain = "WW"

            # read through CSV
            aso_df = pd.read_csv(aso_snotel_data)

            if item in aso_df['Basin'].values:

                if currentYear is True:
                    aso_df = aso_df[aso_df["Year"] == year]
                else:
                    print("Using all years of ASO data")

                # Store paths for this basin
                basin_row = {
                    'ModelDate': rundate,
                    'MainGroup': main_group,
                    'Basin': item,
                    'Domain': domain,
                    'RECENT': None,
                    'GRADE': None,
                    'SENSOR_PATTERN': None,
                    'PATTERN_TYPE': None
                }

                for method in method_list:
                    # getting most recent data
                    if method == "RECENT":
                        print(f"\nMETHOD: {method}")
                        aso_df_basin = aso_df[aso_df["Basin"] == item].copy()
                        target_date  = datetime.strptime(rundate, "%Y%m%d")
                        aso_df_basin['cstm_dte'] = pd.to_datetime(aso_df_basin["Date"], format="%Y%b%d")
                        aso_df_basin["diff_days"] = (aso_df_basin["cstm_dte"] - target_date).abs().dt.days
                        df_filtered = aso_df_basin[aso_df_basin["diff_days"] > 4]

                        # add to the df
                        if not df_filtered.empty:
                            closest_row = df_filtered.loc[df_filtered["diff_days"].idxmin()]
                            # get fractional error path
                            fraErrorPath = (
                                f"{fracErrorWorkspace}/{closest_row['Domain']}_comparison_testing/{closest_row['RunDate']}_{closest_row['modelRun']}/"
                                f"ASO_{closest_row['Basin']}_{closest_row['Date']}_swe_50m_fraErr")
                            print(fraErrorPath)
                            basin_row['RECENT'] = fraErrorPath

                    if method == "GRADE" or method == "GRADES_SPECF":
                        aso_df_basin = aso_df[aso_df["Basin"] == item].copy()
                        aso_df_grade = aso_df_basin[aso_df_basin["GradeDirection"] == grade].copy()
                        if len(aso_df_grade.columns) > 1:
                            if method == "GRADE":
                                print(f"\nMETHOD: {method}")
                                target_date = datetime.strptime(rundate, "%Y%m%d")
                                aso_df_grade['cstm_dte'] = pd.to_datetime(aso_df_grade["Date"], format="%Y%b%d")
                                aso_df_grade["diff_days"] = (aso_df_grade["cstm_dte"] - target_date).abs().dt.days
                                df_filtered = aso_df_grade[aso_df_grade["diff_days"] > 4].copy()

                                if not df_filtered.empty:
                                    closest_row = df_filtered.loc[df_filtered["diff_days"].idxmin()]
                                    fraErrorPath = (
                                        f"{fracErrorWorkspace}/{closest_row['Domain']}_comparison_testing/{closest_row['RunDate']}_{closest_row['modelRun']}/"
                                        f"ASO_{closest_row['Basin']}_{closest_row['Date']}_swe_50m_fraErr")
                                    basin_row['GRADE'] = fraErrorPath

                            # if method == "GRADES_SPECF":
                            #     print(f"\nMETHOD: {method}")
                            #     closest_row = aso_df_grade.loc[(aso_df_grade["GradeDifference"] - grade_amount).abs().idxmin()]
                            #     fraErrorPath = (
                            #         f"{fracErrorWorkspace}/{closest_row['Domain']}_comparison_testing/{closest_row['RunDate']}_{closest_row['modelRun']}/"
                            #         f"ASO_{closest_row['Basin']}_{closest_row['Date']}_swe_50m_fraErr")
                            #     basin_row['GRADES_SPECF'] = fraErrorPath
                            if method == "GRADES_SPECF":
                                print(f"\nMETHOD: {method}")

                                # Check if aso_df_grade has any rows before attempting to find closest match
                                if not aso_df_grade.empty:
                                    closest_row = aso_df_grade.loc[
                                        (aso_df_grade["GradeDifference"] - grade_amount).abs().idxmin()]
                                    fraErrorPath = (
                                        f"{fracErrorWorkspace}/{closest_row['Domain']}_comparison_testing/{closest_row['RunDate']}_{closest_row['modelRun']}/"
                                        f"ASO_{closest_row['Basin']}_{closest_row['Date']}_swe_50m_fraErr")
                                    basin_row['GRADES_SPECF'] = fraErrorPath
                                else:
                                    print(f"No data found for grade direction '{grade}' in basin {item}")
                                    basin_row['GRADES_SPECF'] = None

                    if method == "SENSOR_PATTERN":
                        print(f"\nMETHOD: {method}")
                        aso_df_basin = aso_df[aso_df["Basin"] == item].copy()
                        aso_df_pattern = aso_df_basin[aso_df_basin["OverallTrend"] == sensorTrend].copy()
                        row_count = len(aso_df_pattern)

                        if row_count > 1:
                            print("More than one sensor pattern found. Selecting the most recent one.")
                            target_date = datetime.strptime(rundate, "%Y%m%d")
                            aso_df_pattern['cstm_dte'] = pd.to_datetime(aso_df_pattern["Date"], format="%Y%b%d")
                            aso_df_pattern["diff_days"] = (aso_df_pattern["cstm_dte"] - target_date).abs().dt.days
                            aso_df_pattern = aso_df_pattern[aso_df_pattern["diff_days"] > 4].copy()

                            if not aso_df_pattern.empty:
                                closest_row = aso_df_pattern.loc[aso_df_pattern["diff_days"].idxmin()]
                                fraErrorPath = (
                                    f"{fracErrorWorkspace}/{closest_row['Domain']}_comparison_testing/{closest_row['RunDate']}_{closest_row['modelRun']}/"
                                    f"ASO_{closest_row['Basin']}_{closest_row['Date']}_swe_50m_fraErr")
                                basin_row['SENSOR_PATTERN'] = fraErrorPath


                        elif row_count == 1:
                            closest_row = aso_df_pattern.iloc[0]
                            fraErrorPath = (
                                f"{fracErrorWorkspace}/{closest_row['Domain']}_comparison_testing/{closest_row['RunDate']}_{closest_row['modelRun']}/"
                                f"ASO_{closest_row['Basin']}_{closest_row['Date']}_swe_50m_fraErr")
                            basin_row['SENSOR_PATTERN'] = fraErrorPath
                            basin_row['PATTERN_TYPE'] = sensorTrend

                        else:
                            print(f"No sensor pattern matches found for trend: {sensorTrend}")
                            basin_row['SENSOR_PATTERN'] = "NA"

                # add everything to the df
                all_results.append(basin_row)
            else:
                continue

    results_df = pd.DataFrame(all_results)

    if output_csv == "Y":
        results_df.to_csv(csv_outFile, index=False)
    return results_df

import os
import shutil
import arcpy
import math
import glob
from arcpy.sa import *

def bias_correct(results_workspace, domain, ModelRun, method, rundate, results_df, shapefile_workspace):
    """
    Performs bias correction for ASO SWE data.

    Parameters:
        results_workspace (str): Path to the results workspace.
        ModelRun (str): Name of the model run.
        method (str): Method name used for fractional error.
        rundate (str): Date string for the run (e.g., '20250503').
        results_df (DataFrame): DataFrame containing 'MainGroup', 'Basin', and method column.
        shapefile_workspace (str): Path to shapefiles.
    """
    # first filter by domain
    # Filter by domain if specified
    if domain is not None:
        results_df = results_df[results_df['Domain'] == domain].copy()
        print(f"Filtered to domain: {domain}")

    # Make directories
    os.makedirs(f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/", exist_ok=True)
    os.makedirs(f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/Fix_Files/", exist_ok=True)

    # Copy p8 layer into output directory
    p8Layer = f"{results_workspace}/{ModelRun}/p8_{rundate}_noneg.tif"
    shutil.copy(p8Layer, f"{results_workspace}ASO_BiasCorrect_{ModelRun}/")
    p8_forBC = f"{results_workspace}ASO_BiasCorrect_{ModelRun}/p8_{rundate}_noneg.tif"

    # Get unique main groups
    mainBasins = results_df['MainGroup'].unique().tolist()
    print("Main basins:", mainBasins)

    # Loop through main basins
    for basin in mainBasins:
        print(f"\nProcessing basin: {basin}")
        df_group = results_df[results_df['MainGroup'] == basin]

        for idx, row in df_group.iterrows():
            sub_basin = row["Basin"]
            fraErr = row[method]
            if (fraErr is None or
                    (isinstance(fraErr, float) and math.isnan(fraErr)) or
                    fraErr == "NA" or
                    fraErr == "" or
                    str(fraErr).lower() == "none"):
                print(f"✗ Skipping sub-basin {sub_basin}: {method} has no data (value: '{fraErr}')")
                continue

                # Convert to string and validate it's a real path
            fraErr = str(fraErr).strip()
            if len(fraErr) < 20:  # Valid paths should be longer than 20 characters
                print(f"✗ Skipping sub-basin {sub_basin}: {method} path too short: '{fraErr}'")
                continue

            # Check if directory exists before trying to search it
            fraErr_dir = os.path.dirname(fraErr)
            if not fraErr_dir or not os.path.exists(fraErr_dir):
                print(f"✗ Skipping sub-basin {sub_basin}: directory doesn't exist: '{fraErr_dir}'")
                continue

            # fraErr_dir = os.path.dirname(str(fraErr))
            fraErr_basename = os.path.basename(str(fraErr))

            # Pattern: starts with basename, ends with swe_50m_fraErr.tif
            # This will match: ASO_BoulderCreek_2025Apr09*swe_50m_fraErr.tif
            search_pattern = os.path.join(fraErr_dir, f"{fraErr_basename[:-19]}*swe_50m_fraErr.tif")

            print(f"Basin: {basin} | Sub-Basin: {sub_basin} | Search pattern: {search_pattern}")

            if domain == "SNM":
                basinSHP = f"{shapefile_workspace}/{sub_basin}_albn83.shp"
            if domain == "WW":
                basinSHP = f"{shapefile_workspace}ASO_{sub_basin}_albn83.shp"

            matching_files = glob.glob(search_pattern)

            if matching_files:
                for fraErr_path in matching_files:
                    print(f"Found file: {fraErr_path}")

                    if os.path.exists(fraErr_path):
                        print("File exists!")
                        # Your processing code here
            else:
                print(f"No files found matching pattern: {search_pattern}")

            if os.path.exists(fraErr_path):

                print(f"Basin: {basin} | Sub-Basin: {sub_basin} | fracErr_path: {fraErr_path}")

                # Create p8 mask if it doesn't exist
                mask_path = f"{results_workspace}ASO_BiasCorrect_{ModelRun}/p8_{rundate}_noneg_msk.tif"
                if not os.path.isfile(mask_path):
                    p8masking = Con(Raster(p8_forBC) >= 0, 1, Raster(p8_forBC))
                    p8masking.save(mask_path)
                    print("p8 mask created")
                else:
                    print("p8 mask exists, moving on")

                # Mask just the basin boundary
                arcpy.env.mask = mask_path
                arcpy.env.cellSize = p8_forBC
                arcpy.env.snapRaster = p8_forBC
                basinBound = ExtractByMask(mask_path, basinSHP)
                basinBound.save(f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{sub_basin}_msk.tif")

                # Change all -1 to -0.999
                newFracError = Con(Raster(fraErr_path) == -1, -0.999, Raster(fraErr_path))
                newFracError.save(f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{sub_basin}_fracError_v2.tif")

                # Make all no data values 0
                noNull = Con(IsNull(Raster(f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{sub_basin}_fracError_v2.tif")), 0, Raster(f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{sub_basin}_fracError_v2.tif"))
                noNull.save(f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{sub_basin}_fracError_v3.tif")
                print("non-zero fractional layer created")

                # Snap and clip to the extent of basin
                print("Extracting and setting boundaries")
                extract_fracError = ExtractByMask(f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{sub_basin}_fracError_v3.tif",
                                                 f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{sub_basin}_msk.tif")
                extract_fracError.save(f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{sub_basin}_fracError_v4.tif")

                # Compute LRM fix
                LRMfix = Raster(p8_forBC) / (1 + Raster(f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{sub_basin}_fracError_v4.tif"))
                newFrac = Con(IsNull(Raster(fraErr_path)), 0, Raster(fraErr_path))
                newFix = Con(newFrac == -1, Raster(p8_forBC), LRMfix)
                newFix.save(f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/Fix_Files/{rundate}_{basin}_{sub_basin}_{method}_LRMFix_final.tif")
                print("Fix completed")
            else:
                print("File NOT found.")

    # Mosaic all files in the same main group
    for basin in mainBasins:
        basinFix = []
        fixFiles = os.listdir(f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/Fix_Files/")
        for file in fixFiles:
            if file.endswith(".tif") and file.startswith(f"{rundate}_{basin}_"):
                basinFix.append(f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/Fix_Files/{file}")
            elif not file.endswith(".tif"):
                arcpy.Delete_management(file)
                print(f"Deleted non-TIF file: {file}")

        if basinFix:
            arcpy.env.snapRaster = p8_forBC
            arcpy.env.cellSize = p8_forBC
            out_raster = f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{rundate}_{basin}_{method}_BC_fix_albn83.tif"
            arcpy.MosaicToNewRaster_management(basinFix,
                                               os.path.dirname(out_raster),
                                               os.path.basename(out_raster),
                                               "",
                                               "32_BIT_FLOAT", "", 1, "LAST", "FIRST")
            print(f"Mosaicked raster saved: {out_raster}")










