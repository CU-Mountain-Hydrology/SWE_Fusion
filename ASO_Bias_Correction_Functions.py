# import modules
import arcpy
import pandas as pd
import os
import shutil
from arcpy.sa import *
from arcpy import *
from datetime import datetime
import numpy as np
# from Vetting_functions import *
print('modules imported')

# parameters
# rundate = "20250503"
# ModelRun = "fSCA_RT_CanAdj_rcn_noSW_woCCR"
# currentYear = True
# current_year = datetime.now().year
# method_list = ["RECENT", "GRADE", "SENSOR_PATTERN", "GRADES_SPECF"]
# grade = "positive"
# grade_range = False
# grade_amount = -10
# sensorTrend = "Mixed"
# SNOTEL = "Decreasing"
# domains = ['SNM', 'SOCN']
# basinList = ["SouthPlatte", "Uinta", "Kings"]
# output_csv = "Y"
# csv_outFile = r"W:/Spatial_SWE/ASO/2025/data_testing/FracError_data_test.csv"
# # asoCatalog = r"W:/Spatial_SWE/ASO/2025/data_testing/ASO_SNOTEL_DifferenceStats.csv"
# # basin_List = r"W:/Spatial_SWE/ASO/ASO_Metadata/ASO_in_Basin.txt"
# domainList = r"M:\SWE\WestWide\Spatial_SWE\ASO\ASO_Metadata\State_Basin.txt"
# results_workspace = f"M:/SWE/WestWide/Spatial_SWE/WW_regression/RT_report_data/{rundate}_results/"

## function for fix
## function for validation

def seasonal_diff(date_series, target_date):
    """Day-of-year difference ignoring year, with wrap-around handling."""
    target_doy = target_date.timetuple().tm_yday
    doy = date_series.dt.dayofyear
    diff = (doy - target_doy).abs()
    return diff.apply(lambda x: min(x, 365 - x))

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

            # establish domain
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
                    aso_df = aso_df

                # Store paths for this basin
                basin_row = {
                    'ModelDate': rundate,
                    'MainGroup': main_group,
                    'Basin': item,
                    'Domain': domain,
                    'RECENT': None,
                    'GRADE': None,
                    'SENSOR_PATTERN': None,
                    'PATTERN_TYPE': None,
                    'MAX_INCREASE': None
                }

                # Track paths to detect duplicates
                method_paths = {}

                # for method in method_list:
                #     # getting most recent data
                #     if method == "RECENT":
                #         # print(f"\nMETHOD: {method}")
                #         aso_df_basin = aso_df[aso_df["Basin"] == item].copy()
                #         target_date = datetime.strptime(rundate, "%Y%m%d")
                #         aso_df_basin['cstm_dte'] = pd.to_datetime(aso_df_basin["Date"], format="%Y%b%d")
                #         aso_df_basin["diff_days"] = (aso_df_basin["cstm_dte"] - target_date).abs().dt.days
                #         df_filtered = aso_df_basin[aso_df_basin["diff_days"] > 4]
                #         print(
                #             aso_df_basin[["Basin", "Date", "diff_days"]]
                #         )
                #
                #         # add to the df
                #         if not df_filtered.empty:
                #             closest_row = df_filtered.loc[df_filtered["diff_days"].idxmin()]
                #             # get fractional error path
                #             fraErrorPath = (
                #                 f"{fracErrorWorkspace.rstrip('/')}/{closest_row['Domain']}_comparison/{closest_row['RunDate']}_{closest_row['modelRun']}/"
                #                 f"ASO_{closest_row['Basin']}_{closest_row['Date']}_swe_50m_fraErr")
                #             method_paths['RECENT'] = fraErrorPath
                #
                #     if method == "GRADE" or method == "GRADES_SPECF":
                #         aso_df_basin = aso_df[aso_df["Basin"] == item].copy()
                #         aso_df_grade = aso_df_basin[aso_df_basin["GradeDirection"] == grade].copy()
                #         if len(aso_df_grade.columns) > 1:
                #             if method == "GRADE":
                #                 target_date = datetime.strptime(rundate, "%Y%m%d")
                #                 aso_df_grade['cstm_dte'] = pd.to_datetime(aso_df_grade["Date"], format="%Y%b%d")
                #                 aso_df_grade["diff_days"] = (aso_df_grade["cstm_dte"] - target_date).abs().dt.days
                #                 df_filtered = aso_df_grade[aso_df_grade["diff_days"] > 4].copy()
                #
                #                 if not df_filtered.empty:
                #                     closest_row = df_filtered.loc[df_filtered["diff_days"].idxmin()]
                #                     fraErrorPath = (
                #                         f"{fracErrorWorkspace.rstrip('/')}/{closest_row['Domain']}_comparison/{closest_row['RunDate']}_{closest_row['modelRun']}/"
                #                         f"ASO_{closest_row['Basin']}_{closest_row['Date']}_swe_50m_fraErr")
                #                     method_paths['GRADE'] = fraErrorPath
                #
                #             if method == "GRADES_SPECF":
                #
                #                 # Check if aso_df_grade has any rows before attempting to find closest match
                #                 if not aso_df_grade.empty:
                #                     closest_row = aso_df_grade.loc[
                #                         (aso_df_grade["GradeDifference"] - grade_amount).abs().idxmin()]
                #                     fraErrorPath = (
                #                         f"{fracErrorWorkspace.rstrip('/')}/{closest_row['Domain']}_comparison/{closest_row['RunDate']}_{closest_row['modelRun']}/"
                #                         f"ASO_{closest_row['Basin']}_{closest_row['Date']}_swe_50m_fraErr")
                #                     method_paths['GRADES_SPECF'] = fraErrorPath
                #                 else:
                #                     method_paths['GRADES_SPECF'] = None
                #
                #     if method == "SENSOR_PATTERN":
                #         # print(f"\nMETHOD: {method}")
                #         aso_df_basin = aso_df[aso_df["Basin"] == item].copy()
                #         aso_df_pattern = aso_df_basin[aso_df_basin["OverallTrend"] == sensorTrend].copy()
                #         row_count = len(aso_df_pattern)
                #
                #         if row_count > 1:
                #             # print("More than one sensor pattern found. Selecting the most recent one.")
                #             target_date = datetime.strptime(rundate, "%Y%m%d")
                #             aso_df_pattern['cstm_dte'] = pd.to_datetime(aso_df_pattern["Date"], format="%Y%b%d")
                #             aso_df_pattern["diff_days"] = (aso_df_pattern["cstm_dte"] - target_date).abs().dt.days
                #             aso_df_pattern = aso_df_pattern[aso_df_pattern["diff_days"] > 4].copy()
                #
                #             if not aso_df_pattern.empty:
                #                 closest_row = aso_df_pattern.loc[aso_df_pattern["diff_days"].idxmin()]
                #                 fraErrorPath = (
                #                     f"{fracErrorWorkspace.rstrip('/')}/{closest_row['Domain']}_comparison/{closest_row['RunDate']}_{closest_row['modelRun']}/"
                #                     f"ASO_{closest_row['Basin']}_{closest_row['Date']}_swe_50m_fraErr")
                #                 method_paths['SENSOR_PATTERN'] = fraErrorPath
                #
                #
                #         elif row_count == 1:
                #             closest_row = aso_df_pattern.iloc[0]
                #             fraErrorPath = (
                #                 f"{fracErrorWorkspace.rstrip('/')}/{closest_row['Domain']}_comparison/{closest_row['RunDate']}_{closest_row['modelRun']}/"
                #                 f"ASO_{closest_row['Basin']}_{closest_row['Date']}_swe_50m_fraErr")
                #             method_paths['SENSOR_PATTERN'] = fraErrorPath
                #             basin_row['PATTERN_TYPE'] = sensorTrend
                #
                #         else:
                #             # print(f"No sensor pattern matches found for trend: {sensorTrend}")
                #             method_paths['SENSOR_PATTERN'] = "NA"
                #
                #     if method == "MAX_INCREASE":
                #         aso_df_basin = aso_df[aso_df["Basin"] == item].copy()
                #
                #         # Filter out inf and NaN values in GradeDifference
                #         aso_df_basin = aso_df_basin[
                #             np.isfinite(aso_df_basin["GradeDifference"])
                #         ].copy()
                #
                #         # Also exclude flights too close to the rundate
                #         target_date = datetime.strptime(rundate, "%Y%m%d")
                #         aso_df_basin['cstm_dte'] = pd.to_datetime(aso_df_basin["Date"], format="%Y%b%d")
                #         aso_df_basin["diff_days"] = (aso_df_basin["cstm_dte"] - target_date).abs().dt.days
                #         aso_df_basin = aso_df_basin[aso_df_basin["diff_days"] > 4].copy()
                #
                #         if not aso_df_basin.empty:
                #             # Select flight where SNOTEL was increasing most = ASO likely highest vs model
                #             closest_row = aso_df_basin.loc[aso_df_basin["GradeDifference"].idxmax()]
                #             fraErrorPath = (
                #                 f"{fracErrorWorkspace.rstrip('/')}/{closest_row['Domain']}_comparison/{closest_row['RunDate']}_{closest_row['modelRun']}/"
                #                 f"ASO_{closest_row['Basin']}_{closest_row['Date']}_swe_50m_fraErr")
                #             method_paths['MAX_INCREASE'] = fraErrorPath
                #             print(f"  MAX_INCREASE selected: {closest_row['Basin']} {closest_row['Date']} "
                #                   f"(GradeDiff: {closest_row['GradeDifference']:.1f}%)")
                #         else:
                #             method_paths['MAX_INCREASE'] = None

                for method in method_list:

                    if method == "RECENT":
                        # Current year only — most recent flight before rundate
                        aso_df_basin = aso_df[aso_df["Basin"] == item].copy()
                        target_date = datetime.strptime(rundate, "%Y%m%d")
                        current_year = target_date.year

                        aso_df_basin['cstm_dte'] = pd.to_datetime(aso_df_basin["Date"], format="%Y%b%d")
                        aso_df_basin = aso_df_basin[aso_df_basin["Year"] == current_year].copy()
                        aso_df_basin["diff_days"] = (aso_df_basin["cstm_dte"] - target_date).abs().dt.days
                        df_filtered = aso_df_basin[aso_df_basin["diff_days"] > 4]

                        print(aso_df_basin[["Basin", "Date", "diff_days"]])

                        if not df_filtered.empty:
                            closest_row = df_filtered.loc[df_filtered["diff_days"].idxmin()]
                            fraErrorPath = (
                                f"{fracErrorWorkspace.rstrip('/')}/{closest_row['Domain']}_comparison/{closest_row['RunDate']}_{closest_row['modelRun']}/"
                                f"ASO_{closest_row['Basin']}_{closest_row['Date']}_swe_50m_fraErr")
                            method_paths['RECENT'] = fraErrorPath
                        else:
                            print(f"  No current year flights found for RECENT - {item}")
                            method_paths['RECENT'] = None

                    if method == "GRADE" or method == "GRADES_SPECF":
                        # Full catalogue — seasonal day-of-year matching
                        aso_df_basin = aso_df[aso_df["Basin"] == item].copy()
                        aso_df_grade = aso_df_basin[aso_df_basin["GradeDirection"] == grade].copy()

                        if not aso_df_grade.empty:
                            target_date = datetime.strptime(rundate, "%Y%m%d")
                            aso_df_grade['cstm_dte'] = pd.to_datetime(aso_df_grade["Date"], format="%Y%b%d")

                            if method == "GRADE":
                                # Seasonally closest match across all years
                                aso_df_grade["diff_days"] = seasonal_diff(aso_df_grade["cstm_dte"], target_date)
                                df_filtered = aso_df_grade[aso_df_grade["diff_days"] > 4].copy()

                                if not df_filtered.empty:
                                    closest_row = df_filtered.loc[df_filtered["diff_days"].idxmin()]
                                    fraErrorPath = (
                                        f"{fracErrorWorkspace.rstrip('/')}/{closest_row['Domain']}_comparison/{closest_row['RunDate']}_{closest_row['modelRun']}/"
                                        f"ASO_{closest_row['Basin']}_{closest_row['Date']}_swe_50m_fraErr")
                                    method_paths['GRADE'] = fraErrorPath
                                else:
                                    method_paths['GRADE'] = None

                            if method == "GRADES_SPECF":
                                # Closest grade amount match across all years — filter inf first
                                aso_df_grade = aso_df_grade[np.isfinite(aso_df_grade["GradeDifference"])].copy()
                                if not aso_df_grade.empty:
                                    closest_row = aso_df_grade.loc[
                                        (aso_df_grade["GradeDifference"] - grade_amount).abs().idxmin()]
                                    fraErrorPath = (
                                        f"{fracErrorWorkspace.rstrip('/')}/{closest_row['Domain']}_comparison/{closest_row['RunDate']}_{closest_row['modelRun']}/"
                                        f"ASO_{closest_row['Basin']}_{closest_row['Date']}_swe_50m_fraErr")
                                    method_paths['GRADES_SPECF'] = fraErrorPath
                                else:
                                    method_paths['GRADES_SPECF'] = None
                        else:
                            method_paths[method] = None

                    if method == "SENSOR_PATTERN":
                        # Full catalogue — seasonal day-of-year matching
                        aso_df_basin = aso_df[aso_df["Basin"] == item].copy()
                        aso_df_pattern = aso_df_basin[aso_df_basin["OverallTrend"] == sensorTrend].copy()
                        target_date = datetime.strptime(rundate, "%Y%m%d")

                        if not aso_df_pattern.empty:
                            aso_df_pattern['cstm_dte'] = pd.to_datetime(aso_df_pattern["Date"], format="%Y%b%d")
                            aso_df_pattern["diff_days"] = seasonal_diff(aso_df_pattern["cstm_dte"], target_date)
                            aso_df_pattern = aso_df_pattern[aso_df_pattern["diff_days"] > 4].copy()

                            if not aso_df_pattern.empty:
                                closest_row = aso_df_pattern.loc[aso_df_pattern["diff_days"].idxmin()]
                                fraErrorPath = (
                                    f"{fracErrorWorkspace.rstrip('/')}/{closest_row['Domain']}_comparison/{closest_row['RunDate']}_{closest_row['modelRun']}/"
                                    f"ASO_{closest_row['Basin']}_{closest_row['Date']}_swe_50m_fraErr")
                                method_paths['SENSOR_PATTERN'] = fraErrorPath
                                basin_row['PATTERN_TYPE'] = sensorTrend
                            else:
                                method_paths['SENSOR_PATTERN'] = "NA"
                        else:
                            method_paths['SENSOR_PATTERN'] = "NA"

                    if method == "MAX_INCREASE":
                        # Full catalogue — pick flight with highest GradeDifference
                        aso_df_basin = aso_df[aso_df["Basin"] == item].copy()
                        aso_df_basin = aso_df_basin[np.isfinite(aso_df_basin["GradeDifference"])].copy()

                        target_date = datetime.strptime(rundate, "%Y%m%d")
                        aso_df_basin['cstm_dte'] = pd.to_datetime(aso_df_basin["Date"], format="%Y%b%d")
                        aso_df_basin["diff_days"] = (aso_df_basin["cstm_dte"] - target_date).abs().dt.days
                        aso_df_basin = aso_df_basin[aso_df_basin["diff_days"] > 4].copy()

                        if not aso_df_basin.empty:
                            closest_row = aso_df_basin.loc[aso_df_basin["GradeDifference"].idxmax()]
                            fraErrorPath = (
                                f"{fracErrorWorkspace.rstrip('/')}/{closest_row['Domain']}_comparison/{closest_row['RunDate']}_{closest_row['modelRun']}/"
                                f"ASO_{closest_row['Basin']}_{closest_row['Date']}_swe_50m_fraErr")
                            method_paths['MAX_INCREASE'] = fraErrorPath
                            print(f"  MAX_INCREASE selected: {closest_row['Basin']} {closest_row['Date']} "
                                  f"(GradeDiff: {closest_row['GradeDifference']:.1f}%)")
                        else:
                            method_paths['MAX_INCREASE'] = None

                # Detect duplicates and update basin_row
                # Process methods in order to determine which one gets the actual path
                path_to_first_method = {}  # Maps path -> first method that had it

                for method in method_list:
                    if method in method_paths:
                        path = method_paths[method]

                        if path is None or path == "NA":
                            basin_row[method] = path
                        elif path in path_to_first_method:
                            # This path was already used by another method
                            first_method = path_to_first_method[path]
                            basin_row[method] = f"SAME_FILE_AS_{first_method}"
                            print(f"  -> {method} uses same file as {first_method}")
                        else:
                            # First time seeing this path
                            basin_row[method] = path
                            path_to_first_method[path] = method

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

# def bias_correct(results_workspace, domain, ModelRun, method, rundate, results_df, shapefile_workspace):
#     """
#     Performs bias correction for ASO SWE data.
#
#     Parameters:
#         results_workspace (str): Path to the results workspace.
#         ModelRun (str): Name of the model run.
#         method (str): Method name used for fractional error.
#         rundate (str): Date string for the run (e.g., '20250503').
#         results_df (DataFrame): DataFrame containing 'MainGroup', 'Basin', and method column.
#         shapefile_workspace (str): Path to shapefiles.
#     """
#     # first filter by domain
#     # Filter by domain if specified
#     if domain is not None:
#         results_df = results_df[results_df['Domain'] == domain].copy()
#         print(f"Filtered to domain: {domain}")
#
#     # Make directories
#     os.makedirs(f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/", exist_ok=True)
#     os.makedirs(f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/Fix_Files/", exist_ok=True)
#
#     # Copy p8 layer into output directory
#     p8Layer = f"{results_workspace}/{ModelRun}/p8_{rundate}_noneg.tif"
#     shutil.copy(p8Layer, f"{results_workspace}ASO_BiasCorrect_{ModelRun}/")
#     p8_forBC = f"{results_workspace}ASO_BiasCorrect_{ModelRun}/p8_{rundate}_noneg.tif"
#
#     # Get unique main groups
#     mainBasins = results_df['MainGroup'].unique().tolist()
#     print("Main basins:", mainBasins)
#
#     # Loop through main basins
#     for basin in mainBasins:
#         print(f"\n{'=' * 60}")
#         print(f"Processing main group: {basin}")
#         print(f"{'=' * 60}")
#         df_group = results_df[results_df['MainGroup'] == basin]
#
#         # PRE-CHECK: Validate that ALL basins in this main group have valid files for this method
#         all_basins_valid = True
#         validation_results = []
#
#         for idx, row in df_group.iterrows():
#             sub_basin = row["Basin"]
#             fraErr = row[method]
#
#             # Check if this is a reference to another method's file
#             if isinstance(fraErr, str) and fraErr.startswith("SAME_FILE_AS_"):
#                 validation_results.append(f"  ✗ {sub_basin}: references another method ({fraErr})")
#                 all_basins_valid = False
#                 continue
#
#             # Check for missing/invalid data
#             if (fraErr is None or
#                     (isinstance(fraErr, float) and math.isnan(fraErr)) or
#                     fraErr == "NA" or
#                     fraErr == "" or
#                     str(fraErr).lower() == "none"):
#                 validation_results.append(f"  ✗ {sub_basin}: no data (value: '{fraErr}')")
#                 all_basins_valid = False
#                 continue
#
#             # Convert to string and validate it's a real path
#             fraErr = str(fraErr).strip()
#             if len(fraErr) < 20:
#                 validation_results.append(f"  ✗ {sub_basin}: path too short: '{fraErr}'")
#                 all_basins_valid = False
#                 continue
#
#             # Check if directory exists
#             fraErr_dir = os.path.dirname(fraErr)
#             if not fraErr_dir or not os.path.exists(fraErr_dir):
#                 validation_results.append(f"  ✗ {sub_basin}: directory doesn't exist: '{fraErr_dir}'")
#                 all_basins_valid = False
#                 continue
#
#             validation_results.append(f"  ✓ {sub_basin}: valid path")
#
#         # Print validation results
#         print(f"\nValidation results for method '{method}' in main group '{basin}':")
#         for result in validation_results:
#             print(result)
#
#         # If not all basins are valid, skip this entire main group for this method
#         if not all_basins_valid:
#             print(f"\n  SKIPPING entire main group '{basin}' for method '{method}'")
#             print(f"    Reason: Not all basins have valid fractional error files")
#             print(f"    {sum(1 for r in validation_results if '✓' in r)}/{len(validation_results)} basins valid")
#             continue
#
#         print(f"\n✓ All basins valid for main group '{basin}' - proceeding with processing\n")
#
#         # Now process all basins in this main group
#         for idx, row in df_group.iterrows():
#             sub_basin = row["Basin"]
#             fraErr = row[method]
#
#             # Convert to string (we already validated this above)
#             fraErr = str(fraErr).strip()
#             fraErr_dir = os.path.dirname(fraErr)
#             fraErr_basename = os.path.basename(str(fraErr))
#
#             # Pattern: starts with basename, ends with swe_50m_fraErr.tif
#             # This will match: ASO_BoulderCreek_2025Apr09*swe_50m_fraErr.tif
#             search_pattern = os.path.join(fraErr_dir, f"{fraErr_basename[:-15]}*swe_50m_fraErr.tif")
#
#             print(f"Basin: {basin} | Sub-Basin: {sub_basin} | Search pattern: {search_pattern}")
#
#             if domain == "SNM":
#                 basinSHP = f"{shapefile_workspace}/{sub_basin}_albn83.shp"
#             elif domain == "WW":
#                 basinSHP = f"{shapefile_workspace}ASO_{sub_basin}_albn83.shp"
#
#                 # Handle special basin merging cases
#                 if sub_basin in ["Truckee", "Tahoe"]:
#                     # Merge Truckee and Tahoe - use same merged shapefile for both
#                     merge_output = f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/TruckeeTahoeMerge_albn83.shp"
#                     if not os.path.exists(merge_output):
#                         arcpy.Merge_management(
#                             [f"{shapefile_workspace}/Truckee_albn83.shp",
#                              f"{shapefile_workspace}/Tahoe_albn83.shp"],
#                             merge_output
#                         )
#                         print(f"Created merged shapefile: {merge_output}")
#                     basinSHP = merge_output
#
#                 elif sub_basin in ["ECarson", "WCarson"]:
#                     # Merge East and West Carson - use same merged shapefile for both
#                     merge_output = f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/CarsonMerge_albn83.shp"
#                     if not os.path.exists(merge_output):
#                         arcpy.Merge_management(
#                             [f"{shapefile_workspace}/WCarson_albn83.shp",
#                              f"{shapefile_workspace}/ECarson_albn83.shp"],
#                             merge_output
#                         )
#                         print(f"Created merged shapefile: {merge_output}")
#                     basinSHP = merge_output
#
#             matching_files = glob.glob(search_pattern)
#
#             if matching_files:
#                 for fraErr_path in matching_files:
#                     print(f"Found file: {fraErr_path}")
#
#                     if os.path.exists(fraErr_path):
#                         print("File exists!")
#                         # Your processing code here
#             else:
#                 print(f"No files found matching pattern: {search_pattern}")
#
#             if os.path.exists(fraErr_path):
#
#                 print(f"Basin: {basin} | Sub-Basin: {sub_basin} | fracErr_path: {fraErr_path}")
#
#                 # Create p8 mask if it doesn't exist
#                 mask_path = f"{results_workspace}ASO_BiasCorrect_{ModelRun}/p8_{rundate}_noneg_ASO_msk.tif"
#                 if not os.path.isfile(mask_path):
#                     p8masking = Con(Raster(p8_forBC) >= 0, 1, Raster(p8_forBC))
#                     p8masking.save(mask_path)
#                     # print("p8 mask created")
#                 else:
#                     print("P8 mask exists, moving on")
#
#                 # Mask just the basin boundary
#                 arcpy.env.mask = mask_path
#                 arcpy.env.cellSize = p8_forBC
#                 arcpy.env.snapRaster = p8_forBC
#                 if os.path.exists(f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{sub_basin}_msk.tif"):
#                     print('Basin mask already exists')
#                 else:
#                     basinBound = ExtractByMask(mask_path, basinSHP)
#                     basinBound.save(f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{sub_basin}_msk.tif")
#
#                 # Change all -1 to -0.999
#                 newFracError = Con(Raster(fraErr_path) == -1, -0.999, Raster(fraErr_path))
#                 newFracError.save(
#                     f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{sub_basin}_fracError_v2.tif")
#
#                 # Make all no data values 0
#                 noNull = Con(IsNull(
#                     Raster(f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{sub_basin}_fracError_v2.tif")), 0,
#                              Raster(
#                                  f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{sub_basin}_fracError_v2.tif"))
#                 noNull.save(f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{sub_basin}_fracError_v3.tif")
#                 # print("non-zero fractional layer created")
#
#                 # Snap and clip to the extent of basin
#                 # print("Extracting and setting boundaries")
#                 extract_fracError = ExtractByMask(
#                     f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{sub_basin}_fracError_v3.tif",
#                     f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{sub_basin}_msk.tif")
#                 extract_fracError.save(
#                     f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{sub_basin}_fracError_v4.tif")
#
#                 # Compute LRM fix
#                 LRMfix = Raster(p8_forBC) / (1 + Raster(
#                     f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{sub_basin}_fracError_v4.tif"))
#                 newFrac = Con(IsNull(Raster(fraErr_path)), 0, Raster(fraErr_path))
#                 newFix = Con(newFrac == -1, Raster(p8_forBC), LRMfix)
#                 newFix.save(
#                     f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/Fix_Files/{rundate}_{basin}_{sub_basin}_{method}_LRMFix_final.tif")
#                 print("Fix completed")
#             else:
#                 print("File NOT found.")
#
#     # Mosaic all files in the same main group
#     for basin in mainBasins:
#         basinFix = []
#         fixFiles = os.listdir(f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/Fix_Files/")
#         for file in fixFiles:
#             if file.endswith(".tif") and file.startswith(f"{rundate}_{basin}_"):
#                 basinFix.append(f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/Fix_Files/{file}")
#             elif not file.endswith(".tif"):
#                 arcpy.Delete_management(file)
#                 # print(f"Deleted non-TIF file: {file}")
#
#         if basinFix:
#             arcpy.env.snapRaster = p8_forBC
#             arcpy.env.cellSize = p8_forBC
#             out_raster = f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{rundate}_{basin}_{method}_BC_fix_albn83.tif"
#             arcpy.MosaicToNewRaster_management(basinFix,
#                                                os.path.dirname(out_raster),
#                                                os.path.basename(out_raster),
#                                                "",
#                                                "32_BIT_FLOAT", "", 1, "LAST", "FIRST")
#             print(f"Mosaicked raster saved: {out_raster}")
import os
import shutil
import arcpy
import math
import glob
from arcpy.sa import *

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

    if domain is not None:
        results_df = results_df[results_df['Domain'] == domain].copy()
        print(f"Filtered to domain: {domain}")

    os.makedirs(f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/", exist_ok=True)
    os.makedirs(f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/Fix_Files/", exist_ok=True)

    p8Layer = f"{results_workspace}/{ModelRun}/p8_{rundate}_noneg.tif"
    shutil.copy(p8Layer, f"{results_workspace}ASO_BiasCorrect_{ModelRun}/")
    p8_forBC = f"{results_workspace}ASO_BiasCorrect_{ModelRun}/p8_{rundate}_noneg.tif"

    mainBasins = results_df['MainGroup'].unique().tolist()
    print("Main basins:", mainBasins)

    merged_basins = {
        "Truckee": {"partner": "Tahoe", "merged_name": "TruckeeTahoe"},
        "Tahoe":   {"partner": "Truckee", "merged_name": "TruckeeTahoe"},
        "ECarson": {"partner": "WCarson", "merged_name": "Carson"},
        "WCarson": {"partner": "ECarson", "merged_name": "Carson"}
    }

    for basin in mainBasins:
        print(f"\n{'=' * 60}")
        print(f"Processing main group: {basin}")
        print(f"{'=' * 60}")
        df_group = results_df[results_df['MainGroup'] == basin]

        processed_merged = set()

        for idx, row in df_group.iterrows():
            sub_basin = row["Basin"]
            fraErr = row[method]

            # ---------------------------------------------------------
            # FIX: skip just this basin instead of the whole main group
            # ---------------------------------------------------------
            if (fraErr is None or
                    fraErr == "NA" or
                    fraErr == "" or
                    str(fraErr).lower() == "none" or
                    (isinstance(fraErr, float) and math.isnan(fraErr))):
                print(f"  Skipping {sub_basin} — no data for method {method} (value: '{fraErr}')")
                continue

            if isinstance(fraErr, str) and fraErr.startswith("SAME_FILE_AS_"):
                print(f"  Skipping {sub_basin} — {fraErr}")
                continue

            fraErr = str(fraErr).strip()
            if len(fraErr) < 20:
                print(f"  Skipping {sub_basin} — path too short: '{fraErr}'")
                continue

            fraErr_normalized = fraErr.replace('\\', '/').replace('//', '/')
            fraErr_dir = os.path.dirname(fraErr_normalized)
            fraErr_basename = os.path.basename(fraErr_normalized)

            if not fraErr_dir or not os.path.exists(fraErr_dir):
                print(f"  Skipping {sub_basin} — directory doesn't exist: '{fraErr_dir}'")
                continue
            # ---------------------------------------------------------

            # Handle merged basin groups
            is_merged = sub_basin in merged_basins
            if is_merged:
                merged_name = merged_basins[sub_basin]["merged_name"]
                if merged_name in processed_merged:
                    print(f"  Skipping {sub_basin} — already processed as part of {merged_name}")
                    continue
                processed_merged.add(merged_name)
                output_basin_name = merged_name
            else:
                output_basin_name = sub_basin

            search_pattern = f"{fraErr_dir}/{fraErr_basename[:-15]}*fraErr.tif"
            print(f"Basin: {basin} | Sub-Basin: {sub_basin} | Output Name: {output_basin_name} | Search pattern: {search_pattern}")

            if domain == "SNM":
                basinSHP = f"{shapefile_workspace}/{sub_basin}_albn83.shp"
            elif domain == "WW":
                if sub_basin in ["Truckee", "Tahoe"]:
                    merge_output = f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/TruckeeTahoeMerge_albn83.shp"
                    if not os.path.exists(merge_output):
                        arcpy.Merge_management(
                            [f"{shapefile_workspace}/Truckee_albn83.shp",
                             f"{shapefile_workspace}/Tahoe_albn83.shp"],
                            merge_output
                        )
                    basinSHP = merge_output
                elif sub_basin in ["ECarson", "WCarson"]:
                    merge_output = f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/CarsonMerge_albn83.shp"
                    if not os.path.exists(merge_output):
                        arcpy.Merge_management(
                            [f"{shapefile_workspace}/WCarson_albn83.shp",
                             f"{shapefile_workspace}/ECarson_albn83.shp"],
                            merge_output
                        )
                    basinSHP = merge_output
                else:
                    basinSHP = f"{shapefile_workspace}ASO_{sub_basin}_albn83.shp"

            matching_files = glob.glob(search_pattern)
            if not matching_files:
                print(f"  No files found matching pattern: {search_pattern} — skipping {sub_basin}")
                continue

            fraErr_path = matching_files[0]
            print(f"  Found file: {fraErr_path}")

            if not os.path.exists(fraErr_path):
                print(f"  File not found on disk — skipping {sub_basin}")
                continue

            print(f"Basin: {basin} | Sub-Basin: {sub_basin} | Output Name: {output_basin_name} | fracErr_path: {fraErr_path}")

            # Create p8 mask if it doesn't exist
            mask_path = f"{results_workspace}ASO_BiasCorrect_{ModelRun}/p8_{rundate}_noneg_ASO_msk.tif"
            if not os.path.isfile(mask_path):
                p8masking = Con(Raster(p8_forBC) >= 0, 1, Raster(p8_forBC))
                p8masking.save(mask_path)
            else:
                print("  P8 mask exists, moving on")

            arcpy.env.mask = mask_path
            arcpy.env.cellSize = p8_forBC
            arcpy.env.snapRaster = p8_forBC

            basin_mask_path = f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{output_basin_name}_msk.tif"
            if os.path.exists(basin_mask_path):
                print(f"  Basin mask already exists: {output_basin_name}_msk.tif")
            else:
                basinBound = ExtractByMask(mask_path, basinSHP)
                basinBound.save(basin_mask_path)

            newFracError = Con(Raster(fraErr_path) == -1, -0.999, Raster(fraErr_path))
            newFracError.save(
                f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{output_basin_name}_fracError_v2.tif")

            noNull = Con(IsNull(
                Raster(f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{output_basin_name}_fracError_v2.tif")),
                0,
                Raster(f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{output_basin_name}_fracError_v2.tif"))
            noNull.save(
                f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{output_basin_name}_fracError_v3.tif")

            extract_fracError = ExtractByMask(
                f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{output_basin_name}_fracError_v3.tif",
                basin_mask_path)
            extract_fracError.save(
                f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{output_basin_name}_fracError_v4.tif")

            LRMfix = Raster(p8_forBC) / (1 + Raster(
                f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{output_basin_name}_fracError_v4.tif"))
            newFrac = Con(IsNull(Raster(fraErr_path)), 0, Raster(fraErr_path))
            newFix = Con(newFrac == -1, Raster(p8_forBC), LRMfix)
            newFix.save(
                f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/Fix_Files/{rundate}_{basin}_{output_basin_name}_{method}_LRMFix_final.tif")
            print(f"  Fix completed for {output_basin_name}")

    # Mosaic all fix files per main group
    for basin in mainBasins:
        basinFix = []
        fixFiles = os.listdir(f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/Fix_Files/")
        for file in fixFiles:
            if file.endswith(".tif") and file.startswith(f"{rundate}_{basin}_"):
                basinFix.append(
                    f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/Fix_Files/{file}")
            elif not file.endswith(".tif"):
                arcpy.Delete_management(file)

        if basinFix:
            arcpy.env.snapRaster = p8_forBC
            arcpy.env.cellSize = p8_forBC
            out_raster = f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{rundate}_{basin}_{method}_BC_fix_albn83.tif"
            arcpy.MosaicToNewRaster_management(basinFix,
                                               os.path.dirname(out_raster),
                                               os.path.basename(out_raster),
                                               "",
                                               "32_BIT_FLOAT", "", 1, "LAST", "FIRST")
            print(f"  Mosaicked raster saved: {out_raster}")
        else:
            print(f"  No fix files found for main group '{basin}' method '{method}' — mosaic skipped")
# def bias_correct(results_workspace, domain, ModelRun, method, rundate, results_df, shapefile_workspace):
#     """
#     Performs bias correction for ASO SWE data.
#
#     Parameters:
#         results_workspace (str): Path to the results workspace.
#         ModelRun (str): Name of the model run.
#         method (str): Method name used for fractional error.
#         rundate (str): Date string for the run (e.g., '20250503').
#         results_df (DataFrame): DataFrame containing 'MainGroup', 'Basin', and method column.
#         shapefile_workspace (str): Path to shapefiles.
#     """
#     # first filter by domain
#     # Filter by domain if specified
#     if domain is not None:
#         results_df = results_df[results_df['Domain'] == domain].copy()
#         print(f"Filtered to domain: {domain}")
#
#     # Make directories
#     os.makedirs(f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/", exist_ok=True)
#     os.makedirs(f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/Fix_Files/", exist_ok=True)
#
#     # Copy p8 layer into output directory
#     p8Layer = f"{results_workspace}/{ModelRun}/p8_{rundate}_noneg.tif"
#     shutil.copy(p8Layer, f"{results_workspace}ASO_BiasCorrect_{ModelRun}/")
#     p8_forBC = f"{results_workspace}ASO_BiasCorrect_{ModelRun}/p8_{rundate}_noneg.tif"
#
#     # Get unique main groups
#     mainBasins = results_df['MainGroup'].unique().tolist()
#     print("Main basins:", mainBasins)
#
#     # Loop through main basins
#     for basin in mainBasins:
#         print(f"\n{'=' * 60}")
#         print(f"Processing main group: {basin}")
#         print(f"{'=' * 60}")
#         df_group = results_df[results_df['MainGroup'] == basin]
#
#         # PRE-CHECK: Validate that ALL basins in this main group have valid files for this method
#         all_basins_valid = True
#         validation_results = []
#
#         for idx, row in df_group.iterrows():
#             sub_basin = row["Basin"]
#             fraErr = row[method]
#
#             # Check if this is a reference to another method's file
#             if isinstance(fraErr, str) and fraErr.startswith("SAME_FILE_AS_"):
#                 validation_results.append(f"  ✗ {sub_basin}: references another method ({fraErr})")
#                 all_basins_valid = False
#                 continue
#
#             # Check for missing/invalid data
#             if (fraErr is None or
#                     (isinstance(fraErr, float) and math.isnan(fraErr)) or
#                     fraErr == "NA" or
#                     fraErr == "" or
#                     str(fraErr).lower() == "none"):
#                 validation_results.append(f"  ✗ {sub_basin}: no data (value: '{fraErr}')")
#                 all_basins_valid = False
#                 continue
#
#             # Convert to string and validate it's a real path
#             fraErr = str(fraErr).strip()
#             if len(fraErr) < 20:
#                 validation_results.append(f"  ✗ {sub_basin}: path too short: '{fraErr}'")
#                 all_basins_valid = False
#                 continue
#
#             # Check if directory exists
#             fraErr_dir = os.path.dirname(fraErr)
#             if not fraErr_dir or not os.path.exists(fraErr_dir):
#                 validation_results.append(f"  ✗ {sub_basin}: directory doesn't exist: '{fraErr_dir}'")
#                 all_basins_valid = False
#                 continue
#
#             validation_results.append(f"  ✓ {sub_basin}: valid path")
#
#         # Print validation results
#         # print(f'HELP {fraErr_dir}')
#         print(f"\nValidation results for method '{method}' in main group '{basin}':")
#         for result in validation_results:
#             print(result)
#
#         # If not all basins are valid, skip this entire main group for this method
#         if not all_basins_valid:
#             print(f"\n  SKIPPING entire main group '{basin}' for method '{method}'")
#             print(f"    Reason: Not all basins have valid fractional error files")
#             print(f"    {sum(1 for r in validation_results if '✓' in r)}/{len(validation_results)} basins valid")
#             continue
#
#         print(f"\n✓ All basins valid for main group '{basin}' - proceeding with processing\n")
#
#         # Define merged basin groups - only process the first basin in each group
#         merged_basins = {
#             "Truckee": {"partner": "Tahoe", "merged_name": "TruckeeTahoe"},
#             "Tahoe": {"partner": "Truckee", "merged_name": "TruckeeTahoe"},
#             "ECarson": {"partner": "WCarson", "merged_name": "Carson"},
#             "WCarson": {"partner": "ECarson", "merged_name": "Carson"}
#         }
#
#         processed_merged = set()  # Track which merged basins we've already processed
#
#         # Now process all basins in this main group
#         for idx, row in df_group.iterrows():
#             sub_basin = row["Basin"]
#             fraErr = row[method]
#
#             # Check if this is a merged basin and if we've already processed its partner
#             is_merged = sub_basin in merged_basins
#             if is_merged:
#                 merged_name = merged_basins[sub_basin]["merged_name"]
#                 if merged_name in processed_merged:
#                     print(f"Skipping {sub_basin} - already processed as part of {merged_name}")
#                     continue
#                 processed_merged.add(merged_name)
#                 # Use merged name for output files
#                 output_basin_name = merged_name
#             else:
#                 output_basin_name = sub_basin
#
#             # Convert to string (we already validated this above)
#             fraErr = str(fraErr).strip()
#             fraErr_dir = os.path.dirname(fraErr)
#             fraErr_basename = os.path.basename(str(fraErr))
#
#             # Pattern: starts with basename, ends with swe_50m_fraErr.tif
#             # This will match: ASO_BoulderCreek_2025Apr09*swe_50m_fraErr.tif
#             # search_pattern = os.path.join(fraErr_dir, f"{fraErr_basename[:-15]}*swe_50m_fraErr.tif")
#
#             fraErr_normalized = fraErr.replace('\\', '/').replace('//', '/')
#             fraErr_dir = os.path.dirname(fraErr_normalized)
#             fraErr_basename = os.path.basename(fraErr_normalized)
#             # search_pattern = f"{fraErr_dir}/{fraErr_basename[:-15]}*swe_50m_fraErr.tif"
#             search_pattern = f"{fraErr_dir}/{fraErr_basename[:-15]}*fraErr.tif"
#
#             print(f"Basin: {basin} | Sub-Basin: {sub_basin} | Output Name: {output_basin_name} | Search pattern: {search_pattern}")
#
#             if domain == "SNM":
#                 basinSHP = f"{shapefile_workspace}/{sub_basin}_albn83.shp"
#             elif domain == "WW":
#                 # Handle special basin merging cases
#                 if sub_basin in ["Truckee", "Tahoe"]:
#                     # Merge Truckee and Tahoe
#                     merge_output = f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/TruckeeTahoeMerge_albn83.shp"
#                     if not os.path.exists(merge_output):
#                         arcpy.Merge_management(
#                             [f"{shapefile_workspace}/Truckee_albn83.shp",
#                              f"{shapefile_workspace}/Tahoe_albn83.shp"],
#                             merge_output
#                         )
#                         # print(f"Created merged shapefile: {merge_output}")
#                     basinSHP = merge_output
#
#                 elif sub_basin in ["ECarson", "WCarson"]:
#                     # Merge East and West Carson
#                     merge_output = f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/CarsonMerge_albn83.shp"
#                     if not os.path.exists(merge_output):
#                         arcpy.Merge_management(
#                             [f"{shapefile_workspace}/WCarson_albn83.shp",
#                              f"{shapefile_workspace}/ECarson_albn83.shp"],
#                             merge_output
#                         )
#                         # print(f"Created merged shapefile: {merge_output}")
#                     basinSHP = merge_output
#                 else:
#                     basinSHP = f"{shapefile_workspace}ASO_{sub_basin}_albn83.shp"
#
#             # matching_files = glob.glob(search_pattern)
#             #
#             # if matching_files:
#             #     for fraErr_path in matching_files:
#             #         print(f"Found file: {fraErr_path}")
#             #
#             #         if os.path.exists(fraErr_path):
#             #             print("File exists!")
#             #             # Your processing code here
#             # else:
#             #     print(f"No files found matching pattern: {search_pattern}")
#             matching_files = glob.glob(search_pattern)
#
#             if not matching_files:
#                 print(f"No files found matching pattern: {search_pattern} — SKIPPING {sub_basin}")
#                 continue  # ← skip to next basin, don't fall through with stale fraErr_path
#
#             fraErr_path = matching_files[0]  # explicitly assign here, not inside loop
#             print(f"Found file: {fraErr_path}")
#
#             if os.path.exists(fraErr_path):
#
#                 print(
#                     f"Basin: {basin} | Sub-Basin: {sub_basin} | Output Name: {output_basin_name} | fracErr_path: {fraErr_path}")
#
#                 # Create p8 mask if it doesn't exist
#                 mask_path = f"{results_workspace}ASO_BiasCorrect_{ModelRun}/p8_{rundate}_noneg_ASO_msk.tif"
#                 if not os.path.isfile(mask_path):
#                     p8masking = Con(Raster(p8_forBC) >= 0, 1, Raster(p8_forBC))
#                     p8masking.save(mask_path)
#                     # print("p8 mask created")
#                 else:
#                     print("P8 mask exists, moving on")
#
#                 # Mask just the basin boundary - use output_basin_name for merged basins
#                 arcpy.env.mask = mask_path
#                 arcpy.env.cellSize = p8_forBC
#                 arcpy.env.snapRaster = p8_forBC
#                 basin_mask_path = f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{output_basin_name}_msk.tif"
#                 if os.path.exists(basin_mask_path):
#                     print(f'Basin mask already exists: {output_basin_name}_msk.tif')
#                 else:
#                     basinBound = ExtractByMask(mask_path, basinSHP)
#                     basinBound.save(basin_mask_path)
#                     # print(f'Created basin mask: {output_basin_name}_msk.tif')
#
#                 # Change all -1 to -0.999 - use output_basin_name
#                 newFracError = Con(Raster(fraErr_path) == -1, -0.999, Raster(fraErr_path))
#                 newFracError.save(
#                     f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{output_basin_name}_fracError_v2.tif")
#
#                 # Make all no data values 0
#                 noNull = Con(IsNull(
#                     Raster(
#                         f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{output_basin_name}_fracError_v2.tif")),
#                     0,
#                     Raster(
#                         f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{output_basin_name}_fracError_v2.tif"))
#                 noNull.save(
#                     f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{output_basin_name}_fracError_v3.tif")
#                 # print("non-zero fractional layer created")
#
#                 # Snap and clip to the extent of basin
#                 # print("Extracting and setting boundaries")
#                 extract_fracError = ExtractByMask(
#                     f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{output_basin_name}_fracError_v3.tif",
#                     basin_mask_path)
#                 extract_fracError.save(
#                     f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{output_basin_name}_fracError_v4.tif")
#
#                 # Compute LRM fix
#                 LRMfix = Raster(p8_forBC) / (1 + Raster(
#                     f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{output_basin_name}_fracError_v4.tif"))
#                 newFrac = Con(IsNull(Raster(fraErr_path)), 0, Raster(fraErr_path))
#                 newFix = Con(newFrac == -1, Raster(p8_forBC), LRMfix)
#                 newFix.save(
#                     f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/Fix_Files/{rundate}_{basin}_{output_basin_name}_{method}_LRMFix_final.tif")
#                 # print(f"Fix completed for {output_basin_name}")
#             else:
#                 print("File NOT found.")
#
#     # Mosaic all files in the same main group
#     for basin in mainBasins:
#         basinFix = []
#         fixFiles = os.listdir(f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/Fix_Files/")
#         for file in fixFiles:
#             if file.endswith(".tif") and file.startswith(f"{rundate}_{basin}_"):
#                 basinFix.append(f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/Fix_Files/{file}")
#             elif not file.endswith(".tif"):
#                 arcpy.Delete_management(file)
#                 # print(f"Deleted non-TIF file: {file}")
#
#         if basinFix:
#             arcpy.env.snapRaster = p8_forBC
#             arcpy.env.cellSize = p8_forBC
#             out_raster = f"{results_workspace}ASO_BiasCorrect_{ModelRun}/{method}/{rundate}_{basin}_{method}_BC_fix_albn83.tif"
#             arcpy.MosaicToNewRaster_management(basinFix,
#                                                os.path.dirname(out_raster),
#                                                os.path.basename(out_raster),
#                                                "",
#                                                "32_BIT_FLOAT", "", 1, "LAST", "FIRST")
#             # print(f"Mosaicked raster saved: {out_raster}")







