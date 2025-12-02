# import modules
import arcpy
import pandas as pd
import os
from arcpy import *
from datetime import datetime
print('modules imported')

# parameters
rundate = "20250506"
currentYear = True
current_year = datetime.now().year
methods = ["RECENT", "GRADE", "SENSOR_PATTERN"]
grade = "positive"
grade_range = False
grade_amount = 10
sensorTrend = "Mixed"
SNOTEL = "Decreasing"
domains = ['SNM', 'SOCN']
basinList = ["SouthPlatte", "Uinta"]
out_csv = "Y"
csv_outFile = r"W:/Spatial_SWE/ASO/2025/data_testing/FracError_data_test.csv"
asoCatalog = r"W:/Spatial_SWE/ASO/2025/data_testing/ASO_SNOTEL_DifferenceStats.csv"
asoBasinList = r"W:/Spatial_SWE/ASO/ASO_Metadata/ASO_in_Basin.txt"
fracErrorWorkspace = "W:/Spatial_SWE/ASO/2025/data_testing/"

## function for fix
## function for validation

mapping = {}
with open(asoBasinList) as f:
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

# set the list
all_results = []

for main_group, sub_items in mapping.items():
    print(f"Main group: {main_group}")
    for item in sub_items:
        print(f"{item}")
        aso_df = pd.read_csv(asoCatalog)
        if item in aso_df['Basin'].values:
            if currentYear is True:
                aso_df = aso_df[aso_df["Year"] == current_year]
            else:
                print("Using all years of ASO data")

            # Store paths for this basin
            basin_row = {
                'ModelDate': rundate,
                'MainGroup': main_group,
                'Basin': item,
                'RECENT': None,
                'GRADE': None,
                'SENSOR_PATTERN': None,
                'PATTERN_TYPE': None
            }

            for method in methods:
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
                            f"{fracErrorWorkspace}/{closest_row['Domain']}_comparison_testing/{closest_row['RunDate']}_{closest_row['modelRun']}"
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
                                    f"{fracErrorWorkspace}/{closest_row['Domain']}_comparison_testing/{closest_row['RunDate']}_{closest_row['modelRun']}"
                                    f"ASO_{closest_row['Basin']}_{closest_row['Date']}_swe_50m_fraErr")
                                basin_row['GRADE'] = fraErrorPath

                        if method == "GRADES_SPECF":
                            print(f"\nMETHOD: {method}")
                            closest_row = aso_df_grade.loc[(aso_df_grade["-20"] - grade_amount).abs().idxmin()]
                            fraErrorPath = (
                                f"{fracErrorWorkspace}/{closest_row['Domain']}_comparison_testing/{closest_row['RunDate']}_{closest_row['modelRun']}"
                                f"ASO_{closest_row['Basin']}_{closest_row['Date']}_swe_50m_fraErr")

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
                                f"{fracErrorWorkspace}/{closest_row['Domain']}_comparison_testing/{closest_row['RunDate']}_{closest_row['modelRun']}"
                                f"ASO_{closest_row['Basin']}_{closest_row['Date']}_swe_50m_fraErr")
                            basin_row['SENSOR_PATTERN'] = fraErrorPath


                    elif row_count == 1:
                        closest_row = aso_df_pattern.iloc[0]
                        fraErrorPath = (
                            f"{fracErrorWorkspace}/{closest_row['Domain']}_comparison_testing/{closest_row['RunDate']}_{closest_row['modelRun']}"
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

if out_csv == "Y":
    results_df.to_csv(csv_outFile, index=False)


############################################
# start of new function: Bias correction and fix
############################################
# PARAMETERS
# P8 layer
# method

# get unqiue main groups make a list
results_df

# loop through list:
## for group in unique_list:
    # loop through "Basin" only if "MainGroup" is group:
    # get fraction error layer for Row name, row name = method:


    # LRMfix = Raster(LRMlayer) / (1 + Raster(fracError_v4))
    # newFrac = Con(IsNull(Raster(fracError_layer)), 0, Raster(fracError_layer))
    # newFix = Con(newFrac == -1, Raster(LRMlayer), LRMfix)
    # newFix.save(FixLayerWorkspace + f"{RTdate}_{Basin}_LRMFix_final.tif")
    # print("fix completed")

    ## add fix final to a list
    # mosaic all fix final together

    ## extract main group by mask
    ## Con Is Null with the Fix

    # run vetting code for sensors only
    # check if running code should be surveys
    # output df:
    # domain = WW/SNM
    # RT date = date
    # basin = "main_Group"
    # aso_bains = list of basins in unique list
    # Method_error tuple = ('method', number)
    # Method_AF tuple = ('method', AF)
    # Method_path tuple = ('method', 'file path')

############################################
# start of new function: select bias correction layer
############################################
# open csv
# select for only date
# isolate the basin column and any column that has _error
# go through each row and select the lowest error method
# give a list of: the winningest group, plot a chart on the error differences per basin and AF values pre and post bias correction
# PROMPT: which option to chose

## Make final layer
### select all the method file paths
## do a Con IsNull for the rest of the model run.











