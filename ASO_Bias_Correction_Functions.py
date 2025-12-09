# import modules
import arcpy
import pandas as pd
import os
import shutil
from arcpy.sa import *
from arcpy import *
from datetime import datetime
print('modules imported')

# parameters
rundate = "20250503"
ModelRun = "fSCA_RT_CanAdj_rcn_noSW_woCCR"
currentYear = True
current_year = datetime.now().year
methods = ["RECENT", "GRADE", "SENSOR_PATTERN", "GRADES_SPECF"]
grade = "positive"
grade_range = False
grade_amount = -10
sensorTrend = "Mixed"
SNOTEL = "Decreasing"
domains = ['SNM', 'SOCN']
basinList = ["SouthPlatte", "Uinta"]
out_csv = "Y"
csv_outFile = r"W:/Spatial_SWE/ASO/2025/data_testing/FracError_data_test.csv"
asoCatalog = r"W:/Spatial_SWE/ASO/2025/data_testing/ASO_SNOTEL_DifferenceStats.csv"
asoBasinList = r"W:/Spatial_SWE/ASO/ASO_Metadata/ASO_in_Basin.txt"
fracErrorWorkspace = "W:/Spatial_SWE/ASO/2025/data_testing/"
results_workspace = f"M:/SWE/WestWide/Spatial_SWE/WW_regression/RT_report_data/{rundate}_results_ET/"

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

                        if method == "GRADES_SPECF":
                            print(f"\nMETHOD: {method}")
                            closest_row = aso_df_grade.loc[(aso_df_grade["GradeDifference"] - grade_amount).abs().idxmin()]
                            fraErrorPath = (
                                f"{fracErrorWorkspace}/{closest_row['Domain']}_comparison_testing/{closest_row['RunDate']}_{closest_row['modelRun']}/"
                                f"ASO_{closest_row['Basin']}_{closest_row['Date']}_swe_50m_fraErr")
                            basin_row['GRADES_SPECF'] = fraErrorPath

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

if out_csv == "Y":
    results_df.to_csv(csv_outFile, index=False)


############################################
# start of new function: Bias correction and fix
############################################
# PARAMETERS
# model_run
shapefile_workspace = "W:/data/hydro/WW/ASO_Basin_Shapefiles/"
# method
results_df = pd.read_csv(csv_outFile) # temp
method = "RECENT"

# make directories
os.makedirs(results_workspace + f"ASO_BiasCorrect_{ModelRun}/{method}/")
os.makedirs(results_workspace + f"ASO_BiasCorrect_{ModelRun}/{method}/Fix_Files/")

# copy p8 layer into outdir
p8Layer = f"{results_workspace}/{ModelRun}/p8_{rundate}_noneg.tif"
shutil.copy(p8Layer, results_workspace + f"ASO_BiasCorrect_{ModelRun}/")
p8_forBC = results_workspace + f"ASO_BiasCorrect_{ModelRun}/p8_{rundate}_noneg.tif"

# get unqiue main groups make a list
mainBasins = results_df['MainGroup'].unique().tolist()
print(mainBasins)

# loop through list:
for basin in mainBasins:
    print(f"\nbasin: {basin}")
    # loop through "Basin" only if "MainGroup" is group:
    df_group = results_df[results_df['MainGroup'] == basin]

    # loop through the sub-basins
    for idx, row in df_group.iterrows():
        sub_basin = row["Basin"]
        fraErr = row[method]
        fraErr_path = fraErr + ".tif"
        print(f"Basin: {basin} | Sub-Basin: {sub_basin} | fracErr_path: {fraErr_path}")
        basinSHP = shapefile_workspace + f"ASO_{sub_basin}_albn83.shp"
        if os.path.exists(fraErr_path):
            print("File exists!")

            # make a mask
            if os.path.isfile(results_workspace + f"ASO_BiasCorrect_{ModelRun}/{method}/p8_{rundate}_noneg_msk.tif"):
                print("p8 mask exists, moving on")
            else:
                p8masking = Con(Raster(p8_forBC) >= 0, 1, Raster(p8_forBC))
                p8masking.save(results_workspace + f"ASO_BiasCorrect_{ModelRun}/{method}/p8_{rundate}_noneg_msk.tif")
                print("p8 mask created")

            # mask just the basin boundary
            arcpy.env.mask = results_workspace + f"ASO_BiasCorrect_{ModelRun}/{method}/p8_{rundate}_noneg_msk.tif"
            arcpy.env.cellSize = p8_forBC
            arcpy.env.snapRaster = p8_forBC
            basinBound = ExtractByMask(results_workspace + f"ASO_BiasCorrect_{ModelRun}/{method}/p8_{rundate}_noneg_msk.tif", basinSHP)
            basinBound.save(results_workspace + f"ASO_BiasCorrect_{ModelRun}/{method}/{sub_basin}_msk.tif")

            # change all -1 to -0.999
            newFracError = Con(Raster(fraErr_path) == -1, -0.999, Raster(fraErr_path))
            newFracError.save(results_workspace + f"ASO_BiasCorrect_{ModelRun}/{method}/{sub_basin}_fracError_v2.tif")

            # make all no data values 0
            noNull = Con(IsNull(Raster(results_workspace + f"ASO_BiasCorrect_{ModelRun}/{method}/{sub_basin}_fracError_v2.tif")), 0, Raster(results_workspace + f"ASO_BiasCorrect_{ModelRun}/{method}/{sub_basin}_fracError_v2.tif"))
            noNull.save(results_workspace + f"ASO_BiasCorrect_{ModelRun}/{method}/{sub_basin}_fracError_v3.tif")
            print("non_zero fractional layer created")

            # snap and clip to the extent of basin
            print("extracting and setting boundaries")
            extract_fracError = ExtractByMask(results_workspace + f"ASO_BiasCorrect_{ModelRun}/{method}/{sub_basin}_fracError_v3.tif", results_workspace + f"ASO_BiasCorrect_{ModelRun}/{method}/{sub_basin}_msk.tif")
            extract_fracError.save(results_workspace + f"ASO_BiasCorrect_{ModelRun}/{method}/{sub_basin}_fracError_v4.tif")

            LRMfix = Raster(p8_forBC) / (1 + Raster(results_workspace + f"ASO_BiasCorrect_{ModelRun}/{method}/{sub_basin}_fracError_v4.tif"))
            newFrac = Con(IsNull(Raster(fraErr_path)), 0, Raster(fraErr_path))
            newFix = Con(newFrac == -1, Raster(p8_forBC), LRMfix)
            newFix.save(results_workspace + f"ASO_BiasCorrect_{ModelRun}/{method}/Fix_Files/" + f"{rundate}_{basin}_{sub_basin}_{method}_LRMFix_final.tif")
            print("fix completed")

        else:
            print("File NOT found.")

# mosaic all files that are in the same main group
for basin in mainBasins:
    basinFix = []
    fixFiles = os.listdir(results_workspace + f"ASO_BiasCorrect_{ModelRun}/{method}/Fix_Files/")
    for file in fixFiles:
        if file.endswith(".tif"):
            if file.startswith(f"{rundate}_{basin}_"):
                basinFix.append(results_workspace + f"ASO_BiasCorrect_{ModelRun}/{method}/Fix_Files/{file}")
        else:
            arcpy.Delete_management(file)
            print(file)

    arcpy.env.snapRaster = p8_forBC
    arcpy.env.cellSize = p8_forBC
    arcpy.MosaicToNewRaster_management(basinFix, os.path.dirname(results_workspace + f"ASO_BiasCorrect_{ModelRun}/{method}/"), os.path.basename(f"{rundate}_{basin}_{method}_BC_fix_albn83.tif"),
                                       "",
                                       "32_BIT_FLOAT", "", 1, "LAST", "FIRST")





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











