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
method = "RECENT"
grade = "positive"
grade_range = False
grade_amount = 10
SNOTEL = "Decreasing"
domains = ['SNM', 'SOCN']
basinList = ["SouthPlatte", "Uinta"]
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

fracErrorDict_recent = {}
fracErrorDict_grade = {}
fracErrorDict_grade_specf = {}
# fractional error dictionary
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

            # getting most recent data
            if method == "RECENT":
                aso_df_basin = aso_df[aso_df["Basin"] == item]
                target_date  = datetime.strptime(rundate, "%Y%m%d")
                aso_df_basin['cstm_dte'] = pd.to_datetime(aso_df_basin["Date"], format="%Y%b%d")
                aso_df_basin["diff_days"] = (aso_df_basin["cstm_dte"] - target_date).abs().dt.days
                df_filtered = aso_df_basin[aso_df_basin["diff_days"] > 4]
                closest_row = df_filtered.loc[df_filtered["diff_days"].idxmin()]

                # get fractional error path
                fraErrorPath = (f"{fracErrorWorkspace}/{closest_row['Domain']}_comparison_testing/{closest_row['RunDate']}_{closest_row['modelRun']}"
                            f"ASO_{closest_row['Basin']}_{closest_row['Date']}_swe_50m_fraErr")

                # store in dictionary
                if main_group not in fracErrorDict_recent:
                    fracErrorDict_recent[main_group] = {}
                fracErrorDict_recent[main_group][item] = fraErrorPath

            if method == "GRADE" or method == "GRADES_SPECF":
                aso_df_basin = aso_df[aso_df["Basin"] == item]
                aso_df_grade = aso_df_basin[aso_df_basin["GradDirection"] == grade]
                if len(aso_df_grade.columns) > 1:
                    if method == "GRADE":
                        target_date = datetime.strptime(rundate, "%Y%m%d")
                        aso_df_grade['cstm_dte'] = pd.to_datetime(aso_df_grade["Date"], format="%Y%b%d")
                        aso_df_grade["diff_days"] = (aso_df_grade["cstm_dte"] - target_date).abs().dt.days
                        df_filtered = aso_df_grade[aso_df_grade["diff_days"] > 4]
                        closest_row = df_filtered.loc[df_filtered["diff_days"].idxmin()]

                        fraErrorPath = (
                            f"{fracErrorWorkspace}/{closest_row['Domain']}_comparison_testing/{closest_row['RunDate']}_{closest_row['modelRun']}"
                            f"ASO_{closest_row['Basin']}_{closest_row['Date']}_swe_50m_fraErr")

                    if method == "GRADES_SPECF":
                        closest_row = aso_df_grade.loc[(aso_df_grade["X"] - grade_amount).abs().idxmin()]
                        fraErrorPath = (
                            f"{fracErrorWorkspace}/{closest_row['Domain']}_comparison_testing/{closest_row['RunDate']}_{closest_row['modelRun']}"
                            f"ASO_{closest_row['Basin']}_{closest_row['Date']}_swe_50m_fraErr")

                # get fractional error path
                fraErrorPath = (
                    f"{fracErrorWorkspace}/{closest_row['Domain']}_comparison_testing/{closest_row['RunDate']}_{closest_row['modelRun']}"
                    f"ASO_{closest_row['Basin']}_{closest_row['Date']}_swe_50m_fraErr")

                # store in dictionary
                if main_group not in fracErrorDict_recent:
                    fracErrorDict_recent[main_group] = {}
                fracErrorDict_recent[main_group][item] = fraErrorPath
        else:
            continue




## have a dictionary that shows the ASO bias corrected basins that are listed within the basin -- NEED TO DO
# makes a list of all the basins
# loop through the ASO flights within that basin

    ## if len(list) == 0:
        # continue
    ## else

# most recent flight
## open the csv
## loop through the list
    ## open an empty fraction error list
    ## subset any rows that are in the basins
    ## make a new column that has the date in the MMDDYYYY format
    ## exclude any dates that are within 5 days of the run date
    ## find the most recent date
    ## get that fractional error layer in the list based on the file path

# percent grade
## loop through the list
    ## open an empty fraction error list
    ## subset any rows that are in the basins
    ## check to see if it's within the year = in_current_year=True/False
    ### if True: df['Year'] == Year
    ## negative postive or mixed for "GradeDirection"
    ## get the percentage grade within percentile

## select the one that your want through a csv
