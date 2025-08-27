# import modules
import os
import shutil
import geopandas as gpd

# from SWE_FUSION import WW_results_workspace

print('modules imported')

user = "Emma"
report_date = "20250315"
pillow_date = "15Mar2025"
model_run = "RT_CanAdj_rcn_noSW_woCCR"
domainList = ["NOCN", "PNW", "SNM", "SOCN", "INMT"]
workspaceBase = fr"M:/SWE/WestWide/Spatial_SWE/WW_regression/"
WW_results_workspace = workspaceBase + f"RT_report_data/{report_date}_results_ET/"
model_workspace = fr"W:/Spatial_SWE/WW_regression/RT_report_data/ModelWS_Test/"

def geopackage_to_shapefile(report_date, pillow_date, model_run, user, domainList, model_workspace, results_workspace):
    for domain in domainList:
        model_workspace_domain = f"{model_workspace}/{domain}/{user}/StationSWERegressionV2/data/outputs/{model_run}/"
        pillow_gpkg = model_workspace_domain + f"{domain}_pillow-{pillow_date}.gpkg"
        snowPillow = results_workspace + f"{report_date}_sensors_{domain}_corrupt.shp"
        pillow_copy = results_workspace + f"{domain}_pillow-{pillow_date}.gpkg"
        pillow_copy_rnme = results_workspace + f"pillow-{pillow_date}_{domain}.gpkg"
        pillowGPKG = results_workspace + f"{domain}_pillow-{pillow_date}.gpkg/pillow_data"

        # # Copy the .gpkg file to the new folder
        shutil.copy2(pillow_gpkg, results_workspace)

        # Path to the copied .gpkg file
        new_gpkg_path = os.path.join(results_workspace, f"{domain}_pillow-{pillow_date}.gpkg")

        # Read the .gpkg file (you can list the layers or specify the one you want)
        gdf = gpd.read_file(new_gpkg_path)
        gdf = gdf.rename(columns={
            "nwbDistance": "nwbDist",
            "regionaleastness": "regEast",
            "regionalnorthness": "regNorth",
            "regionalzness": "regZn",
            "northness4km" : "north4km",
            "eastness4km" : "east4km"
        })

        # Specify the path for the shapefile output
        shapefile_output = os.path.join(results_workspace, f"{report_date}_sensors_{domain}.shp")

        # Convert the GeoDataFrame to a shapefile
        gdf.to_file(shapefile_output)

        print(f"Shapefile saved to: {shapefile_output}")

print("\nProcessing GeoPackage")
geopackage_to_shapefile(report_date=report_date, pillow_date=pillow_date, model_run=model_run,
                        user=user, domainList=domainList, model_workspace=model_workspace,
                        results_workspace=WW_results_workspace)



# sensor processing code
import arcpy
import pandas as pd
import geopandas as gpd
print('modules imported')
arcpy.ClearWorkspaceCache_management()

# establish paths and dates
surveys = "N" # this should be "T" if it's the first of the month and you're using surveys, "F" if not
survey_date = "20250315"
# report_date = "20250526"
# prev_report_date = "20250517"
difference = "N"

projIn = arcpy.SpatialReference(4269)
projOut = arcpy.SpatialReference(102039)
merge = "Y"
# domainList = ["PNW", "NOCN", "SOCN", "INMT", "SNM"]

watershed_shapefile = "M:/SWE/WestWide/data/hydro/WW_Basins_noSNM_notahoe_albn83_sel_new.shp"
band_shapefile = "M:/SWE/WestWide/data/hydro/WW_BasinsBanded_noSNM_notahoe_albn83_sel_new.shp"
case_field_wtrshd = "SrtName"
case_field_band = "SrtNmeBand"

workspaceBase = fr"M:/SWE/WestWide/Spatial_SWE/WW_regression/"
results_workspace = workspaceBase + f"RT_report_data/{report_date}_results_ET/"
# prev_results_workspace = workspaceBase + f"RT_report_data/{prev_report_date}_results/"
##########################

def merge_sort_sensors_surveys(report_date, results_workspace, surveys, difference, watershed_shapefile,
                               band_shapefile, merge, projOut, projIn=None, domainList = None, domain_shapefile=None, prev_report_date=None, prev_results_workspace=None):

    # Set up snow pillow and snow survey shapefiles
    snowPillow_merge = results_workspace + f"{report_date}_sensors_WW_merge.shp"
    snowSurveys = results_workspace + f"{report_date}_surveys.shp"
    snowSurveys_proj = results_workspace + f"{report_date}_surveys_albn83.shp"


    # Create temp view for a join
    snowPillowView = results_workspace + f"{report_date}_sensors_view.dbf"

    # Create joined tables
    snowPillowsJoin = results_workspace + f"{report_date}_sensors_join.dbf"
    calcField = f"{report_date}_sensors.Diff_In"

    # snow and survey file names
    SensorWtshdInt = results_workspace + f"{report_date}_sensors_Wtshd_Intersect.shp"
    SnwSurvWtshdInt = results_workspace + f"{report_date}_surveys_Wtshd_Intersect.shp"
    SensorBandWtshdInt = results_workspace + f"{report_date}_sensors_BandWtshd_Intersect.shp"
    SnwSurvBandWtshdInt = results_workspace + f"{report_date}_surveys_BandWtshd_Intersect.shp"
    SensorWtshdIntStat = f"{SensorWtshdInt[:-4]}_stat.dbf"
    SnwSurvWtshdIntStat = f"{SnwSurvWtshdInt[:-4]}_stat.dbf"
    SensorBandWtshdIntStat = f"{SensorBandWtshdInt[:-4]}_stat.dbf"
    SnwSurvBandWtshdIntStat = f"{SnwSurvBandWtshdInt[:-4]}_stat.dbf"
    SensorBandWtshdIntStat_save = f"{SensorBandWtshdInt[:-4]}_save.dbf"
    SensorWtshdIntStat_save = f"{SensorWtshdIntStat[:-4]}_save.dbf"
    SnwSurvBandWtshdIntStat_save = f"{SnwSurvBandWtshdInt[:-4]}_save.dbf"
    SnwSurvWtshdIntStat_save = f"{SnwSurvWtshdIntStat[:-4]}_save.dbf"

    # final outputs
    SensorWtshdIntStat_CSV = f"{SensorWtshdIntStat[:-4]}.csv"
    SensorBandWtshdIntStat_CSV = f"{SensorBandWtshdInt[:-4]}.csv"
    SnwSurvWtshdIntStat_CSV = f"{SnwSurvWtshdInt[:-4]}.csv"
    SnwSurvBandWtshdIntStat_CSV = f"{SnwSurvBandWtshdInt[:-4]}.csv"
    SnwPillowsJoin_CSV = results_workspace + f"{report_date}_sensors_Join.csv"

    # set up intersect lists

    IntersctLstSurvey = [snowSurveys_proj, watershed_shapefile]
    IntersctLstBandSurvey = [snowSurveys_proj, band_shapefile]

    ############################################################################
    # Processing begins
    ############################################################################
    # ## set paths
    # merge and delete duplicates
    if merge == "Y":
        snowPillow_proj = results_workspace + f"{report_date}_sensors_albn83.shp"
        IntersctLst = [snowPillow_proj, watershed_shapefile]
        IntersctLstBand = [snowPillow_proj, band_shapefile]
        arcpy.Merge_management([results_workspace + f"{report_date}_sensors_{domainList[0]}.shp", results_workspace + f"{report_date}_sensors_{domainList[1]}.shp",
                                results_workspace + f"{report_date}_sensors_{domainList[2]}.shp", results_workspace + f"{report_date}_sensors_{domainList[3]}.shp",
                                results_workspace + f"{report_date}_sensors_{domainList[4]}.shp"], snowPillow_merge)

        # delete duplicates
        arcpy.DeleteIdentical_management(snowPillow_merge, "Site_ID")

        # reproject to Albers
        arcpy.Project_management(snowPillow_merge, snowPillow_proj, projOut)

    if merge == "N":
        snowPillow_proj = results_workspace + f"{report_date}_sensors_albn83.shp"
        IntersctLst = [snowPillow_proj, watershed_shapefile]
        IntersctLstBand = [snowPillow_proj, band_shapefile]
        arcpy.Project_management(domain_shapefile, snowPillow_proj, projOut, "", projIn)

    ## first add SWE inches, don't need to do this for surveys, it's already in there and then calculate field
    arcpy.AddField_management(snowPillow_proj, "SWE_In", "DOUBLE", "#", "#", "#",
                              "#", "NULLABLE", "NON_REQUIRED", "#")
    arcpy.CalculateField_management(snowPillow_proj, "SWE_In", "!pillowswe! * 39.370079", "PYTHON")

    ## Intersect with watersheds
    arcpy.Intersect_analysis(IntersctLst, SensorWtshdInt, "ALL", "-1 Unknown", "POINT")

    ## Create statistics
    arcpy.Statistics_analysis(SensorWtshdInt, SensorWtshdIntStat, "SWE_In MEAN", case_field_wtrshd)
    arcpy.AddField_management(SensorWtshdIntStat, "SWE_freq", "TEXT", "#", "#",
                              "#", "#", "NULLABLE", "NON_REQUIRED", "#")
    arcpy.CalculateField_management(SensorWtshdIntStat, "SWE_freq",
                                   '"{} ( {} )".format(round( !MEAN_SWE_I! ,1) , !FREQUENCY! )', "PYTHON", "")
    if surveys == "Y":
        arcpy.Project_management(snowSurveys, snowSurveys_proj, projOut)
        arcpy.Intersect_analysis(IntersctLstSurvey, SnwSurvWtshdInt, "ALL", "-1 Unknown", "POINT")
        arcpy.Statistics_analysis(SnwSurvWtshdInt, SnwSurvWtshdIntStat, "SWE_in MEAN", case_field_wtrshd)
        arcpy.AddField_management(SnwSurvWtshdIntStat, "SWE_freq", "TEXT", "#", "#",
                                  "#", "#", "NULLABLE", "NON_REQUIRED",
                                  "#")
        arcpy.CalculateField_management(SnwSurvWtshdIntStat, "SWE_freq",
                                        '"{} ( {} )".format(round( !MEAN_SWE_i! ,1) , !FREQUENCY! )', "PYTHON", "")

    arcpy.Intersect_analysis(IntersctLstBand, SensorBandWtshdInt, "ALL", "-1 Unknown", "POINT")
    if surveys == "Y":
        arcpy.Intersect_analysis(IntersctLstBandSurvey, SnwSurvBandWtshdInt, "ALL", "-1 Unknown", "POINT")

    arcpy.Statistics_analysis(SensorBandWtshdInt, SensorBandWtshdIntStat, "SWE_In MEAN", case_field_band)
    arcpy.AddField_management(SensorBandWtshdIntStat, "SWE_freq", "TEXT", "#", "#",
                              "#", "#", "NULLABLE", "NON_REQUIRED",
                              "#")
    ## Calculate Field
    arcpy.CalculateField_management(SensorBandWtshdIntStat, "SWE_freq",
                                    '"{} ( {} )".format(round( !MEAN_SWE_I! ,1) , !FREQUENCY! )', "PYTHON", "")
    if surveys == "Y":
        arcpy.Statistics_analysis(SnwSurvBandWtshdInt, SnwSurvBandWtshdIntStat, "SWE_in MEAN", case_field_band)
        arcpy.AddField_management(SnwSurvBandWtshdIntStat, "SWE_freq", "TEXT", "#",
                                  "#", "#", "#", "NULLABLE",
                                  "NON_REQUIRED", "#")
        arcpy.CalculateField_management(SnwSurvBandWtshdIntStat, "SWE_freq",
                                        '"{} ( {} )".format(round( !MEAN_SWE_I! ,1) , !FREQUENCY! )', "PYTHON", "")
    # Sort by bandname and watershed name, 2 tables
    arcpy.Sort_management(SensorBandWtshdIntStat, SensorBandWtshdIntStat_save, [[case_field_band, "ASCENDING"]])
    arcpy.Sort_management(SensorWtshdIntStat, SensorWtshdIntStat_save, [[case_field_wtrshd, "ASCENDING"]])
    if surveys == "Y":
        arcpy.Sort_management(SnwSurvBandWtshdIntStat, SnwSurvBandWtshdIntStat_save, [[case_field_band, "ASCENDING"]])
        arcpy.Sort_management(SnwSurvWtshdIntStat, SnwSurvWtshdIntStat_save, [[case_field_wtrshd, "ASCENDING"]])

    ## Make tables into table views for joins
    arcpy.MakeTableView_management(snowPillow_proj, snowPillowView)

    # creating a data frame of just the last SWE inches
    if difference == "Y":
        lastPillowView = prev_results_workspace + f"{prev_report_date}_sensors_view.dbf"
        lastPillow = prev_results_workspace + f"{prev_report_date}_sensors_albn83.shp"
        arcpy.MakeTableView_management(lastPillow, lastPillowView)
        arcpy.TableToTable_conversion(lastPillowView, results_workspace, f"{report_date}_temp.csv")
        temp_df = pd.read_csv(results_workspace + f"{report_date}_temp.csv")
        temp_df = temp_df[["Site_ID", "SWE_In"]]
        temp_df.rename(columns={"SWE_In" : "LastSWE_in"}, inplace=True)

        arcpy.TableToTable_conversion(snowPillowView, results_workspace, f"{report_date}_sensors.csv")
        curr_df = pd.read_csv(results_workspace + f"{report_date}_sensors.csv")
        merged_df = pd.merge(curr_df, temp_df[["Site_ID", "LastSWE_in"]], how="left", on="Site_ID")
        merged_df.to_csv(results_workspace + f"{report_date}_sensors_Join.csv", index=False)

    sensorBand_dbf = gpd.read_file(SensorBandWtshdIntStat_save)
    sensorBand_dbf = pd.DataFrame(sensorBand_dbf)
    sensorBand_dbf.to_csv(SensorBandWtshdIntStat_CSV, index=False)

    sensorWtshd_dbf = gpd.read_file(SensorWtshdIntStat_save)
    sensorWtshd_dbf = pd.DataFrame(sensorWtshd_dbf)
    sensorWtshd_dbf.to_csv(SensorWtshdIntStat_CSV, index=False)

    if surveys == "Y":
        surveyBand_dbf = gpd.read_file(SnwSurvBandWtshdIntStat_save)
        surveyBand_dbf = pd.DataFrame(surveyBand_dbf)
        surveyBand_dbf.to_csv(SnwSurvBandWtshdIntStat_CSV, index=False)

        surveyWtshd_dbf = gpd.read_file(SnwSurvWtshdIntStat_save)
        surveyWtshd_dbf = pd.DataFrame(surveyWtshd_dbf)
        surveyWtshd_dbf.to_csv(SnwSurvWtshdIntStat_CSV, index=False)

print('\nProcessing and sorting the sensors ... ')
merge_sort_sensors_surveys(report_date=report_date, results_workspace=WW_results_workspace, surveys="N", difference="N",
                           watershed_shapefile=watershed_shapefile, band_shapefile=band_shapefile, projOut=projOut, merge="Y",
                           domainList=domainList)

print('\nProcessing and sorting the sensors for the Sierra... ')
SNM_results_workspace = rf"M:/SWE/Sierras/Spatial_SWE/SNM_regression/RT_report_data/{report_date}_results_ET/"
SNM_sensors = rf"M:/SWE/WestWide/Spatial_SWE/WW_regression/RT_report_data/{report_date}_results_ET/{report_date}_sensors_SNM.shp"
merge_sort_sensors_surveys(report_date=report_date, results_workspace=SNM_results_workspace, surveys="N", difference="N",
                           watershed_shapefile=watershed_shapefile, band_shapefile=band_shapefile, projOut=projOut, projIn=projIn,
                            merge="N", domain_shapefile=SNM_sensors)
