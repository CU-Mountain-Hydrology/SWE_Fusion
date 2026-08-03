# run/sensor_processing.py

import arcpy

from config import Config
from run.utils import get_previous_model_run, get_water_year
from SWE_Fusion_functions import merge_sort_sensors_surveys

def sensor_survey_processing(date: int, domain: str, surveys_use: str, cfg: Config):
    """
    This is a wrapper to call the merge_sort_sensor_survey() function from SWE_Fusion_functions.py,
    with added error handling. Eventually I would like to refactor the function entirely and then this won't be needed.

    :param date:
    :param domain:
    :param surveys_use:
    :param cfg:
    """
    if domain == "WW":
        # Get previous run date to know if differencing should be used
        prev_ww_rundate, _ = get_previous_model_run(date=date, domain="WW", cfg=cfg)

        if get_water_year(prev_ww_rundate) == get_water_year(date):
            difference = "Y"  # If the previous report date was from this water year, then difference="Y" (compare to last report)
        else:
            difference = "N"

        # Process & Sort WW sensors & surveys
        merge_sort_sensors_surveys(
            report_date=str(date), results_workspace=f"{cfg.ww_results_workspace}/{date}_results/",
            surveys=surveys_use, difference=difference,
            watershed_shapefile=cfg.ww_watershed_shapefile, case_field_wtrshd=cfg.case_field_watershed,
            case_field_band=cfg.case_field_band, band_shapefile=cfg.ww_band_shapefile,
            projOut=arcpy.SpatialReference(cfg.proj_alb), merge="Y",
            domainList=cfg.domain_list, prev_report_date=str(prev_ww_rundate),
            prev_results_workspace=f"{cfg.ww_results_workspace}/{prev_ww_rundate}_results/")

    elif domain == "SNM":
        # Get previous run date to know if differencing should be used
        prev_snm_rundate, _ = get_previous_model_run(date=date, domain="SNM", cfg=cfg)

        if get_water_year(prev_snm_rundate) == get_water_year(date):
            difference = "Y"  # If the previous report date was from this water year, then difference="Y" (compare to last report)
        else:
            difference = "N"

        merge_sort_sensors_surveys(
            report_date=str(date), results_workspace=f"{cfg.snm_results_workspace}/{date}_results/",
            surveys=surveys_use, difference=difference,
            watershed_shapefile=cfg.snm_watershed_shapefile, band_shapefile=cfg.snm_band_shapefile,
            case_field_wtrshd=cfg.case_field_watershed, case_field_band=cfg.case_field_band,
            projOut=arcpy.SpatialReference(cfg.proj_alb), projIn=arcpy.SpatialReference(cfg.proj_geo),
            domain="SNM", merge="N", domain_shapefile=cfg.snm_sensors_shp.format(date=date),
            prev_report_date=str(prev_snm_rundate),
            prev_results_workspace=f"{cfg.snm_results_workspace}/{prev_snm_rundate}_results/")

    else:
        raise ValueError(f"Domain {domain} not recognized! Must be 'WW' or 'SNM'.")


# TODO: refactor the following function pulled from SWE_Fusion_functions.py to work directly into pipeline
"""
def merge_sort_sensors_surveys(report_date, results_workspace, surveys, difference, watershed_shapefile, case_field_wtrshd,
                               band_shapefile, case_field_band, merge, projOut, projIn=None, domainList=None, domain= None, domain_shapefile=None,
                               prev_report_date=None, prev_results_workspace=None):

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
        arcpy.Merge_management([results_workspace + f"{report_date}_sensors_{domainList[0]}.shp",
                                results_workspace + f"{report_date}_sensors_{domainList[1]}.shp",
                                results_workspace + f"{report_date}_sensors_{domainList[2]}.shp",
                                results_workspace + f"{report_date}_sensors_{domainList[3]}.shp",
                                results_workspace + f"{report_date}_sensors_{domainList[4]}.shp"], snowPillow_merge)

        # delete duplicates
        arcpy.DeleteIdentical_management(snowPillow_merge, "Site_ID")

        # reproject to Albers
        arcpy.Project_management(snowPillow_merge, snowPillow_proj, projOut)

    if merge == "N":
        snowPillow_proj = results_workspace + f"SNM_{report_date}_sensors_albn83.shp"
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
    arcpy.Delete_management("in-memory")
    gc.collect()

    arcpy.AddField_management(SensorWtshdIntStat, "SWE_freq", "TEXT", "#", "#",
                              "#", "#", "NULLABLE", "NON_REQUIRED", "#")
    arcpy.Delete_management("in-memory")
    gc.collect()

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
    arcpy.Delete_management("in-memory")
    gc.collect()

    arcpy.AddField_management(SensorBandWtshdIntStat, "SWE_freq", "TEXT", "#", "#",
                              "#", "#", "NULLABLE", "NON_REQUIRED",
                              "#")
    arcpy.Delete_management("in-memory")
    gc.collect()
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

        if domain == "SNM":

            lastPillowView = prev_results_workspace + f"SNM_{prev_report_date}_sensors_view.dbf"
            lastPillow = prev_results_workspace + f"SNM_{prev_report_date}_sensors_albn83.shp"

        else:
            lastPillowView = prev_results_workspace + f"{prev_report_date}_sensors_view.dbf"
            lastPillow = prev_results_workspace + f"{prev_report_date}_sensors_albn83.shp"

        arcpy.MakeTableView_management(lastPillow, lastPillowView)
        arcpy.TableToTable_conversion(lastPillowView, results_workspace, f"{report_date}_temp.csv")
        temp_df = pd.read_csv(results_workspace + f"{report_date}_temp.csv")
        temp_df = temp_df[["Site_ID", "SWE_In"]]
        temp_df.rename(columns={"SWE_In": "LastSWE_in"}, inplace=True)

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

"""