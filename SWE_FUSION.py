# THIS IS THE CODE THAT'S THE OVERALL FUNCTION FOR MODEL POST PROCESSING
from time import sleep

# import modules
# import modules
import arcpy
import arcpy
from arcpy import env
from arcpy.ra import ZonalStatisticsAsTable
from arcpy.sa import *
import pandas as pd
import geopandas as gpd
import os
# from tables_layers_testing_code import *
from SWE_Fusion_functions import *
# from ASO_Bias_Correction_Functions import *
from ASO_Bias_Correction_Functions import *
from Vetting_functions import *
print('modules imported')
start = time.time()
print(f"\n START TIME: {start}")

# TURN OFF PROCESSING SO IT DOESN'T CRASH FOR NO REASON WHATSOEVER
arcpy.env.parallelProcessingFactor = "0"

######################################
# RUNDATE VARIABLES
######################################
## date info
user = "Olaf"
year = 2026
rundate = "20260101"
survey_date = "20260101"
pillow_date = "01Jan2026"
mean_date = "0101"
prev_rundate = "20260115"

# flags
difference = "N" # should be Y if you want to compare against a previous model run
biasCorrection = "N"
surveys_use = "Y"

# model run information
domainList = ["NOCN", "PNW", "SNM", "SOCN", "INMT"]
ChosenModelRun = "RT_CanAdj_rcn_woCCR_nofscamskSens" ## TEMP
model_wCCR = "RT_CanAdj_rcn_wCCR_nofscamskSens"
model_woCCR = "RT_CanAdj_rcn_woCCR_nofscamskSens"
modelRuns = [model_woCCR, model_wCCR]
model_labels = ["woCCR", "wCCR"]
prev_model_run = "RT_CanAdj_rcn_woCCR_nofscamskSens_UseThis_UseAvg"

######################################
# VETTING VARIABLES
######################################
# bias correction
aso_snotel_data = r"W:/Spatial_SWE/ASO/2025/data_testing/ASO_SNOTEL_DifferenceStats.csv"
currentYear = True
# current_year = datetime.now().year
# year = 2026
methods = ["RECENT", "GRADE", "SENSOR_PATTERN", "GRADES_SPECF"]
grade = "positive"
grade_range = False
grade_amount = -10
sensorTrend = "Mixed"
SNOTEL = "Decreasing"
output_csv = "Y"
csv_outFile = fr"W:/Spatial_SWE/ASO/{year}/data_testing/FracError_data_test.csv"
asoCatalog = fr"W:/Spatial_SWE/ASO/{year}/data_testing/ASO_SNOTEL_DifferenceStats.csv"
basin_List = r"W:/Spatial_SWE/ASO/ASO_Metadata/ASO_in_Basin.txt"
fracErrorWorkspace = f"W:/Spatial_SWE/ASO/{year}/data_testing/"
domain_textFile = r"M:\SWE\WestWide\Spatial_SWE\ASO\ASO_Metadata\State_Basin.txt"
SNM_shapefile_workspace = "M:/SWE/WestWide/data/hydro/SNM/Basin_Shapefiles/"
WW_shapefile_workspace = "M:/SWE/WestWide/data/hydro/WW/ASO_Basin_Shapefiles/"

# vetting
snowTrax_csv = r"J:\Spatial_SWE\SNM_regression\RT_report_data\wsfr_snow.csv"
elevation_tif =r"M:\SWE\WestWide\data\topo\ww_DEM_albn83_feet_banded_100.tif"
aspect_path = r"M:\SWE\WestWide\data\topo\ww_ASPECT_albn83.tif"
reference_col = "SNOW17_SWE_AF"
elev_bins = np.arange(1500, 15000, 100, dtype=float)
swe_col_surv = 'SWE_m'
id_col_surv = 'Station_Na'
# error_statistic = ""

######################################
# WORKSPACES
######################################
meanWorkspace = fr"M:/SWE/WestWide/Spatial_SWE/WW_regression/mean_2000_2025_WY26_glacMask/"
clipbox_WS = "M:/SWE/WestWide/data/boundaries/Domains/DomainShapefiles/"

## results workspaces
SNM_results_workspace = "J:/Spatial_SWE/SNM_regression/RT_report_data/"
WW_results_workspace = "W:/Spatial_SWE/WW_regression/RT_report_data/"

## reporting workspaces
WW_reports_workspace = f"W:/documents/{year}_RT_Reports/"
SNM_reports_workspace = f"J:/paperwork/0_UCSB_DWR_Project/{year}_RT_Reports/"
model_workspace = fr"H:/WestUS_Data/Regress_SWE/"

######################################
# GEOSPATIAL PARAMETERS
######################################
# projections
projGEO = arcpy.SpatialReference(4269)
projALB = arcpy.SpatialReference(102039)
ProjOut_UTM = arcpy.SpatialReference(26911)

# snap rasters
snapRaster_geon83 = "M:/SWE/WestWide/data/boundaries/SnapRaster_geon83.tif"
snapRaster_albn83 = "M:/SWE/WestWide/data/boundaries/SnapRaster_albn83.tif"
SNM_snapRaster_albn83 = "M:/SWE/WestWide/data/boundaries/SNM_SnapRaster_albn83.tif"

# basins tifs and shapefiles
case_field_wtrshd = "SrtName"
case_field_band = "SrtNmeBand"
statelines_file = r"W:\data\basemap\USA_statelines.shp"
## sierras
SNM_band_zones = "M:/SWE/WestWide/data/hydro/SNM/dwr_band_basins_geoSort_albn83_delin.tif"
SNM_watershed_zones = "M:/SWE/WestWide/data/hydro/SNM/dwr_basins_geoSort_albn83.tif"
SNM_watershed_shapefile = "M:/SWE/WestWide/data/hydro/SNM/dwr_basins_albn83_GT5000.shp"
SNM_band_shapefile = "M:/SWE/WestWide/data/hydro/SNM/dwr_band_basins_albn83_delin.shp"
SNM_regions = "M:/SWE/WestWide/data/hydro/SNM/dwr_regions_albn83.tif"
SNM_domain_msk = "M:/SWE/WestWide/data/hydro/SNM/dwr_mask_null_albn83.tif"
SNM_clipbox = "M:/SWE/WestWide/data/boundaries/Domains/DomainShapefiles/WW_SNM_Clipbox_albn83.shp"
## west wide
HUC6_zones = "M:/SWE/WestWide/data/hydro/WW_HUC6_albn83_ras_msked.tif"
region_zones = "M:/SWE/WestWide/data/hydro/WW_Regions_albn83_v2.tif"
WW_band_zones = r"M:\SWE\WestWide\data\hydro\outdated\WW_BasinBanded_noSNM_notahoe_albn83_sel.tif"
WW_watershed_zones = r"M:\SWE\WestWide\data\hydro\outdated\WW_BasinBanded_noSNM_notahoe_albn83_sel.tif"
WW_watershed_shapefile = r"M:\SWE\WestWide\data\hydro\outdated\WW_Basins_noSNM_notahoe_sel_albn83.shp"
WW_band_shapefile = r"M:\SWE\WestWide\data\hydro\outdated\WW_BasinsBanded_noSNM_notahoe_sel_albn83.shp"
# WW_band_zones = "M:/SWE/WestWide/data/hydro/20260113_BasinUpdatesNoSNM/WW_BasinBanded_noSNM_notahoe_albn83_sel_v2.tif"
# WW_watershed_zones = "M:/SWE/WestWide/data/hydro/20260113_BasinUpdatesNoSNM/WW_BasinBanded_noSNM_notahoe_albn83_sel_v2.tif"
# WW_watershed_shapefile = "M:/SWE/WestWide/data/hydro/20260113_BasinUpdatesNoSNM/WW_Basins_noSNM_notahoe_albn83_sel_v2.shp"
# WW_band_shapefile = "M:/SWE/WestWide/data/hydro/20260113_BasinUpdatesNoSNM/WW_BasinsBanded_noSNM_notahoe_albn83_sel_v2.shp"


# masks
watermask = "M:/SWE/WestWide//data/mask/watermask_outdated/outdated/WW/Albers/WW_watermaks_10_albn83_clp_final.tif"
glacierMask = "M:/SWE/WestWide/data/mask/glacierMask/glims_glacierMask_null_GT10_final.tif"

######################################
#SENSOR AND SURVEY DATA
######################################
# surveys paths
survey_workspace = "W:/data/Snow_Instruments/surveys_courses/"
cdec_shapefile = survey_workspace + "20240207_Courses.shp"
basin_list = ["SHASTA RIVER", "SCOTT RIVER", "TRINITY RIVER", "SACRAMENTO RIVER", "MC CLOUD RIVER", "STONY CREEK",
             "FEATHER RIVER", "YUBA RIVER", "AMERICAN RIVER", "MOKELUMNE RIVER", "STANISLAUS RIVER", "TUOLUMNE RIVER",
             "MERCED RIVER", "SAN JOAQUIN RIVER", "KINGS RIVER", "KAWEAH RIVER", "TULE RIVER", "KERN RIVER"]
WW_url_file = survey_workspace + r"stateSurveyURLS.txt"
NRCS_shp = survey_workspace + "20240114_nrcs_course.shp"
WW_state_list = ["AZ", "CO", "ID", "MT", "NV", "NM", "OR", "SD", "UT", "WA", "WY"]
WW_surveys = WW_results_workspace + f"{rundate}_results_ET/{rundate}_surveys_albn83.shp"

# sensors
SNM_sensors = rf"M:/SWE/WestWide/Spatial_SWE/WW_regression/RT_report_data/{rundate}_results_ET/{rundate}_sensors_SNM.shp"

######################################
## SNODAS Variables
######################################
SNODAS_Type = "masked"
WW_NOHRSC_workspace = r"M:/SWE/WestWide/Spatial_SWE/NOHRSC/"
SNM_NOHRSC_workspace = r"M:/SWE/Sierras/Spatial_SWE/NOHRSC/"
Cellsize = "500"

############################################################################################################
# Processing Starts Now
############################################################################################################

# make results and reports directory
os.makedirs(WW_results_workspace + f"/{rundate}_results_ET", exist_ok=True)
print(WW_results_workspace + f"/{rundate}_results_ET")
os.makedirs(SNM_results_workspace + f"/{rundate}_results_ET", exist_ok=True)
print("\nResults directories made")

os.makedirs(WW_reports_workspace + f"/{rundate}_RT_report_ET", exist_ok=True)
os.makedirs(SNM_reports_workspace + f"/{rundate}_RT_report_ET", exist_ok=True)
print("\nReports directories made")

# download surveys if requested
if surveys_use == "Y":
    print("\nGetting SNM Surveys")
    download_cdec_snow_surveys(report_date=rundate, survey_date=survey_date, survey_workspace=survey_workspace,
                               SNM_results_workspace=SNM_results_workspace,
                               cdec_shapefile=cdec_shapefile, basin_list=basin_list)

    print("\nGetting WW Surveys")
    download_snow_surveys(report_date=rundate, survey_date=survey_date, survey_workspace=survey_workspace, results_workspace=WW_results_workspace,
                          WW_url_file=WW_url_file, NRCS_shp=NRCS_shp, WW_state_list=WW_state_list)


# get geopackage converted to shapefile
print("\nProcessing GeoPackage")
geopackage_to_shapefile(report_date=rundate, pillow_date=pillow_date, model_run=model_woCCR,
                        user=user, domainList=domainList, model_workspace=model_workspace,
                        results_workspace=WW_results_workspace + f"/{rundate}_results_ET/")
# clear memory
sleep(30)
clear_arcpy_locks()

# organize and reprocess the sensors for West Wide
print('\nProcessing and sorting the sensors for West Wide ... ')
merge_sort_sensors_surveys(report_date=rundate, results_workspace=WW_results_workspace + f"/{rundate}_results_ET/", surveys=surveys_use, difference=difference,
                           watershed_shapefile=WW_watershed_shapefile, case_field_wtrshd=case_field_wtrshd,
                           case_field_band=case_field_band, band_shapefile=WW_band_shapefile, projOut=projALB, merge="Y",
                           domainList=domainList, prev_report_date=prev_rundate, prev_results_workspace=WW_results_workspace + f"/{prev_rundate}_results_ET/")

# clear memory
sleep(30)
clear_arcpy_locks()

# run SNODAS for West Wide
print("\nSNODAS for WW...")
SNODAS_Processing(report_date=rundate, RunName=model_woCCR, NOHRSC_workspace=WW_NOHRSC_workspace,
                      results_workspace=WW_results_workspace,
                      projin=projGEO, projout=projALB, Cellsize=500, snapRaster=snapRaster_albn83, watermask=watermask,
                      glacierMask=glacierMask,
                      band_zones=WW_band_zones, watershed_zones=WW_watershed_zones, unzip_SNODAS="Y")

# clear memory
sleep(30)
clear_arcpy_locks()

# run tables and layers code for the woCCR model run for West Wide
print(f'\nRunning Tables and Layers Code for all domains for {model_woCCR}')
tables_and_layers(user=user, year=year, report_date=rundate, mean_date = mean_date, meanWorkspace = meanWorkspace, model_run=model_woCCR,
                  masking="N", watershed_zones=WW_watershed_zones, band_zones=WW_band_zones, HUC6_zones=HUC6_zones,
                  region_zones=region_zones, case_field_wtrshd=case_field_wtrshd,case_field_band=case_field_band,
                  watermask=watermask, glacierMask=glacierMask, snapRaster_geon83=snapRaster_geon83, snapRaster_albn83=snapRaster_albn83,
                  projGEO=projGEO, projALB=projALB, ProjOut_UTM=ProjOut_UTM, bias="N")

# clear memory
sleep(30)
clear_arcpy_locks()

# run tables and layers code for the wCCR model run for West Wide
print(f'\nRunning Tables and Layers Code for all domains for {model_wCCR}')
tables_and_layers(user=user, year=year, report_date=rundate, mean_date = mean_date, meanWorkspace = meanWorkspace, model_run=model_wCCR, masking="N", watershed_zones=WW_watershed_zones,
                  band_zones=WW_band_zones, HUC6_zones=HUC6_zones, region_zones=region_zones, case_field_wtrshd=case_field_wtrshd,
                  case_field_band=case_field_band, watermask=watermask, glacierMask=glacierMask, snapRaster_geon83=snapRaster_geon83,
                  snapRaster_albn83=snapRaster_albn83, projGEO=projGEO, projALB=projALB, ProjOut_UTM=ProjOut_UTM, bias="N")

# get zero sensors for all domains
for domain in domainList:
    zero_CCR_sensors(rundate=rundate, results_workspace=WW_results_workspace, pillow_date=pillow_date, domain=domain,
                     sensors=WW_results_workspace + f"{rundate}_results_ET/{rundate}_sensors_{domain}.shp", zero_sensors=True,
                     CCR=False, model_workspace_domain=model_workspace + f"{domain}/{user}/StationSWERegressionV2/data/outputs/{model_wCCR}/")

# clear memory
sleep(30)
clear_arcpy_locks()

# organize and reprocess the sensors for Sierras
print('\nProcessing and sorting the sensors for the Sierra... ')
merge_sort_sensors_surveys(report_date=rundate, results_workspace=SNM_results_workspace + f"/{rundate}_results_ET/", surveys=surveys_use, difference=difference,
                           watershed_shapefile=SNM_watershed_shapefile, case_field_wtrshd=case_field_wtrshd, band_shapefile=SNM_band_shapefile,
                           case_field_band=case_field_band, projOut=projALB, projIn=projGEO, domain = "SNM",
                            merge="N", domain_shapefile=SNM_sensors, prev_report_date=prev_rundate,
                           prev_results_workspace=SNM_results_workspace + f"/{prev_rundate}_results_ET/")

# Run SNODAS for SNM
print("\nSNODAS for SNM...")
SNODAS_Processing(report_date=rundate, RunName=model_woCCR, NOHRSC_workspace=WW_NOHRSC_workspace,
                  results_workspace=SNM_results_workspace,
                  projin=projGEO, projout=projALB, Cellsize=500, snapRaster=SNM_snapRaster_albn83,
                  watermask=watermask, glacierMask=glacierMask,
                  band_zones=SNM_band_zones, watershed_zones=SNM_watershed_zones, unzip_SNODAS="N")

# clear memory
sleep(30)
clear_arcpy_locks()

print(f'\nRunning Tables and Layers Code for Sierra {model_woCCR}...')
tables_and_layers_SNM(year=year, rundate=rundate, mean_date=mean_date, WW_model_run=model_woCCR, SNM_results_workspace=SNM_results_workspace,
                      watershed_zones=SNM_watershed_zones, band_zones=SNM_band_zones, region_zones=SNM_regions,
                      case_field_wtrshd=case_field_wtrshd, case_field_band=case_field_band, watermask=watermask,
                      glacier_mask=glacierMask, domain_mask=SNM_domain_msk, run_type="Normal",
                      snap_raster=SNM_snapRaster_albn83, WW_results_workspace=WW_results_workspace,
                      Difference=difference, prev_report_date=prev_rundate, previous_model_run=prev_model_run)
# clear memory
sleep(30)
clear_arcpy_locks()

print(f'\nRunning Tables and Layers Code for Sierra {model_wCCR}...')
tables_and_layers_SNM(year=year, rundate=rundate, mean_date=mean_date, WW_model_run=model_wCCR, SNM_results_workspace=SNM_results_workspace,
                      watershed_zones=SNM_watershed_zones, band_zones=SNM_band_zones, region_zones=SNM_regions,
                      case_field_wtrshd=case_field_wtrshd, case_field_band=case_field_band, watermask=watermask,
                      glacier_mask=glacierMask, domain_mask=SNM_domain_msk, run_type="Normal",
                      snap_raster=SNM_snapRaster_albn83, WW_results_workspace=WW_results_workspace,
                      Difference=difference, prev_report_date=prev_rundate, previous_model_run=prev_model_run)
# clear memory
clear_arcpy_locks()
sleep(30)

zero_CCR_sensors(rundate=rundate, results_workspace=SNM_results_workspace, pillow_date=pillow_date, domain="SNM",
                     sensors=SNM_results_workspace + f"{rundate}_results_ET/SNM_{rundate}_sensors_albn83.shp", zero_sensors=True,
                     CCR=True, model_workspace_domain=model_workspace + f"SNM/{user}/StationSWERegressionV2/data/outputs/{model_wCCR}/")

############### START VETTING ######################
# loop through domains
for modelRun in modelRuns:
    for domain in domainList:
        if domain == "SNM":
            raster = f"{SNM_results_workspace}/{rundate}_results_ET/{modelRun}/p8_{rundate}_noneg.tif"
            sensors_SNM = SNM_results_workspace + f"{rundate}_results_ET/SNM_{rundate}_sensors_albn83.shp"
            surveys_SNM = SNM_results_workspace + f"{rundate}_results_ET/{rundate}_surveys_albn83.shp"

            ## make vetting folder
            outVettingWS_SNM = SNM_reports_workspace + f"{rundate}_RT_report_ET/{modelRun}/vetting_domains/"
            os.makedirs(outVettingWS_SNM, exist_ok=True)

            # fsca variables
            fSCA_raster = f"{SNM_results_workspace}/{rundate}_results_ET/{model_woCCR}/SNM_fSCA_{rundate}_albn83.tif"
            prev_vetting_WS = f"J:/paperwork/0_UCSB_DWR_Project/{year}_RT_Reports/{prev_rundate}_RT_report_ET/{prev_model_run}/vetting_domains/"
            vetting_WS = f"J:/paperwork/0_UCSB_DWR_Project/{year}_RT_Reports/{rundate}_RT_report_ET/{ChosenModelRun}/vetting_domains/"
            prev_raster = prev_vetting_WS + f"SNM_fSCA_{prev_rundate}_albn83.tif"

            # move p8 to vetting space
            arcpy.CopyRaster_management(raster, outVettingWS_SNM + f"p8_{rundate}_noneg.tif")

            # move fsca to vetting space
            arcpy.CopyRaster_management(fSCA_raster, outVettingWS_SNM + f"SNM_fSCA_{rundate}_albn83.tif")

            # add in snowTrax comparison
            snowtrax_comparision(rundate=rundate, snowTrax_csv=snowTrax_csv, results_WS=SNM_results_workspace,
                                 output_csv = outVettingWS_SNM + f"{rundate}_snowTrax_comparison.csv", model_list=modelRuns,
                                 model_labels=model_labels, reference_col=reference_col, output_png= SNM_reports_workspace + f"{rundate}_RT_report_ET/{rundate}_snowTrax_comparison.png")

        else:
            # extract by mask
            arcpy.env.snapRaster = snapRaster_albn83
            arcpy.env.cellSize = snapRaster_albn83
            raster = f"{WW_results_workspace}/{rundate}_results_ET/{modelRun}/p8_{rundate}_noneg.tif"
            fSCA_raster = f"{WW_results_workspace}/{rundate}_results_ET/{model_woCCR}/fSCA_{rundate}_albn83.tif"
            prev_vetting_WS = f"W:/documents/2026_RT_Reports/{prev_rundate}_RT_report_ET/{prev_model_run}/vetting_domains/"
            sensors_WW = WW_results_workspace + f"{rundate}_results_ET/{rundate}_sensors_albn83.shp"
            surveys_WW = WW_results_workspace + f"{rundate}_results_ET/{rundate}_surveys_albn83.shp"
            prev_fSCA_raster = prev_vetting_WS + f"fSCA_{prev_rundate}_{domain}_clp.tif"

            ## make vetting folder
            outVettingWS_WW = f"{WW_reports_workspace}/{rundate}_RT_report_ET/{modelRun}/vetting_domains/"
            os.makedirs(outVettingWS_WW, exist_ok=True)
            outMask = ExtractByMask(raster, clipbox_WS + f"WW_{domain}_Clipbox_albn83.shp")
            outMask.save(outVettingWS_WW + f"p8_{rundate}_noneg_{domain}_clp.tif")
            print(f"{domain} clipped and saved")

            ## make vetting folder
            outMask = ExtractByMask(fSCA_raster, clipbox_WS + f"WW_{domain}_Clipbox_albn83.shp")
            outMask.save(outVettingWS_WW + f"fSCA_{rundate}_{domain}_clp.tif")
            print(f"{domain} clipped and saved for FSCA")

    for domain in domainList:
        if domain == "SNM":
            print('domain is SNM')
            raster = outVettingWS_SNM + f"p8_{rundate}_noneg.tif"
            if surveys_use == "Y":


                model_domain_vetting(raster=raster, point=surveys_SNM, swe_col=swe_col_surv, id_col=id_col_surv,
                                     rundate=rundate, domain=domain, modelRun=modelRun,
                                     out_csv=SNM_reports_workspace + f"{rundate}_RT_report_ET/{rundate}_surveys_error.csv")

            swe_col_sens = 'pillowswe'
            id_col_sens = 'Site_ID'
            model_domain_vetting(raster=raster, point=sensors_SNM, swe_col=swe_col_sens, id_col=id_col_sens,
                                 rundate=rundate, domain=domain,
                                 modelRun=modelRun, out_csv=SNM_reports_workspace + f"{rundate}_RT_report_ET/{rundate}_sensors_error.csv")
        else:
            raster = outVettingWS_WW + f"p8_{rundate}_noneg_{domain}_clp.tif"
            swe_col_sens = 'pillowswe'
            id_col_sens = 'Site_ID'
            if surveys_use == "Y":


                model_domain_vetting(raster=raster, point=surveys_WW, swe_col=swe_col_surv, id_col=id_col_surv, rundate=rundate,
                                     domain=domain, modelRun=modelRun, out_csv=f"{WW_reports_workspace}/{rundate}_RT_report_ET/{rundate}_surveys_error.csv")

            model_domain_vetting(raster=raster, point=sensors_WW, swe_col=swe_col_sens, id_col=id_col_sens, rundate=rundate, domain=domain,
                                 modelRun=modelRun, out_csv=f"{WW_reports_workspace}/{rundate}_RT_report_ET/{rundate}_sensors_error.csv")

#####
#PROMPT TO PICK THE BEST ERROR FROM THE CSV

if difference == "Y":
    print("Running Plots...")
    for modelRun in modelRuns:
        for domain in domainList:

            if domain == "SNM":
                print('analyzing Sierras')
                prev_vetting_WS = f"J:/paperwork/0_UCSB_DWR_Project/{year}_RT_Reports/{prev_rundate}_RT_report_ET/{prev_model_run}/vetting_domains/"
                vetting_WS = f"J:/paperwork/0_UCSB_DWR_Project/{year}_RT_Reports/{rundate}_RT_report_ET/{modelRun}/vetting_domains/"
                fSCA_vetting_WS = f"J:/paperwork/0_UCSB_DWR_Project/{year}_RT_Reports/{rundate}_RT_report_ET/"
                raster = vetting_WS + f"p8_{rundate}_noneg.tif"
                fSCA_raster = vetting_WS + f"SNM_fSCA_{rundate}_albn83.tif"
                prev_raster = prev_vetting_WS + f"p8_{prev_rundate}_noneg.tif"
                prev_fSCA_raster = prev_vetting_WS + f"SNM_fSCA_{prev_rundate}_albn83.tif"
                point_dataset = fr"M:\SWE\Sierras\Spatial_SWE\SNM_regression\RT_report_data\{rundate}_results_ET\SNM_{rundate}_sensors_albn83.shp"
                prev_pointDataset = fr"M:\SWE\Sierras\Spatial_SWE\SNM_regression\RT_report_data\{prev_rundate}_results_ET\SNM_{prev_rundate}_sensors_albn83.shp"

                sensor_difference_map(rundate=rundate, prev_rundate=prev_rundate,
                                      sensors=rf"{WW_results_workspace}/{rundate}_results_ET/{rundate}_sensors_{domain}.shp",
                                      prev_sensors=rf"{WW_results_workspace}/{prev_rundate}_results_ET/{prev_rundate}_sensors_{domain}.shp",
                                      domain=domain, point_value='pillowswe',
                                      outfile=SNM_reports_workspace + f"{rundate}_RT_report_ET/{domain}_{rundate}_{prev_rundate}_sensor_diff.png",
                                      basemap_file_1=statelines_file, basemap_name_1="State Lines",
                                      basemap_file_2=SNM_watershed_shapefile, basemap_name_2="Watersheds")


            else:
                print(f"analyzing {domain}")
                prev_vetting_WS = f"W:/documents/{year}_RT_Reports/{prev_rundate}_RT_report_ET/{prev_model_run}/vetting_domains/"
                vetting_WS = f"W:/documents/{year}_RT_Reports/{rundate}_RT_report_ET/{modelRun}/vetting_domains/"
                fSCA_vetting_WS = f"W:/documents/{year}_RT_Reports/{rundate}_RT_report_ET/"
                raster = vetting_WS + f"p8_{rundate}_noneg_{domain}_clp.tif"
                fSCA_raster = f"{WW_results_workspace}/{rundate}_results_ET/{model_woCCR}/fSCA_{rundate}_albn83.tif"
                prev_raster = prev_vetting_WS + f"p8_{prev_rundate}_noneg_{domain}_clp.tif"
                prev_fSCA_raster = prev_vetting_WS + f"fSCA_{prev_rundate}_{domain}_clp.tif"
                point_dataset = fr"M:\SWE\WestWide\Spatial_SWE\WW_regression\RT_report_data\{rundate}_results_ET\{rundate}_sensors_albn83.shp"
                prev_pointDataset = fr"M:\SWE\WestWide\Spatial_SWE\WW_regression\RT_report_data\{prev_rundate}_results_ET\{prev_rundate}_sensors_albn83.shp"

                sensor_difference_map(rundate=rundate, prev_rundate=prev_rundate,
                                      sensors=rf"{WW_results_workspace}/{rundate}_results_ET/{rundate}_sensors_{domain}.shp",
                                      prev_sensors=rf"{WW_results_workspace}/{prev_rundate}_results_ET/{prev_rundate}_sensors_{domain}.shp",
                                      domain=domain, point_value='pillowswe',
                                      outfile=SNM_reports_workspace + f"{rundate}_RT_report_ET/{domain}_{rundate}_{prev_rundate}_sensor_diff.png",
                                      basemap_file_1=statelines_file, basemap_name_1="State Lines",
                                      basemap_file_2=WW_watershed_shapefile, basemap_name_2="Watersheds")

            ## engage plots
            print("Plotting pillow change comparison...")
            pillow_date_comparison(rundate=rundate, prev_model_date=prev_rundate, raster=raster,
                                   point_dataset=point_dataset,
                                   prev_pointDataset=prev_pointDataset, id_column="Site_ID", swe_col="pillowswe",
                                   elev_col="dem",
                                   output_png=vetting_WS + f"{domain}_sensor_difference.png", convert_meters_feet="Y")

            print('Creating box and whiskers plot...')
            raster_box_whisker_plot(rundate=rundate, prev_model_date=prev_rundate, raster=raster, prev_raster=prev_raster,
                                    domain=domain, variable="SWE", unit="m", output_png=vetting_WS + f"{domain}_{rundate}_box_whisker.png")

            print('Creating elevation step plot...')
            swe_elevation_step_plot(rundate=rundate, prev_model_date=prev_rundate, domain=domain, raster=raster, prev_raster=prev_raster,
                                    output_png=vetting_WS + f"{domain}_{rundate}_elevation_step.png",
                                    elevation_tif=elevation_tif,
                                    elev_bins=elev_bins, variable="SWE", unit="m")

            print('Creating aspect compass...')
            create_aspect_comparison(aspect_path=aspect_path, raster=raster, prev_raster=prev_raster, label_1=rundate,
                                     label_2=prev_rundate, title="Difference of SWE", variable='SWE', unit="m",
                                     output_path=vetting_WS + f"{rundate}_{domain}_aspect_comparison.png", num_bins=16)


            print('Creating box and whiskers plot...')
            raster_box_whisker_plot(rundate=rundate, prev_model_date=prev_rundate, raster=fSCA_raster, prev_raster=prev_fSCA_raster,
                                    domain=domain, variable="fSCA", unit="%", output_png=fSCA_vetting_WS + f"{domain}_{rundate}_fSCA_box_whisker.png")

            print('Creating elevation step plot...')
            swe_elevation_step_plot(rundate=rundate, prev_model_date=prev_rundate, domain=domain, raster=fSCA_raster, prev_raster=prev_fSCA_raster,
                                    output_png=fSCA_vetting_WS + f"{domain}_{rundate}_fSCA_elevation_step.png",
                                    elevation_tif=elevation_tif,
                                    elev_bins=elev_bins, variable="fSCA", unit="%")

            print('Creating aspect compass...')
            create_aspect_comparison(aspect_path=aspect_path, raster=fSCA_raster, prev_raster=prev_fSCA_raster, label_1=rundate,
                                 label_2=prev_rundate, title="Difference of fSCA", variable='fSCA', unit="%",
                                     output_path=fSCA_vetting_WS + f"{rundate}_{domain}_fSCA_aspect_comparison.png", num_bins=16)


## ERIC: prompt for best model run with sensor counts and % error
sensors_SNM = SNM_results_workspace + f"{rundate}_results_ET/SNM_{rundate}_sensors_albn83.shp"
sensors_WW = WW_results_workspace + f"{rundate}_results_ET/{rundate}_sensors_albn83.shp"
if biasCorrection == "Y":

    out_csv = rf"W:/Spatial_SWE/ASO/ASO_Metadata/{rundate}_ASO_biasCorrection_stats.csv"
    # list of methods
    results_df = bias_correction_selection(rundate=rundate, aso_snotel_data = aso_snotel_data, basin_List=basin_List, domainList=domain_textFile, method_list=method_list,
                                           fracErrorWorkspace=fracErrorWorkspace, output_csv=output_csv, csv_outFile=csv_outFile,
                                           currentYear=True, year=year, grade_amount=grade_amount, sensorTrend=sensorTrend, SNOTEL=SNOTEL,
                                           grade=grade, grade_range=grade_range)
    results_df.to_csv(out_csv, index=False)

    #################################
    # BIAS CORRECTION CODE FOR WW
    #################################
    for method in methods:
        print(f"\nProcessing method:", method)
        bias_correct(WW_results_workspace + f"{rundate}_results_ET/", domain="WW", ModelRun=ChosenModelRun, method=method, rundate=rundate, results_df=results_df, shapefile_workspace=WW_shapefile_workspace)

    # got through methods to find the best version for vetting
    prefix = rundate
    unique_names = set()  # use a set to keep unique values
    file_mapping = {}
    for root, dirs, files in os.walk(rf"W:/Spatial_SWE/WW_regression/RT_report_data/{rundate}_results_ET/ASO_BiasCorrect_{ChosenModelRun}/"):
        for file in files:
            if file.startswith(prefix):

                # Split by "_" and take the first two parts
                parts = file.split("_")
                if len(parts) >= 2:
                    name = "_".join(parts[:2])
                    unique_names.add(name)
                    print(name)

    # Convert to list if needed
    unique_names = list(unique_names)
    print(unique_names)

    print("\nFull file names by prefix:")
    for name, files in file_mapping.items():
        print(f"{name}:")
        for f in files:
            print(f"  {f}")

    control_raster_WW = rf"M:/SWE/WestWide/Spatial_SWE/WW_regression/RT_report_data/{rundate}_results_ET/{ChosenModelRun}/p8_{rundate}_noneg.tif"
    os.makedirs(f"W:/documents/{year}_RT_Reports/{rundate}_RT_report_ET/ASO_BiasCorrect_{ChosenModelRun}/", exist_ok=True)

    WW_out_csv_vetting = f"W:/documents/{year}_RT_Reports/{rundate}_RT_report_ET/ASO_BiasCorrect_{ChosenModelRun}/{rundate}_ASO_bias_correction_stats.csv"
    for method in methods:
        print(f"\nMethod: {method}"'')
        BC_path = rf"W:/Spatial_SWE/WW_regression/RT_report_data/{rundate}_results_ET/ASO_BiasCorrect_{ChosenModelRun}/{method}/"
        control_out_folder = rf"W:/Spatial_SWE/WW_regression/RT_report_data/{rundate}_results_ET/ASO_BiasCorrect_{ChosenModelRun}/"

        for name in unique_names:
            print(f"Name: {name}")
            raster = BC_path + f"{name}_{method}_BC_fix_albn83.tif"

            if os.path.exists(raster):
                bias_correction_vetting(raster=raster, point=sensors_WW, domain="WW", swe_col="pillowswe",
                                        id_col="Site_ID", rundate=rundate,
                                        name=name, method=method, out_csv=WW_out_csv_vetting, folder=BC_path,
                                        control_out_folder=control_out_folder, control_raster=control_raster_WW)
            else:
                print(f'{raster} RASTER DOES NOT EXISTS')

    # figures and vetting
    df = pd.read_csv(WW_out_csv_vetting)
    aso_df = df[df["Domain"] == "WW"]

    # get a list of unique values
    basins_bc = aso_df["Basin"].unique()

    for basin in basins_bc:
        print(basin)
        file_paths = []
        labels = []

        # get control file
        control_list = os.listdir(f"M:/SWE/WestWide/Spatial_SWE/WW_regression/RT_report_data/{rundate}_results_ET/ASO_BiasCorrect_{ChosenModelRun}/")
        for file in control_list:
            if file.startswith(f"{rundate}_{basin}") and file.endswith("Control_clp.tif"):
                file_paths.append(f"M:/SWE/WestWide/Spatial_SWE/WW_regression/RT_report_data/{rundate}_results_ET/ASO_BiasCorrect_{ChosenModelRun}/" + file)
                labels.append("Control")

        # get bias corrected
        for method in methods:
            for bc_file in os.listdir(f"M:/SWE/WestWide/Spatial_SWE/WW_regression/RT_report_data/{rundate}_results_ET/ASO_BiasCorrect_{ChosenModelRun}/" + f"{method}/"):
                if bc_file.startswith(f"{rundate}_{basin}") and bc_file.endswith(f"{method}_BC_fix_albn83.tif"):
                    file_paths.append(f"M:/SWE/WestWide/Spatial_SWE/WW_regression/RT_report_data/{rundate}_results_ET/ASO_BiasCorrect_{ChosenModelRun}/" + f"{method}/{bc_file}")
                    labels.append(method)

        # get metadata
        print(len(file_paths))

        # get box and whiskers plot
        if len(file_paths) < 2:
            print("Skipping — not enough rasters")
            continue

        output_png = f"W:/documents/{year}_RT_Reports/{rundate}_RT_report_ET/ASO_BiasCorrect_{ChosenModelRun}/{rundate}_{basin}_SWE_boxplot.png"

        raster_box_whisker_plot_multi(
            rundate=rundate,
            raster_paths=file_paths,
            labels=labels,
            domain=basin,
            variable="SWE",
            unit="m",
            output_png=output_png
        )
        # plot rasters
        titles = ["CONTROL"] + methods[:len(file_paths) - 1]

        plot_rasters_side_by_side(
            rundate=rundate,
            basin=basin ,
            raster_paths=file_paths,
            titles=labels,
            variable="SWE",
            unit="m",
            output_png=f"W:/documents/{year}_RT_Reports/{rundate}_RT_report_ET/ASO_BiasCorrect_{ChosenModelRun}/{rundate}_{basin}_SWE_maps.png"
        )

    #################################
    # BIAS CORRECTION CODE FOR SNM
    #################################
    for method in methods:
        print(f"\nprocessing method:", method)
        bias_correct(results_workspace=SNM_results_workspace + f"{rundate}_results_ET/", domain="SNM", ModelRun=ChosenModelRun, method=method, rundate=rundate, results_df=results_df, shapefile_workspace=SNM_shapefile_workspace)

    # got through methods to find the best version for vetting
    folder = rf"J:/Spatial_SWE/SNM_regression/RT_report_data/{rundate}_results_ET/ASO_BiasCorrect_{ChosenModelRun}/"
    prefix = rundate
    unique_names = set()  # use a set to keep unique values
    file_mapping = {}
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.startswith(prefix):

                # Split by "_" and take the first two parts
                parts = file.split("_")
                if len(parts) >= 2:
                    name = "_".join(parts[:2])
                    unique_names.add(name)

    # Convert to list if needed
    unique_names = list(unique_names)
    print(unique_names)

    print("\nFull file names by prefix:")
    for name, files in file_mapping.items():
        print(f"{name}:")
        for f in files:
            print(f"  {f}")

    methods = ["RECENT", "GRADE", "SENSOR_PATTERN", "GRADES_SPECF"]
    control_raster_SNM = rf"M:/SWE/Sierras/Spatial_SWE/SNM_regression/RT_report_data/{rundate}_results_ET/{ChosenModelRun}/p8_{rundate}_noneg.tif"
    os.makedirs(f"J:/paperwork/0_UCSB_DWR_Project/{year}_RT_Reports/{rundate}_RT_report_ET/ASO_BiasCorrect_{ChosenModelRun}/", exist_ok=True)
    SNM_out_csv_vetting = f"J:/paperwork/0_UCSB_DWR_Project/{year}_RT_Reports/{rundate}_RT_report_ET/ASO_BiasCorrect_{ChosenModelRun}/{rundate}_ASO_error_stats.csv"
    for method in methods:
        print(f"\nMethod: {method}")
        BC_path = rf"J:/Spatial_SWE/SNM_regression/RT_report_data/{rundate}_results_ET/ASO_BiasCorrect_{ChosenModelRun}/{method}/"

        for name in unique_names:
            print(f"Name: {name}")
            raster = BC_path + f"{name}_{method}_BC_fix_albn83.tif"

            if os.path.exists(raster):
                bias_correction_vetting(raster=raster, point=sensors_SNM, domain="SNM", swe_col="pillowswe",
                                        id_col="Site_ID", rundate=rundate,
                                        name=name, method=method, out_csv=SNM_out_csv_vetting, folder=BC_path, control_out_folder=rf"J:/Spatial_SWE/SNM_regression/RT_report_data/{rundate}_results_ET/ASO_BiasCorrect_{ChosenModelRun}/",
                                        control_raster=control_raster_SNM)
            else:
                print(f'{raster} RASTER DOES NOT EXISTS')

    # figures and vetting
    df = pd.read_csv(SNM_out_csv_vetting)
    aso_df = df[df["Domain"] == "SNM"]

    # get a list of unique values
    basins_bc = aso_df["Basin"].unique()

    for basin in basins_bc:
        print(basin)
        file_paths = []
        labels = []

        # get control file
        control_list = os.listdir(f"J:/Spatial_SWE/SNM_regression/RT_report_data/{rundate}_results_ET/ASO_BiasCorrect_{ChosenModelRun}/")
        for file in control_list:
            if file.startswith(f"{rundate}_{basin}") and file.endswith("Control_clp.tif"):
                file_paths.append(f"J:/Spatial_SWE/SNM_regression/RT_report_data/{rundate}_results_ET/ASO_BiasCorrect_{ChosenModelRun}/" + file)
                labels.append("Control")

        # get bias corrected
        for method in methods:
            for bc_file in os.listdir(f"J:/Spatial_SWE/SNM_regression/RT_report_data/{rundate}_results_ET/ASO_BiasCorrect_{ChosenModelRun}/{method}"):
                if bc_file.startswith(f"{rundate}_{basin}") and bc_file.endswith(f"{method}_BC_fix_albn83.tif"):
                    file_paths.append(f"J:/Spatial_SWE/SNM_regression/RT_report_data/{rundate}_results_ET/ASO_BiasCorrect_{ChosenModelRun}/" + f"{method}/{bc_file}")
                    labels.append(method)

        # get metadata
        print(len(file_paths))

        # get box and whiskers plot
        if len(file_paths) < 2:
            print("Skipping — not enough rasters")
            continue

        output_png = f"J:/paperwork/0_UCSB_DWR_Project/{year}_RT_Reports/{rundate}_RT_report_ET/ASO_BiasCorrect_{ChosenModelRun}/{rundate}_{basin}_SWE_boxplot.png"

        raster_box_whisker_plot_multi(
            rundate=rundate,
            raster_paths=file_paths,
            labels=labels,
            domain=basin,
            variable="SWE",
            unit="m",
            output_png=output_png
        )
        # plot rasters
        titles = ["CONTROL"] + methods[:len(file_paths) - 1]

        plot_rasters_side_by_side(
            rundate=rundate,
            basin=basin ,
            raster_paths=file_paths,
            titles=labels,
            variable="SWE",
            unit="m",
            output_png=f"J:/paperwork/0_UCSB_DWR_Project/{year}_RT_Reports/{rundate}_RT_report_ET/ASO_BiasCorrect_{ChosenModelRun}/{rundate}_{basin}_SWE_maps.png"
        )

    ## pick the best file
    print("\n Choosing and mosaic for WW...")
    aso_choice_and_mosaic(rundate=rundate, aso_error_csv=WW_out_csv_vetting, error_metric="MAE", aso_region="WW",
                          bias_correction_workspace=f"M:/SWE/WestWide/Spatial_SWE/WW_regression/RT_report_data/{rundate}_results_ET/ASO_BiasCorrect_{ChosenModelRun}/",
                          snapRaster=snapRaster_albn83, control_raster=f"M:/SWE/WestWide/Spatial_SWE/WW_regression/RT_report_data/{rundate}_results_ET/ASO_BiasCorrect_{ChosenModelRun}/p8_{rundate}_noneg.tif")

    print("\n Choosing and mosaic for SNM...")
    aso_choice_and_mosaic(rundate=rundate, aso_error_csv=SNM_out_csv_vetting, error_metric="MAE", aso_region="SNM",
                          bias_correction_workspace=f"M:/SWE/Sierras/Spatial_SWE/SNM_regression/RT_report_data/{rundate}_results_ET/ASO_BiasCorrect_{ChosenModelRun}/",
                          snapRaster=snapRaster_albn83,
                          control_raster=f"M:/SWE/Sierras/Spatial_SWE/SNM_regression/RT_report_data/{rundate}_results_ET/ASO_BiasCorrect_{ChosenModelRun}/p8_{rundate}_noneg.tif")

    ###### TO DO ##############3
    # mosaic and run vetting again

# run tables code
SNM_tables_for_report(rundate=rundate, modelRunName=ChosenModelRun, averageRunName=model_woCCR, results_workspace=SNM_results_workspace + f"{rundate}_results_ET/",
                     reports_workspace=SNM_reports_workspace + f"{rundate}_RT_report_ET/", difference="N")

WW_tables_for_report(rundate=rundate, modelRunName=ChosenModelRun, averageRunName=model_woCCR, results_workspace=WW_results_workspace + f"{rundate}_results_ET/",
                     reports_workspace=WW_reports_workspace + f"{rundate}_RT_report_ET/", difference="N")

# Run vetting code
end = time.time()
time_elapsed = (end - start) /60
print(f"\n Elapsed Time: {time_elapsed} minutes")