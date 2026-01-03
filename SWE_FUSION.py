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
from Vetting_functions import *
print('modules imported')
start = time.time()
print(f"\n START TIME: {start}")

# tables and layers -- establish paths
user = "Olaf"
year = "2025"
rundate = "20251227"
pillow_date = "27Dec2025"
mean_date = "1227"
# prev_rundate = "20250517"
model_wCCR = "RT_CanAdj_rcn_wCCR_nofscamskSens_testReport"
model_woCCR = "RT_CanAdj_rcn_woCCR_nofscamskSens_testReport"
SNM_mask = "M:/SWE/WestWide/data/hydro/SNM/dwr_mask_null_albn83.tif"
# model_run = "RT_CanAdj_rcn_noSW_woCCR_nofscamsk"
# prev_model_run = "ASO_FixLayers_fSCA_RT_CanAdj_rcn_noSW_woCCR_UseThis"
# masking = "N"
# bias = "N"
surveys_use = "N"
snapRaster_geon83 = "M:/SWE/WestWide/data/boundaries/SnapRaster_geon83.tif"
snapRaster_albn83 = "M:/SWE/WestWide/data/boundaries/SnapRaster_albn83.tif"
SNM_snapRaster_albn83 = "M:/SWE/WestWide/data/boundaries/SNM_SnapRaster_albn83.tif"
projGEO = arcpy.SpatialReference(4269)
projALB = arcpy.SpatialReference(102039)
ProjOut_UTM = arcpy.SpatialReference(26911)
meanWorkspace = fr"M:/SWE/WestWide/Spatial_SWE/WW_regression/mean_2000_2025_WY26_glacMask/"
HUC6_zones = "M:/SWE/WestWide/data/hydro/WW_HUC6_albn83_ras_msked.tif"
region_zones = "M:/SWE/WestWide/data/hydro/WW_Regions_albn83_v2.tif"
watermask = "M:/SWE/WestWide//data/mask/watermask_outdated/outdated/WW/Albers/WW_watermaks_10_albn83_clp_final.tif"
glacierMask = "M:/SWE/WestWide/data/mask/glacierMask/glims_glacierMask_null_GT10_final.tif"
WW_band_zones = "M:/SWE/WestWide/data/hydro/WW_BasinBanded_noSNM_notahoe_albn83_sel.tif"
WW_watershed_zones = "M:/SWE/WestWide/data/hydro/WW_BasinBanded_noSNM_notahoe_albn83_sel.tif"
SNM_band_zones = "M:/SWE/WestWide/data/hydro/SNM/dwr_band_basins_geoSort_albn83_delin.tif"
SNM_watershed_zones = "M:/SWE/WestWide/data/hydro/SNM/dwr_basins_geoSort_albn83.tif"
WW_watershed_shapefile = "M:/SWE/WestWide/data/hydro/WW_Basins_noSNM_notahoe_albn83_sel_new.shp"
WW_band_shapefile = "M:/SWE/WestWide/data/hydro/WW_BasinsBanded_noSNM_notahoe_albn83_sel_new.shp"
SNM_watershed_shapefile = "M:/SWE/WestWide/data/hydro/SNM/dwr_basins_albn83_GT5000.shp"
SNM_band_shapefile = "M:/SWE/WestWide/data/hydro/SNM/dwr_band_basins_albn83_delin.shp"
SNM_regions = "M:/SWE/WestWide/data/hydro/SNM/dwr_regions_albn83.tif"
SNM_domain_msk = "M:/SWE/WestWide/data/hydro/SNM/dwr_mask_null_albn83.tif"
domainList = ["NOCN", "PNW", "SNM", "SOCN", "INMT"]
SNM_clipbox = "M:/SWE/WestWide/data/boundaries/Domains/DomainShapefiles/WW_SNM_Clipbox_albn83.shp"

## results workspaces
SNM_results_workspace = "J:/Spatial_SWE/SNM_regression/RT_report_data/"
WW_results_workspace = "W:/Spatial_SWE/WW_regression/RT_report_data/"

## reporting workspaces
WW_reports_workspace = f"W:/documents/{year}_RT_Reports/"
SNM_reports_workspace = f"J:/paperwork/0_UCSB_DWR_Project/{year}_RT_Reports/"

# surveys paths
survey_workspace = "W:/data/Snow_Instruments/surveys_courses/"
cdec_shapefile = survey_workspace + "20240207_Courses.shp"
basin_list = ["SHASTA RIVER", "SCOTT RIVER", "TRINITY RIVER", "SACRAMENTO RIVER", "MC CLOUD RIVER", "STONY CREEK",
             "FEATHER RIVER", "YUBA RIVER", "AMERICAN RIVER", "MOKELUMNE RIVER", "STANISLAUS RIVER", "TUOLUMNE RIVER",
             "MERCED RIVER", "SAN JOAQUIN RIVER", "KINGS RIVER", "KAWEAH RIVER", "TULE RIVER", "KERN RIVER"]

WW_url_file = survey_workspace + r"stateSurveyURLS.txt"
NRCS_shp = survey_workspace + "20240114_nrcs_course.shp"
WW_state_list = ["AZ", "CO", "ID", "MT", "NV", "NM", "OR", "SD", "UT", "WA", "WY"]

## SNODAS Variables
# RunName = "RT_CanAdj_rcn_noSW_woCCR_nofscamsk"
SNODAS_Type = "masked"
WW_NOHRSC_workspace = r"M:/SWE/WestWide/Spatial_SWE/NOHRSC/"
SNM_NOHRSC_workspace = r"M:/SWE/Sierras/Spatial_SWE/NOHRSC/"
Cellsize = "500"

watermask = "M:/SWE/WestWide//data/mask/watermask_outdated/outdated/WW/Albers/WW_watermaks_10_albn83_clp_final.tif"
glacierMask = "M:/SWE/WestWide/data/mask/glacierMask/glims_glacierMask_null_GT10_final.tif"
# band_zones = "M:/SWE/WestWide/data/hydro/WW_BasinBanded_noSNM_notahoe_albn83_sel.tif"
# watershed_zones = "M:/SWE/WestWide/data/hydro/WW_BasinBanded_noSNM_notahoe_albn83_sel.tif"

####################################
# Processing Starts Now
####################################

# make results and reports directory
os.makedirs(WW_results_workspace + f"/{rundate}_results_ET", exist_ok=True)
print(WW_results_workspace + f"/{rundate}_results_ET")
os.makedirs(SNM_results_workspace + f"/{rundate}_results_ET", exist_ok=True)
print("\nResults directories made")

os.makedirs(WW_reports_workspace + f"/{rundate}_RT_report_ET", exist_ok=True)
os.makedirs(SNM_reports_workspace + f"/{rundate}_RT_report_ET", exist_ok=True)
print("\nReports directories made")

#download sensors (look into r version of this code and see if we can have a python version)
if surveys_use == "Y":
    print("\nGetting CDEC Surveys")
    download_cdec_snow_surveys(report_date=rundate, survey_workspace=survey_workspace,
                               SNM_results_workspace=SNM_results_workspace,
                               cdec_shapefile=cdec_shapefile, basin_list=basin_list)

    print("\nGetting WW Surveys")
    download_snow_surveys(report_date=rundate, surveyWorkspace=survey_workspace, results_workspace=WW_results_workspace,
                          WW_url_file=WW_url_file, NRCS_shp=NRCS_shp, WW_state_list=WW_state_list)




## Maybe this section should be looped through CCR and woCCR
# run sensors code

# run R model

########### START #########

## sensor_code variables:
workspaceBase = fr"M:/SWE/WestWide/Spatial_SWE/WW_regression/"
model_workspace = fr"H:/WestUS_Data/Regress_SWE/"
case_field_wtrshd = "SrtName"
case_field_band = "SrtNmeBand"

print("\nProcessing GeoPackage")
geopackage_to_shapefile(report_date=rundate, pillow_date=pillow_date, model_run=model_woCCR,
                        user=user, domainList=domainList, model_workspace=model_workspace,
                        results_workspace=WW_results_workspace + f"/{rundate}_results_ET/")
# clear memory
sleep(30)
clear_arcpy_locks()

print('\nProcessing and sorting the sensors for West Wide ... ')
merge_sort_sensors_surveys(report_date=rundate, results_workspace=WW_results_workspace + f"/{rundate}_results_ET/", surveys="N", difference="N",
                           watershed_shapefile=WW_watershed_shapefile, case_field_wtrshd=case_field_wtrshd,
                           case_field_band=case_field_band, band_shapefile=WW_band_shapefile, projOut=projALB, merge="Y",
                           domainList=domainList)

# clear memory
sleep(30)
clear_arcpy_locks()

# run SNODAS for WW
print("SNODAS for WW...")
SNODAS_Processing(report_date=rundate, RunName=model_woCCR, NOHRSC_workspace=WW_NOHRSC_workspace,
                      results_workspace=WW_results_workspace,
                      projin=projGEO, projout=projALB, Cellsize=500, snapRaster=snapRaster_albn83, watermask=watermask,
                      glacierMask=glacierMask,
                      band_zones=WW_band_zones, watershed_zones=WW_watershed_zones, unzip_SNODAS="Y")

# run tables and layers
modelRuns = [model_woCCR, model_wCCR]
# clear memory
sleep(30)
clear_arcpy_locks()

print(f'\nRunning Tables and Layers Code for all domains for {model_woCCR}')
tables_and_layers(user=user, year=year, report_date=rundate, mean_date = mean_date, meanWorkspace = meanWorkspace, model_run=model_woCCR,
                  masking="N", watershed_zones=WW_watershed_zones, band_zones=WW_band_zones, HUC6_zones=HUC6_zones,
                  region_zones=region_zones, case_field_wtrshd=case_field_wtrshd,case_field_band=case_field_band,
                  watermask=watermask, glacierMask=glacierMask, snapRaster_geon83=snapRaster_geon83, snapRaster_albn83=snapRaster_albn83,
                  projGEO=projGEO, projALB=projALB, ProjOut_UTM=ProjOut_UTM, bias="N")

# clear memory
sleep(30)
print('done sleeping')
clear_arcpy_locks()

print(f'\nRunning Tables and Layers Code for all domains for {model_wCCR}')
tables_and_layers(user=user, year=year, report_date=rundate, mean_date = mean_date, meanWorkspace = meanWorkspace, model_run=model_wCCR, masking="N", watershed_zones=WW_watershed_zones,
                  band_zones=WW_band_zones, HUC6_zones=HUC6_zones, region_zones=region_zones, case_field_wtrshd=case_field_wtrshd,
                  case_field_band=case_field_band, watermask=watermask, glacierMask=glacierMask, snapRaster_geon83=snapRaster_geon83,
                  snapRaster_albn83=snapRaster_albn83, projGEO=projGEO, projALB=projALB, ProjOut_UTM=ProjOut_UTM, bias="N")

# clear memory
sleep(30)
print('done sleeping')
clear_arcpy_locks()

print('\nProcessing and sorting the sensors for the Sierra... ')
# SNM_results_workspace = rf"M:/SWE/Sierras/Spatial_SWE/SNM_regression/RT_report_data/{rundate}_results_ET/"
SNM_sensors = rf"M:/SWE/WestWide/Spatial_SWE/WW_regression/RT_report_data/{rundate}_results_ET/{rundate}_sensors_SNM.shp"
merge_sort_sensors_surveys(report_date=rundate, results_workspace=SNM_results_workspace + f"/{rundate}_results_ET/", surveys="N", difference="N",
                           watershed_shapefile=SNM_watershed_shapefile, case_field_wtrshd=case_field_wtrshd, band_shapefile=SNM_band_shapefile,
                           case_field_band=case_field_band, projOut=projALB, projIn=projGEO,
                            merge="N", domain_shapefile=SNM_sensors)

# Run SNODAS for SNM
print("\nSNODAS for SNM...")
SNODAS_Processing(report_date=rundate, RunName=model_woCCR, NOHRSC_workspace=WW_NOHRSC_workspace,
                  results_workspace=SNM_results_workspace,
                  projin=projGEO, projout=projALB, Cellsize=500, snapRaster=SNM_snapRaster_albn83,
                  watermask=watermask, glacierMask=glacierMask,
                  band_zones=SNM_band_zones, watershed_zones=SNM_watershed_zones, unzip_SNODAS="N")


print(f'\nRunning Tables and Layers Code for Sierra {model_woCCR}...')

# clear memory
sleep(30)
clear_arcpy_locks()

print('memory cleared')
# SNM_results_workspace = rf"M:/SWE/Sierras/Spatial_SWE/SNM_regression/RT_report_data/{rundate}_results_ET/"
tables_and_layers_SNM(year=year, rundate=rundate, mean_date=mean_date, WW_model_run=model_woCCR, SNM_results_workspace=SNM_results_workspace,
                      watershed_zones=SNM_watershed_zones, band_zones=SNM_band_zones, region_zones=SNM_regions,
                      case_field_wtrshd=case_field_wtrshd, case_field_band=case_field_band, watermask=watermask,
                      glacier_mask=glacierMask, domain_mask=SNM_domain_msk, run_type="Normal",
                      snap_raster=SNM_snapRaster_albn83, WW_results_workspace=WW_results_workspace,
                      Difference="N")
# clear memory
# clear memory
sleep(30)
clear_arcpy_locks()

print('memory cleared again')
print(f'\nRunning Tables and Layers Code for Sierra {model_wCCR}...')
tables_and_layers_SNM(year=year, rundate=rundate, mean_date=mean_date, WW_model_run=model_wCCR, SNM_results_workspace=SNM_results_workspace,
                      watershed_zones=SNM_watershed_zones, band_zones=SNM_band_zones, region_zones=SNM_regions,
                      case_field_wtrshd=case_field_wtrshd, case_field_band=case_field_band, watermask=watermask,
                      glacier_mask=glacierMask, domain_mask=SNM_domain_msk, run_type="Normal",
                      snap_raster=SNM_snapRaster_albn83, WW_results_workspace=WW_results_workspace,
                      Difference="N")
# clear memory
clear_arcpy_locks()
sleep(30)
print('memory cleared again')

# sensor vetting function
# parameters
domains = ["SNM", "PNW", "INMT", "SOCN", "NOCN"]
clipbox_WS = "M:/SWE/WestWide/data/boundaries/Domains/DomainShapefiles/"
# surveys_use ="N"
# resultsWorkspace = f"W:/Spatial_SWE/WW_regression/RT_report_data/"

print('folder created')

# loop through domains
for modelRun in modelRuns:
    for domain in domains:
        if domain == "SNM":
            raster = f"{SNM_results_workspace}/{rundate}_results_ET/{modelRun}/p8_{rundate}_noneg.tif"
            sensors_SNM = SNM_results_workspace + f"{rundate}_results_ET/SNM_{rundate}_sensors_albn83.shp"
            surveys_SNM = SNM_results_workspace + f"{rundate}_results_ET/{rundate}_surveys_albn83.shp"

            ## make vetting folder
            outVettingWS_SNM = SNM_reports_workspace + f"{rundate}_RT_report_ET/{modelRun}/vetting_domains/"
            os.makedirs(outVettingWS_SNM, exist_ok=True)

            # move p8 to vetting space
            arcpy.CopyRaster_management(raster, outVettingWS_SNM + f"p8_{rundate}_noneg.tif")

        else:
            # extract by mask
            arcpy.env.snapRaster = snapRaster_albn83
            arcpy.env.cellSize = snapRaster_albn83
            raster = f"{WW_results_workspace}/{rundate}_results_ET/{modelRun}/p8_{rundate}_noneg.tif"
            sensors = WW_results_workspace + f"{rundate}_results_ET/{rundate}_sensors_albn83.shp"
            surveys = WW_results_workspace + f"{rundate}_results_ET/{rundate}_surveys_albn83.shp"

            ## make vetting folder
            outVettingWS_WW = f"{WW_reports_workspace}/{rundate}_RT_report_ET/{modelRun}/vetting_domains/"
            os.makedirs(outVettingWS_WW, exist_ok=True)
            outMask = ExtractByMask(raster, clipbox_WS + f"WW_{domain}_Clipbox_albn83.shp")
            outMask.save(outVettingWS_WW + f"p8_{rundate}_noneg_{domain}_clp.tif")
            print(f"{domain} clipped and saved")

    for domain in domains:
        if domain == "SNM":
            print('domain is SNM')
            raster = outVettingWS_SNM + f"p8_{rundate}_noneg.tif"
            if surveys_use == "Y":
                swe_col_surv = 'SWE_m'
                id_col_surv = 'Station_Id'

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
            if surveys_use == "Y":
                swe_col_surv = 'SWE_m'
                id_col_surv = 'Station_Id'

                model_domain_vetting(raster=raster, point=surveys, swe_col=swe_col_surv, id_col=id_col_surv, rundate=rundate,
                                     domain=domain, modelRun=modelRun, out_csv=f"{WW_reports_workspace}/{rundate}_RT_report_ET/{rundate}_surveys_error.csv")

            swe_col_sens = 'pillowswe'
            id_col_sens = 'Site_ID'
            model_domain_vetting(raster=raster, point=sensors, swe_col=swe_col_sens, id_col=id_col_sens, rundate=rundate, domain=domain,
                                 modelRun=modelRun, out_csv=f"{WW_reports_workspace}/{rundate}_RT_report_ET/{rundate}_sensors_error.csv")
#
#
# ## ERIC: prompt for best model run with sensor counts and % error
# ChosenModelRun = "fSCA_RT_CanAdj_rcn_noSW_woCCR"
# biasCorrection = "Y"
#
# # ASO Bias correction
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
# # domains = ['SNM', 'SOCN']
# # basinList = ["SouthPlatte", "Uinta", "Kings"]
# output_csv = "Y"
# csv_outFile = r"W:/Spatial_SWE/ASO/2025/data_testing/FracError_data_test.csv"
# asoCatalog = r"W:/Spatial_SWE/ASO/2025/data_testing/ASO_SNOTEL_DifferenceStats.csv"
# basin_List = r"W:/Spatial_SWE/ASO/ASO_Metadata/ASO_in_Basin.txt"
# fracErrorWorkspace = "W:/Spatial_SWE/ASO/2025/data_testing/"
# domainList = r"M:\SWE\WestWide\Spatial_SWE\ASO\ASO_Metadata\State_Basin.txt"
# results_workspace = f"M:/SWE/WestWide/Spatial_SWE/WW_regression/RT_report_data/{rundate}_results_ET/"
#
# if biasCorrection == "Y":
#
#     # list of methods
#     results_df = bias_correction_selection(rundate=rundate, basin_List=basin_List, domainList=domainList, method_list=method_list,
#                                            fracErrorWorkspace=fracErrorWorkspace, output_csv=output_csv, csv_outFile=csv_outFile,
#                                            currentYear=True, grade_amount=grade_amount, sensorTrend=sensorTrend, SNOTEL=SNOTEL,
#                                            grade=grade, grade_range=grade_range)
#
#     #################################
#     # BIAS CORRECTION CODE FOR WW
#     #################################
#     methods = ["RECENT", "GRADE", "SENSOR_PATTERN", "GRADES_SPECF"]
#     for method in methods:
#         print(f"\nprocessing method:", method)
#         bias_correct(results_workspace, domain="WW", ModelRun=ChosenModelRun, method=method, rundate=rundate, results_df=results_df, shapefile_workspace)
#
#     # got through methods to find the best version for vetting
#     folder = r"W:\Spatial_SWE\WW_regression\RT_report_data\20250503_results_ET\ASO_BiasCorrect_fSCA_RT_CanAdj_rcn_noSW_woCCR/"
#     prefix = rundate
#     unique_names = set()  # use a set to keep unique values
#     file_mapping = {}
#     for root, dirs, files in os.walk(folder):
#         for file in files:
#             if file.startswith(prefix):
#
#                 # Split by "_" and take the first two parts
#                 parts = file.split("_")
#                 if len(parts) >= 2:
#                     name = "_".join(parts[:2])
#                     unique_names.add(name)
#
#     # Convert to list if needed
#     unique_names = list(unique_names)
#     print(unique_names)
#
#     print("\nFull file names by prefix:")
#     for name, files in file_mapping.items():
#         print(f"{name}:")
#         for f in files:
#             print(f"  {f}")
#
#     methods = ["RECENT", "GRADE", "SENSOR_PATTERN", "GRADES_SPECF"]
#     for method in methods:
#         print(f"\nMethod: {method}"'')
#         BC_path = folder + f"/{method}/"
#         for name in unique_names:
#             print(f"Name: {name}")
#             raster = BC_path + f"{name}_{method}_BC_fix_albn83.tif"
#
#             if os.path.exists(raster):
#                 bias_correction_vetting(raster=raster, point=surveys, swe_col="SWE_m", id_col="Station_Id", rundate="20250503",
#                     name=name, method=method, out_csv=out_csv, folder=folder, control_raster=control_raster)
#
#         #################################
#         # BIAS CORRECTION CODE FOR SNM
#         #################################
#         methods = ["RECENT", "GRADE", "SENSOR_PATTERN", "GRADES_SPECF"]
#         for method in methods:
#             print(f"\nprocessing method:", method)
#             bias_correct(results_workspace, domain="SNM", ModelRun=ChosenModelRun, method=method, rundate=rundate, results_df=results_df, shapefile_workspace)
#
#         # got through methods to find the best version for vetting
#         folder = r"W:\Spatial_SWE\WW_regression\RT_report_data\20250503_results_ET\ASO_BiasCorrect_fSCA_RT_CanAdj_rcn_noSW_woCCR/"
#         prefix = rundate
#         unique_names = set()  # use a set to keep unique values
#         file_mapping = {}
#         for root, dirs, files in os.walk(folder):
#             for file in files:
#                 if file.startswith(prefix):
#
#                     # Split by "_" and take the first two parts
#                     parts = file.split("_")
#                     if len(parts) >= 2:
#                         name = "_".join(parts[:2])
#                         unique_names.add(name)
#
#         # Convert to list if needed
#         unique_names = list(unique_names)
#         print(unique_names)
#
#         print("\nFull file names by prefix:")
#         for name, files in file_mapping.items():
#             print(f"{name}:")
#             for f in files:
#                 print(f"  {f}")
#
#         methods = ["RECENT", "GRADE", "SENSOR_PATTERN", "GRADES_SPECF"]
#         for method in methods:
#             print(f"\nMethod: {method}"'')
#             BC_path = folder + f"/{method}/"
#             for name in unique_names:
#                 print(f"Name: {name}")
#                 raster = BC_path + f"{name}_{method}_BC_fix_albn83.tif"
#
#                 if os.path.exists(raster):
#                     bias_correction_vetting(raster=raster, point=surveys, swe_col="SWE_m", id_col="Station_Id",
#                                             rundate="20250503",
#                                             name=name, method=method, out_csv=out_csv, folder=folder,
#                                             control_raster=control_raster)

### ADD PROMPTS

    # run the vetting
    ## go into the folder
    ## make a list of unique starts to the file names
    ## if file name starts with that


# Run vetting code

end = time.time()
time_elapsed = (end - start) /60
print(f"\n Elapsed Time: {time_elapsed} minutes")