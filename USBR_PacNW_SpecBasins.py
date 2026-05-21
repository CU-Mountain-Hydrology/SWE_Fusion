# import modules
import os
import arcpy
from arcpy.sa import *
import pandas as pd
print('modules import')

# report information
rundate = '20260426'
model_run = 'ASO_BiasCorrect_RT_CanAdj_rcn_woCCR_nofscamskSens_noMdlFsca_UseThis'
avg_model_run = ''
year = '2026'

raster = fr"W:/Spatial_SWE/WW_regression/RT_report_data/{rundate}_results/{model_run}/p8_{rundate}_noneg.tif"
# fsca_raster = r"U:\Emma\basin_spec_data\fSCA_20260308_albn83.tif"
# avg_raster = ''
workspace = fr"W:/documents/{year}_RT_Reports/{rundate}_RT_report/{model_run}/Tables/Tables_PacNW/"
os.makedirs(workspace, exist_ok=True)

# paths
shapefile_basins = r"W:\data\hydro\USBR_PacNW_basins\USBR_PacNW_Basin_albn83.shp"
shapefile_bands = r"W:\data\hydro\USBR_PacNW_basins\USBR_PacNW_BasinBanded_albn83.shp"
zone_raster = r"W:\data\hydro\USBR_PacNW_basins\USBR_PacNW_BasinBanded_albn83.tif"

# case fields
watershed_caseField = 'SrtName'
band_caseField = 'SrtNmeBand'

elevationBands = {
        "-1000": "< 0", "00000": "0-1,000'", "01000": "1,000-2,000'", "02000": "2,000-3,000'", "03000": "3,000-4,000'",
        "04000": "4,000-5,000'",
        "05000": "5,000-6,000'", "06000": "6,000-7,000'", "07000": "7,000-8,000'", "08000": "8,000-9,000'",
        "09000": "9,000-10,000'",
        "10000": "10,000-11,000'", "11000": "11,000-12,000'", "12000": "12,000-13,000'", "13000": "13,000-14,000'",
        "14000": "14,000-15,000'", "GT05000":">5,000'", "GT06000":">6,000'", "GT07000":">7,000'", "GT08000":">8,000'",
        "GT09000":">9,000'", "GT10000":">10,000'", "GT11000":">11,000'", "LT02000":"<2,000'", "LT03000":"<3,000'",
        "LT04000":"<4,000'",  "LT05000":"<5,000'"}

################################################
# get zonal stats
################################################
print('Processing for elevation bands')
ZonalStatisticsAsTable(zone_raster, band_caseField, raster, workspace + f"{rundate}_SWEbands_PacNW.dbf",
                                  "", "ALL")
# ZonalStatisticsAsTable(zone_raster, band_caseField, fsca_raster, workspace + f"{rundate}_SCAbands_PacNW.dbf",
#                                   "", "ALL")
# table to table
arcpy.TableToTable_conversion(workspace + f"{rundate}_SWEbands_PacNW.dbf", workspace, f"{rundate}_SWEbands_PacNW.csv")
# arcpy.TableToTable_conversion(workspace + f"{rundate}_SCAbands_PacNW.dbf", workspace, f"{rundate}_SCAbands_PacNW.csv")

# editing tables
df_swe = pd.read_csv(workspace + f"{rundate}_SWEbands_PacNW.csv")

# calc fields
df_swe['VOL_m3'] = df_swe['AREA'] * df_swe['MEAN']
df_swe['Area_mi2'] = (df_swe['AREA'] * 0.00000038610216).round(2)
df_swe['VOL_AF'] = (df_swe['VOL_m3'] * 0.000810714).round(0)
df_swe['SWE_in'] = (df_swe['MEAN'] * 39.370079).round(2)

# organize table
df_swe = df_swe[[band_caseField, 'ZONE_CODE', 'MEAN', 'SWE_in','VOL_m3', 'AREA', 'Area_mi2', 'VOL_AF']]
df_swe_rename = df_swe.rename(columns={'MEAN': 'SWE_m', 'AREA': 'Area_m2'})
df_swe_rename.to_csv(workspace + f"{rundate}_SWEbands_PacNW_raw.csv")

# final table processing
df_swe_final = df_swe_rename.sort_values(by=band_caseField, ascending=True)
df_swe_final['Num'] = df_swe_final[band_caseField].str.split("_").str[0]
df_swe_final['BasinCode'] = df_swe_final[band_caseField].str.split("_").str[1]
df_swe_final['ElevationCode'] = df_swe_final[band_caseField].str.split("_").str[2]
df_swe_final['ElevationBand'] = df_swe_final['ElevationCode'].map(elevationBands)
df_swe_final = df_swe_final[['Num', 'BasinCode', 'ElevationBand', 'Area_mi2', 'SWE_in', 'VOL_AF']]
df_swe_final = df_swe_final.rename(columns={'Num':'No.', 'BasinCode':'Basin Code', 'ElevationBand':'Elevation Band', 'SWE_in': 'SWE (in.)', 'Area_mi2': 'Area (mi2)', 'VOL_AF': 'Vol (AF)'})
df_swe_final.to_csv(workspace + f"{rundate}_SWE_ElevationBands_PacNW.csv", index=False)

####
print('Processing for watershed')
ZonalStatisticsAsTable(zone_raster, watershed_caseField, raster, workspace + f"{rundate}_SWEWtshd_PacNW.dbf",
                                  "", "ALL")

# table to table
arcpy.TableToTable_conversion(workspace + f"{rundate}_SWEWtshd_PacNW.dbf", workspace, f"{rundate}_SWEWtshd_PacNW.csv")

# editing tables
df_swe_wtshd = pd.read_csv(workspace + f"{rundate}_SWEWtshd_PacNW.csv")

# calc fields
df_swe_wtshd['VOL_m3'] = df_swe_wtshd['AREA'] * df_swe['MEAN']
df_swe_wtshd['Area_mi2'] = (df_swe_wtshd['AREA'] * 0.00000038610216).round(2)
df_swe_wtshd['VOL_AF'] = (df_swe_wtshd['VOL_m3'] * 0.000810714).round(0)
df_swe_wtshd['SWE_in'] = (df_swe_wtshd['MEAN'] * 39.370079).round(2)

# organize table
df_swe_wtshd = df_swe_wtshd[[watershed_caseField, 'ZONE_CODE', 'MEAN', 'SWE_in','VOL_m3', 'AREA', 'Area_mi2', 'VOL_AF']]
df_swe_wtshd_rename = df_swe_wtshd.rename(columns={'MEAN': 'SWE_m', 'AREA': 'Area_m2'})
df_swe_wtshd_rename.to_csv(workspace + f"{rundate}_SWEWtshd_PacNW_raw.csv")

# final table processing
df_swe_wtshd_final = df_swe_wtshd_rename.sort_values(by=watershed_caseField, ascending=True)
df_swe_wtshd_final['Num'] = df_swe_wtshd_final[watershed_caseField].str.split("_").str[0]
df_swe_wtshd_final['BasinCode'] = df_swe_wtshd_final[watershed_caseField].str.split("_").str[1]
df_swe_wtshd_final = df_swe_wtshd_final[['Num', 'BasinCode', 'Area_mi2', 'SWE_in', 'VOL_AF']]
df_swe_wtshd_final = df_swe_wtshd_final.rename(columns={'Num':'No.', 'BasinCode':'Basin Code', 'SWE_in': 'SWE (in.)', 'Area_mi2': 'Area (mi2)', 'VOL_AF': 'Vol (AF)'})

# contigency for OWY
df_swe_wtshd_final['Basin Code'] = df_swe_wtshd_final['Basin Code'].replace({
    'OWYGT': "OWY (Above 5,750')",
    'OWYLT': "OWY (Below 5,750')"
})


df_swe_wtshd_final.to_csv(workspace + f"{rundate}_SWE_Watershed_PacNW.csv", index=False)
