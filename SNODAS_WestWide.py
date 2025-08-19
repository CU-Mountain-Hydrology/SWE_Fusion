# snm_SNODAS_tables_layers_final.py

# Import system modules
import os, sys, string
import arcinfo
import arcpy
import geopandas as gpd
import pandas as pd
from arcpy import env
from arcpy.sa import *
class LicenseError(Exception):
    pass
if arcpy.CheckExtension("Spatial") == "Available":
    arcpy.CheckOutExtension("Spatial")
else:
    # Raise a custom exception
    #
    raise LicenseError

print("modules imported")

##############################################################

## Set these variables

##############################################################

# Set the compression environment to NONE.
arcpy.env.compression = "NONE"
user = "Leanne"
#  date = SNODAS image date
date = "20250526"

## Use this run name when creating SNODAS for fSCA, nt for final model output
RunName = "RT_CanAdj_rcn_noSW_woCCR"

## Is SNODAS masked or unmasked? "masked" or "unmasked"
SType = "masked"

## Set workspaces
ModelWorkspace = r"M:/SWE/Sierras/Spatial_SWE/NOHRSC/"
if SType == "unmasked":
    SubWorkspace = "SNODAS_" + SType + "_" + date + "/"
if SType == "masked":
    SubWorkspace = "SNODAS_" + date + "/"

# RegressModelWorkspaceBase = rf"H:/WestUS_Data/Regress_SWE/SNM/{user}/StationSWERegressionV2/data/"
# RegressModelWorkspace = RegressModelWorkspaceBase + r"outputs/" + RunName + "/"
SNODASWorkspace = ModelWorkspace + SubWorkspace
WorkspaceBase = r"M:/SWE/Sierras/Spatial_SWE/SNM_regression/"
SWEWorkspaceBase = WorkspaceBase + r"RT_report_data/" + date + rf"_results/{RunName}/"
resultsWorkspace = WorkspaceBase + r"RT_report_data/" + date + rf"_results/"
SWEWorkspace = WorkspaceBase + r"RT_report_data/" + date + r"_results/SNODAS/"
MODWorkspace = SWEWorkspaceBase + date + "_MODSCAG_create/"
WatShdWorkspace = r"M:/SWE/Sierras/data/hydro/"
MskWorkspace = r"M:/SWE/Sierras/data/mask/"
arcpy.env.workspace = SWEWorkspace
InWorkspace = SWEWorkspace
OutWorkspace = SWEWorkspace

## Set regression SWE image for the same date
RegressSWE = SWEWorkspaceBase + "p8_" + date + "_noneg.tif"

## Set snapraster
snapRaster = "M:/SWE/WestWide/data/boundaries/SnapRaster_albn83.tif"

## Set the water mask used by the LRM model
watermask = "M:/SWE/WestWide//data/mask/watermask_outdated/outdated/WW/Albers/WW_watermaks_10_albn83_clp_final.tif"
glacierMask = "M:/SWE/WestWide/data/mask/glacierMask/glims_glacierMask_null_GT10_final.tif"

## Set up band watershed and watershed raster layers
band_zones = "M:/SWE/WestWide/data/hydro/WW_BasinBanded_noSNM_notahoe_albn83_sel.tif"
watershed_zones = "M:/SWE/WestWide/data/hydro/WW_BasinBanded_noSNM_notahoe_albn83_sel.tif"

## Set up input and output projections
projin = arcpy.SpatialReference(4269) #GCS NAD
projout = arcpy.SpatialReference(102039) #Albers
Cellsize = "500"

print("variables established")
##############################################################
## End of Set these variables
##############################################################

##### Set automatic local variables
arcpy.CreateFolder_management(resultsWorkspace, "SNODAS")
product8 = SWEWorkspace + "p8_" + date + "_noneg.tif"
arcpy.CopyRaster_management(RegressSWE, product8)

## Output tables
SWEbandtable = SWEWorkspace + date + "_band_SNODAS_swe_table.dbf"
SWEtable = SWEWorkspace + date + "_SNODAS_swe_table.dbf"
SWEbandtable_save = SWEWorkspace + date + "_band_SNODAS_swe_table_save.dbf"
SWEtable_save = SWEWorkspace + date + "_SNODAS_swe_table_save.dbf"
CNRFC_SWEbandtable = SWEWorkspace + date + "_CNRFC_band_SNODAS_swe_table.dbf"
CNRFC_SWEbandtable_save = SWEWorkspace + date + "_CNRFC_band_SNODAS_swe_table_save.dbf"

## Statistic type for zonal statistics to table commands
statisticType = "MEAN"

# Final outut CSV tables
SWEbandtableCSV = SWEWorkspace + date + "_band_SNODAS_swe_table.csv"
SWEtableCSV = SWEWorkspace + date + "_SNODAS_swe_table.csv"
CNRFC_SWEbandtableCSV = SWEWorkspace + date + "_CNRFC_band_SNODAS_swe_table.csv"

## SNODAS SWE Files
OutSNODAS = "SWE_" + date + ".tif"
OutSNODASplus = SWEWorkspace + OutSNODAS
FloatSNODAS = SWEWorkspace + "SWE_" + date + "_Cp.tif"
MeterSNODAS = SWEWorkspace + "SWE_" + date + "_Cp_m.tif"
ProjSNODAS = SWEWorkspace + "SWE_" + date + "_Cp_m_albn83.tif"
ClipSNODAS = SWEWorkspace + "SWE_" + date + "_Cp_m_albn83_clp.tif"
SCA_SNODAS = SWEWorkspace + "SWE_" + date + "_fSCA.tif"
MOD_SCA_SNODAS = MODWorkspace + "SNODAS_" + date + "_fSCA.tif"

## Regress SWE and SNODAS SWE overlap layer
SWE_both = SWEWorkspace + "SWE_" + date + "_both.tif"

###### End of setting up variables
# To Do: need to rename and move the HDR file
# maybe extract the zip
print("Creating masked SWE: " + product8)
arcpy.env.workspace = SNODASWorkspace
print("Input Workspace: " + SNODASWorkspace)

## Add .dat file to file list
dats = arcpy.ListFiles("*.dat")

## Process all applicable .dat files
for dat in dats:

    ## Create geoTIF file from .dat file
    OutTif = dat[0:-4] + ".tif"
    print("Creating: " + OutSNODAS)

    ## Check to see if geoTIF file exists, if not create it.
    if arcpy.Exists(OutTif):
        print(" ")
    else:
        ## Create a geotif from the .dat file
        arcpy.RasterToOtherFormat_conversion(dat, SNODASWorkspace, "TIFF")

    # define projection
    arcpy.DefineProjection_management(OutTif, projin)

    ## Get rid of -9999 values and change to NODATA values
    NoData = SetNull(Raster(OutTif) == -9999, OutTif)
    NoData.save(OutSNODASplus)

    ## Copy to floating point raster
    arcpy.CopyRaster_management(OutSNODASplus, FloatSNODAS, "", "", "-2147483648", "NONE", "NONE", "32_BIT_FLOAT",
                                "NONE", "NONE")

    print("Creating SWE in meters ...")

    ## Divide by 1000 to get value in meters not mm
    SWEm = Raster(FloatSNODAS) / 1000
    SWEm.save(MeterSNODAS)

    print("Projecting and snapping to regression SWE ...")

    ## Define projection again b/c arcpy can't deal
    arcpy.DefineProjection_management(MeterSNODAS, projin)

    ## Project to WGS84, match to UCRB domain cellsize, extent and snapraster
    arcpy.env.snapRaster = snapRaster
    arcpy.env.extent = snapRaster
    arcpy.env.cellSize = snapRaster

    arcpy.ProjectRaster_management(MeterSNODAS, ProjSNODAS, projout, "NEAREST", Cellsize,
                                   "", "", projin)

    # set extent and apply masks
    arcpy.env.extent = snapRaster
    SNODASwatMsk = Raster(ProjSNODAS) * Raster(watermask)
    SNODASallMsk = SNODASwatMsk * Raster(glacierMask)

    ## If test run previously then SCA_SNODAS will exist, delete and then create
    if arcpy.Exists(SCA_SNODAS):
        arcpy.Delete_management(SCA_SNODAS, "#")
        SNODASfSCA = Con(SNODASallMsk > .001, 100, 0)
        SNODASfSCA.save(SCA_SNODAS)
    ## Else if this is a test run create it
    else:
        SNODASfSCA = Con(SNODASallMsk > .001, 100, 0)
        SNODASfSCA.save(SCA_SNODAS)

## If test run delete all layers b/c they are not final, and copy fSCA image
## to create_MODSCAG directory
if RunName == "test":
    arcpy.CopyRaster_management(SCA_SNODAS, MOD_SCA_SNODAS, "#", "#", "-128", "NONE", "NONE", "#", "NONE", "NONE", "#",
                                "NONE")
    arcpy.Delete_management(product8, "#")
    arcpy.Delete_management(OutSNODASplus, "#")
    arcpy.Delete_management(FloatSNODAS, "#")
    arcpy.Delete_management(MeterSNODAS, "#")
    arcpy.Delete_management(ProjSNODAS, "#")
    arcpy.Delete_management(ClipSNODAS, "#")

## If this is the final run, then proceed, otherwise if test run don't create
## final layers
if RunName != "test":

    # Do zonal stats for real time swe layer table
    print("creating zonal stats for SNODAS swe = " + SWEtable)
    outswetbl = ZonalStatisticsAsTable(band_zones, "SrtNmeBand", ClipSNODAS, SWEbandtable, "DATA", "MEAN")
    outswetbl2 = ZonalStatisticsAsTable(watershed_zones, "SrtName", ClipSNODAS, SWEtable, "DATA", "MEAN")
    # outswetbl3 = ZonalStatisticsAsTable(CNRFC_band_watershed, "NameBand", ProjSNODAS, CNRFC_SWEbandtable, "DATA",
    #                                     "MEAN")

    # Add SWE in inches fields to 2 tables above
    arcpy.AddField_management(SWEbandtable, "SWE_IN", "FLOAT", "#", "#", "#", "#", "NULLABLE", "NON_REQUIRED", "#")
    arcpy.AddField_management(SWEtable, "SWE_IN", "FLOAT", "#", "#", "#", "#", "NULLABLE", "NON_REQUIRED", "#")
    arcpy.AddField_management(CNRFC_SWEbandtable, "SWE_IN", "FLOAT", "#", "#", "#", "#", "NULLABLE", "NON_REQUIRED",
                              "#")

    # Calculate SWE in inches from meters
    arcpy.CalculateField_management(SWEbandtable, "SWE_IN", "!MEAN! * 39.370079", "PYTHON")
    arcpy.CalculateField_management(SWEtable, "SWE_IN", "!MEAN! * 39.370079", "PYTHON")
    arcpy.CalculateField_management(CNRFC_SWEbandtable, "SWE_IN", "!MEAN! * 39.370079", "PYTHON")

    # Sort by bandname and watershed name, 2 tables
    arcpy.Sort_management(SWEbandtable, SWEbandtable_save, [["SrtNmeBand", "ASCENDING"]])
    arcpy.Sort_management(SWEtable, SWEtable_save, [["SrtName", "ASCENDING"]])
    arcpy.Sort_management(CNRFC_SWEbandtable, CNRFC_SWEbandtable_save, [["NameBand", "ASCENDING"]])

    print("Creating SNODAS and Regress diff layers ...")

    SNODAS1000 = Con(Raster(ClipSNODAS) > 0.001, 1000, 0)
    RSWE100 = Con(Raster(RegressSWE) > 0.001, 100, 0)

    ## Then add them together to create a layer showing where they overlap and
    ## where they're different
    SWEboth = SNODAS1000 + RSWE100

    ## Then save both layers
    SWEboth.save(SWE_both)

    print("Creating CSV tables ...")

    snodas_wtshd_dbf = gpd.read_file(SWEtable_save)
    snodas_wtshd_df = pd.DataFrame(snodas_wtshd_dbf)
    snodas_wtshd_df.to_csv(SWEtableCSV, index=False)

    snodas_band_dbf = gpd.read_file(SWEbandtable_save)
    snodas_band_df = pd.DataFrame(snodas_band_dbf)
    snodas_band_df.to_csv(SWEbandtableCSV, index=False)

    snodas_cnrfc_dbf = gpd.read_file(CNRFC_SWEbandtable_save)
    snodas_cnrfc_df = pd.DataFrame(snodas_cnrfc_dbf)
    snodas_cnrfc_df.to_csv(CNRFC_SWEbandtableCSV, index=False)


