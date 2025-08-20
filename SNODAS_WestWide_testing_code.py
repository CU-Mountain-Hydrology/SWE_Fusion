# snm_SNODAS_tables_layers_final.py

# Import system modules
import os, sys, string
import arcinfo
import arcpy
import geopandas as gpd
import pandas as pd
from arcpy import env
import gzip
import shutil
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
report_date = "20250525"

## Use this run name when creating SNODAS for fSCA, nt for final model output
RunName = "RT_CanAdj_rcn_noSW_woCCR_nofscamsk"

## Is SNODAS masked or unmasked? "masked" or "unmasked"
SNODAS_Type = "masked"
WW_workspaceBase = r"M:/SWE/WestWide/Spatial_SWE/WW_regression/"
## Set workspaces
WW_NOHRSC_workspace = r"M:/SWE/WestWide/Spatial_SWE/NOHRSC/"
WW_results_workspace = WW_workspaceBase + "RT_report_data/"
projin = arcpy.SpatialReference(4269) #GCS NAD
projout = arcpy.SpatialReference(102039) #Albers
Cellsize = "500"
unzip_SNODAS = "Y"


snapRaster = "M:/SWE/WestWide/data/boundaries/SnapRaster_albn83.tif"
watermask = "M:/SWE/WestWide//data/mask/watermask_outdated/outdated/WW/Albers/WW_watermaks_10_albn83_clp_final.tif"
glacierMask = "M:/SWE/WestWide/data/mask/glacierMask/glims_glacierMask_null_GT10_final.tif"
band_zones = "M:/SWE/WestWide/data/hydro/WW_BasinBanded_noSNM_notahoe_albn83_sel.tif"
watershed_zones = "M:/SWE/WestWide/data/hydro/WW_BasinBanded_noSNM_notahoe_albn83_sel.tif"

#######################################
#PROCESSING
#######################################
def SNODAS_Processing(report_date, RunName, NOHRSC_workspace, results_workspace,
                         projin, projout, Cellsize, snapRaster, watermask, glacierMask, band_zones, watershed_zones, unzip_SNODAS):
    SNODASWorkspace = NOHRSC_workspace + f"SNODAS_{report_date}/"
    SWEWorkspaceBase = results_workspace + f"{report_date}_results_ET/{RunName}/"
    resultsWorkspace = results_workspace +f"{report_date}_results_ET/"
    SWEWorkspace = results_workspace + f"{report_date}_results_ET/SNODAS/"
    arcpy.env.workspace = SWEWorkspace

    ## Set regression SWE image for the same date
    RegressSWE = SWEWorkspaceBase + f"p8_{report_date}_noneg.tif"

    ##### Set automatic local variables
    arcpy.CreateFolder_management(resultsWorkspace, "SNODAS")
    product8 = SWEWorkspace + f"p8_{report_date}_noneg.tif"
    arcpy.CopyRaster_management(RegressSWE, product8)

    # unzip and move HDR file
    if unzip_SNODAS == "Y":
        gz_datFile = SNODASWorkspace + f"us_ssmv11034tS__T0001TTNATS{report_date}05HP001.dat.gz"
        gz_unzipDat = SNODASWorkspace + f"us_ssmv11034tS__T0001TTNATS{report_date}05HP001.dat"
        print("\nUnzipping SNODAS file...")
        with gzip.open(gz_datFile, "rb") as f_in:
            with open(gz_unzipDat, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        print("SNODAS file unzipped")

        hdrSNODAS = WW_NOHRSC_workspace + f"us_ssmv11034tS_masked.hdr"
        hdrSNODAS_copy = f"us_ssmv11034tS__T0001TTNATS{report_date}05HP001.hdr"
        shutil.copy(hdrSNODAS, os.path.join(SNODASWorkspace, hdrSNODAS_copy))
        print("HDR file moved")

    ## Output tables
    SWEbandtable = SWEWorkspace + f"{report_date}_band_SNODAS_swe_table.dbf"
    SWEtable = SWEWorkspace + f"{report_date}_SNODAS_swe_table.dbf"
    SWEbandtable_save = SWEWorkspace + f"{report_date}_band_SNODAS_swe_table_save.dbf"
    SWEtable_save = SWEWorkspace + f"{report_date}_SNODAS_swe_table_save.dbf"

    ## Statistic type for zonal statistics to table commands
    statisticType = "MEAN"

    # Final outut CSV tables
    SWEbandtableCSV = SWEWorkspace + f"{report_date}_band_SNODAS_swe_table.csv"
    SWEtableCSV = SWEWorkspace +f"{report_date}_SNODAS_swe_table.csv"

    ## SNODAS SWE Files
    OutSNODAS = f"SWE_{report_date}.tif"
    OutSNODASplus = SWEWorkspace + OutSNODAS
    FloatSNODAS = SWEWorkspace + f"SWE_{report_date}_Cp.tif"
    MeterSNODAS = SWEWorkspace + f"SWE_{report_date}_Cp_m.tif"
    ProjSNODAS = SWEWorkspace + f"SWE_{report_date}_Cp_m_albn83.tif"
    ClipSNODAS = SWEWorkspace + f"SWE_{report_date}_Cp_m_albn83_clp.tif"
    SCA_SNODAS = SWEWorkspace + f"SWE_{report_date}_fSCA.tif"

    ## Regress SWE and SNODAS SWE overlap layer
    SWE_both = SWEWorkspace + f"SWE_{report_date}_both.tif"

    ###### End of setting up variables
    print("Creating masked SWE: " + product8)
    arcpy.env.workspace = SNODASWorkspace

    if unzip_SNODAS == "Y":
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

        SNODASmsk = ExtractByMask(ProjSNODAS, snapRaster, "INSIDE")
        SNODASmsk.save(ClipSNODAS)

        ## If test run previously then SCA_SNODAS will exist, delete and then create
        if arcpy.Exists(SCA_SNODAS):
            arcpy.Delete_management(SCA_SNODAS, "#")
            SNODASfSCA = Con(SNODASallMsk > .001, 100, 0)
            SNODASfSCA.save(SCA_SNODAS)
        ## Else if this is a test run create it
        else:
            SNODASfSCA = Con(SNODASallMsk > .001, 100, 0)
            SNODASfSCA.save(SCA_SNODAS)


    # Do zonal stats for real time swe layer table
    print("creating zonal stats for SNODAS swe = " + SWEtable)
    outswetbl = ZonalStatisticsAsTable(band_zones, "SrtNmeBand", ClipSNODAS, SWEbandtable, "DATA", "MEAN")
    outswetbl2 = ZonalStatisticsAsTable(watershed_zones, "SrtName", ClipSNODAS, SWEtable, "DATA", "MEAN")

    # Add SWE in inches fields to 2 tables above
    arcpy.AddField_management(SWEbandtable, "SWE_IN", "FLOAT", "#", "#", "#",
                              "#", "NULLABLE", "NON_REQUIRED", "#")
    arcpy.AddField_management(SWEtable, "SWE_IN", "FLOAT", "#", "#", "#",
                              "#", "NULLABLE", "NON_REQUIRED", "#")

    # Calculate SWE in inches from meters
    arcpy.CalculateField_management(SWEbandtable, "SWE_IN", "!MEAN! * 39.370079", "PYTHON")
    arcpy.CalculateField_management(SWEtable, "SWE_IN", "!MEAN! * 39.370079", "PYTHON")

    # Sort by bandname and watershed name, 2 tables
    arcpy.Sort_management(SWEbandtable, SWEbandtable_save, [["SrtNmeBand", "ASCENDING"]])
    arcpy.Sort_management(SWEtable, SWEtable_save, [["SrtName", "ASCENDING"]])

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


SNODAS_Processing(report_date=report_date, RunName=RunName, NOHRSC_workspace=WW_NOHRSC_workspace, results_workspace=WW_results_workspace,
                     projin=projin, projout=projout, Cellsize=Cellsize, snapRaster=snapRaster, watermask=watermask, glacierMask=glacierMask,
                     band_zones=band_zones, watershed_zones=watershed_zones, unzip_SNODAS="Y")