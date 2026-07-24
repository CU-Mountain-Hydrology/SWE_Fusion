# run/snodas_processing.py
# This file is a refactoring of the SNODAS_Processing() function from SWE_Fusion_functions.py

import arcpy
import os

from config import Config
from utils import make_directories
from download.download_snodas import download_snodas
from SWE_Fusion_functions import SNODAS_Processing

def snodas_processing(date: int, domain: str, model_woCCR: str, cfg: Config):
    """
    This is a wrapper to call the SNODAS_Processing() function from SWE_Fusion_functions.py, with added error handling.
    Once the function is refactored completely this won't be necessary

    :param date: (int) YYYYMMDD. The date to process (model run date)
    :param domain: (str) "WW" or "SNM". The domain to proces
    :param model_woCCR: (str) Name of model run without CoCoRaHs sensors
    :param cfg: Configuration object containing environment variables from the .env.
    """

    # Confirm that SNODAS data has been downloaded
    if not os.path.exists(f"{cfg.snodas_local_path}/SNODAS_{date}"):
        download_snodas(date, cfg)

    if domain == "WW":
        # Confirm results directory exists
        if not os.path.exists(f"{cfg.ww_results_workspace}/{date}_results"):
            make_directories(date, cfg)

        try:
            SNODAS_Processing(
                report_date=str(date), domain="WW", RunName=model_woCCR, NOHRSC_workspace=cfg.ww_nohrsc_workspace,
                results_workspace=cfg.ww_results_workspace,
                projin=arcpy.SpatialReference(cfg.proj_geo), projout=arcpy.SpatialReference(cfg.proj_alb), Cellsize=500,
                snapRaster=cfg.ww_snap_raster_albn83, watermask=cfg.water_mask, glacierMask=cfg.glacier_mask,
                band_zones=cfg.ww_band_zones, watershed_zones=cfg.ww_watershed_zones, unzip_SNODAS="Y"
            )
        except Exception as e:
            # TODO: figure out error handling framework with checkpoint system. Return code? Retry? Pass exception?
            print(e)

    elif domain == "SNM":
        # Confirm results directory exists
        if not os.path.exists(f"{cfg.snm_results_workspace}/{date}_results"):
            make_directories(date, cfg)

        try:
            # TODO: why does this use ww_nohrsc_workspace? snm_nohrsc_workspace is never used
            SNODAS_Processing(
                report_date=str(date), domain="SNM", RunName=model_woCCR, NOHRSC_workspace=cfg.ww_nohrsc_workspace,
                results_workspace=cfg.snm_results_workspace,
                projin=arcpy.SpatialReference(cfg.proj_geo), projout=arcpy.SpatialReference(cfg.proj_alb), Cellsize=500,
                snapRaster=cfg.snm_snap_raster_albn83,
                watermask=cfg.water_mask, glacierMask=cfg.glacier_mask,
                band_zones=cfg.snm_band_zones, watershed_zones=cfg.snm_watershed_zones, unzip_SNODAS="N",
                dwr_mask=cfg.snm_domain_mask
            )
        except Exception as e:
            # TODO: error handling
            print(e)

    else:
        raise ValueError(f"Domain {domain} is not supported.")


if __name__ == "__main__":
     config = Config()
     snodas_processing(20260630, "WW", "RT_CanAdj_rcn_woCCR_nofscamskSens_noMdlFsca", config)

# TODO: refactor the copied-over function below for improved error handling, performance, and integration with Config class
"""
def SNODAS_Processing(report_date, domain, RunName, NOHRSC_workspace, results_workspace,
                         projin, projout, Cellsize, snapRaster, watermask, glacierMask, band_zones, watershed_zones, unzip_SNODAS,
                      dwr_mask=None):
    SNODASWorkspace = NOHRSC_workspace + f"SNODAS_{report_date}/"
    SWEWorkspaceBase = results_workspace + f"{report_date}_results/{RunName}/"
    resultsWorkspace = results_workspace +f"{report_date}_results/"
    SWEWorkspace = results_workspace + f"{report_date}_results/SNODAS/"

    ## Set regression SWE image for the same date
    # RegressSWE = SWEWorkspaceBase + f"p8_{report_date}_noneg.tif"

    ##### Set automatic local variables
    arcpy.CreateFolder_management(resultsWorkspace, "SNODAS")
    # product8 = SWEWorkspace + f"p8_{report_date}_noneg.tif"
    # arcpy.CopyRaster_management(RegressSWE, product8)

    OutSNODAS = f"SWE_{report_date}.tif"
    OutSNODASplus = SNODASWorkspace + OutSNODAS
    FloatSNODAS = SWEWorkspace + f"SWE_{report_date}_Cp.tif"
    MeterSNODAS = SWEWorkspace + f"SWE_{report_date}_Cp_m.tif"
    ProjSNODAS = SWEWorkspace + f"SWE_{report_date}_Cp_m_albn83.tif"
    ClipSNODAS = SWEWorkspace + f"SWE_{report_date}_Cp_m_albn83_clp.tif"
    SCA_SNODAS = SWEWorkspace + f"SWE_{report_date}_fSCA.tif"

    SWEbandtable = SWEWorkspace + f"{report_date}_band_SNODAS_swe_table.dbf"
    SWEtable = SWEWorkspace + f"{report_date}_SNODAS_swe_table.dbf"
    SWEbandtable_save = SWEWorkspace + f"{report_date}_band_SNODAS_swe_table_save.dbf"
    SWEtable_save = SWEWorkspace + f"{report_date}_SNODAS_swe_table_save.dbf"
    SWEbandtableCSV = SWEWorkspace + f"{report_date}_band_SNODAS_swe_table.csv"
    SWEtableCSV = SWEWorkspace +f"{report_date}_SNODAS_swe_table.csv"

    ###### End of setting up variables


    # unzip and move HDR file
    if unzip_SNODAS == "Y":
        arcpy.env.workspace = SNODASWorkspace
        gz_datFile = SNODASWorkspace + f"us_ssmv11034tS__T0001TTNATS{report_date}05HP001.dat.gz"
        gz_unzipDat = SNODASWorkspace + f"us_ssmv11034tS__T0001TTNATS{report_date}05HP001.dat"

        print("\nUnzipping SNODAS file...")
        with gzip.open(gz_datFile, "rb") as f_in:
            with open(gz_unzipDat, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        print("SNODAS file unzipped")

        hdrSNODAS = NOHRSC_workspace + f"us_ssmv11034tS_masked.hdr"
        hdrSNODAS_copy = f"us_ssmv11034tS__T0001TTNATS{report_date}05HP001.hdr"
        shutil.copy(hdrSNODAS, os.path.join(SNODASWorkspace, hdrSNODAS_copy))
        print("HDR file moved")

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

        arcpy.env.workspace = None
    clear_arcpy_locks()
    arcpy.env.workspace = None
    arcpy.ClearEnvironment("workspace")
    arcpy.ClearEnvironment("extent")

    import gc
    gc.collect()
    import time
    time.sleep(2)
    clear_arcpy_locks()
    # Verify source file exists
    print(f"Checking source file: {OutSNODASplus}")
    if not arcpy.Exists(OutSNODASplus):
        raise FileNotFoundError(f"ERROR: Source SNODAS file not found: {OutSNODASplus}")

    ## Copy to floating point raster
    # print(f"Copying to: {FloatSNODAS}")
    ## Copy to floating point raster
    if os.path.exists(FloatSNODAS):
        print(f"{FloatSNODAS} exists")
    if not os.path.exists(FloatSNODAS):
        print(f"Copying to: {FloatSNODAS}")
        arcpy.CopyRaster_management(OutSNODASplus, FloatSNODAS)
    # arcpy.CopyRaster_management(OutSNODASplus, FloatSNODAS)
    clear_arcpy_locks()
    # Verify source file exists
    print(f"Checking source file: {OutSNODASplus}")
    if not arcpy.Exists(OutSNODASplus):
        raise FileNotFoundError(f"ERROR: Source SNODAS file not found: {OutSNODASplus}")

    clear_arcpy_locks()
    ## Copy to floating point raster
    # print(FloatSNODAS)
    # # arcpy.CopyRaster_management(OutSNODASplus, FloatSNODAS, "", "", "-2147483648", "NONE", "NONE", "32_BIT_FLOAT",
    # #                             "NONE", "NONE")
    # arcpy.CopyRaster_management(OutSNODASplus, FloatSNODAS)

    print("Creating SWE in meters ...")
    clear_arcpy_locks()
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
    # arcpy.env.extent = snapRaster
    SNODASwatMsk = Raster(ProjSNODAS) * Raster(watermask)
    SNODASallMsk = SNODASwatMsk * Raster(glacierMask)

    if domain == "SNM":
        SNODASmsk = ExtractByMask(ProjSNODAS, dwr_mask, "INSIDE")
        SNODASmsk.save(ClipSNODAS)
        clear_arcpy_locks()

    else:
        SNODASmsk = ExtractByMask(ProjSNODAS, snapRaster, "INSIDE")
        SNODASmsk.save(ClipSNODAS)
        clear_arcpy_locks()
    ## If test run previously then SCA_SNODAS will exist, delete and then create
    if arcpy.Exists(SCA_SNODAS):
        arcpy.Delete_management(SCA_SNODAS, "#")
        SNODASfSCA = Con(SNODASallMsk > .001, 100, 0)
        SNODASfSCA.save(SCA_SNODAS)
    ## Else if this is a test run create it
    else:
        SNODASfSCA = Con(SNODASallMsk > .001, 100, 0)
        SNODASfSCA.save(SCA_SNODAS)

    clear_arcpy_locks()
    # Do zonal stats for real time swe layer table
    print("creating zonal stats for SNODAS swe = " + SWEtable)
    ZonalStatisticsAsTable(band_zones, "SrtNmeBand", ClipSNODAS, SWEbandtable, "DATA", "MEAN")
    ZonalStatisticsAsTable(watershed_zones, "SrtName", ClipSNODAS, SWEtable, "DATA", "MEAN")
    arcpy.Delete_management("in-memory")
    gc.collect()
    clear_arcpy_locks()
    # Add SWE in inches fields to 2 tables above
    arcpy.AddField_management(SWEbandtable, "SWE_IN", "FLOAT", "#", "#", "#",
                              "#", "NULLABLE", "NON_REQUIRED", "#")
    arcpy.AddField_management(SWEtable, "SWE_IN", "FLOAT", "#", "#", "#",
                              "#", "NULLABLE", "NON_REQUIRED", "#")
    arcpy.Delete_management("in-memory")
    gc.collect()

    # Calculate SWE in inches from meters
    arcpy.CalculateField_management(SWEbandtable, "SWE_IN", "!MEAN! * 39.370079", "PYTHON")
    arcpy.CalculateField_management(SWEtable, "SWE_IN", "!MEAN! * 39.370079", "PYTHON")
    arcpy.Delete_management("in-memory")
    gc.collect()

    # Sort by bandname and watershed name, 2 tables
    arcpy.Sort_management(SWEbandtable, SWEbandtable_save, [["SrtNmeBand", "ASCENDING"]])
    arcpy.Sort_management(SWEtable, SWEtable_save, [["SrtName", "ASCENDING"]])
    arcpy.Delete_management("in-memory")
    gc.collect()

    # print("Creating SNODAS and Regress diff layers ...")
    # SNODAS1000 = Con(Raster(ClipSNODAS) > 0.001, 1000, 0)
    # RSWE100 = Con(Raster(RegressSWE) > 0.001, 100, 0)
    #
    # ## Then add them together to create a layer showing where they overlap and
    # ## where they're different
    # SWEboth = SNODAS1000 + RSWE100
    #
    # ## Then save both layers
    # SWEboth.save(SWE_both)

    print("Creating CSV tables ...")

    snodas_wtshd_dbf = gpd.read_file(SWEtable_save)
    snodas_wtshd_df = pd.DataFrame(snodas_wtshd_dbf)
    snodas_wtshd_df.to_csv(SWEtableCSV, index=False)
    arcpy.Delete_management("in-memory")
    gc.collect()

    snodas_band_dbf = gpd.read_file(SWEbandtable_save)
    snodas_band_df = pd.DataFrame(snodas_band_dbf)
    snodas_band_df.to_csv(SWEbandtableCSV, index=False)

    arcpy.env.workspace = None
"""