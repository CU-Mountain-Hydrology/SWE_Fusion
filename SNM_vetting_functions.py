
# import modules
import arcpy
import arcpy
from arcpy import env
from arcpy.ra import ZonalStatisticsAsTable
from arcpy.sa import *
import pandas as pd
import geopandas as gpd
import os
print('modules imported')
# establish paths
# user = "Leanne"
# year = "2025"

# set dates
# rundate = "20250525"
# mmddDate = "0528"
# prevRepDate = "20250517"
# modscagDate = "20250525"

# set model run info
# WW_modelRun = "RT_CanAdj_rcn_noSW_woCCR_nofscamsk"
# modelRun = "ASO_FixLayers_RT_CanAdj_rcn_noSW_woCCR_nofscamsk"
# bias_model_run =
# previous_model_run = "ASO_FixLayers_RT_CanAdj_rcn_noSW_woCCR_UseThis"

# set code parameters
# SNODAS = "N"
# DIF = "Y"
# runType = "Bias"
# where_clause = """"POLY_AREA" > 100"""
# part_area = "100 SquareKilometers"

# set projection info
# snapRaster_geon83 = "M:/SWE/WestWide/data/boundaries/SnapRaster_geon83.tif"
# SNM_snapRaster_albn83 = "M:/SWE/WestWide/data/boundaries/SNM_SnapRaster_albn83.tif"
# projGEO = arcpy.SpatialReference(4269)
# projALB = arcpy.SpatialReference(102039)
# ProjOut_UTM = arcpy.SpatialReference(26911)

# workspaceBase = fr"M:/SWE/WestWide/Spatial_SWE/WW_regression/"
# results_base = rf"M:/SWE/Sierras/Spatial_SWE/SNM_regression/RT_report_data/"
# WW_results = workspaceBase + f"RT_report_data/{rundate}_results/{WW_model_run}/"
# meanWorkspace = "M:/SWE/WestWide/Spatial_SWE/WW_regression/mean_2001_2021_Nodmfsca/"

# watermask = "M:/SWE/WestWide/data/mask/watermask_outdated/outdated/WW/Albers/WW_watermaks_10_albn83_clp_final.tif"
# glacierMask = "M:/SWE/WestWide/data/mask/glacierMask/glims_glacierMask_null_GT10_final.tif"
# domain_mask = "M:/SWE/WestWide/data/hydro/SNM/dwr_mask_null_albn83.tif"

# basin files -- NEED TO CHANGE
# banded_elev100 = "M:/SWE/WestWide/data/topo/ww_DEM_albn83_feet_banded_100.tif"

#######################################################################
# End of Setting Variables
#######################################################################
import arcpy
from arcpy import env
from arcpy.ra import ZonalStatisticsAsTable
from arcpy.sa import *
import pandas as pd
import geopandas as gpd

def tables_and_layers_SNM(year, rundate, mean_date, prev_report_date, SNM_results_workspace, WW_model_run, run_type, snap_raster,
                          WW_results_workspace, watermask, glacier_mask, domain_mask, Difference, previous_model_run, bias_model_run=None):
        # create directory
        prevRepWorkspace = SNM_results_workspace + f"{prev_report_date}_results/{previous_model_run}/"
        where_clause = """"POLY_AREA" > 100"""
        part_area = "100 SquareKilometers"
        ProjOut_UTM = arcpy.SpatialReference(26911)

        # raster paths
        watershedRas = "M:/SWE/WestWide/data/hydro/SNM/dwr_basins_geoSort_albn83.tif"
        band_raster = "M:/SWE/WestWide/data/hydro/SNM/dwr_band_basins_geoSort_albn83_delin.tif"
        region = "M:/SWE/WestWide/data/hydro/SNM/dwr_regions_albn83.tif"
        SNM_clipbox_alb = "M:/SWE/WestWide/data/boundaries/Domains/DomainShapefiles/WW_SNM_Clipbox_albn83.shp"
        case_field_wtrshd = "SrtName"
        case_field_band = "SrtNmeBand"

        if run_type == "Normal":
            arcpy.CreateFolder_management(SNM_results_workspace + f"/{rundate}_results/", WW_model_run)
            outWorkspace = SNM_results_workspace + f"/{rundate}_results/" + WW_model_run + "/"
            print("model run workspace created")

        if run_type == "Vetting":
            outWorkspace = SNM_results_workspace + f"/{rundate}_results/" + WW_model_run + "/"

        if run_type == "Bias":
            outWorkspace = SNM_results_workspace + f"/{rundate}_results/" + bias_model_run + "/"
        SNM_results_workspace + f"/{rundate}_results/"

        ## project and clip SNODAS
        SNODASWorkspace = SNM_results_workspace + f"/{rundate}_results/" + "SNODAS/"
        ClipSNODAS = SNODASWorkspace + "SWE_" + rundate + "_Cp_m_albn83_clp.tif"
        SWE_Diff = outWorkspace + "SNODAS_Regress_" + rundate + ".tif"


        meanMask = outWorkspace + f"{mean_date}_mean_msk.tif"
        MODSCAG_tif_plus = f"H:/WestUS_Data/Rittger_data/fsca_v2024.0d/NRT_FSCA_WW_N83/{year}/{rundate}.tif"
        MODSCAG_tif_plus_proj_WW = WW_results_workspace +  f"{rundate}_results/{WW_model_run}/" + f"fSCA_{rundate}_albn83.tif"
        MODSCAG_tif_plus_proj = outWorkspace + f"SNM_fSCA_{rundate}_albn83.tif"

        # define snow-no snow layer
        modscag_0_1 = outWorkspace + f"modscag_0_1_{rundate}.tif"
        modscag_per = outWorkspace + f"modscag_per_{rundate}.tif"
        modscag_per_msk = outWorkspace + f"modscag_per_{rundate}_msk.tif"

        # define snow/no snow null layer
        mod_null = outWorkspace + f"modscag_0_1_{rundate}msk_null.tif"
        mod_poly = outWorkspace + f"modscag_0_1_{rundate}msk_null_poly.shp"
        ### ASK LEANNE ABOUT UTM
        mod_poly_utm = outWorkspace + f"modscag_0_1_{rundate}_msk_null_poly_utm.shp"

        snowPolySel = outWorkspace + f"modscag_{rundate}_snowline_Sel.shp"
        snowPolyElim = outWorkspace + f"modscag_{rundate}_snowline_Sel_elim.shp"

        # define snow pillow gpkg
        meanMap_proj_WW = WW_results_workspace +  f"{rundate}_results/{WW_model_run}/" + f"WW_{mean_date}_mean_albn83.tif"
        meanMap_proj = outWorkspace + f"SNM_{mean_date}_mean_albn83.tif"
        meanMapMask = outWorkspace + f"SNM_{mean_date}_mean_msk_albn83.tif"
        lastRast = prevRepWorkspace + f"p8_{prev_report_date}_noneg.tif"
        DiffRaster = outWorkspace + f"Diff_{rundate}_{prev_report_date}.tif"

        ## define rasters
        WW_product8 = WW_results_workspace +  f"{rundate}_results/{WW_model_run}/" + f"p8_{rundate}_noneg.tif"
        rcn_glacMask_WW = WW_results_workspace +  f"{rundate}_results/{WW_model_run}/" + f"WW_{rundate}_phvrcn_mos_masked.tif"
        rcn_raw_proj_WW = WW_results_workspace +  f"{rundate}_results/{WW_model_run}/" + f"WW_{rundate}_phvrcn_albn83.tif"
        WW_p8_SNM = outWorkspace + f"WW_p8_{rundate}_noneg.tif"
        rcnFinal = outWorkspace + f"phvrcn_{rundate}_final.tif"
        product7 = outWorkspace + f"p7_{rundate}.tif"
        product7_noFsca = outWorkspace + f"p7_{rundate}_nofsca.tif"
        prod7msk = outWorkspace + f"p7_{rundate}_msk.tif"
        product8 = outWorkspace + f"p8_{rundate}_noneg.tif"
        prod8msk = outWorkspace + f"p8_{rundate}_noneg_msk.tif"
        product9 = outWorkspace + f"p9_{rundate}.tif"
        prod9msk = outWorkspace + f"p9_{rundate}_msk.tif"
        prod10msk = outWorkspace + f"p10_{rundate}_msk.tif"
        product10 = outWorkspace + f"p10_{rundate}.tif"
        rcn_raw_proj = outWorkspace + f"SNM_{rundate}_phvrcn_albn83.tif"

        # output Tables
        SWEbandtable = outWorkspace + f"{rundate}band_swe_table.dbf"
        SWEtable = outWorkspace + f"{rundate}swe_table.dbf"
        SWEbandtable100 = outWorkspace + f"{rundate}swe_table_100.dbf"
        SWEbandtable_save = outWorkspace + f"{rundate}band_swe_table_save.dbf"
        SWEtable_save = outWorkspace + f"{rundate}swe_table_save.dbf"
        SWEbandtable100_save = outWorkspace + f"{rundate}swe_table_100_save.dbf"

        # anomoly tables
        anombandTable = outWorkspace + f"{rundate}band_anom_table.dbf"
        anomTable = outWorkspace + f"{rundate}anom_table.dbf"
        meanTable = outWorkspace + f"{rundate}mean_table.dbf"
        anombandTable_save = outWorkspace + f"{rundate}band_anom_table_save.dbf"
        anomTable_save = outWorkspace + f"{rundate}anom_table_save.dbf"
        meanTable_save = outWorkspace + f"{rundate}mean_table_save.dbf"

        # region tables
        anomRegionTable = outWorkspace + f"{rundate}anomRegion_table.dbf"
        anomRegionTable_save = outWorkspace + f"{rundate}anomRegion_table_save.dbf"

        # Modscag 0/1 tables and % tables
        scabandtable = outWorkspace + f"{rundate}band_sca_table.dbf"
        scatable = outWorkspace + f"{rundate}sca_table.dbf"
        scabandtable_save = outWorkspace + f"{rundate}band_sca_table_save.dbf"
        scatable_save = outWorkspace + f"{rundate}_sca_table_save.dbf"
        perbandtable = outWorkspace + f"{rundate}band_per_table.dbf"
        pertable = outWorkspace + f"{rundate}_per_table.dbf"

        # create tempoary view for join
        SWEbandtableView = outWorkspace + f"{rundate}band_swe_table_view.dbf"
        SWEtableView = outWorkspace + f"{rundate}swe_table_view.dbf"

        # create joined tables
        BandtableJoin = outWorkspace + f"{rundate}band_table.dbf"
        WtshdTableJoin = outWorkspace + f"{rundate}Wtshd_table.dbf"

        # Anomaly maps
        anomMap = outWorkspace + f"{rundate}_anom.tif"
        anom0_100map = outWorkspace + f"{rundate}anom0_200.tif"
        anom0_100msk = outWorkspace + f"{rundate}anom0_200_msk.tif"

        #SWE maps
        SWEzoneMap = outWorkspace + f"{rundate}_swe_wshd.tif"
        MeanzoneMap = outWorkspace + f"{rundate}_mean_wshd.tif"
        SWEbandzoneMap = outWorkspace + f"{rundate}_swe_band_wshd.tif"
        MeanBandZoneMap = outWorkspace + f"{rundate}_mean_band_wshd.tif"

        # mean layer masked for use in creating anomly map
        anomMask = outWorkspace + f"{rundate}_anom_mask.tif"

        # statistic
        statisticType = "MEAN"

        # final output csv tables
        WtshdTableJoinCSV = outWorkspace + f"{rundate}Wtshd_table.csv"
        BandtableJoinCSV = outWorkspace + f"{rundate}band_table.csv"
        anomRegionTableCSV = outWorkspace + f"{rundate}anomRegion_table.csv"
        Band100TableCSV = outWorkspace + f"{rundate}band_table_100.csv"
        SCATableJoinCSV = outWorkspace + f"{rundate}sca_Wtshd_table.csv"
        BandSCAtableJoinCSV = outWorkspace + f"{rundate}sca_band_table.csv"
        anomWtshdTableCSV = outWorkspace + f"{rundate}anomWtshd_table.csv"
        anomBandTableCSV = outWorkspace + f"{rundate}anomBand_table.csv"
        print("file paths established")

        #start with envirnonment settings
        arcpy.env.snapRaster = snap_raster
        arcpy.env.cellSize = snap_raster
        arcpy.env.extent = snap_raster
        # arcpy.env.snapRaster = snapRaster_albn83
        print("starting")
        # domain model runs
        if run_type == "Normal":
            print('clip and bring over the other files')
            # set snap raster
            # copy over files
            arcpy.CopyRaster_management(WW_product8, WW_p8_SNM)

            outSNM_phvrcn = ExtractByMask(WW_p8_SNM, SNM_clipbox_alb, "INSIDE")
            print('clipped')
            outSNM_phvrcn.save(product8)
            print("saved SWE")

        # fsca
        outSNM_fsca = ExtractByMask(MODSCAG_tif_plus_proj_WW, SNM_clipbox_alb, "INSIDE")
        outSNM_fsca.save(MODSCAG_tif_plus_proj)
        print("saved fSCA")
        # mean
        outSNM_mean = ExtractByMask(meanMap_proj_WW, SNM_clipbox_alb, "INSIDE")
        outSNM_mean.save(meanMap_proj)
        print("clipped domain")

        ########################
        print(f"Processing begins...")
        # create snow/no snow layer
        mod_01 = Con((Raster(MODSCAG_tif_plus_proj) < 101) & (Raster(MODSCAG_tif_plus_proj) > 0),
                     1, 0)
        mod_01_Wtrmask = mod_01 * Raster(watermask)
        mod_01_AllMaks = mod_01_Wtrmask * Raster(glacier_mask)
        mod_01_AllMaks.save(modscag_0_1)
        print(f"fSCA mask tif saved")

        # create fSCA percent layer
        Mod_per = (Float(SetNull(Raster(MODSCAG_tif_plus_proj) > 100, Raster(MODSCAG_tif_plus_proj))) / 100)
        Mod_per.save(modscag_per)
        print(f"fSCA percent layer saved")

        # create fsca percent layer ASK LEANNE, WHAT'S THE DIFFERENT BETWEEN LAKES MASK AND WATER MASK
        mod_01_mask = Con(Raster(modscag_per) > 0.0001, 1, 0)
        mod_per_msk = Raster(watermask) * mod_01_mask
        mod_per_Allmsk = Raster(glacier_mask) * mod_per_msk
        mod_per_Allmsk.save(modscag_per_msk)
        print("fSCA percent layer created")


        # ASK LEANNE, WHAT'S THE DIFFERENCE BETWEEN TIF MASK AND WATER MASK
        print(f"Creating snowline shapefile: {snowPolyElim}")
        mod_01_mask = mod_01 + Raster(watermask)
        mod_01_mask_glacier = mod_01 + Raster(glacier_mask)
        mod_01_msk_null = SetNull(mod_01_mask_glacier == 0, mod_01_mask_glacier)
        mod_01_msk_null.save(mod_null)

        # Convert raster to polygon
        arcpy.RasterToPolygon_conversion(mod_null, mod_poly, "NO_SIMPLIFY", "Value")
        arcpy.Project_management(mod_poly, mod_poly_utm, ProjOut_UTM)
        arcpy.AddGeometryAttributes_management(mod_poly_utm, "AREA", "", "SQUARE_KILOMETERS", "")
        arcpy.Select_analysis(mod_poly_utm, snowPolySel, where_clause)
        arcpy.EliminatePolygonPart_management(snowPolySel, snowPolyElim, "AREA", part_area, "0", "CONTAINED_ONLY")

        print(f"creating masked SWE product")
        if Difference == "Y":
            Diff_rgrs = Raster(product8) - Raster(lastRast)
            Diff_rgrs.save(DiffRaster)

        # create difference with SNODAS
        Diff_SNODAS = Raster(ClipSNODAS) - Raster(product8)
        Diff_SNODAS.save(SWE_Diff)

        print("creating mean mask")
        MeanMapMsk = Raster(meanMap_proj) * Raster(watermask)
        MeanMapALlMsk = MeanMapMsk * Raster(glacier_mask)
        MeanMapALlMsk_2 = MeanMapALlMsk * Raster(domain_mask)
        MeanMapALlMsk_2.save(meanMapMask)

        # Create GT 0 mean blended swe and make mask
        con01 = Con(Raster(meanMapMask) > 0.00, 1, 0)
        con01.save(anomMask)
        #
        # # make anomoly mask
        AnomProd = (Raster(product8) / Raster(meanMapMask)) * 100
        AnomProd.save(anomMap)
        print(f"anomaly map made")

        # # make noneg anomoly map ## ASK LEANNE, DOES THIS NEED TO BE ADJUSTED?
        connoeg = Con(Raster(anomMap) > 200, 200, Raster(anomMap))
        connoeg.save(anom0_100map)

        # # mask with watermaks
        anomnoneg = connoeg * Raster(watermask)
        anomnoneg_Mask = anomnoneg * Raster(glacier_mask)
        prod8Msk = anomnoneg_Mask / Raster(domain_mask)
        prod8Msk.save(anom0_100msk)


        outBandTable = ZonalStatisticsAsTable(band_raster, case_field_band, product8, SWEbandtable, "DATA",
                                              "MEAN")
        outSWETable = ZonalStatisticsAsTable(watershedRas, case_field_wtrshd, product8, SWEtable, "DATA",
                                             "MEAN")
        outSCABand = ZonalStatisticsAsTable(band_raster, case_field_band, modscag_per, scabandtable, "DATA",
                                            "ALL")
        outSCAWtshd = ZonalStatisticsAsTable(watershedRas, case_field_wtrshd, modscag_per, scatable, "DATA",
                                             "ALL")
        arcpy.AddField_management(SWEbandtable, "SWE_IN", "DOUBLE", "", "", "",
                                  "", "NULLABLE", "NON_REQUIRED")
        arcpy.AddField_management(SWEbandtable100, "SWE_IN", "DOUBLE", "", "", "",
                                  "", "NULLABLE", "NON_REQUIRED")
        arcpy.AddField_management(SWEtable, "SWE_IN", "DOUBLE", "#", "#", "#",
                                  "#", "NULLABLE", "NON_REQUIRED", "#")
        arcpy.AddField_management(SWEbandtable, "AREA_MI2", "DOUBLE", "#", "#", "#",
                                  "#", "NULLABLE", "NON_REQUIRED", "#")
        arcpy.AddField_management(SWEtable, "AREA_MI2", "DOUBLE", "#", "#", "#",
                                  "#", "NULLABLE", "NON_REQUIRED", "#")
        arcpy.AddField_management(SWEbandtable100, "VOL_M3", "DOUBLE", "#", "#", "#",
                                  "#", "NULLABLE", "NON_REQUIRED", "#")
        arcpy.AddField_management(SWEbandtable, "VOL_M3", "DOUBLE", "#", "#", "#",
                                  "#", "NULLABLE", "NON_REQUIRED", "#")
        arcpy.AddField_management(SWEtable, "VOL_M3", "DOUBLE", "#", "#", "#",
                                  "#", "NULLABLE", "NON_REQUIRED", "#")
        arcpy.AddField_management(SWEbandtable100, "VOL_AF", "DOUBLE", "#", "#", "#",
                                  "#", "NULLABLE", "NON_REQUIRED", "#")
        arcpy.AddField_management(SWEbandtable, "VOL_AF", "DOUBLE", "#", "#", "#",
                                  "#", "NULLABLE", "NON_REQUIRED", "#")
        arcpy.AddField_management(SWEtable, "VOL_AF", "DOUBLE", "#", "#", "#",
                                  "#", "NULLABLE", "NON_REQUIRED", "#")

        # calculate fields
        arcpy.CalculateField_management(SWEbandtable, "SWE_IN", "!MEAN! * 39.370079", "PYTHON")
        arcpy.CalculateField_management(SWEbandtable100, "SWE_IN", "!MEAN! * 39.370079", "PYTHON")
        arcpy.CalculateField_management(SWEtable, "SWE_IN", "!MEAN! * 39.370079", "PYTHON")

        # Calculate area in sq miles
        arcpy.CalculateField_management(SWEbandtable, "AREA_MI2", "0.00000038610216 * !AREA!", "PYTHON")
        arcpy.CalculateField_management(SWEtable, "AREA_MI2", "0.00000038610216 * !AREA!", "PYTHON")

        # Calculate volume in cubic meters
        arcpy.CalculateField_management(SWEbandtable, "VOL_M3", "!MEAN! * !AREA!", "PYTHON")
        arcpy.CalculateField_management(SWEtable, "VOL_M3", "!MEAN! * !AREA!", "PYTHON")
        arcpy.CalculateField_management(SWEbandtable100, "VOL_M3", "!MEAN! * !AREA!", "PYTHON")

        # Calculate volume in acre feet
        arcpy.CalculateField_management(SWEbandtable100, "VOL_AF", "!VOL_M3! * 0.000810714", "PYTHON")
        arcpy.CalculateField_management(SWEbandtable, "VOL_AF", "!VOL_M3! * 0.000810714", "PYTHON")
        arcpy.CalculateField_management(SWEtable, "VOL_AF", "!VOL_M3! * 0.000810714", "PYTHON")

        ### Sort by bandname and watershed name, 2 tables
        arcpy.Sort_management(SWEbandtable, SWEbandtable_save, [[case_field_band, "ASCENDING"]])
        arcpy.Sort_management(SWEtable, SWEtable_save, [[case_field_wtrshd, "ASCENDING"]])
        arcpy.Sort_management(SWEbandtable100, SWEbandtable100_save, [["Value", "ASCENDING"]])

        ## work on SCA tables
        arcpy.AddField_management(scabandtable, "Percent", "DOUBLE", "", "", "",
                                  "", "NULLABLE", "NON_REQUIRED")
        arcpy.AddField_management(scatable, "Percent", "DOUBLE", "", "", "",
                                  "", "NULLABLE", "NON_REQUIRED")
        # calculate percent
        arcpy.CalculateField_management(scabandtable, "Percent", "( !SUM! / !COUNT! ) * 100", "PYTHON", "")
        arcpy.CalculateField_management(scatable, "Percent", "( !SUM! / !COUNT! ) * 100", "PYTHON", "")

        # sort
        arcpy.Sort_management(scabandtable, scabandtable_save, [[case_field_band, "ASCENDING"]])
        arcpy.Sort_management(scatable, scatable_save, [[case_field_wtrshd, "ASCENDING"]])

        print("Create SWE and mean zonal maps...")
        # NEED TO ADD IN MEAN MASK
        swezmap = ZonalStatistics(watershedRas, case_field_wtrshd, product8, "MEAN", "DATA")
        meanzmap = ZonalStatistics(watershedRas, case_field_wtrshd, meanMapMask, "MEAN", "DATA")
        swezmap.save(SWEzoneMap)
        meanzmap.save(MeanzoneMap)

        # NEED TO ADD IN MEAN MASK
        print("creating product 9...")
        proj9 = (Raster(SWEzoneMap) / Raster(MeanzoneMap)) * 100
        proj9.save(product9)

        # # make anomoly mask
        prod8Msk = (Raster(product8) / Raster(domain_mask))
        prod8Msk.save(prod8msk)

        # creating banded watershed mean and swe
        tswebzmap = ZonalStatistics(band_raster, case_field_band, product8, statisticType, "DATA")
        tmeanbzmap = ZonalStatistics(band_raster, case_field_band, meanMapMask, statisticType, "DATA")
        tswebzmap.save(SWEbandzoneMap)
        tmeanbzmap.save(MeanBandZoneMap)

        print("creating product 10 = " + product10)
        prod10 = (Raster(SWEbandzoneMap) / Raster(MeanBandZoneMap)) * 100
        prod10.save(product10)

        # # # make anomoly mask
        # prod7Msk = (Raster(product7) / Raster(domain_mask)) * 100
        # prod7Msk.save(prod7msk)
        # print(f"anomaly map made")

        print("create anomaly layer table = " + anomTable)
        anomt = ZonalStatisticsAsTable(watershedRas, case_field_wtrshd, product9, anomTable, "DATA", "MEAN")
        anomRt = ZonalStatisticsAsTable(region, "Regions", product9, anomRegionTable, "DATA", "MEAN")
        anombt = ZonalStatisticsAsTable(band_raster, case_field_band, product10, anombandTable, "DATA", "MEAN")


        # Sort by bandname and watershed name, 3 tables
        arcpy.Sort_management(anombandTable, anombandTable_save, [[case_field_band, "ASCENDING"]])
        arcpy.Sort_management(anomTable, anomTable_save, [[case_field_wtrshd, "ASCENDING"]])

        # add field for anom
        arcpy.AddField_management(anombandTable_save, "Average", "DOUBLE", "", "", "",
                                  "", "NULLABLE", "NON_REQUIRED")
        arcpy.AddField_management(anomTable_save, "Average", "DOUBLE", "", "", "",
                                  "", "NULLABLE", "NON_REQUIRED")

        # calculate field
        arcpy.CalculateField_management(anombandTable_save, "Average", f"!MEAN!", "PYTHON3")
        arcpy.CalculateField_management(anomTable_save, "Average", f"!MEAN!", "PYTHON3")

        print("Joining sorted tables ... ")
        ## Delete extra fields from tables before joining them
        ## Banded Tables
        arcpy.DeleteField_management(SWEbandtable_save, "ZONE_CODE")
        arcpy.DeleteField_management(anombandTable_save, "ZONE_CODE;COUNT")
        arcpy.DeleteField_management(scabandtable_save,
                                     "ZONE_CODE;COUNT;MIN;MAX;RANGE;MEAN;STD;SUM;VARIETY;MAJORITY;MINORITY;MEDIAN")
        ## Watershed tables
        arcpy.DeleteField_management(SWEtable_save, "ZONE_CODE")
        arcpy.DeleteField_management(anomTable_save, "ZONE_CODE;COUNT")
        arcpy.DeleteField_management(scatable_save,
                                     "ZONE_CODE;COUNT;MIN;MAX;RANGE;MEAN;STD;SUM;VARIETY;MAJORITY;MINORITY;MEDIAN")

        ## Make tables into table views for joins
        arcpy.MakeTableView_management(SWEbandtable_save, SWEbandtableView)
        arcpy.MakeTableView_management(SWEtable_save, SWEtableView)

        arcpy.JoinField_management(SWEtable_save, case_field_wtrshd, scatable_save, case_field_wtrshd, "Percent")
        arcpy.JoinField_management(SWEbandtable_save, case_field_band, scabandtable_save, case_field_band, "Percent")


        print("Making csvs...")
        # wtshd_dbf = gpd.read_file(SWEtableView)
        wtshd_dbf = gpd.read_file(SWEtable_save)
        wtshd_df = pd.DataFrame(wtshd_dbf)
        wtshd_df.to_csv(WtshdTableJoinCSV, index=False)

        band_dbf = gpd.read_file(SWEbandtable_save)
        band_df = pd.DataFrame(band_dbf)
        band_df.to_csv(BandtableJoinCSV, index=False)

        anom_dbf = gpd.read_file(anomTable_save)
        anom_df = pd.DataFrame(anom_dbf)
        anom_df.to_csv(anomWtshdTableCSV, index=False)

        anom_band_dbf = gpd.read_file(anombandTable_save)
        anom_band_df = pd.DataFrame(anom_band_dbf)
        anom_band_df.to_csv(anomBandTableCSV, index=False)
