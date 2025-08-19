# This is a test to work on the sensors and layers code for the differnt domains

# import modules
import arcpy
import arcpy
from arcpy import env
from arcpy.ra import ZonalStatisticsAsTable
from arcpy.sa import *
import pandas as pd
import geopandas as gpd
import os
# from tables_layer_testing_code import *
print('modules imported')

# establish paths
user = "Leanne"
year = "2025"

# set dates
report_date = "20250525"
mean_date = "0528"
prev_report_date = "20250517"

# set model run info
model_run = "RT_CanAdj_rcn_noSW_woCCR_nofscamsk"
prev_model_run = "ASO_FixLayers_fSCA_RT_CanAdj_rcn_noSW_woCCR_UseThis"
masking = "N"
bias = "n"

#zones
watershedRas = "M:/SWE/WestWide/data/hydro/WW_BasinBanded_noSNM_notahoe_albn83_sel.tif"
huc6_raster = "M:/SWE/WestWide/data/hydro/WW_HUC6_albn83_ras_msked.tif"
band_raster = "M:/SWE/WestWide/data/hydro/WW_BasinBanded_noSNM_notahoe_albn83_sel.tif"
region_raster = "M:/SWE/WestWide/data/hydro/WW_Regions_albn83_v2.tif"
watermask = "M:/SWE/WestWide//data/mask/watermask_outdated/outdated/WW/Albers/WW_watermaks_10_albn83_clp_final.tif"
glacierMask = "M:/SWE/WestWide/data/mask/glacierMask/glims_glacierMask_null_GT10_final.tif"

case_field_wtrshd = "SrtName"
case_field_band = "SrtNmeBand"

def tables_and_layers(user, year, report_date, mean_date, prev_report_date, model_run, prev_model_run, masking, watershed_zones,
                      band_zones, HUC6_zones, region_zones, case_field_wtrshd, case_field_band, watermask, glacierMask, bias):

    # set code parameters
    where_clause = """"POLY_AREA" > 100"""
    part_area = "100 SquareKilometers"

    snapRaster_geon83 = "M:/SWE/WestWide/data/boundaries/SnapRaster_geon83.tif"
    snapRaster_albn83 = "M:/SWE/WestWide/data/boundaries/SnapRaster_albn83.tif"
    projGEO = arcpy.SpatialReference(4269)
    projALB = arcpy.SpatialReference(102039)
    ProjOut_UTM = arcpy.SpatialReference(26911)

    #######################################################################
    # End of Setting Variables
    #######################################################################
    workspaceBase = fr"M:/SWE/WestWide/Spatial_SWE/WW_regression/"
    resultsWorkspace = workspaceBase + f"RT_report_data/{report_date}_results_ET/"

    os.makedirs(resultsWorkspace, exist_ok=True)

    # create directory for model run
    if masking == "Y":
        RunNameMod = f"fSCA_{model_run}"
    else:
        RunNameMod = model_run

    # create directory
    if bias == "N":
        arcpy.CreateFolder_management(resultsWorkspace, RunNameMod)
        outWorkspace = resultsWorkspace + RunNameMod + "/"
        print("model run workspace created")

    if bias == "Y":
        outWorkspace = resultsWorkspace + RunNameMod + "/"

    meanWorkspace = workspaceBase + "mean_2001_2021_Nodmfsca/"
    prevRepWorkspace = workspaceBase + f"RT_report_data/{prev_report_date}_results/{prev_model_run}/"

    meanMask = outWorkspace + f"{mean_date}_mean_msk.tif"
    MODSCAG_tif_plus = f"H:/WestUS_Data/Rittger_data/fsca_v2024.0d/NRT_FSCA_WW_N83/{year}/{report_date}.tif"
    MODSCAG_tif_plus_proj = outWorkspace + f"fSCA_{report_date}_albn83.tif"

    # define snow-no snow layer
    modscag_0_1 = outWorkspace + f"modscag_0_1_{report_date}.tif"
    modscag_per = outWorkspace + f"modscag_per_{report_date}.tif"
    modscag_per_msk = outWorkspace + f"modscag_per_{report_date}_msk.tif"

    # define snow/no snow null layer
    mod_null = outWorkspace + f"modscag_0_1_{report_date}msk_null.tif"
    mod_poly = outWorkspace + f"modscag_0_1_{report_date}msk_null_poly.shp"
    ### ASK LEANNE ABOUT UTM
    mod_poly_utm = outWorkspace + f"modscag_0_1_{report_date}_msk_null_poly_utm.shp"

    snowPolySel = outWorkspace + f"modscag_{report_date}_snowline_Sel.shp"
    snowPolyElim = outWorkspace + f"modscag_{report_date}_snowline_Sel_elim.shp"

    # define snow pillow gpkg
    meanMap = meanWorkspace + f"WW_{mean_date}_mean_geon83.tif"
    meanMap_copy = outWorkspace + f"WW_{mean_date}_mean_geon83.tif"
    meanMap_proj = outWorkspace + f"WW_{mean_date}_mean_albn83.tif"
    meanMapMask = outWorkspace + f"WW_{mean_date}_mean_msk_albn83.tif"
    lastRast = prevRepWorkspace + f"p8_{prev_report_date}_noneg.tif"
    DiffRaster = outWorkspace + f"Diff_{report_date}_{prev_report_date}.tif"

    ## define rasters
    rcn_raw = outWorkspace + f"WW_{report_date}_phvrcn_mos_noMask.tif"
    rcn_glacMask = outWorkspace + f"WW_{report_date}_phvrcn_mos_masked.tif"
    rcn_raw_proj = outWorkspace + f"WW_{report_date}_phvrcn_albn83.tif"
    rcnFinal = outWorkspace + f"phvrcn_{report_date}_final.tif"
    product7 = outWorkspace + f"p7_{report_date}.tif"
    product7_noFsca = outWorkspace + f"p7_{report_date}_nofsca.tif"
    product8 = outWorkspace + f"p8_{report_date}_noneg.tif"
    prod8msk = outWorkspace + f"p8_{report_date}_noneg_msk.tif"
    product9 = outWorkspace + f"p9_{report_date}.tif"
    product10 = outWorkspace + f"p10_{report_date}.tif"
    product11 = outWorkspace + f"p11_{report_date}.tif"
    product12 = outWorkspace + f"p12_{report_date}.tif"

    # output Tables
    SWEbandtable = outWorkspace + f"{report_date}band_swe_table.dbf"
    SWEtable = outWorkspace + f"{report_date}swe_table.dbf"
    SWEbandtable100 = outWorkspace + f"{report_date}swe_table_100.dbf"
    SWEbandtable_save = outWorkspace + f"{report_date}band_swe_table_save.dbf"
    SWEtable_save = outWorkspace + f"{report_date}swe_table_save.dbf"
    SWEbandtable100_save = outWorkspace + f"{report_date}swe_table_100_save.dbf"

    # anomoly tables
    anombandTable = outWorkspace + f"{report_date}band_anom_table.dbf"
    anomTable = outWorkspace + f"{report_date}anom_table.dbf"
    anomHuc6Table = outWorkspace + f"{report_date}huc6_anom_table.dbf"
    anomHuc6Table_save = outWorkspace + f"{report_date}huc6_anom_table_save.dbf"
    meanTable = outWorkspace + f"{report_date}mean_table.dbf"
    anombandTable_save = outWorkspace + f"{report_date}band_anom_table_save.dbf"
    anomTable_save = outWorkspace + f"{report_date}anom_table_save.dbf"
    meanTable_save = outWorkspace + f"{report_date}mean_table_save.dbf"

    # region tables
    anomRegionTable = outWorkspace + f"{report_date}anomRegion_table.dbf"
    anomRegionTable_save = outWorkspace + f"{report_date}anomRegion_table_save.dbf"

    # Modscag 0/1 tables and % tables
    scabandtable = outWorkspace + f"{report_date}band_sca_table.dbf"
    scatable = outWorkspace + f"{report_date}sca_table.dbf"
    scabandtable_save = outWorkspace + f"{report_date}band_sca_table_save.dbf"
    scatable_save = outWorkspace + f"{report_date}_sca_table_save.dbf"
    perbandtable = outWorkspace + f"{report_date}band_per_table.dbf"
    pertable = outWorkspace + f"{report_date}_per_table.dbf"

    # create tempoary view for join
    SWEbandtableView = outWorkspace + f"{report_date}band_swe_table_view.dbf"
    SWEtableView = outWorkspace + f"{report_date}swe_table_view.dbf"

    # create joined tables
    BandtableJoin = outWorkspace + f"{report_date}band_table.dbf"
    WtshdTableJoin = outWorkspace + f"{report_date}Wtshd_table.dbf"

    # Anomaly maps
    anomMap = outWorkspace + f"{report_date}_anom.tif"
    anom0_100map = outWorkspace + f"{report_date}anom0_200.tif"
    anom0_100msk = outWorkspace + f"{report_date}anom0_200_msk.tif"

    #SWE maps
    SWEzoneMap = outWorkspace + f"{report_date}_swe_wshd.tif"
    SWEHuc6Map = outWorkspace + f"{report_date}_swe_huc6.tif"
    MeanHuc6Map = outWorkspace + f"{report_date}_mean_huc6.tif"
    MeanzoneMap = outWorkspace + f"{report_date}_mean_wshd.tif"
    SWEbandzoneMap = outWorkspace + f"{report_date}_swe_band_wshd.tif"
    MeanBandZoneMap = outWorkspace + f"{report_date}_mean_band_wshd.tif"
    SWEregionMap = outWorkspace + f"{report_date}_swe_region.tif"
    MeanRegionMap = outWorkspace + f"{report_date}_mean_region.tif"

    # mean layer masked for use in creating anomly map
    anomMask = outWorkspace + f"{report_date}_anom_mask.tif"

    # statistic
    statisticType = "MEAN"

    # final output csv tables
    WtshdTableJoinCSV = outWorkspace + f"{report_date}Wtshd_table.csv"
    BandtableJoinCSV = outWorkspace + f"{report_date}band_table.csv"
    anomRegionTableCSV = outWorkspace + f"{report_date}anomRegion_table.csv"
    Band100TableCSV = outWorkspace + f"{report_date}band_table_100.csv"
    SCATableJoinCSV = outWorkspace + f"{report_date}sca_Wtshd_table.csv"
    BandSCAtableJoinCSV = outWorkspace + f"{report_date}sca_band_table.csv"
    anomWtshdTableCSV = outWorkspace + f"{report_date}anomWtshd_table.csv"
    anomBandTableCSV = outWorkspace + f"{report_date}anomBand_table.csv"
    anomHUC6TableCSV = outWorkspace + f"{report_date}anomHUC6_table.csv"
    print("file paths established")

    # domain model runs
    if bias == "N":
        print("Starting process for clipping files....")

        domains = ["SNM", "PNW", "INMT", "SOCN", "NOCN"]
        clipFilesWorkspace = "M:/SWE/WestWide/data/boundaries/Domains/DomainCutLines/complete/"

        print("making clip workspace...")
        arcpy.CreateFolder_management(outWorkspace, "cutlines")
        cutLinesWorkspace = outWorkspace + "cutlines/"

        for domain in domains:
            MODWorkspace = fr"H:/WestUS_Data/Regress_SWE/{domain}/{user}/StationSWERegressionV2/"
            arcpy.env.snapRaster = snapRaster_geon83
            arcpy.env.cellSize = snapRaster_geon83
            modelTIF = MODWorkspace + f"data/outputs/{model_run}/{domain}_phvrcn_{report_date}.tif"

            # extract by mask
            outCut = ExtractByMask(modelTIF, clipFilesWorkspace + f"WW_{domain}_cutline_v2.shp", 'INSIDE')
            outCut.save(cutLinesWorkspace + f"{domain}_{report_date}_clp.tif")
            print(f"{domain} clipped")

        # mosaic all tifs together
        arcpy.env.snapRaster = snapRaster_geon83
        arcpy.env.cellSize = snapRaster_geon83
        outCutsList = [os.path.join(cutLinesWorkspace, f) for f in os.listdir(cutLinesWorkspace) if f.endswith(".tif")]
        arcpy.MosaicToNewRaster_management(outCutsList, outWorkspace, f"WW_{report_date}_phvrcn_mos_noMask.tif",
                                           projGEO, "32_BIT_FLOAT", ".005 .005", "1", "LAST")
        print('mosaicked raster created. ')

        ## apply glacier mask
        outGlaciers = Raster(rcn_raw) * Raster(glacierMask)
        outGlaciers.save(rcn_glacMask)
        print("data glaciers masks")

    ########################
    print(f"Processing begins...")
    ## copy in mean map
    arcpy.CopyRaster_management(meanMap, meanMap_copy)

    print("Project both fSCA and phvRaster...")
    # project fSCA image
    arcpy.env.snapRaster = snapRaster_albn83
    arcpy.env.cellSize = snapRaster_albn83
    arcpy.env.extent = snapRaster_albn83
    arcpy.ProjectRaster_management(meanMap_copy, meanMap_proj, projALB,
                                   "NEAREST", "500 500",
                                   "", "", projGEO)

    if bias == "N":
        arcpy.ProjectRaster_management(rcn_glacMask, rcn_raw_proj, projALB,
                                       "NEAREST", "500 500",
                                       "", "")
        arcpy.ProjectRaster_management(MODSCAG_tif_plus, MODSCAG_tif_plus_proj, projALB,
                                       "NEAREST", "500 500",
                                       "", "")
        print("fSCA and rcn raw image and mean map projected")

        mod_01 = Con((Raster(MODSCAG_tif_plus_proj) < 101) & (Raster(MODSCAG_tif_plus_proj) > 0),
                     1, 0)
        mod_01_Wtrmask = mod_01 * Raster(watermask)
        mod_01_AllMaks = mod_01_Wtrmask * Raster(glacierMask)
        mod_01_AllMaks.save(modscag_0_1)
        print(f"fSCA mask tif saved")

        # create fSCA percent layer
        Mod_per = (Float(SetNull(Raster(MODSCAG_tif_plus_proj) > 100, Raster(MODSCAG_tif_plus_proj))) / 100)
        Mod_per.save(modscag_per)
        print(f"fSCA percent layer saved")

        # create fsca percent layer ASK LEANNE, WHAT'S THE DIFFERENT BETWEEN LAKES MASK AND WATER MASK
        mod_01_mask = Con(Raster(modscag_per) > 0.0001, 1, 0)
        mod_per_msk = Raster(watermask) * mod_01_mask
        mod_per_Allmsk = Raster(glacierMask) * mod_per_msk
        mod_per_Allmsk.save(modscag_per_msk)
        print("fSCA percent layer created")

        rcn_final = Raster(rcn_raw_proj) * Raster(watermask)
        rcn_final_wtshd = (Con((IsNull(rcn_final)) & (Raster(modscag_per_msk) >= 0), 0, rcn_final))
        rcn_final_wtshd.save(rcnFinal)
        print("rcn final created")

    # ASK LEANNE, WHAT'S THE DIFFERENCE BETWEEN TIF MASK AND WATER MASK
    print(f"Creating snowline shapefile: {snowPolyElim}")
    mod_01_mask = Raster(modscag_0_1) * Raster(watermask)
    mod_01_mask_glacier = Raster(modscag_0_1) * Raster(glacierMask)
    mod_01_msk_null = SetNull(mod_01_mask_glacier == 0, mod_01_mask_glacier)
    mod_01_msk_null.save(mod_null)

    # Convert raster to polygon
    arcpy.RasterToPolygon_conversion(mod_null, mod_poly, "NO_SIMPLIFY", "Value")
    arcpy.Project_management(mod_poly, mod_poly_utm, ProjOut_UTM)
    arcpy.AddGeometryAttributes_management(mod_poly_utm, "AREA", "", "SQUARE_KILOMETERS", "")
    arcpy.Select_analysis(mod_poly_utm, snowPolySel, where_clause)
    arcpy.EliminatePolygonPart_management(snowPolySel, snowPolyElim, "AREA", part_area, "0", "CONTAINED_ONLY")

    print(f"creating masked SWE product")
    if bias == "N":
        rcn_LT_200 = SetNull(Raster(rcnFinal) > 200, rcnFinal)
        rcn_GT_0 = Con(rcn_LT_200 < 0.0001, 0, rcn_LT_200)
        rcn_GT_0.save(product7)
        # ASK LEANNE ABOUT MASK VS WATERMASK
        rcn_mask = rcn_GT_0 * Raster(watermask)
        rcn_allMask = rcn_mask * Raster(glacierMask)

        if masking == "Y":
            rcn_mask_final = rcn_allMask * modscag_per
        else:
            rcn_mask_final = rcn_allMask
        rcn_mask_final.save(product8)

    print("creating mean mask")
    MeanMapMsk = Raster(meanMap_proj) * Raster(watermask)
    MeanMapALlMsk = MeanMapMsk * Raster(glacierMask)
    MeanMapALlMsk.save(meanMapMask)

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
    anomnoneg_Mask = anomnoneg * Raster(glacierMask)
    anomnoneg_Mask.save(anom0_100msk)

    print("create zonal stats and tables")
    outBandTable = ZonalStatisticsAsTable(band_zones, case_field_band, product8, SWEbandtable, "DATA",
                                          "MEAN")
    outSWETable = ZonalStatisticsAsTable(watershed_zones, case_field_wtrshd, product8, SWEtable, "DATA",
                                         "MEAN")
    outSCABand = ZonalStatisticsAsTable(band_zones, case_field_band, modscag_per, scabandtable, "DATA",
                                        "ALL")
    outSCAWtshd = ZonalStatisticsAsTable(watershed_zones, case_field_wtrshd, modscag_per, scatable, "DATA",
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
    print("fields added")
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
    swezmap = ZonalStatistics(watershed_zones, case_field_wtrshd, product8, "MEAN", "DATA")
    meanzmap = ZonalStatistics(watershed_zones, case_field_wtrshd, meanMapMask, "MEAN", "DATA")
    swezmap.save(SWEzoneMap)
    meanzmap.save(MeanzoneMap)

    # NEED TO ADD IN MEAN MASK
    print("creating product 9...")
    proj9 = (Raster(SWEzoneMap) / Raster(MeanzoneMap)) * 100
    proj9.save(product9)

    # creating banded watershed mean and swe
    # NEED TO ADD IN MEAN MASK
    tswebzmap = ZonalStatistics(band_zones, case_field_band, product8, statisticType, "DATA")
    tmeanbzmap = ZonalStatistics(band_zones, case_field_band, meanMapMask, statisticType, "DATA")
    tswebzmap.save(SWEbandzoneMap)
    tmeanbzmap.save(MeanBandZoneMap)

    # NEED TO ADD IN MEAN MASK
    print("creating product 10 = " + product10)
    prod10 = (Raster(SWEbandzoneMap) / Raster(MeanBandZoneMap)) * 100
    prod10.save(product10)

    print("created product 11 = HUC 6 percent of average")
    swezmap = ZonalStatistics(HUC6_zones, "name", product8, "MEAN", "DATA")
    meanzmap = ZonalStatistics(HUC6_zones, "name", meanMapMask, "MEAN", "DATA")
    swezmap.save(SWEHuc6Map)
    meanzmap.save(MeanHuc6Map)

    prod11 = (Raster(SWEHuc6Map) / Raster(MeanHuc6Map)) * 100
    prod11.save(product11)

    print("created product 12 = region percent of average")
    swezmap = ZonalStatistics(region_zones, "RegionAll", product8, "MEAN", "DATA")
    meanzmap = ZonalStatistics(region_zones, "RegionAll", meanMapMask, "MEAN", "DATA")
    swezmap.save(SWEregionMap)
    meanzmap.save(MeanRegionMap)

    prod11 = (Raster(SWEregionMap) / Raster(MeanRegionMap)) * 100
    prod11.save(product12)


    print("create anomaly layer table = " + anomTable)
    # NEED TO ADD IN MEAN MASK
    anomt = ZonalStatisticsAsTable(watershed_zones, case_field_wtrshd, product9, anomTable, "DATA", "MEAN")
    anombt = ZonalStatisticsAsTable(band_zones, case_field_band, product10, anombandTable, "DATA", "MEAN")
    anomh6 = ZonalStatisticsAsTable(HUC6_zones, "name", product11, anomHuc6Table, "DATA", "MEAN")
    anomreg = ZonalStatisticsAsTable(region_zones, "RegionAll", product12, anomRegionTable, "DATA", "MEAN")

    # NEED TO ADD IN MEAN MASK
    # Sort by bandname and watershed name, 3 tables
    arcpy.Sort_management(anombandTable, anombandTable_save, [[case_field_band, "ASCENDING"]])
    arcpy.Sort_management(anomTable, anomTable_save, [[case_field_wtrshd, "ASCENDING"]])
    arcpy.Sort_management(anomHuc6Table, anomHuc6Table_save, [["name", "ASCENDING"]])
    arcpy.Sort_management(anomRegionTable, anomRegionTable_save, [["RegionAll", "ASCENDING"]])

    # add field for anom
    arcpy.AddField_management(anombandTable_save, "Average", "DOUBLE", "", "", "",
                              "", "NULLABLE", "NON_REQUIRED")
    arcpy.AddField_management(anomTable_save, "Average", "DOUBLE", "", "", "",
                              "", "NULLABLE", "NON_REQUIRED")
    arcpy.AddField_management(anomHuc6Table_save, "Average", "DOUBLE", "", "", "",
                              "", "NULLABLE", "NON_REQUIRED")
    arcpy.AddField_management(anomRegionTable_save, "Average", "DOUBLE", "", "", "",
                              "", "NULLABLE", "NON_REQUIRED")

    # calculate field
    arcpy.CalculateField_management(anombandTable_save, "Average", f"!MEAN!", "PYTHON3")
    arcpy.CalculateField_management(anomTable_save, "Average", f"!MEAN!", "PYTHON3")
    arcpy.CalculateField_management(anomHuc6Table_save, "Average", f"!MEAN!", "PYTHON3")
    arcpy.CalculateField_management(anomRegionTable_save, "Average", f"!MEAN!", "PYTHON3")

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

    band100_dbf = gpd.read_file(SWEbandtable100_save)
    band100_df = pd.DataFrame(band100_dbf)
    band100_df.to_csv(Band100TableCSV)

    anom_dbf = gpd.read_file(anomTable_save)
    anom_df = pd.DataFrame(anom_dbf)
    anom_df.to_csv(anomWtshdTableCSV, index=False)

    anom_band_dbf = gpd.read_file(anombandTable_save)
    anom_band_df = pd.DataFrame(anom_band_dbf)
    anom_band_df.to_csv(anomBandTableCSV, index=False)

    anom_huc_dbf = gpd.read_file(anomHuc6Table_save)
    anom_huc_df = pd.DataFrame(anom_huc_dbf)
    anom_huc_df.to_csv(anomHUC6TableCSV, index=False)

    anom_region_dbf = gpd.read_file(anomRegionTable_save)
    anom_region_df = pd.DataFrame(anom_region_dbf)
    anom_region_dbf.to_csv(anomRegionTableCSV, index=False)


