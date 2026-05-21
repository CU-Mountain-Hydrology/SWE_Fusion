import arcpy
from arcpy.sa import *
import os
print('modules imported')

huc2 = r"C:\Users\etyrr\OneDrive\Documents\CU_Grad\forNYT\HUC2_raster_albn83_500.tif"
huc6 = "M:/SWE/WestWide/data/hydro/WW_HUC6_albn83_ras_msked.tif"

dates = ['20260301', '20260308', '20260315']
mean_map = ['0301', '0308', '0315']

in_workspace = r'W:/Spatial_SWE/WW_regression/RT_report_data/'
out_workspace = 'C:/Users/etyrr/OneDrive/Documents/CU_Grad/forNYT/'

for date, mean_date in zip(dates, mean_map):
    os.makedirs(f"{out_workspace}/{date}/", exist_ok=True)
    SWEHuc2Map = f"{out_workspace}/{date}/HUC2_{date}_SWE.tif"
    MeanHuc2Map = f"{out_workspace}/{date}/HUC2_{date}_MeanSWE.tif"
    Anom_Huc2 = f"{out_workspace}/{date}/HUC2_{date}_anom.tif"
    SWEHuc6Map = f"{out_workspace}/{date}/HUC6_{date}_SWE.tif"
    MeanHuc6Map = f"{out_workspace}/{date}/HUC6_{date}_MeanSWE.tif"
    Anom_Huc6 = f"{out_workspace}/{date}/HUC6_{date}_anom.tif"


    mean_layer = in_workspace + f"{date}_results/RT_CanAdj_rcn_woCCR_nofscamskSens_UseAvg/WW_{mean_date}_mean_msk_albn83.tif"
    swe_layer = in_workspace + f"{date}_results/RT_CanAdj_rcn_woCCR_nofscamskSens_UseAvg/p8_{date}_noneg.tif"

    print("created product = HUC 2 percent of average")
    swezmap = ZonalStatistics(huc2, "Name", swe_layer, "MEAN", "DATA")
    meanzmap = ZonalStatistics(huc2, "Name", mean_layer, "MEAN", "DATA")
    swezmap.save(SWEHuc2Map)
    meanzmap.save(MeanHuc2Map)

    prod11 = (Raster(SWEHuc2Map) / Raster(MeanHuc2Map)) * 100
    prod11.save(Anom_Huc2)
    print('saved huc 2')

    print("created product = HUC 6 percent of average")
    swezmap_huc6 = ZonalStatistics(huc6, "name", swe_layer, "MEAN", "DATA")
    meanzmap_huc6 = ZonalStatistics(huc6, "Name", mean_layer, "MEAN", "DATA")
    swezmap.save(SWEHuc6Map)
    meanzmap.save(MeanHuc6Map)

    prod11 = (Raster(SWEHuc6Map) / Raster(MeanHuc6Map)) * 100
    prod11.save(Anom_Huc6)
    print('saved huc 6')