
## Import system modules
import os, sys, string
import arcinfo
import arcpy
import shutil
from arcpy import env
from arcpy.sa import *
from datetime import datetime, timedelta
snapRaster_geon83 = "M:/SWE/WestWide/data/boundaries/SnapRaster_geon83.tif"
snapRaster_albn83 = "M:/SWE/WestWide/data/boundaries/SnapRaster_albn83.tif"
print("modules imported")

##############################################################

## Set these variables

##############################################################

# Set the compression environment to NONE.
# arcpy.env.compression = "NONE"

startDate = datetime(2000, 11, 1)
endDate = datetime(2000, 12, 31)
start_yyyymmdd = startDate.strftime("%Y%m%d")
end_yyyymmdd = endDate.strftime("%Y%m%d")

domains = ["SNM", "PNW", "INMT", "SOCN", "NOCN"]

mainWorkspace = "H:/WestUS_Data/Regress_SWE/"
output_folder = r"H:/WestUS_Data/Regress_SWE/HistoricalDaily/"

current = startDate
while current <= endDate:
    current_yyyymmdd = current.strftime("%Y%m%d")
    # cutLinesWorkspace = mainWorkspace + f"April1_Mean/cliplines/"
    clipFilesWorkspace = "M:/SWE/WestWide/data/boundaries/Domains/DomainCutLines/complete/"
    projGEO = arcpy.SpatialReference(4269)
    projALB = arcpy.SpatialReference(102039)

    # make output folder
    year = current.year
    year_dir = os.path.join(output_folder, str(year))
    os.makedirs(year_dir, exist_ok=True)

    print(f"\nProcessing {current_yyyymmdd}")
    date_dir = os.path.join(year_dir, str(current_yyyymmdd))
    os.makedirs(date_dir, exist_ok=True)

    for domain in domains:
        arcpy.env.snapRaster = snapRaster_geon83
        arcpy.env.cellSize = snapRaster_geon83
        arcpy.env.outputCoordinateSystem = projGEO
        folder = mainWorkspace + f"{domain}/Leanne/StationSWERegressionV2/data/outputs/Hist_CanAdj_rcn_woCCR_nofscamsk/wMsk/"
        file = folder + f"{domain}_phvrcn_{current_yyyymmdd}_fscamsk.tif"

        if os.path.exists(file):
            print(f"Processing {domain}")
            # extract by mask
            outCut = ExtractByMask(file, clipFilesWorkspace + f"WW_{domain}_cutline_v2.shp", 'INSIDE')
            outCut.save(date_dir + f"/{domain}_phvrcn_{current_yyyymmdd}_fscamsk_clp.tif")
            print(f"{domain} clipped for {current_yyyymmdd}")

        else:
            print(f"{file} DOES NOT EXIST")

    # mosaic all tifs together
    if not os.path.exists(file):
        print('not date, moving on')
    else:
        arcpy.env.snapRaster = snapRaster_geon83
        arcpy.env.cellSize = snapRaster_geon83
        arcpy.env.outputCoordinateSystem = projGEO
        outCutsList = [os.path.join(date_dir, f) for f in os.listdir(date_dir) if f.endswith(f"{current_yyyymmdd}_fscamsk_clp.tif")]
        arcpy.MosaicToNewRaster_management(outCutsList, year_dir, f"/WW_phvrcn_{current_yyyymmdd}_fscamsk_clp.tif",
                                           projGEO, "32_BIT_FLOAT", ".005 .005", "1", "LAST")
        print('mosaicked raster created. ')


    # move to next date
    current += timedelta(days=1)
    print(rf"new current date = {current}")