# import modules
import arcpy
from arcpy.sa import *
import pandas as pd
import os
from datetime import datetime, timedelta

from sqlalchemy.sql.functions import current_date

print("modules imported")

startDate = datetime(2000, 10, 1)
endDate = datetime(2000, 12, 31)
start_yyyymmdd = startDate.strftime("%Y%m%d")
end_yyyymmdd = endDate.strftime("%Y%m%d")
domains = ["SNM", "PNW", "INMT", "SOCN", "NOCN"]
mainWorkspace = "H:/WestUS_Data/Regress_SWE/"
output_folder = r"H:/WestUS_Data/Regress_SWE/HistoricalDaily/"
no_data_csv = r"H:/WestUS_Data/Regress_SWE/HistoricalDaily/NoData_Domains.csv"

# set snapRasters
projGEO = arcpy.SpatialReference(4269)
projALB = arcpy.SpatialReference(102039)
snapRaster_geon83 = "M:/SWE/WestWide/data/boundaries/SnapRaster_geon83.tif"
snapRaster_albn83 = "M:/SWE/WestWide/data/boundaries/SnapRaster_albn83.tif"
glacierMask = "M:/SWE/WestWide/data/mask/glacierMask/glims_glacierMask_null_GT10_final.tif"
arcpy.env.snapRaster = snapRaster_geon83
arcpy.env.cellSize = snapRaster_geon83
arcpy.env.outputCoordinateSystem = projGEO

# read in csv
df = pd.read_csv(no_data_csv)
df["Date"] = df["Date"].astype(str)
nd_dates_list = df["Date"].tolist()
print(f"Number of dates with No Data Swaps: {len(nd_dates_list)}")

current = startDate
while current <= endDate:

    print('\nProcessing date: ', current)
    # set variables and folders
    current_yyyymmdd = current.strftime("%Y%m%d")
    year = current.year
    year_dir = os.path.join(output_folder, str(year))
    date_dir = os.path.join(year_dir, str(current_yyyymmdd))

    if os.path.exists(year_dir + f"/WW_phvrcn_{current_yyyymmdd}_fscamsk_clp.tif"):
        print(f"Model run exists for {current_yyyymmdd}")
        # check to see if there list a data list
        if current_yyyymmdd in nd_dates_list:
            print(f"Swapping in No Data values for {current_yyyymmdd}")
            matching_rows = df.loc[df["Date"] == current_yyyymmdd, "Domain_List"]
            domain_list = matching_rows.iloc[0]

            if isinstance(domain_list, str):
                # Remove brackets and split
                domain_list = domain_list.strip('[]').replace("'", "").split(',')
                domain_list = [d.strip() for d in domain_list if d.strip()]

            print(f"Domains to process: {domain_list}")

            for domain in domain_list:
                # set to no data
                domain_file_path = f"{date_dir}/{domain}_phvrcn_{current_yyyymmdd}_fscamsk_clp.tif"
                old_vals_path = f"{domain_file_path[:-4]}_old_vals.tif"
                temp_nodata_path = f"{domain_file_path[:-4]}_temp_nodata.tif"

                nodata_ras = SetNull(Raster(domain_file_path) >= -99999, Raster(domain_file_path))
                nodata_ras.save(temp_nodata_path)

                # rename original raster
                arcpy.management.Rename(domain_file_path, old_vals_path)
                arcpy.management.Rename(temp_nodata_path, domain_file_path)
                print('No data values swapped for', domain)

            outCutsList = [os.path.join(date_dir, f) for f in os.listdir(date_dir) if
                           f.endswith(f"{current_yyyymmdd}_fscamsk_clp.tif")]
            arcpy.MosaicToNewRaster_management(outCutsList, date_dir, f"/WW_phvrcn_{current_yyyymmdd}_fscamsk_clp.tif",
                                               projGEO, "32_BIT_FLOAT", ".005 .005", "1", "LAST")
            print('new mosaicked raster created.')

        else:
            print('No data values needed swapped for', current_yyyymmdd)

        # add glacier mask
        if os.path.exists(date_dir + f"/WW_phvrcn_{current_yyyymmdd}_fscamsk_clp.tif"):
            non_masked_WW = date_dir + f"/WW_phvrcn_{current_yyyymmdd}_fscamsk_clp.tif"
        else:
            non_masked_WW = year_dir + f"/WW_phvrcn_{current_yyyymmdd}_fscamsk_clp.tif"

        outGlaciers = Raster(non_masked_WW) * Raster(glacierMask)
        outGlaciers.save(year_dir + f"/WW_phvrcn_{current_yyyymmdd}_fscamsk_glacMask.tif")

    else:
        print('No model run for this date')

    # moving forward a date
    current += timedelta(days=1)
    print(rf"new current date = {current}")







