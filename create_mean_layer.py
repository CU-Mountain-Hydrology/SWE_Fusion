
## Import system modules
import os, sys, string
import arcinfo
import rasterio
import numpy as np
import shutil
from arcpy import env
from arcpy.sa import *

# from WW_fileCountsChecks import historicalWorkspace

snapRaster_geon83 = "M:/SWE/WestWide/data/boundaries/SnapRaster_geon83.tif"
snapRaster_albn83 = "M:/SWE/WestWide/data/boundaries/SnapRaster_albn83.tif"
# out_folder = "W:/Spatial_SWE/WW_regression/mean_2000_2025/"
# print("modules imported")


##############################################################

## Set these variables

##############################################################

# Set the compression environment to NONE.
# new_folder = f"H:/WestUS_Data/Regress_SWE/HistoricalDaily_mask/{year}/"
start_year = 2000 
end_year =  2025
dateList = ["0101"]
input_workspace = f"H:/WestUS_Data/Regress_SWE/HistoricalDaily_mask/"
output_folder = "W:/Spatial_SWE/WW_regression/mean_2000_2025_WY26_glacMask/"


import arcpy
import rasterio
import numpy as np
import os

def create_mean_layer(input_workspace, output_folder, dateList, start_year, end_year):
    years = list(range(start_year, (end_year + 1)))
    
    for date in dateList:
        file_list = []
        for year in years:
            print('year ', year)
            folder = input_workspace + str(year) + "/"
            file = f"WW_phvrcn_{year}{date}_fscamsk_glacMask.tif"
    
            # append to list
            if file in os.listdir(folder):
                file_list.append(folder + file)
    
        arrays = []
        meta = None
    
        for r in file_list:
            with rasterio.open(r) as src:
                arr = src.read(1).astype("float32")
    
                # Convert NoData to NaN
                if src.nodata is not None:
                    arr[arr == src.nodata] = np.nan
    
                arrays.append(arr)
    
                if meta is None:
                    meta = src.meta.copy()
    
        # Stack → (n_rasters, rows, cols)
        stack = np.stack(arrays)
        mean_raster = np.nanmean(stack, axis=0)
        meta.update(dtype="float32", nodata=np.nan)
        out_raster = output_folder + f"WW_{date}_fscamsk_glacMask_mean.tif"
    
        with rasterio.open(out_raster, "w", **meta) as dst:
            dst.write(mean_raster, 1)
    
        print("Mean raster written:", out_raster)
