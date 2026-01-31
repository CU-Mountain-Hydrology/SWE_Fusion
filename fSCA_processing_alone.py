import os
import pandas as pd
import datetime as dt
import shutil
import time
from SWE_Fusion_functions import *

start_date = datetime(2026, 1, 26)
end_date = datetime(2026, 1, 30)
dateList = ["0126", "0127", "0128", "0129", "0130"]
##############
netCDF_WS = "H:/WestUS_Data/Rittger_data/fsca_v2025.0.1_ops/NRT_NetCDFs/netcdf/"
tile_list = ['h08v04', 'h08v05', 'h09v04', 'h09v05', 'h10v04']
output_fscaWS = "H:/WestUS_Data/Rittger_data/fsca_v2025.0.1_ops/NRT_FSCA_WW_N83/"
SinuModisProj = r"M:/SWE/WestWide/data/basemap/CoordinateSystems/Sinusodial_MODIS_custom_ET.prj"
proj_in = arcpy.SpatialReference(SinuModisProj)
snap_raster = "M:/SWE/WestWide/data/boundaries/SnapRaster_geon83.tif"
extent = rf"M:/SWE/WestWide/data/boundaries/Margulis_domain_geon83.shp"
proj_out = arcpy.SpatialReference(4269)
start_year = 2000
end_year =  2025
input_workspace = f"H:/WestUS_Data/Regress_SWE/HistoricalDaily_mask/"
output_folder = "W:/Spatial_SWE/WW_regression/mean_2000_2025_WY26_glacMask/"

print('Processing fSCA Data...')
fsca_processing_tif(start_date=start_date, end_date=end_date, netCDF_WS=netCDF_WS, tile_list=tile_list,
                    output_fscaWS=output_fscaWS, proj_in=proj_in, snap_raster=snap_raster, extent=extent, proj_out=proj_out)


# Calculate DMFSCA for December 2024
print('Processing DMFSCA Data...')
calculate_dmfsca(
    fSCA_folder="H:/WestUS_Data/Rittger_data/fsca_v2025.0.1_ops/NRT_FSCA_WW_N83/",
    DMFSCA_folder="H:/WestUS_Data/Rittger_data/fsca_v2025.0.1_ops/NRT_DMFSCA_WW_N83/",
    wateryear_start=datetime(2025, 10, 1),
    process_start_date=start_date,
    process_end_date=end_date
)

# calculate mean layer
print('Creating Mean Layer...')
create_mean_layer(input_workspace=input_workspace, output_folder=output_folder, dateList=dateList, start_year=start_year,
                  end_year=end_year)

# sleep and remove layers
clear_arcpy_locks()

# remove intermediary directories
shutil.rmtree(output_fscaWS + "/2026/intermediary/")
shutil.rmtree(output_fscaWS + "/2026/projected/")
shutil.rmtree(output_fscaWS + "/2026/outTifs/")
