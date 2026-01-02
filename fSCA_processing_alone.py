import os
import pandas as pd
import datetime as dt
from SWE_Fusion_functions import *

print("\nProcessing fSCA data...")
start_date = datetime(2025, 12, 11)
end_date = datetime(2026, 1, 1)
netCDF_WS = "H:/WestUS_Data/Rittger_data/fsca_v2024.1.0_ops/NRT_NetCDFs/netcdf/"
tile_list = ['h08v04', 'h08v05', 'h09v04', 'h09v05', 'h10v04']
output_fscaWS = "H:/WestUS_Data/Rittger_data/fsca_v2024.1.0_ops/NRT_FSCA_WW_N83/"
SinuModisProj = r"M:/SWE/WestWide/data/basemap/CoordinateSystems/Sinusodial_MODIS_custom_ET.prj"
proj_in = arcpy.SpatialReference(SinuModisProj)
snap_raster = "M:/SWE/WestWide/data/boundaries/SnapRaster_geon83.tif"
extent = rf"M:/SWE/WestWide/data/boundaries/Margulis_domain_geon83.shp"
proj_out = arcpy.SpatialReference(4269)

print('processing fSCA Data...')
fsca_processing_tif(start_date=start_date, end_date=end_date, netCDF_WS=netCDF_WS, tile_list=tile_list,
                    output_fscaWS=output_fscaWS, proj_in=proj_in, snap_raster=snap_raster, extent=extent, proj_out=proj_out)