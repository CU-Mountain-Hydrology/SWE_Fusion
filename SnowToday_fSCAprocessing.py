# This fuction is used to process Snow Today fSCA data from MODIS Sinusodial tiles [h08v04, h08v05, h09v04, h09v05, h10v04].
# This code needs the following folder structures in order to run properly:
### netCDF_WS = f"{root_directory}/{tile_from_tileList}/{year}", for example: C:\fSCA_processing\netcdf\h10v04\2025
### output_fscaWS needs subdirectories as well including: intermediary, outTifs, projected, for example: H:\WestUS_Data\NRT_FSCA_WW_N83\2025\intermediary, outTifs, projected.
#### if these folders aren't created, the code will create them for you. The final output should be in the root of output_fSCAWS.
### this is an arcpy code and the projections should be in the arcpy format, i.e.: projOut = arcpy.SpatialReference(4269)

import arcpy
import os
from arcpy.sa import *
import time
import rasterio
from rasterio.merge import merge
from datetime import datetime, timedelta

def process_fsca_for_date_range(start_date, end_date,
                                 netCDF_WS,
                                 tile_list,
                                 output_fscaWS,
                                 proj_in,
                                 snap_raster,
                                 extent,
                                proj_out):
    proj_out = proj_out
    proj_in = proj_in

    while start_date <= end_date:
        print("\n")
        time.sleep(10)
        arcpy.env.workspace = None
        arcpy.ResetEnvironments()

        yyyymmddd = start_date.strftime("%Y%m%d")
        yyyymmddd_str = str(yyyymmddd)
        print(f"Working on {yyyymmddd} fSCA file")
        current_year = start_date.year

        server = os.path.join(output_fscaWS, str(current_year))
        out_intermediary = os.path.join(server, "intermediary")
        out_files_mos = os.path.join(server, "outTifs")
        out_projected = os.path.join(server, "projected")

        os.makedirs(out_intermediary, exist_ok=True)
        os.makedirs(out_files_mos, exist_ok=True)
        os.makedirs(out_projected, exist_ok=True)

        # NetCDF and output paths
        netCDF_list = [os.path.join(netCDF_WS, tile, str(current_year),
                        f"SPIRES_NRT_{tile}_MOD09GA061_{yyyymmddd_str}_V1.0.nc")
                       for tile in tile_list]

        outTIF_list = [os.path.join(out_intermediary, f"{tile}_Terra_{yyyymmddd_str}.v2024.0d.tif")
                       for tile in tile_list]

        # Convert NetCDF to GeoTIFF
        for netCDF, geotif in zip(netCDF_list, outTIF_list):
            layer_name = f"{geotif[:-4]}_layer"
            arcpy.MakeNetCDFRasterLayer_md(netCDF, "snow_fraction", "x", "y", layer_name, "", "", "BY_VALUE", "CENTER")
            arcpy.CopyRaster_management(layer_name, geotif, pixel_type="32_BIT_FLOAT", format="TIFF")
            if arcpy.Exists(layer_name):
                arcpy.Delete_management(layer_name)
            arcpy.DefineProjection_management(geotif, proj_in)

        # Mosaic
        tif_files_to_mosaic = [rasterio.open(fp) for fp in outTIF_list]
        mosaic_array, out_transform = merge(tif_files_to_mosaic)
        out_meta = tif_files_to_mosaic[0].meta.copy()
        out_meta.update({'driver': 'GTiff', 'count': 1,
                         'height': mosaic_array.shape[1],
                         'width': mosaic_array.shape[2],
                         'transform': out_transform})

        out_mosaic = os.path.join(out_files_mos, f"WUS_Terra_{yyyymmddd_str}_sinu.v2024.0d.tif")
        with rasterio.open(out_mosaic, 'w', **out_meta) as dest:
            dest.write(mosaic_array[0], 1)

        for tif in tif_files_to_mosaic:
            tif.close()

        # Project raster
        arcpy.env.snapRaster = snap_raster
        arcpy.env.extent = snap_raster
        arcpy.env.outputCoordinateSystem = snap_raster
        arcpy.env.cellSize = snap_raster

        projected_tif = os.path.join(out_projected, f"WUS_Terra_{yyyymmddd_str}_sinu.v2024.0d_geon83.tif")
        arcpy.ProjectRaster_management(out_mosaic, projected_tif, proj_out, "NEAREST", ".005")

        # Clip and mask
        clipped = ExtractByMask(projected_tif, extent, "INSIDE")
        clipped_path = os.path.join(out_projected, f"WUS_Terra_{yyyymmddd_str}_sinu.v2024.0d_geon83_GT100.tif")
        clipped.save(clipped_path)

        # fSCA > 100 masking
        final_fsca = Con(Raster(clipped_path) > 100, 100, Raster(clipped_path))
        final_output = os.path.join(server, f"{yyyymmddd_str}.tif")
        final_fsca.save(final_output)
        print("fSCA file created")

        # Cleanup
        deletes = []
        for folder in [out_projected, out_files_mos, out_intermediary]:
            for file in os.listdir(folder):
                if not file.endswith(".tif"):
                    deletes.append(os.path.join(folder, file))
        for f in deletes:
            os.remove(f)

        # Move to next day
        start_date += timedelta(days=1)
