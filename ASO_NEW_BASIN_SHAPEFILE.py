# import modules
import arcpy
from arcpy import env
from arcpy.sa import *
import os

## establish file paths
raw_tifs = r"M:/SWE/WestWide/Spatial_SWE/ASO/2026/data/NEW_BASINS/"
SnapRaster = r"W:\data\boundaries\SnapRaster_albn83.tif"

projout = arcpy.SpatialReference(102039)

# reproject to albers and snap to the snap rasters
for tif in os.listdir(raw_tifs):
    if tif.endswith('.tif'):
        arcpy.env.snapRaster = SnapRaster
        arcpy.ProjectRaster_management(f"{raw_tifs}/{tif}", f"{raw_tifs}/{tif[:-4]}_albn83.tif",
                                       projout, "NEAREST", "50 50", "WGS_1984_(ITRF00)_To_NAD_1983")
        print(f"{tif} reprojected")

        # convert to int
        outInt = Int(f"{raw_tifs}/{tif[:-4]}_albn83.tif")
        outInt.save(f"{raw_tifs}/{tif[:-4]}_albn83_int.tif")
        print("converted to INT")

        # raster to poylgon
        arcpy.RasterToPolygon_conversion(f"{raw_tifs}/{tif[:-4]}_albn83_int.tif", f"{raw_tifs}/{tif[:-4]}.shp",
                                         "SIMPLIFY")
        print(f"{tif} converted to polygon")

        # add and calculate field
        arcpy.AddField_management(f"{raw_tifs}/{tif[:-4]}.shp", "DIS", "TEXT")
        with arcpy.da.UpdateCursor(f"{raw_tifs}/{tif[:-4]}.shp", "DIS") as cursor:
            for row in cursor:
                row[0] == "Y"
                cursor.updateRow(row)

        # disolve
        arcpy.Dissolve_management(f"{raw_tifs}/{tif[:-4]}.shp",f"{raw_tifs}/{tif[:-4]}_dis.shp",
                                  "DIS")
        print("dissolved")

        # elimate parts inside
        arcpy.EliminatePolygonPart_management(f"{raw_tifs}/{tif[:-4]}_dis.shp", f"{raw_tifs}/{tif[:-4]}_final.shp",
                                              "AREA", 200000000, "", "CONTAINED_ONLY")
        print('done with final')