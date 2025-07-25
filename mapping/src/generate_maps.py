'''
Automatically generates report maps in ArcPy using post-processed rasters.
'''

import arcpy
# from arcpy import env
from arcpy.sa import * # Spatial Analyst module

#########################
#         CONFIG        #
#########################
# TODO: config should not need to be changed for each run, but may vary based on system filepaths
# templateAprx = "M:/.../ReportTemplate.aprx"
# productSourceDir = f"M:/SWE/WestWide/{date}_..."
# jpgOutputDir = "../output/{date}_figs"

#########################
#       END CONFIG      #
#########################

'''
@brief Creates a clone of the map template APRX file
'''
def cloneTemplate():
    pass

'''
@brief Sets the data source for the given figure
'''
def setDataSource(figure, new_data_source):
    pass

'''
@brief Exports the given layout to JPG
'''
def exportToJpg(layout):
    pass

''' 
@brief Parses inputs and generates appropriate maps
'''
# TODO: update function documentation
def main():
    pass
    # Parse file arguments and flags, as well as any config variables defined in this file
    # Check --output != none
    # Generate temporary directory to store aprx files in
    # Open a copy of the template aprx file
    #   call cloneTemplate()
    # For each map in --output:
    #   call setDataSource()
    #   call exportToJpg()
    # Save all jpgs to a subdirectory
    # If --preview open the jpg (option to go through one by one and generate as needed?)
    # Delete temporary directory and clean up any files (del aprx)

if __name__ == '__main__':
    main()