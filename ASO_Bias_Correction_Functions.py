# import modules
import arcpy
import pandas as pd
import os
from arcpy import *
print('modules imported')

# parameters
# rundate = ""

# loop through SNM and SOCO domain
# loop through basin list
## have a dictionary that shows the ASO bias corrected basins that are listed within the basin -- NEED TO DO
# makes a list of all the basins
# loop through the ASO flights within that basin
    ## if len(list) == 0:
        # continue
    ## else

# most recent flight
## open the csv
## loop through the list
    ## open an empty fraction error list
    ## subset any rows that are in the basins
    ## make a new column that has the date in the MMDDYYYY format
    ## exclude any dates that are within 5 days of the run date
    ## find the most recent date
    ## get that fractional error layer in the list based on the file path

# percent grade
## loop through the list
    ## open an empty fraction error list
    ## subset any rows that are in the basins
    ## check to see if it's within the year = in_current_year=True/False
    ### if True: df['Year'] == Year
    ## negative postive or mixed for "GradeDirection"
    ## get the percentage grade within percentile

## select the one that your want

