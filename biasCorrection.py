# import modules
import arcpy
import os
import pandas as pd
import sys
from datetime import datetime
print('modules imported')

# establish paths
# list the basins for bias correction
# establish method for bias correction
## METHOD ONE, MOST RECENT FRACTIONAL ERROR LAYER
## METHOD TWO, BASED ON SNOTEL
## METHOD THREE, PICK YOUR OWN