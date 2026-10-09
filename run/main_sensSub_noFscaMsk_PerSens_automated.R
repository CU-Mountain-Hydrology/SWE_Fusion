# Main function to run LRM (Linear Regression Model)
# Kehan Yang  (kyang33@uw.edu, kehan.yang@colorado.edu, 2022-08-01)
# Updated 1/27/23 by Leanne Lestak to run using new features
# Updated 8/1/24 by Leanne Lestak to port to the PNW
# Updated 11/26/24 by Leanne Lestak to add ecoregion sensors and port to INMT
# Updated 5/28/25 by Leanne Lestak to include a flag to turn on/off fsca masking of sensors
#     Raster masking must be done after the model runs in python code.
# Updated 10/8/25 by Leanne Lestak to check that there are GT 40% of sensors & added SW
#   Barrier height & distance layers. Check that there are GT 40% of sensors for each domain
#   that are reporting GT 0 before running the model for that date. Can change variable
#   SensPer <- 0.4, to change percentage. This is for all domains. Cannot change for each
#   domain individually.
#
# stationsweRegression which is an old package developed by Dominik Schneider which
# may not be working - KY - 20220801. It was written using an old version of R.
# This needs to be called otherwise this code won't work. If not installed, must
# be installed first. See instructions below.
#
# StationSWERegressionV2 must be installed before running this code. In 2022 Kehan Yang
# wrote this code to include a new method, using Karl Rittger's gap-filled daily fSCA and then
# deriving the mean daily fSCA, for any given pixel, this is the mean of all pixels from the
# beginning of the the water year (10/01).
#
# How to install stationsweRegression:
# install.packages("devtools")
# devtools::install_github("hoargroup/stationsweRegression", build_vignettes = TRUE)
#
# How to install StationSWERegressionV2:
# Before the first run of this code you must install the new regression package.
# otherwise comment this line out
# Before running the line below, make sure I'm in the
# "/Volumes/hydroData/WestUS_Data/Regress_SWE/SNM/Leanne/StationSWERegressionV2/main/" directory
# install.packages("../StationSWERegressionV2-version0.1.tar", type = "source", repos = NULL)
#
#
# load functions and libraries ----------------------------------------------------------

library(stationsweRegression)
library(StationSWERegressionV2)

pickAlpha=function(dF,myformula,nfolds,cl=cl){
  cvalpha=cva.glmnet(myformula,data=dF,nfolds=nfolds,type.measure='mse',outerParallel=cl,use.model.frame=T)

  maxlambda=100
  alpha=cvalpha$alpha
  ialpha=1
  cvdf=tibble(alpha) %>%
    bind_cols(data.frame(matrix(NA_real_,ncol=maxlambda,nrow=length(alpha)))) %>%
    bind_cols(as.data.frame(matrix(NA_real_,ncol=maxlambda,nrow=length(alpha))))#
  for(ialpha in seq_along(alpha)){
    cvlambdas=cvalpha$modlist[[ialpha]]$lambda
    numlambda=length(cvlambdas)
    cvdf[ialpha,2:(numlambda+1)] <- as.list(cvlambdas)
    cvdf[ialpha,maxlambda+1+(1:numlambda)] <- as.list(cvalpha$modlist[[ialpha]]$cvm)
  }
  cvresults <-
    cvdf %>%
    mutate(alpha=alpha) %>%
    gather(lambdaid,lambdaval,num_range('X',1:100)) %>%
    gather(mseid,mseval,num_range('V',1:100))

  med_mse <-
    cvresults %>%
    group_by(alpha) %>%
    summarise(
      medmse=median(mseval,na.rm=T)
    )

  med_se <- med_mse %>%
    summarise(
      se=sd(medmse,na.rm=T)/sqrt(n())
    ) %>% as.numeric

  alphaind=which.min(abs(med_mse$medmse-(min(med_mse$medmse,na.rm=T)+med_se)))[1]
  bestalpha=alpha[alphaind]
  # bestalpha=1
  return(bestalpha)
}

## adding in function;
gnet_phvfsca=function(dF,formula,cl=NULL, model = 'glmnet'){
  if(is.null(cl)) yesParallel=FALSE else yesParallel=TRUE
  nfolds=floor(nrow(dF)/20)
  if(nfolds<4)
    nfolds=4
  if(nfolds>10)
    nfolds=10
  myformula=as.formula(formula)

  if(model == 'glm'){
    cvfit = glm(formula = myformula, data = dF) # use the default gaussian distribution
  }else{
    # if model == 'glmnet'
    bestalpha = pickAlpha(dF,myformula,nfolds,cl)
    cvfit = glmnetUtils::cv.glmnet(myformula,data=dF,nfolds=nfolds,type.measure='mse',alpha=bestalpha,
                                   parallel = yesParallel,use.model.frame=TRUE)# use the default gaussian distribution
    cvfit$alpha=bestalpha
  }

  return(cvfit)
}


# do not show warnings
options(warn=-1)
# show warnings
# options(warn=-0)

# Load library dplyr
library(dplyr, warn.conflicts = FALSE)
# Suppress summarise info
options(dplyr.summarise.inform = FALSE) # get rid of all warnings

###############################################################################################
#
#  Load model parameters from Python (set in .env)
#
###############################################################################################
library(jsonlite)

require_param <- function(config, name) {
  if (is.null(config[[name]])) stop(paste("Missing required parameter:", name))
  config[[name]]
}

args   <- commandArgs(trailingOnly = TRUE)
config <- fromJSON(args[1])

# Setup main Regression SWE directory
PATH_regress <- require_param(config, "PATH_regress")
# Set working directory
setwd(PATH_regress)



# TODO: move ModDomLst to .env file
# Create a list of the model domains that you want to run
ModDomLst <- list("INMT", "NOCN", "PNW", "SNM", "SOCN")
#ModDomLst <- "SNM"






# Percent of sensors that are greater than 0, set this number to cancel model run if there aren't
# enough sensors. Total sensors for each domain are; INMT (225), NOCN (213), PNW (115), SNM (111), SOCN (333)
# Set this number as a decimal, ie. 0.4 = 40%
SensPer <- require_param(config, "SensPer")

# Flag T/F to mask the snow pillows or cocorahs with the fsca pixel value, this is done before
# the model is run. Dom recommended masking the sensors before running the model.
#
# 8/20/25 - We have decided we will not mask the sensors w/fSCA for all historical model runs (set to T)
# We will produce 3 possible outputs, 1) that is masked by raster fSCA values after the model runs (see isfscaMask below)
# and 2) that isnt masked with raster fsca after the model runs. All runs will be masked
# with the 0/1 fsca, as usual. Output file names are as follows;
# 'phv',SNOW_VAR,'_',datestr,'_nofscamsk.tif' (T - no sensor masking, no raster fsca masking)
# 'phv',SNOW_VAR,'_',datestr,'_sensormsk.tif' (F- sensor masking w/fsca, no raster fsca masking)

# Values are T (don't mask sensors) or F (mask sensors) for isfscaFlag
isfscaFlag <- require_param(config, "isfscaFlag")

# This flag will mask or not mask the model output with the raster fsca image. If the flag
# is set to T = mask with raster fsca, if set to F = dont mask with raster fsca. If set to T output file name is as follows;
# 'phv',SNOW_VAR,'_',datestr,'_fscamsk.tif' (no sensor fsca masking, raster fsca masking final model output). If this is set
# to T, 2 output geoTIFs will be produced, one from above and then above output is used to create raster fsca masked output
# file.
isfscaMask <- require_param(config, "isfscaMask")

# This flag will write out the model inputs by sensor to a .gpkg database file (T) or not (F). For historic runs set to F for real
# time runs set to T.
isGPKG <- require_param(config, "isGPKG")

# This is the name of the user that is contained in the model directory path here;
# hydroData/WestUs_Data/Regress_SWE/{ModelDomain}/{UserName}
UserName <- require_param(config, "UserName")

# change to 'T' if you know the best historic reconstructed SWE; Not for the first run, use 'F'
ishisday <- require_param(config, "ishisday")

# If set to 'T' above, choose the date here
# format of besthisdate, 'YYYY-MM-DD'
besthisdate <- require_param(config, "besthisdate")

# change to F if you don't want to use cocorahs data, or to T if you do. Make sure to change RUNNAME below.
isCCR <- require_param(config, "isCCR")

# change the name of output folder to separate different simulations
# If not masking sensors with fsca, make sure it goes into the correct directory
# RUNNAME = 'test_woCCR_RT_CanAdj_rcn_noSW_woCCR_nofscamsk'
# RUNNAME = 'test2'
# RUNNAME = 'RT_CanAdj_wCCR_noScale2'
# RUNNAME = 'RT_CanAdj_rcn_noSW_woCCR'
# RUNNAME = 'RT_CanAdj_rcn_noSW_woCCR_nofscamsk'
# RUNNAME = 'RT_CanAdj_rcn_wCCR_nofscamskSens_noMdlFsca_20'
RUNNAME <- require_param(config, "RUNNAME")

# This is the date that sensors were downloaded, download late morning as some sensors don't
# report until later in the day. This is the date that is on the sensor CSV file name which
# is sitting in PATH_SNOTEL, for example; cdec_BKL_2024-12-19.csv, date is 12/19/2024.
# This date must be = or GT than the latest model simulation date, which is set in the file;
# Regress_SWE/simulation_date.txt
# Format 'YYYYMMDD'
oldestDate <- as.Date(require_param(config, "oldestDate"), "%Y%m%d")

# read in the list of the dates that you are running the model for
# this file must be created in the proper format before running the model
# put this file in the Regress_SWE directory, file format:
# "simdate"
# 2019-04-01 (YYYY-MM-DD)
# simulationday <- read.csv(paste0(PATH_root,'/inputs/simulation_date_',ModDom,'.txt'), stringsAsFactors = F, header = T)
# simulationday <- read.csv(paste0(PATH_regress,'simulation_date_historic_ET.txt'), stringsAsFactors = F, header = T)
# Update July 2026: The automated version of this model no longer reads in the run date from
# the csv since it is passed directly from python, however the file is still updated if needed.
# This should always match oldestDate for automated daily runs
# simulationday <- require_param(config, "simulationday")
simulationday <- data.frame(
  simdate = require_param(config, "simulationday"),
  stringsAsFactors = FALSE
)
irow <- nrow(simulationday)
simulationday$datestr <- sapply(simulationday[, 1], fdate2str)
# print all model simulation dates to the screen
# print(paste0('Creating LRM for these dates: ', simulationday))

# Set which fSCA to use 'MODSCAG' image or 'Rittger' gap-filled fsca
# Make sure there is a MODSCAG or Rittger fSCA first
# For Rittger, files are in directory path PATH_FSCA
# For MODSCAG, files are in directory path PATH_MODSCAGDOWNLOAD
# Files need to be the entire extent of the western U.S. correct cellsize and
# extent, because the model just crops, it doesn't reproject. The function
# get_modscag_data.R will reproject and put tiles together if needed, but is
# currently not being called.
fscaType <- require_param(config, "fscaType")

# rcn or fsca (reconstruction or fSCA data)
SNOW_VAR <- require_param(config, "SNOW_VAR")

# If fsca type is MODSCAG then this path is used for location of geotif files, always need this.
# path should point 1 level above /yr/doy/*.tif
# can point to historic modscag or NRT (near real time) modscag. Make sure directory and MODSCAG_TYPE
# are the same. This is currently not being used.

# 'NRT' or 'historic'
MODSCAG_TYPE <- require_param(config, "MODSCAG_TYPE")

# 'snow_fraction_canadj = snow fraction (fSCA) canadj is canopy adjusted (vegetation adj)
# 'snow_fraction' = snow fraction w/out canopy adjustment, then use FVEG_Correction = T
MODSCAG_FILE <- require_param(config, "MODSCAG_FILE") # can be 'snow_fraction_canadj' or 'snow_fraction'

# If fsca images aren't canopy adjusted set this to 'T'.
# 'F' or 'T', fveg correction is F if 'snow_fraction_canadj' is used
FVEG_CORRECTION <- require_param(config, "FVEG_CORRECTION")

# Set up directories for fSCA and DMFSCA, this needs to be streamlined, MODSCAG is currently not correct
if (fscaType == "MODSCAG"){
  # Directory location of the MODSCAG FSCA images, minus the year
  # This is incomplete, we need to add doy, but we're not using MODSCAG now
  # PATH_FSCA_main=paste0(PATH, 'MODSCAG/modscag/')
  PATH_FSCA_main <- require_param(config, "MODSCAG_PATH")
  # Directory location of the daily mean fSCA images, minus the year
  # PATH_DMFSCA_main = paste0(PATH, 'Rittger_data/fsca_v2025.0.1_ops/NRT_DMFSCA_WW_N83/')
  PATH_DMFSCA_main <- require_param(config, "DMFSCA_PATH")
}
if (fscaType == "Rittger"){
  # Directory location of the rittger FSCA images, minus the year
  # PATH_FSCA_main = paste0(PATH, 'Rittger_data/fsca_v2025.0.1_ops/NRT_FSCA_WW_N83/')
  PATH_FSCA_main <- require_param(config, "FSCA_PATH")
  # Directory location of the daily mean fSCA images, minus the year
  # PATH_DMFSCA_main = paste0(PATH, 'Rittger_data/fsca_v2025.0.1_ops/NRT_DMFSCA_WW_N83/')
  PATH_DMFSCA_main <- require_param(config, "DMFSCA_PATH")
}

###############################################################################################
#
#  End of Change these parameters before running the model
#
###############################################################################################

# Initialize i
i = 1

# Create a file that contains the information about a model run if the run is skipped b/c there
# aren't enough sensors to run, based on the percent value set in the variable SensPer above
# Define the file name and then write the header line
SkipFile <- paste0(PATH_regress,'SkipModelRun_olafTest.txt')
Header <- c("ModDom", "Date", "GT0_Sens","%_Sens")
cat(Header, file = SkipFile)

# Run the model for each domain
for(i in i:length(ModDomLst)){
  print("************************")
  print("************************")
  print(paste0("Model Domain: ", ModDomLst[i]))
  print("************************")
  ModDom <- as.character(ModDomLst[i])

  # set up model running environment and input directories -----
  PATH_root <- paste0(PATH_regress, ModDom, "/", UserName, '/StationSWERegressionV2/data')
  PATH_functions <- paste0(PATH_regress, ModDom, "/", UserName, '/StationSWERegressionV2/R/')
  PATH_input <-paste0(PATH_root,'/inputs/')

  # Source user defined functions, these will be used instead of those in package
  source(paste0(PATH_functions,'get_best_historical_date_ww_no0.R'))
  source(paste0(PATH_functions,'get_best_historical_date_ww.R'))
  source(paste0(PATH_functions,'get_ccr_sub.R'))
  source(paste0(PATH_functions,'get_station_inventory_cdec.R'))
  source(paste0(PATH_functions,'get_station_inventory_snotel.R'))
  source(paste0(PATH_functions,'get_stationswe_data_cdec_snotel.R'))

  # Set which package to use for fsca masking of snow pillows
  if (isfscaFlag){
    source(paste0(PATH_functions,'setup_modeldata_ww_noMask.R'))

  }else{
    source(paste0(PATH_functions,'setup_modeldata_ww.R'))
  }

  RESO = 18/3600 #cell size resolution in degrees

  # Water mask layer, this is cropped to the model domain, cellsize .005
  fwatermask <- paste0(PATH_input,'/',ModDom,'_watermask.tif')
  watermask <- raster(fwatermask)

  # Historical reconstructed SWE directory location, geoTIF files
  # file contains historical reconstructed SWE for the entire western U.S. in model
  # space, .005 cellsize and full western extent, same as input PHV files
  # This file is created for each new historical image chosen in setup_modeldata
  PATH_RCNDOWNLOAD = paste0(PATH_regress,"WGS100_ori")

  # Directories containing all model inputs, the independent regression variables
  PATH_PHV=paste0(PATH_input,'phv')

  # directory to store downloaded CCR data
  PATH_CCR <- paste0(PATH_input,'CCR')
  # directory to store snotel and CDEC downloaded CSV data files
  PATH_SNOTEL <- paste0(PATH_regress,'snow_sensors')
  # directory containing CSV file determining which sensors use for this domain
  PATH_Pillows_Sel <- paste0(PATH_input,'sensor_files')

  # temporary directory to save cropped reconstructed SWE and FSCA
  PATH_temp <- paste0(PATH_input,'temp')
  PATH_FSCA_WGS <- paste0(PATH_temp,'/fsca_resa') # cropped fsca files
  PATH_REC_WGS <- paste0(PATH_temp,'/rec_resa') # cropped reconstruction files

  # Define output directories ----
  PATH_OUTPUT=paste0(PATH_root,'/outputs/',RUNNAME)
  PATH_XVAL=file.path(PATH_OUTPUT,'crossval_stats_dates')

  # Create new directories defined above
  dir.create(path=PATH_XVAL,rec=TRUE, showWarnings = F)
  dir.create(PATH_temp)
  dir.create(PATH_FSCA_WGS)
  dir.create(PATH_REC_WGS)
  # create new CCR directory
  dir.create(PATH_CCR)

  # get phv variables and make a stack for domain from /data/phv folder and convert to a dataframe
  # see use_package and Make_PHV_Inputs vignettes

  print("Creating PHV stack ......")

  phvfilenames=dir(PATH_PHV,pattern='.tif$',full.names=TRUE)
  phvstack=stack(phvfilenames)
  names(phvstack) <- sapply(strsplit(names(phvstack),'_'),'[',2)
  phvstack_scaled <- scale(phvstack) # centers and/or scales the columns of a numeric matrix.
  ucophv <- as.data.frame(phvstack_scaled) # change to data frame

  # get domain corner extents based on phv inputs - 20200117 KH
  EXTENT_NORTH = extent(phvstack[[1]])[4]
  EXTENT_EAST = extent(phvstack[[1]])[2]
  EXTENT_SOUTH = extent(phvstack[[1]])[3]
  EXTENT_WEST = extent(phvstack[[1]])[1]

  print("Create sensor location file.")
  PATH_Pillows_Sel <- paste0(PATH_input,'sensor_files')
  print(PATH_Pillows_Sel)
  print(paste0(PATH_Pillows_Sel,'/',ModDom,'_SNOTEL_inventory.csv'))
  # Get CDEC station locations using domain specific CSV file
  station_locations_cdec <- get_station_inventory_cdec(SensorDir=PATH_Pillows_Sel,ModDom)

  # Get snotel station locations using domain specific CSV file
  station_locations_snotel <- get_station_inventory_snotel(SensorDir=PATH_Pillows_Sel,ModDom)

  # Create a single file containing necessary columns for merging with PHV variables
  # First subset the CDEC file so they have the same columns
  sta_locs_cdec <- station_locations_cdec %>%
    dplyr::select(Site_ID,site_name,Longitude,Latitude)
  # Then subset the snotel file so they have the same columns
  sta_locs_snotel <- station_locations_snotel %>%
    dplyr::select(Site_ID,site_name,Longitude,Latitude)
  # Join the CDEC and snotel files
  sta_locs_all <- sta_locs_cdec %>%
    dplyr::bind_rows(sta_locs_snotel)

  # create a spatial object of the locations ----
  snotellocs=as.data.frame(sta_locs_all)
  coordinates(snotellocs)= ~Longitude+Latitude
  coordnames(snotellocs)=c('x','y')
  proj4string(snotellocs)='+proj=longlat +datum=NAD83'

  # extract the phv variable values for the station locations ----
  print("Extract the phv variable values for the station locations")
  phvsnotel=raster::extract(phvstack_scaled,snotellocs,sp=T)
  phvsnotel=phvsnotel %>%
    tbl_df %>%
    mutate_if(is.factor,as.character) %>%
    dplyr::select(Site_ID,site_name,dem:zness)

  # the model will run through the dates from the bottom of the file to the first date
  # this file needs to be setup before running the model, simulation date file

  # simulationday$datestr = sapply(simulationday[,1], fdate2str)


  # Loop through each date in the simulation date file
  for(irow in irow:1){  #simulate in reverse will download less data

    # Set up date variables
    simdate <- simulationday[irow,1]
    simdate <- as.Date(as.character(simdate), '%Y-%m-%d')
    yr=strftime(simdate,'%Y')
    doy=strftime(simdate,'%j')
    mth=strftime(simdate,'%m')
    dy=strftime(simdate,'%d')
    datestr=paste0(yr,mth,dy)

    # Check to be sure the simulation date is less than or equal to the date sensors were downloaded
    if (simdate > oldestDate){
      print('                                               ')
      print("***** Model run aborted, Fix This first *****")
      print(paste0('simulation date: ', simdate, ' is newer than sensor file date: ', oldestDate))
      print('Download sensors first using /Regress_SWE/snow_sensors/0_get_All_stationswe_data.R')
      print("                                                                                  ")
      next
    }


    # Print simulation date to the screen
    print("         ")
    print('*************')
    print(paste0('Processing: ', simdate))

    # Set up directories for fSCA, this needs to be streamlined, MODSCAG is currently not correct
    if (fscaType == "MODSCAG"){
      # Directory location of the MODSCAG FSCA images
      # This is incomplete, we need to add doy, but we're not using MODSCAG now
      PATH_FSCA=paste0(PATH_FSCA_main, yr,'/')
      # Directory location of the daily mean fSCA images
      PATH_DMFSCA = paste0(PATH_DMFSCA_main, yr,'/')
    }
    if (fscaType == "Rittger"){
      # Directory location of the rittger FSCA images
      PATH_FSCA = paste0(PATH_FSCA_main,yr,'/')
      # Directory location of the daily mean fSCA images
      PATH_DMFSCA = paste0(PATH_DMFSCA_main, yr,'/')
    }

    # Setup output geotif model run filename and check to see if it already exists
    mapfn=file.path(PATH_OUTPUT,paste0(ModDom,'_phvrcn_',datestr,'.tif'))
    fe.logical=file.exists(mapfn)
    if(fe.logical) {
      print(paste0('swe exists in ', PATH_OUTPUT,'. skipping.'))
      next
    }

    ## download station swe data for the date of simulation and merge with station locations ----
    #
    print('Create station data----')

    station_data=get_stationswe_data_cdec_snotel(PATH_SNOTEL,station_locations_cdec, station_locations_snotel,simdate,oldestDate)

    station_xmin <- min(station_data$Longitude)
    station_xmax <- max(station_data$Longitude)
    station_ymin <- min(station_data$Latitude)
    station_ymax <- max(station_data$Latitude)

    # If CoCoRAHs sensors are used then calculate density here
    if(isCCR){
      cos_density = 0.1 # the default density for freshsnow
      ccrdata <- get_ccr_sub(datestr, PATH_CCR,cos_density, station_xmin, station_xmax,station_ymin, station_ymax)
      print('The transformed cocorahs data:')
      print(ccrdata)
    }

    # # If CoCoRAHs sensors are used then calculate density here
    # if(isCCR){
    #   cos_density = 0.1 # the default density for freshsnow
    #   ccrdata <- get_ccr(datestr, PATH_CCR,cos_density, phvstack[[1]])
    #   print('The transformed cocorahs data:')
    #   print(ccrdata)
    # }

    ## subset snotel data for simulation date ----p
    snoteltoday <-
      station_data %>%
      filter(!is.na(Longitude),!is.na(Latitude)) %>%
      dplyr::filter_(~dte == datestr) #%>%
    #filter(!is.na(pillowswe)) # This will filter out pillowse that are NA, I want those now!

    # If using CCR then bind it to the snotel data
    if(isCCR){
      ccr <- cbind.data.frame(Site_ID = ccrdata$Site_ID, Longitude = ccrdata$Longitude,
                              Latitude = ccrdata$Latitude, dte = fdate2str(ccrdata$dte), pillowswe = ccrdata$swe)

      alltoday <- rbind(snoteltoday, ccr)
      # Print CCR data to the screen
      print(paste0(nrow(ccr),' CCR obs used in the model with mean SWE of ',
                   round(mean(ccr$pillowswe),2),
                   ' and ', sum(ccr$pillowswe>0), ' positive obs'))

      ccrlocs=as.data.frame(ccr)
      coordinates(ccrlocs)= ~Longitude+Latitude
      coordnames(ccrlocs)=c('x','y')
      proj4string(ccrlocs)='+proj=longlat +datum=WGS84'

      # extract the phv variable values for the CCR station locations ----
      phvccr=raster::extract(phvstack_scaled,ccrlocs,sp=T)
      phvccr=phvccr %>%
        tbl_df %>%
        mutate_if(is.factor,as.character) %>%
        dplyr::select(Site_ID, dem:zness)
      phvccr$site_name = phvccr$Site_ID

      # Add the CCR PHV values to the pillow PHV values
      phvall<- rbind(phvsnotel, phvccr)

    }else{
      alltoday <- snoteltoday
      phvall <- phvsnotel
    }

    # We used to check the pillows and CCR points to make sure they are complete, not doing that now
    # alltoday <- alltoday[complete.cases(alltoday),]

    # Change the pillow/CCR file to a dataframe, add lat/long coordinates to the points, and add projection info
    alltoday.sp=data.frame(alltoday)
    sp::coordinates(alltoday.sp)=~Longitude+Latitude
    proj4string(alltoday.sp)='+proj=longlat +datum=WGS84'

    # Check how many snow stations there are recording SWE GT 0
    truenum <- table(alltoday$pillowswe>0)["TRUE"]

    # If there are no pillows > 0 and a NA in the table then truenum will be NA, it needs to be a number
    # so then if truenum = NA replace it with 0
    if (is.na(truenum)) {
      truenum <- 0
    }

    # Print to screen
    #
    print(paste0('Num of pillows > 0: ', truenum))

    # Setup number of sensors that are 40% of total
    if (ModDom == 'INMT') {SensNum <- round(225 * SensPer)}
    if (ModDom == 'NOCN') {SensNum <- round(213 * SensPer)}
    if (ModDom == 'PNW') {SensNum <- round(115 * SensPer)}
    if (ModDom == 'SNM') {SensNum <- round(111 * SensPer)}
    if (ModDom == 'SOCN') {SensNum <- round(333 * SensPer)}

    # If LT 40% of the stations are GT 0 then don't model for this day
    if( truenum < SensNum){
      print(paste0('Skip this day! The number of stations with snow is less than ', SensNum))
      print(paste0('Domain/Date/#Sens: ',ModDom,' / ',datestr,' / ',truenum,' < ',SensNum))
      # If model run is aborted then append values to SkipFile here
      cat("\n",ModDom,datestr,truenum,SensNum, file = SkipFile, append = TRUE)
      next
    }

    # setup the best historical date, either one chosen manually or one chosen by the model
    # Print to screen
    #
    print('Choosing best Historical SWE date ... ')

    # If historic rcn was chosen manually
    if(ishisday){

      besthisdate <- besthisdate

      # Otherwise the model will chose the best historical rcn from the best fit for today's snow pillows
      # The dates and r2 values that are used to select the best historical date from the pillows is
      # written out here; PATH_OUTPUT/{ModDom}_date_selectrcn_r2.csv
      # We are only using today's sensors that are GT 0. If you want to use all sensors call this:
      # get_best_historical_date_ww.R

    }else{

      besthisdate = get_best_historical_date_ww_no0(datestr, station_data, ModDom)

    }

    # Print to screen
    #
    print(paste0('The historical reconstruction SWE on ', besthisdate, ' will be used.'))

    # Choose which fsca image to use, MODSCAG or Rittger gap-filled
    if (fscaType == "MODSCAG"){

      # function to create the MODSCAG image
      simfsca <- get_modscag_data(doy,yr,MODSCAG_TYPE,PATH_FSCA,MODSCAG_FILE,FVEG_CORRECTION,RESO,EXTENT_WEST,EXTENT_EAST,EXTENT_SOUTH,EXTENT_NORTH)

      # Independent regression variables with barrier variables, full list
         PHV_VARS = ~lon+lat+dem+eastness+northness+regionaleastness+regionalnorthness+regionalzness+zness+wbDistance+wbHeight+wd2ocean+nwbDistance+
            nwbHeight+nwd2ocean+swbDistance+swbHeight+swd2ocean+dmfsca

      # Independent regression variables partial list
      # PHV_VARS = ~lon+lat+dem+eastness+northness+regionaleastness+regionalnorthness+regionalzness+zness+wbDistance+wbHeight+wd2ocean+nwbDistance+
      #   nwbHeight+nwd2ocean+swd2ocean+dmfsca
    }

    if (fscaType == "Rittger"){
      #or if you have a fsca mosaic created already, read it directly;
      fsimfsca <- list.files(PATH_FSCA, glob2rx(paste0('*',datestr,'*.tif$')), full.names = T, recursive = T)

      # Independent regression variables with barrier variables, full list
      PHV_VARS = ~lon+lat+dem+eastness+northness+regionaleastness+regionalnorthness+regionalzness+zness+wbDistance+wbHeight+wd2ocean+nwbDistance+
        nwbHeight+nwd2ocean+swbDistance+swbHeight+swd2ocean+dmfsca

      # Independent regression variables partial list
      # PHV_VARS = ~lon+lat+dem+eastness+northness+regionaleastness+regionalnorthness+regionalzness+zness+wbDistance+wbHeight+wd2ocean+nwbDistance+
      #   nwbHeight+nwd2ocean+swd2ocean+dmfsca

      # Check to see if fsimfsca file exists, if not jump out of code
      fsim=file.path(PATH_FSCA,paste0(datestr,'.tif'))
      fs.logical=file.exists(fsim)
      if(!fs.logical) {
        print(paste0('No fSCA : ',PATH_FSCA,yr,'/',datestr,'.tif','. skipping.'))
        next
      }

      # Set output to fsimfsca
      simfsca <- raster(fsimfsca)

      # if input fsca has different extent than watermask, we need to crop the fsca
      # Compare file that is entire extent of westwide, correct cellsize and cell location
      # Then crop to the model domain
      if(!compareRaster(watermask,simfsca, extent=T,rowcol=T, crs=T, res=T, rotation=T,stopiffalse = F)){

        # Crop fsca to the same extent as watermask
        simfsca_crop <- crop(simfsca,extent(watermask))

      }
      # Set cropped image to output image
      simfsca <- simfsca_crop
    }

    # Prep PHV and pillows stacked dataframe for running the model
    phvall2 <- phvall
    ucophv2 <- ucophv

    # Add dmfsca
    if (fscaType == "Rittger"){

      # add fscddm data to ucophv (create variable ucophv2) and phvsnotel
      target <- paste0('*',datestr,"*.tif$")
      fn_dmfsca <- list.files(PATH_DMFSCA, glob2rx(target), full.names = T, recursive = T)

      if(length(fn_dmfsca) != 1){
        stop('no DMFSCA swe data!')
      }

      r_demfsca <- raster(fn_dmfsca)
      r_demfsca[r_demfsca>100] <- NA

      # if input dmfsca has different extent than watermask, we need to crop the dmfsca
      # Compare file that is entire extent of westwide, correct cellsize and cell location
      # Then crop to the model domain using the watermask
      if(!compareRaster(watermask,r_demfsca, extent=T,rowcol=T, crs=T, res=T, rotation=T,stopiffalse = F)){

        # Crop fsca to the same extent as watermask
        r_demfsca_crop <- crop(r_demfsca,extent(watermask))

      }
      # Set cropped image to output image
      r_demfsca <- r_demfsca_crop

      r_demfsca_scale <- scale(r_demfsca)
      v_demfsca_scale <- getValues(r_demfsca_scale)

      ucophv2 <- cbind.data.frame(ucophv, dmfsca=v_demfsca_scale)

      vfscdsnotel=raster::extract(scale(simfsca),alltoday.sp,sp=F)
      df_dmfsca = data.frame(Site_ID = alltoday.sp$Site_ID, dmfsca  =vfscdsnotel)

      phvall2 <- merge(phvall,df_dmfsca, by = 'Site_ID')
    }

    ## run model ---
    ## Choose whether to mask snow pillows by fsca as point values or after model is run
    ## as raster values
    if (isfscaFlag){
      modelingdFs <- setup_modeldata_ww_noMask(alltoday.sp,phvall2,simfsca,
                                               SNOW_VAR,PHV_VARS,PATH_OUTPUT,PATH_RCNDOWNLOAD,
                                               fdate2str(besthisdate),
                                               datestr,phvstack,ucophv2, 'glmnet',PATH_REC_WGS,ModDom,isGPKG)
    }else{
      modelingdFs <- setup_modeldata_ww(alltoday.sp,phvall2,simfsca,
                                               SNOW_VAR,PHV_VARS,PATH_OUTPUT,PATH_RCNDOWNLOAD,
                                               fdate2str(besthisdate),
                                               datestr,phvstack,ucophv2, 'glmnet',PATH_REC_WGS,ModDom,isGPKG)
    }

    doidata=modelingdFs[[1]]
    predictdF=modelingdFs[[2]]
    myformula=modelingdFs[[3]]

    ## fit glmnet model ----
    mdl <- gnet_phvfsca(doidata,myformula,model = 'glmnet')

    ## predict on swe for domain and mask with 0/1 fsca and watermask ----
    yhat=predict(mdl,predictdF,na.action=na.pass)

    simyhat=simfsca
    values(simyhat) <- yhat

    simyhat=simfsca
    values(simyhat) <- yhat
    simyhat <- mask(simyhat,watermask,maskvalue=1,updatevalue=NA)
    simyhat <- mask(simyhat,simfsca)
    simyhat <- mask(simyhat,simfsca,maskvalue=235,updatevalue=NA)
    simyhat <- mask(simyhat,simfsca,maskvalue=250,updatevalue=NA)
    simyhat <- mask(simyhat,simfsca,maskvalue=0,updatevalue=0)

    simyhat[simyhat<0] <- 0
    # simyhat[simyhat>20] <- NA
    plot(simyhat)
    ## save prediction to file ----
    # outfile=paste0(ModDom,'_phv',SNOW_VAR,'_',datestr,'.tif')

    if (isfscaFlag){
      outfile=paste0(ModDom,'_phv',SNOW_VAR,'_',datestr,'_nofscamsk.tif')

    }else{
      outfile=paste0(ModDom,'_phv',SNOW_VAR,'_',datestr,'_sensormsk.tif')

    }

    # Write out the raster (geotif) model run to disk
    writeRaster(simyhat,file.path(PATH_OUTPUT,outfile),NAflag=-99,overwrite=T)

    if (isfscaMask){
      outfile2=paste0(ModDom,'_phv',SNOW_VAR,'_',datestr,'_fscamsk.tif')
      simyhatmask <- simyhat * (simfsca/100)
      writeRaster(simyhatmask,file.path(PATH_OUTPUT,outfile2),NAflag=-99,overwrite=T)
    }

    # Write out the raster (geotif) model run to disk
    #writeRaster(simyhatmask,file.path(PATH_OUTPUT,outfile2),NAflag=-99,overwrite=T)

    ## save coefficients from model and write to file ----
    coef_dF <-
      as.data.frame(as.matrix((coef(mdl)))) %>%
      mutate(predictor=rownames(.)) %>%
      setNames(c('coefficient','predictor'))

    write_tsv(format(coef_dF,sci=FALSE),
              path=paste0(PATH_OUTPUT,'/',ModDom,'_phv',SNOW_VAR,'_coefs_',datestr,'.txt')
    )

    print(' - Computing crossvalidation statistics...')
    allmdls <-
      doidata %>%
      crossv_mc(.,n=30,test=0.1) %>%
      mutate(
        dte=datestr,
        phvfsca_obj_glmmdl=map(train,gnet_phvfsca,myformula),
        phvfsca_r2_glmmdl=map2_dbl(phvfsca_obj_glmmdl,test,myr2),
        phvfsca_pctmae_glmmdl=map2_dbl(phvfsca_obj_glmmdl,test,mypctmae2),
        phvfsca_pbias_glmmdl=map2_dbl(phvfsca_obj_glmmdl,test,mypbias)

      )

    stat_r2 <-
      allmdls %>%
      group_by(dte) %>%
      summarise(
        avg_r2=mean(phvfsca_r2_glmmdl,na.rm=T),
        sd_r2=sd(phvfsca_r2_glmmdl,na.rm=T),
        uci_r2=avg_r2+1.96*sd_r2/sqrt(n()),
        lci_r2=avg_r2-1.96*sd_r2/sqrt(n())
      )

    write_tsv(StationSWERegressionV2::format_numeric(stat_r2,sci=FALSE),
              path=file.path(PATH_XVAL,paste0(ModDom,'_phv',SNOW_VAR,'_r2_',datestr,'.txt')))

    stat_pctmae <-
      allmdls %>%
      group_by(dte) %>%
      summarise(
        avg_pctmae=mean(phvfsca_pctmae_glmmdl,na.rm=T),
        sd_pctmae=sd(phvfsca_pctmae_glmmdl,na.rm=T),
        uci_pctmae=avg_pctmae+1.96*sd_pctmae/sqrt(n()),
        lci_pctmae=avg_pctmae-1.96*sd_pctmae/sqrt(n())
      )

    write_tsv(StationSWERegressionV2::format_numeric(stat_pctmae,sci=FALSE),
              path=file.path(PATH_XVAL,paste0(ModDom,'_phv',SNOW_VAR,'_pctmae_',datestr,'.txt')))


    stat_pbias <-
      allmdls %>%
      group_by(dte) %>%
      summarise(
        avg_pbias=mean(phvfsca_pbias_glmmdl,na.rm=T),
        sd_pbias=sd(phvfsca_pbias_glmmdl,na.rm=T),
        uci_pbias=avg_pbias+1.96*sd_pbias/sqrt(n()),
        lci_pbias=avg_pbias-1.96*sd_pbias/sqrt(n())
      )

    write_tsv(StationSWERegressionV2::format_numeric(stat_pbias,sci=FALSE),
              path=file.path(PATH_XVAL,paste0(ModDom,'_phv',SNOW_VAR,'_pbias_',datestr,'.txt')))

    ## Write out sensors used in model run to a file, this is different than all the sensors
    ## which are in the .gpkg
    write_tsv(doidata,path=file.path(PATH_XVAL,paste0(ModDom, '_doidata_',SNOW_VAR,'_',datestr,'.txt')))

  }
}
