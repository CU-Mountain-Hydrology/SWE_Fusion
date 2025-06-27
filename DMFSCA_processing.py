## This code processing DMFSCA from the start of the Water Year (Oct. 1) through the date needed. The first step is that fSCA needs to be processed daily from Oct. 1 
## to the date needed. The two folders required are "fSCA_folder" which is the fodler with the daily fSCA geoTif files. The "output_folder" is output DMFSCA folder. 
## The code starts at Oct. 1 and calculates the sum of fSCA values for each pixel and then divides it by the number of dates to get a daily mean fSCA. 
## Variables: wateryear_start = this should typically always be Oct. 1 unless you want the average to start from a different date
### process_start_date = this is the date you would want to start processing. It can be Oct. 1 if you are starting from scratch. However it can also be any other date if you
#### need to pick up from Apr. 4, for example.
### process_end_date = This is the final date you want to process. If you want DMFSCA from Apr. 4 through Apr. 11, the end date would Apr. 11. The end date is not exclusive.
#### If you want to only calculate one date, the process_start_date and process_end_date would be the same. 

import rasterio
import numpy as np
import os
from datetime import datetime, timedelta

def calculate_dmfsca(
    fSCA_folder,
    output_folder,
    wateryear_start,
    process_start_date,
    process_end_date
):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Sort .tif files
    raster_files = sorted([f for f in os.listdir(fSCA_folder) if f.endswith(".tif")])

    # Build dictionary mapping date -> filepath
    raster_dict = {}
    for f in raster_files:
        try:
            date_str = os.path.splitext(f)[0]
            file_date = datetime.strptime(date_str, "%Y%m%d")
            raster_dict[file_date] = os.path.join(original_folder, f)
        except ValueError:
            continue

    sorted_dates = sorted(raster_dict.keys())

    for current_date in sorted_dates:
        if current_date < process_start_date or current_date > process_end_date:
            continue

        # Get all dates from wateryear start to current date
        date_subset = [d for d in sorted_dates if wateryear_start <= d <= current_date]

        sum_array = None
        count = 0

        for d in date_subset:
            with rasterio.open(raster_dict[d]) as src:
                data = src.read(1).astype(np.float32)
                mask = data == src.nodata
                data[mask] = 0
                if sum_array is None:
                    sum_array = np.zeros_like(data)
                    valid_mask = np.zeros_like(data, dtype=np.int32)
                sum_array += data
                valid_mask += ~mask
                profile = src.profile

        # Calculate average
        with np.errstate(invalid='ignore'):
            avg_array = np.divide(sum_array, valid_mask, where=valid_mask != 0)
            avg_array[valid_mask == 0] = profile['nodata']

        out_filename = os.path.join(output_folder, f"{current_date.strftime('%Y%m%d')}_dmfsca.tif")
        with rasterio.open(out_filename, "w", **profile) as dst:
            dst.write(avg_array, 1)

        print(f"Wrote: {out_filename}")
