# download/download_modis.py

from datetime import datetime, timedelta
from owslib.wms import WebMapService
from PIL import Image
import numpy as np
import io

from config import Config

def percent_cloud_cover(date: int, cfg: Config) -> float:
    """
    Calculates the percentage of pixels in the MODIS image that are cloud-covered, within the MODIS_BBOX set in the .env.
    This is done by applying a confidence threshold to the MOD06_L2 Terra Cloud product provided by LP DAAC, to get a
    binary cloud/no-cloud classification for each pixel before being averaged to get the total cloud cover percent.
    This is done at a low, and spatially distorted, resolution (64 x 64 pixels) to reduce computational complexity and
    download times. There seems to be relatively little improvement in the accuracy when using higher resolution.

    :param date: YYYYMMDD date to analyze cloud cover on
    :param cfg: Configuration object containing environment variables from the .env.
    """
    date_str = datetime.strptime(str(date), "%Y%m%d").strftime("%Y-%m-%d")
    min_lon, min_lat, max_lon, max_lat = cfg.modis_bbox

    print(f"Calculating MODIS cloud cover percentage for {date}: ", end="")

    wms = WebMapService(
        "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?",
        version="1.1.1"
    )
    response = wms.getmap(
        layers=["MODIS_Terra_Cloud_Fraction_Day"],
        srs="EPSG:4326",
        bbox=(min_lon, min_lat, max_lon, max_lat),
        size=(64, 64), # A larger image (128,128) or (256,256) can be used for higher resolution, but I've seen minimal improvement.
        time=date_str,
        format="image/png",
        transparent=False,
    )
    img = Image.open(io.BytesIO(response.read())).convert("L")  # Grayscale
    # img.save(f"cloud_fraction_{date_str}.png")
    arr = np.array(img, dtype=float)

    # GIBS renders cloud fraction as 0 (0%) to 255 (100%). A value of 77 or greater is typical for full visual cloud cover.
    cloud_cover_threshold = 70
    cloud_cover_percentage = (arr > cloud_cover_threshold).mean() * 100

    print(f"{cloud_cover_percentage:.2f}%")
    return cloud_cover_percentage


def _download_truecolor_geotiff(date: int, cfg: Config, output_path: str, resolution_m: float = 500.0):
    """
    Downloads the MODIS Terra Corrected Reflectance True Color image for the date, based on the MODIS_BBOX set in the .env

    :param date: YYYYMMDD date of the image to download
    :param cfg: Configuration object containing environment variables from the .env.
    :param output_path: Full filepath for the location to download the file to
    :param resolution_m: Resolution of the image in meters (default 500)
    """
    date_str = datetime.strptime(str(date),"%Y%m%d").strftime("%Y-%m-%d")
    min_lon, min_lat, max_lon, max_lat = cfg.modis_bbox

    # Target cell size in degrees
    deg_per_pixel = resolution_m / 111_320

    height_px = max(1, round((max_lat - min_lat) / deg_per_pixel))
    width_px = max(1, round((max_lon - min_lon) / deg_per_pixel))

    print(f"Downloading MODIS Terra true color imagery for {date}...", end="")

    wms = WebMapService("https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi", version="1.1.1")

    response = wms.getmap(
        layers=["MODIS_Terra_CorrectedReflectance_TrueColor"],
        srs="EPSG:4326",
        bbox=(min_lon, min_lat, max_lon, max_lat),
        size=(width_px, height_px),
        time=date_str,
        format="image/tiff",
        transparent=False,
    )

    with open(output_path, "wb") as f:
        f.write(response.read())

    print(f"\033[32m Done.\033[0m")

    return output_path


def download_modis(date: int, cfg: Config, cloud_cover_max: int = 30, loopback_days: int = 5):
    """
    Downloads MODIS Terra true color imagery from NASA Global Imagery Browse Services.
    NASA Common Metadata Repository is queried to select the date with the most cloud-free imagery.
    
    Starting from {date}, the previous {loopback_days} number of days are checked in order until a date with less
    than {cloud_cover_max percent} of the image is cloudy. If none of the last {loopback_days} have a cloud cover
    percentage below that threshold, then the least cloudy date is used.
    
    To always download imagery from the specified date, regardless of cloud cover, set loopback_days to 1.


    :param date: (YYYYMMDD) Date the model is run on, and the first day to check for cloud free imagery.
    :param cfg: Configuration object containing environment variables from the .env.
    :param cloud_cover_max: (default 20) The maximum percentage of cloud cover over the defined region.
    :param loopback_days: (default 5) Number of previous days to select most cloud free image from.
    """

    # TODO: move hardcoded path to .env
    year = str(date)[:4]
    rt_report_folder = f"J:/paperwork/0_UCSB_DWR_Project/{year}_RT_Reports/{date}_RT_report"

    # Sequentially check the previous days until a date with low enough cloud cover is found
    for days_back in range(loopback_days):
        check_date = datetime.strptime(str(date), "%Y%m%d") - timedelta(days=days_back)
        check_date = int(check_date.strftime("%Y%m%d"))

        # Calculate rough cloud cover percentage
        cloud_cover = percent_cloud_cover(check_date, cfg)

        if cloud_cover < cloud_cover_max:
            modis_date = datetime.strptime(str(check_date), "%Y%m%d").strftime('%Y-%m-%d')
            modis_path = f"{rt_report_folder}{cfg.modis_path}/snapshot_{modis_date}T00_00_00Z_UseThis.tif"
            _download_truecolor_geotiff(check_date, cfg, modis_path)
            # Stop checking previous days
            break
    else:
        # Download the most recent date, despite a high cloud cover percentage
        print(f"No imagery with less than {cloud_cover_max}% cloud cover found within {loopback_days} days of {date}.")
        modis_date = datetime.strptime(str(date), "%Y%m%d").strftime('%Y-%m-%d')
        modis_path = f"{rt_report_folder}{cfg.modis_path}/snapshot_{modis_date}T00_00_00Z_UseThis.tif"
        _download_truecolor_geotiff(date, cfg, modis_path)


if __name__ == "__main__":
    config = Config()
    download_modis(20260517, config, cloud_cover_max=30, loopback_days=5)