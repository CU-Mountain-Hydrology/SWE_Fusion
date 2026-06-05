# download/download_sensors.py

import pandas as pd
from datetime import datetime, timedelta
import zoneinfo
from pathlib import Path
import requests
import time
import re

from config import Config

def _extract_date_from_filename(filepath: Path) -> datetime | None:
    """
    Parse the YYYY-MM-DD portion out of a filename like cdec_ADM_2026-05-17.csv.

    :param filepath: Filepath to parse.
    :return: Parsed date or None
    """
    parts = re.split(r'[._]', filepath.stem)  # stem drops .csv
    for part in parts:
        try:
            return datetime.fromisoformat(part)
        except ValueError:
            continue
    return None


def _download_file(url: str, filepath: Path, retries = 3, delay_s = 5) -> bool:
    """
    Downloads a file from url and saves it to filepath.

    :param url: URL request to download.
    :param filepath: Path to save the downloaded file to.
    :param retries: Number of times to retry download (default 3).
    :param delay_s: Number of seconds to delay before downloading (default 5).

    :return: True if the download was successful, False otherwise.
    """
    for attempt in range(1, retries+1):
        try:
            response = requests.get(url, timeout=500)
            response.raise_for_status()
            filepath.write_bytes(response.content)
            print(f"   Saved to {filepath.name}")
            return True

        except requests.RequestException as e:
            print(f"   Download attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                print(f"   Retrying in {delay_s} seconds...")
                time.sleep(delay_s)

    print("   All {retries} attempts failed, skipping.")
    return False


def download_sensors(date: int, cfg: Config):
    """
    Downloads all SNOTEL and CDEC sensors for a specific date. Mimics the process from 0_get_All_stationswe_data_OLAF.R
    Download sensors after 11am MST to ensure all values are updated.

    :param date: YYYYMMDD date the model is being run on
    :param cfg: Configuration object containing environment variables from the .env.
    """
    date_d = datetime.strptime(str(date), "%Y%m%d")
    mst = zoneinfo.ZoneInfo("America/Denver")
    now = datetime.now(tz=mst)
    # Confirm date is not in the future
    if now.date() < date_d.date():
        print(f"{date} is in the future!")
        print(f"Attempting sensor download for today ({now.strftime("%Y%m%d")}) instead.")
        date = int(now.strftime("%Y%m%d"))
        date_d = datetime.strptime(str(date), "%Y%m%d")

    # Confirm it is past 11am on the specified date
    cutoff = datetime(date_d.year, date_d.month, date_d.day, 11, 0, 0, tzinfo=mst)
    if now.date() == datetime.strptime(str(date), "%Y%m%d") and now < cutoff:
        print(f"Download sensors after 11am MST to ensure today's reading are updated! It is currently {now.strftime("%H:%M")} MST.")
        yesterday = datetime.strftime((now-timedelta(days=1)).date(), "%Y%m%d")
        print(f"Attempting sensor download for yesterday ({yesterday}) instead.")
        date = int(yesterday)
        date_d = datetime.strptime(str(date), "%Y%m%d")

    # Read master sensor csv file containing CDEC and SNOTEL
    # One row per sensor, with columns site_id, 2 digit state_id, and network (snotel or cdec)
    master_sensor_path = f"{cfg.sensor_path}/{cfg.sensor_list}"
    sensor_list = pd.read_csv(master_sensor_path)

    # Loop through every sensor
    print("Downloading snow pillow sensor data...")
    failed_sensors = []
    for i, (_, row) in enumerate(sensor_list.iterrows(), start=1):
        site_id = str(row["site_id"])
        network = str(row["network"]).lower()
        progress = i / len(sensor_list) * 100

        print(f"{progress:.1f}%, Site ID: {site_id}, Network: {network}")

        new_file = Path(f"{cfg.sensor_path}/{network}_{site_id}_{date_d.strftime("%Y-%m-%d")}.csv")
        if new_file.exists():
           print(f"   File already current, skipping.")
           continue

        # Find the most recent existing file date
        old_files = list(Path(cfg.sensor_path).glob(f'{network}_{site_id}*.csv'))
        if old_files:
            most_recent_date = max([_extract_date_from_filename(f) for f in old_files])
            if most_recent_date.date() >= date_d.date():
                print(f"   Existing file is already up to date with {date_d.strftime('%Y%m%d')}. Skipping.")
                continue

        # Build download URL
        download_url = None
        match network:
            case "snotel":
                state_id = str(row['state_id'])
                download_url = (
                    f'https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/'
                    f'customMultiTimeSeriesGroupByStationReport/daily/start_of_period/'
                    f'{site_id}:{state_id}:SNTL|id=%22%22|name/'
                    f'POR_BEGIN,POR_END/WTEQ::value'
                )
            case "cdec":
                download_url = (
                    f'http://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet'
                    f'?Stations={site_id}&SensorNums=82&dur_code=D'
                    f'&Start=1984-12-20&End={date_d.strftime("%Y-%m-%d")}'
                )
            case _:
                print(f"   Network {network} not recognized. Skipping.")
                continue

        # Download sensor data
        success = _download_file(download_url, new_file)

        if success:
            # Remove stale files after successful download
            for f in old_files:
                try:
                    f.unlink()
                except OSError as e:
                    print(f"   Failed to remove old file {f.name}: {e}")
        else:
            # Report failed sensor download
            failed_sensors.append((site_id, network))

    print("Snow pillow sensor data downloaded.")
    if failed_sensors:
        print(f"The following sensors failed to download: {failed_sensors}.")


if __name__ == "__main__":
    config = Config()
    download_sensors(20260604, config)