# download/download_snowtrax.py

import os
import requests
import json

from config import Config

def download_snowtrax(cfg: Config):
    """
    Downloads the most recent CCSS SnowTrax data my mimicking the network request sent by visiting https://snow.water.ca.gov/fcast_resources
    and clicking Download Data > Snow Data.
    The downloaded data is written to SNOWTRAX_FILENAME and saved to SNOWTRAX_PATH, as set in the .env

    :param cfg: Configuration object containing environment variables from the .env.
    """
    # Create network request
    url = "https://snow.water.ca.gov/service/plotly/dash/fcast_resources/_dash-update-component"

    payload = {
        "output": "download-data.data",
        "outputs": {"id": "download-data", "property": "data"},
        "inputs": [
            {"id": "data-export-snow", "property": "n_clicks", "value": 1},
            {"id": "data-export-flow", "property": "n_clicks", "value": 0}
        ],
        "changedPropIds": ["data-export-snow.n_clicks"],
        "parsedChangedPropIds": ["data-export-snow.n_clicks"]
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Referer": "https://snow.water.ca.gov/service/plotly/dash/fcast_resources",
    }

    response = requests.post(url, json=payload, headers=headers, stream=True)

    # Define local path
    local_path = f"{cfg.snowtrax_path}/{cfg.snowtrax_filename}"

    # Estimate download size to get percentage
    if os.path.exists(local_path):
        total_bytes = os.path.getsize(local_path)
    else:
        total_bytes = 10_000_000  # approximate expected size (10MB)

    # Stream download of SWE data
    raw = b""
    downloaded = 0
    for chunk in response.iter_content(chunk_size=65536):
        raw += chunk
        downloaded += len(chunk)
        print(f"\rDownloading SnowTrax SWE data... {min(100, round(downloaded / total_bytes * 100))}%", end="", flush=True)

    print("\rDownloading SnowTrax SWE data... \033[32m100%\033[0m")

    # Write downloaded content to csv file
    data = json.loads(raw)
    csv_content = data["response"]["download-data"]["data"]["content"]

    with open(local_path, "wb") as f:
        f.write(csv_content.encode())


if __name__ == '__main__':
    config = Config()
    download_snowtrax(config)
