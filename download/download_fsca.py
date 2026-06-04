# download/download_fsca.py

"""
This file contains two different methods for downloading the Rittger fSCA, along with necessary helper functions.
The primary method used is to download the files using an SSH connection to PetaLibrary. This requires setting up an
    SSH key on your device and connecting it to the CU Research Computing system using the CILogon. More information on
    how to do this can be found here: https://curc.readthedocs.io/en/latest/additional-resources/registrycilogon-instructions.html

As a fallback in case authentication fails, I have provided a function that downloads the data from the Snow Today FTP
    endpoint. This data is open to the public and does not require any authentication or setup, however the data that is
    published to Snow Today is not necessarily as up to date as the data shared with us directly through PetaLibrary.

While it would be possible to Globus and mimic the manual download process, this requires by far the most set up
    including having Karl or someone else on his team give the client permission to view files. Due to the unnecessary
    complexity of this method, I have omitted the functions from this file. See download/README.md for details on how to
    implement this if necessary.


TODO: error handling for if the remote source does not contain fsca data for the run date
"""

import os
import re
from datetime import datetime
import subprocess
import ftplib

from config import Config

##### Config #####
tiles = {"h08v04", "h08v05", "h09v04", "h09v05", "h10v04"}
##################

def _date_from_filename(filename: str) -> int | None:
    """
    Parses the first valid YYYYMMDD date from a filename

    :param filename: The filename to parse
    :return: The date parsed from the filename
    """
    # Set bounds on valid years
    current_year = datetime.today().year
    year_max = current_year + 1
    year_min = 1980

    # Search for pattern
    for match in re.finditer(r"(\d{8})", filename):
        date = match.group(1)
        year, month, day = int(date[:4]), int(date[4:6]), int(date[6:])
        if not (year_min <= year <= year_max):
            continue
        if not (1 <= month <= 12):
            continue
        try:
            datetime(year, month, day)  # catches invalid days (e.g. Feb 30)
        except ValueError:
            continue
        return int(date)
    return None


def _tile_from_filename(filename: str) -> str | None:
    """
    Return the tile ID found in a filename, or None if not recognised.
    """
    match = re.search(r"(h\d{2}v\d{2})", filename)
    if match and match.group(1) in tiles:
        return match.group(1)
    return None


def _list_remote_files(src_dir: str, method: str, cfg: Config) -> list[str]:
    """
    Returns a list of bare filenames in src_dir on the remote host.
    """
    match method:
        case "ssh":
            cmd = ["ssh", "-o", "StrictHostKeyChecking=no", f"{cfg.curc_identikey}@{cfg.curc_host}", f"ls -1 {src_dir}"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return [line.strip() for line in result.stdout.splitlines() if line.strip()]
        case "ftp":
            with ftplib.FTP() as ftp:
                # Connect to Snow Today server
                ftp.connect(cfg.snow_today_host, port=21)
                ftp.login(cfg.snow_today_username, cfg.snow_today_password)
                ftp.cwd(src_dir)
                return ftp.nlst()
        case _:
            print(f"Invalid method '{method}'. Valid methods are: ssh, ftp.")
            return []


def _scp_file(filepath: str, cfg: Config) -> bool:
    """
    Copies a file from the source directory to the destination directory over SSH.
    """
    filename = filepath.split("/")[-1]
    date = _date_from_filename(filename)
    tile = _tile_from_filename(filename)

    remote = f"{cfg.curc_identikey}@{cfg.curc_host}:{filepath}"
    local = f"{cfg.local_fsca_path}/{tile}/{str(date)[:4]}"

    if not os.path.isdir(local):
        print(f"Directory not found: {local}")
        exit(1)

    cmd = ["scp", "-o", "StrictHostKeyChecking=no", remote, local]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return True
    else:
        print(result.stderr.strip())
        return False


def _ftp_file(filepath: str, cfg: Config) -> bool:
    filename = filepath.split("/")[-1]
    date = _date_from_filename(filename)
    tile = _tile_from_filename(filename)

    remote_dir = f"{cfg.snow_today_fsca_path}/{tile}/{str(date)[:4]}/"
    local_dir = f"{cfg.local_fsca_path}/{tile}/{str(date)[:4]}"
    local_path = os.path.join(local_dir, filename)

    try:
        with ftplib.FTP() as ftp:
            # Connect to Snow Today
            ftp.connect(cfg.snow_today_host, port=21)
            ftp.login(cfg.snow_today_username, cfg.snow_today_password)
            ftp.cwd(remote_dir)

            # Download fSCA file
            with open(local_path, "wb") as f:
                ftp.retrbinary(f"RETR {filename}", f.write)

        return True

    except ftplib.all_errors as e:
        print(f"FTP error downloading {filename}: {e}")
        # Clean up partial file if it was created
        if os.path.exists(local_path):
            os.remove(local_path)
        return False

    except OSError as e:
        print(f"File system error downloading {filename}: {e}")
        return False



def _download_file(filepath: str, method: str, cfg: Config) -> bool:
    match method:
        case "ssh":
            return _scp_file(filepath, cfg)
        case "ftp":
            return _ftp_file(filepath, cfg)
        case _:
            print(f"Invalid method '{method}'. Valid methods are: ssh, ftp.")
            return False


def download_fsca(date: int, method: str, cfg: Config, prompt_user: bool = False):
    """
    Uses a CU research computing ssh connection to access and copy Rittger fSCA data from PetaLibrary to snowserver.
    Given a date, this function finds all fSCA netcdf's prior to and including the date, of the same year, that have not
    already been copied over to snowserver.
    See download/README.md for more information on how to set up the SSH key.

    :param date: (YYYYMMDD) Date the model will be run on. fSCA data will be downloaded for this date and all previous undownloaded dates of the same year.
    :param method: "ssh" (default) downloads fSCA from PetaLibrary and requires CURC authentication. "ftp" downloads fSCA from Snow Today and does not require authentication.
    :param cfg: Configuration object containing environment variables from the .env.
    :param prompt_user: Ask the user for confirmation before downloading files. Default: False
    """
    year = str(date)[:4]

    # Add each file from each tile to a queue
    to_download = []
    print(f"Checking for new fSCA data from {year}0101 until {date}...")
    for tile in tiles:
        # Define source and destination filepaths
        if method == "ssh":
            src_dir = f"{cfg.curc_fsca_path}/{tile}/{year}"
        else : # method == "ftp"
            src_dir = f"{cfg.snow_today_fsca_path}/{tile}/{year}"
        dst_dir = f"{cfg.local_fsca_path}/{tile}/{year}"

        for file in _list_remote_files(src_dir, method, cfg):
            if _date_from_filename(file) <= date:
                # Check if it exists on snowserver
                if os.path.exists(f"{dst_dir}/{file}"):
                    continue
                else:
                    to_download.append(f"{src_dir}/{file}")
            else:
                # Stop checking files newer than the model run date
                break

    if not to_download:
        print(f"No new fSCA files to download.")
        return

    new_fsca_dates = sorted(set(_date_from_filename(file) for file in to_download))
    if prompt_user:
        print(f"fSCA data will be downloaded for the following dates: {new_fsca_dates}")
        print(f"Continue? (y/N)", end=" ")
        while True:
            user_answer = input("> ").strip().lower()
            if user_answer in ["y", "yes"]:
                break
            elif user_answer in ["", "n", "no"]:
                print(f"Aborting download.")
                exit(1)
            else:
                print("Invalid input. Please enter y or n.")

    failed = []
    for new_date in new_fsca_dates:
        print(f"Downloading fSCA data from {new_date}...", end="")
        for file in sorted(to_download):
            if _date_from_filename(file) == new_date:
                # Download file
                if _download_file(file, method, cfg):
                    print(f" {_tile_from_filename(file)}",end="")
                else:
                    failed.append(file)
                to_download.remove(file)
        print(". \033[32mDone.\033[0m")

    if len(to_download) > 0:
        # TODO: error handling to retry these files
        print(f"Some fSCA files in download queue were never copied: {to_download}")

    if len(failed) > 0:
        # TODO: error handling to retry these files
        print(f"\033[31mSome fSCA files were not downloaded: {failed}\033[0m")


if __name__ == "__main__":
    config = Config()
    download_fsca(20260522, "ftp", config)
