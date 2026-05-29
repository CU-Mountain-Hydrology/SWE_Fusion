# download/download_snodas.py

import os
import ftplib
import tarfile

from config import Config


def download_snodas(date: int, cfg: Config):
    """
    Downloads Snodas data for a specific date using FTP

    :param date: (YYYYMMDD) Date the model will be run on. SNODAS data will be downloaded for this date.
    :param cfg: Configuration object containing environment variables from the .env.
    """
    # Check that the local parent directory exists
    if not os.path.exists(cfg.snodas_local_path):
        print(f"SNODAS local path does not exist: {cfg.snodas_local_path}. "
              f"Ensure SNODAS_LOCAL_PATH is set correctly in the .env")
        # TODO: better error handling such as make the new directory
        exit(1)

    # Don't download new data if there is already a SNODAS folder for the date
    if os.path.exists(f"{cfg.snodas_local_path}/SNODAS_{date}"):
        print(f"SNODAS folder already exists for {date}!")
        return

    # Define filepaths
    year = str(date)[:4]
    month = str(date)[4:6]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    src_dir = f"{cfg.snodas_remote_path}/{year}/{month}_{months[int(month)-1]}/"

    tar_file = f"SNODAS_{date}.tar"
    local_tar = f"{cfg.snodas_local_path}/{tar_file}"

    # Connect to SNODAS server
    try:
        with ftplib.FTP() as ftp:
            ftp.connect(cfg.snodas_host, port=21)
            ftp.login()
            ftp.cwd(src_dir)

            # Download tar file
            print(f"Downloading SNODAS for {date}...", end="")
            with open(local_tar, "wb") as f:
                ftp.retrbinary(f"RETR {tar_file}", f.write)
            print(f"\033[32m Done.\033[0m")

    except ftplib.all_errors as e:
        print(f"\nError downloading SNODAS: {e}")
        # TODO: error handling
        # Clean up partial file if it was created
        if os.path.exists(local_tar):
            os.remove(local_tar)
        exit(1)

    # Confirm file was downloaded
    if not os.path.exists(local_tar):
        print(f"Local zip file not found! {local_tar}")
        # TODO: error handling
        exit(1)

    # Unzip SNODAS file
    print(f"Unzipping SNODAS for {date}...", end="")
    with tarfile.open(local_tar, "r:*") as tar:
        tar.extractall(f"{cfg.snodas_local_path}/SNODAS_{date}", filter='data')
    print(f"\033[32m Done.\033[0m")

    # Delete tar file
    if os.path.exists(local_tar):
        os.remove(local_tar)

if __name__ == "__main__":
    config = Config()
    download_snodas(20260522, config)