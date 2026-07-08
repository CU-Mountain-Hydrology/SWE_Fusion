# run/main.py

import argparse
from datetime import datetime

from config import Config
from download.download_fsca import download_fsca
from download.download_snodas import download_snodas
from download.download_snowtrax import download_snowtrax
from download.download_sensors import download_sensors
from run.fsca_processing import fsca_processing
from run.run_R_model import write_simulation_date, run_R_model

def run_model(date: int, prompt_user: bool=False):
    """
    This is the master script to automatically run the whole model. Each step is called sequentially, with checkpoints
    in order to resume mid way through in case of a fatal error.
    TODO: add checkpoints

    :param date: (YYYYMMDD) Date the model is run on.
    :param prompt_user: Ask the user for confirmation before downloading files. Default: False

    """
    # Set up configuration object containing parameters from the .env
    cfg = Config()

    # Download fSCA data
    download_fsca(date, "ssh", cfg, prompt_user=prompt_user)

    # Download SNODAS data
    download_snodas(date, cfg)

    # Download SnowTrax SWE data
    download_snowtrax(cfg)

    # fSCA Processing
    fsca_processing(date, cfg)

    # Set date in text file at SIMULATION_DATE_PATH from the .env
    # This is never read by the automated daily model, only kept for consistency and manual runs.
    write_simulation_date(date, cfg)

    # Download snow sensor data
    download_sensors(date, cfg)

    # Run model with CCR
    run_R_model(date=date, isCCR=True, cfg=cfg)

    # Run model without CCR
    run_R_model(date=date, isCCR=False, cfg=cfg)

    # Run SWE_Fusion
    # TODO


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("date", type=int, help="YYYYMMDD Date the model is run on.")
    parser.add_argument("-u", "--prompt_user", action="store_true",
                        help="Prompt the user before overwriting or automatically selecting files")

    args = parser.parse_args()

    # Sanity checks
    date_d = args.date.strftime("%Y%m%d")
    if date_d > datetime.today():
        print(f"Date ({args.date} is in the future! Aborting.")
        exit(1)

    # Run Model
    run_model(args.date, args.prompt_user)

