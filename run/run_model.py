# run/run_model.py

import argparse
from datetime import datetime

from config import Config
from download.download_fsca import download_fsca
from download.download_snodas import download_snodas
from download.download_snowtrax import download_snowtrax
from download.download_sensors import download_sensors

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
    # TODO

    # Set date in simulation_date_historic_ET.txt
    # TODO

    # Download snow sensor data
    download_sensors(date, cfg)

    # Run model with CCR
    # TODO

    # Run model without CCR
    # TODO

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

