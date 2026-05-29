import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

class Config:
    """
    Configuration class for running the model

    Reads environment variables in from a .env file (searched for upwards from the current working directory). Stores
    all input/output paths, runtime parameters, and API keys/tokens. Any variable can be overridden by passing it as
    an argument, which takes priority over the value from the .env file.

    See .env.example for how each variable should be defined in the .env.
    """
    def __init__(self,
        ##### fSCA Download Configs #####
        local_fsca_path: str = None,
         # PetaLibrary (ssh method)
        curc_fsca_path: str = None,
        curc_host: str = None,
        curc_identikey: str = None,
        # Snow Today (ftp method)
        snow_today_fsca_path: str = None,
        snow_today_host: str = None,
        snow_today_username: str = None,
        snow_today_password: str = None,

        ##### SNODAS Download Configs #####
        snodas_local_path: str = None,
        snodas_remote_path: str = None,
        snodas_host: str = None,

        ##### SnowTrax Download Configs #####
        snowtrax_path: str = None,
        snowtrax_filename: str = None,
    ):
        # Load .env file into os.environment
        load_dotenv(find_dotenv(usecwd=True))

        ##### fSCA Download Configs #####
        self.local_fsca_path = local_fsca_path or str(os.environ.get("LOCAL_FSCA_PATH"))
        # PetaLibrary (ssh method)
        self.curc_fsca_path = curc_fsca_path or str(os.environ.get("CURC_FSCA_PATH"))
        self.curc_host = curc_host or str(os.environ.get("CURC_HOST"))
        self.curc_identikey = curc_identikey or str(os.environ.get("CURC_IDENTIKEY"))
        # Snow Today (ftp method)
        self.snow_today_fsca_path = snow_today_fsca_path or str(os.environ.get("SNOW_TODAY_FSCA_PATH"))
        self.snow_today_host = snow_today_host or str(os.environ.get("SNOW_TODAY_HOST"))
        self.snow_today_username = snow_today_username or str(os.environ.get("SNOW_TODAY_USERNAME"))
        self.snow_today_password = snow_today_password or str(os.environ.get("SNOW_TODAY_PASSWORD"))

        ##### SNODAS Download Configs #####
        self.snodas_local_path = snodas_local_path or str(os.environ.get("SNODAS_LOCAL_PATH"))
        self.snodas_remote_path = snodas_remote_path or str(os.environ.get("SNODAS_REMOTE_PATH"))
        self.snodas_host = snodas_host or str(os.environ.get("SNODAS_HOST"))

        ##### SnowTrax Download Configs #####
        self.snowtrax_path = snowtrax_path or str(os.environ.get("SNOWTRAX_PATH"))
        self.snowtrax_filename = snowtrax_filename or str(os.environ.get("SNOWTRAX_FILENAME"))
