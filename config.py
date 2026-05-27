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
        # fSCA Download Configs
        fsca_src_path: str = None,
        fsca_dst_path: str = None,
        curc_identikey: str = None,
    ):
        # Load .env file into os.environment
        load_dotenv(find_dotenv(usecwd=True))

        # fSCA Download Configs
        self.fsca_src_path = ( fsca_src_path or str(os.environ.get("FSCA_SRC_PATH")))
        self.fsca_dst_path = ( fsca_dst_path or str(os.environ.get("FSCA_DST_PATH")))
        self.curc_identikey = ( curc_identikey or str(os.environ.get("CURC_IDENTIKEY")))