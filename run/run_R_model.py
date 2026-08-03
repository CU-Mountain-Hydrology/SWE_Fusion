# run/run_R_model.py

from dataclasses import dataclass, field, InitVar, asdict
from datetime import datetime
import os
import math
import json
import subprocess


from config import Config

# Convert the R-style values of "T"/"F" from the .env to booleans
def _parse_r_bool(value: str, default: str = "F") -> bool:
    return (value or default).strip().upper() in ("T", "TRUE")

@dataclass
class RModelConfig:
    # Passed every run
    oldestDate: str # 'YYYYMMDD' Same as run date
    isCCR: bool
    cfg: InitVar['Config'] # injected Config class

    # Pulled from .env
    UserName: str = field(init=False)
    PATH_regress: str = field(init=False)
    SensPer: float = field(init=False)
    FSCA_PATH: str = field(init=False)
    DMFSCA_PATH: str = field(init=False)
    fscaType: str = field(init=False)
    isfscaFlag: bool = field(init=False)
    isfscaMask: bool = field(init=False)
    isGPKG: bool = field(init=False)
    ishisday: bool = field(init=False)
    besthisdate: str = field(init=False)
    SNOW_VAR: str = field(init=False)
    MODSCAG_PATH: str = field(init=False)
    MODSCAG_TYPE: str = field(init=False)
    MODSCAG_FILE: str = field(init=False)
    FVEG_CORRECTION: bool = field(init=False)

    # Derived from other configs
    simulationday: str = field(init=False)
    RUNNAME: str = field(init=False)

    # ModDomLst: list = field(init=False, default_factory=lambda: ["INMT", "NOCN", "PNW", "SNM", "SOCN"])

    def __post_init__(self, cfg: 'Config'):
        # Pulled from .env
        self.UserName = cfg.rmodel_username
        self.PATH_regress = cfg.regress_path
        self.SensPer = cfg.rmodel_sens_per
        self.FSCA_PATH = cfg.processed_fsca_path
        self.DMFSCA_PATH = cfg.dmfsca_path
        self.fscaType = cfg.rmodel_fsca_type
        self.isfscaFlag = _parse_r_bool(cfg.rmodel_is_fsca_flag)
        self.isfscaMask = _parse_r_bool(cfg.rmodel_is_fsca_mask)
        self.isGPKG = _parse_r_bool(cfg.rmodel_is_gpkg)
        self.ishisday = _parse_r_bool(cfg.rmodel_is_his_day)
        self.besthisdate = cfg.rmodel_best_his_date
        self.SNOW_VAR = cfg.rmodel_snow_var
        self.MODSCAG_PATH = cfg.modscag_path
        self.MODSCAG_TYPE = cfg.rmodel_modscag_type
        self.MODSCAG_FILE = cfg.rmodel_modscag_file
        self.FVEG_CORRECTION = _parse_r_bool(cfg.rmodel_fveg_correction)

        # Derived from other configs
        self.simulationday = datetime.strptime(self.oldestDate, "%Y%m%d").strftime("%Y-%m-%d")
        self.RUNNAME = self._build_runname()

    def _build_runname(self) -> str:
        parts = ["RT"]
        if not self.FVEG_CORRECTION:
            parts.append("CanAdj")
        parts.append(self.SNOW_VAR)
        parts.append("wCCR" if self.isCCR else "woCCR")
        if self.isfscaFlag:
            parts.append("nofscamskSens")
        if not self.isfscaMask:
            parts.append("noMdlFsca")
        if not math.isclose(self.SensPer, 0.3):
            parts.append(str(round(self.SensPer * 100)))
        return "_".join(parts)



def write_simulation_date(date: int, cfg: Config):
    """
    Writes the simulation date to the text file found at SIMULATION_DATE_PATH in the .env.
    This value is read into the manual version of the main R code, however it is no longer used for automated daily
    runs. This function is only kept for clarity in case the model needs to be run manually.

    :param date: (YYYYMMDD) Date the model will be run on. This is what is set in the file.
    :param cfg: Configuration object containing environment variables from the .env.
    """
    formatted_date = datetime.strptime(str(date), "%Y%m%d").strftime("%Y-%m-%d")
    with open(cfg.simulation_date_path, "w") as f:
        f.write('"simdate"\n')
        f.write(formatted_date)



def run_R_model(date: int, isCCR: bool, cfg: Config) -> tuple[int, str]:
    """
    Runs the R model given the parameters defined by date and isCCR, as well as all those set in the .env
    The path to the R code is RMODEL_SCRIPT_PATH in the .env
    A JSON copy of all the config settings used is saved to H:/WestUS_Data/Regress_SWE/config_logs/WY{year}/{date}/{runname}

    :param date: (YYYYMMDD) Date the model will be run on.
    :param isCCR: (bool) True if the model will be run with CoCoRaHs sensors, False otherwise.
    :param cfg: Configuration object containing environment variables and all remaining R model configs from the .env.

    :return: Tuple of (return code, model run name) where a return code of 0 indicates success.
    """
    # Parse water year from date
    date_f = datetime.strptime(str(date), "%Y%m%d")
    water_year = date_f.year if date_f.month < 10 else date_f.year + 1

    print(f"Generating R model config file for {date} with isCCR={isCCR}...", end="")
    model_config = RModelConfig(oldestDate=str(date), isCCR=isCCR, cfg=cfg)

    # Save JSON with all configs for this run
    os.makedirs(f"{cfg.rmodel_config_log_dir}/WY{water_year}/{date}", exist_ok=True)
    config_path = f"{cfg.rmodel_config_log_dir}/WY{water_year}/{date}/{model_config.RUNNAME}.json"
    with open(config_path, "w") as f:
        json.dump(asdict(model_config), f, indent=4)
    print(". \033[32mDone.\033[0m")
    print(f"Config saved to {config_path}")

    # Run R model
    print(f"Running R model for {date} with isCCR={isCCR}...", end="")
    result = subprocess.run(
        [cfg.rscript_exe_path, cfg.rmodel_script_path, str(config_path)],
        capture_output=True,
        text=True,
        cwd=cfg.regress_path
    )
    print(". \033[32mDone.\033[0m")

    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(
            f"R model failed!\n"
            f"Config: {config_path}\n"
            f"{result.stderr}"
        )

    return result.returncode, model_config.RUNNAME

if __name__ == "__main__":
    config = Config()
    # write_simulation_date(20260517, config)
    run_R_model(20260517, False, config)
