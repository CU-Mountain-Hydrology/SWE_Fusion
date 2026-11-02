# run/run_R_model.py

from dataclasses import dataclass, field, InitVar, asdict
from datetime import datetime
import os
import math
import json
import subprocess
import re
import csv
from collections import Counter
from pathlib import Path
from run.utils import get_water_year


from config import Config

# Convert the R-style values of "T"/"F" from the .env to booleans
def _parse_r_bool(value: str, default: str = "F") -> bool:
    return (value or default).strip().upper() in ("T", "TRUE")

@dataclass
class RModelConfig:
    # Passed every run
    oldestDate: str # 'YYYYMMDD' Same as run date unless running in the past in which case it is the date which sensors
                    #  were last updated for (ex. 20260604 if Regress_SWE/snow_sensors/cdec_ADM_2026-06-04.csv)
    runDate: str    # 'YYYYMMDD' Date the model is simulating
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
        self.simulationday = datetime.strptime(self.runDate, "%Y%m%d").strftime("%Y-%m-%d")
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


def get_sensor_date(run_date: int, cfg: Config) -> str:
    """
    Returns the download date str(YYYYMMDD) of the sensor CSVs the R model should read.

    Sensor files are named like cdec_ADM_2026-06-04.csv. This picks the date >= run_date that has the most files,
    in case of a corrupted or partial download. Falls back to run_date if nothing is found.
    """
    sensor_dir = Path(cfg.regress_path) / "snow_sensors"   # same dir R reads (PATH_regress + 'snow_sensors')
    pattern = re.compile(r"_(\d{4})-(\d{2})-(\d{2})\.csv$")

    counts = Counter()
    for f in sensor_dir.glob("*.csv"):
        m = pattern.search(f.name)
        if m:
            d = int("".join(m.groups()))
            if d >= run_date:
                counts[d] += 1

    if not counts:
        return str(run_date)
    return str(min(counts, key=lambda d: (-counts[d], d)))


DOMAIN_SENSOR_TOTALS = {"INMT": 225, "NOCN": 213, "PNW": 115, "SNM": 111, "SOCN": 333}


def get_domain_log_path(date: int, cfg: Config) -> Path:
    return Path(cfg.model_run_log_dir.format(water_year=get_water_year(date))) / f"domains_{date}.csv"


def write_domain_log(r_output: str, date: int, cfg: Config) -> None:
    """
    Parses the R model output and writes domains_YYYYMMDD.csv with the number of sensors reporting > 0,
    the % of the domain's total sensors, and whether the model ran for each domain.
    """
    skip_markers = ("Skip this day!", "Model run aborted", "No fSCA")
    parts = re.split(r"Model Domain: (\w+)", r_output)
    rows = []
    for domain, block in zip(parts[1::2], parts[2::2]):
        match = re.search(r"Num of pillows > 0: (\d+)", block)
        sensors = int(match.group(1)) if match else ""
        total = DOMAIN_SENSOR_TOTALS.get(domain)
        pct = round(100 * sensors / total, 1) if match and total else ""
        ran = not any(marker in block for marker in skip_markers)
        rows.append([domain, sensors, pct, ran])

    log_path = get_domain_log_path(date, cfg)
    os.makedirs(log_path.parent, exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "sensors_gt0", "pct_of_total", "ran"])
        writer.writerows(rows)


def apply_domain_log(date: int, cfg: Config) -> None:
    """
    Sets cfg.domain_list to the domains that ran, according to domains_YYYYMMDD.csv. Does nothing if the file
    does not exist.
    """
    log_path = get_domain_log_path(date, cfg)
    if not log_path.exists():
        return
    with open(log_path, newline="") as f:
        ran = {row["domain"] for row in csv.DictReader(f) if row["ran"] == "True"}
    cfg.domain_list = [d for d in cfg.domain_list if d in ran]


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
    # Parse the oldest date from the snow sensor filepaths. This should always be >= model run date.
    oldest_date = get_sensor_date(date, cfg)
    model_config = RModelConfig(oldestDate=oldest_date, runDate=str(date), isCCR=isCCR, cfg=cfg)

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
    if result.returncode == 0:
        print(". \033[32mDone.\033[0m")
    else:
        print(". \033[31mFailed.\033[0m")

    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(
            f"R model failed!\n"
            f"Config: {config_path}\n"
            f"{result.stderr}"
        )

    write_domain_log(result.stdout, date, cfg)
    apply_domain_log(date, cfg)

    return result.returncode, model_config.RUNNAME

if __name__ == "__main__":
    config = Config()
    # write_simulation_date(20260517, config)
    run_R_model(20260517, False, config)
