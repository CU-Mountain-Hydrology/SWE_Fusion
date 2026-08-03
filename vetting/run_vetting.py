# vetting/run_vetting.py
"""
This is the main script to generate all the vetting plots for our weekly meetings. This does not include any of the
plots that are generated daily. See run/main.py for those.

TODO: docs about what plots are created
"""

from datetime import datetime, timedelta

from config import Config
from vetting.snm_plots import weekly_swe_trend_snm

def run_vetting(date: int, cfg: Config):
    """
    TODO: docs

    :param date: (YYYYMMDD) Date that the vetting is run on. This will be the final day in weekly plots.
    :param cfg: Configuration object containing environment variables set in the .env
    """
    date_f = datetime.strptime(str(date), "%Y%m%d")
    week_start = date_f - timedelta(days=7)
    weekly_swe_trend_snm(date=int(week_start.strftime("%Y%m%d")), cfg=cfg, basin_means=False, volume_total=True, show_plot=True)


if __name__ == "__main__":
    config = Config()
    run_vetting(config)