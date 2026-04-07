"""Process raw sap flux density to 30-min and daily time series."""

import logging
from pathlib import Path

from dehar.physiology.sapflow import compute_daily_sapflow, read_all_trees
from dehar.utils.io import save_processed

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/physiology/sap_flux_density")
OUT_DIR = Path("data/processed/physiology/sap_flux_density")


def main():
    log.info("Reading sap flux density files from %s", RAW_DIR)
    half_hourly = read_all_trees(RAW_DIR)
    log.info(
        "Loaded %d rows, %d trees: %s",
        len(half_hourly),
        len(half_hourly.columns),
        list(half_hourly.columns),
    )

    save_processed(half_hourly, OUT_DIR / "sapflow_dehar_30min.csv")

    daily = compute_daily_sapflow(half_hourly)
    log.info("Daily aggregation: %d days", len(daily))
    save_processed(daily, OUT_DIR / "sapflow_dehar_daily.csv")

    log.info("Done. Outputs written to %s", OUT_DIR)


if __name__ == "__main__":
    main()
