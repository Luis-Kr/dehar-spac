"""Process raw eddy covariance / met station data to meteo, fluxes, and soil moisture."""

import logging
from pathlib import Path

from dehar.atmosphere.flux import extract_fluxes
from dehar.atmosphere.meteo import extract_meteorology, read_atmosphere_soil_raw
from dehar.soil.vwc import extract_soil_moisture
from dehar.utils.io import save_processed

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

RAW_FILE = Path("data/raw/atmosphere_soil/DE-HARTH_M-2025-Results_v1.txt")
OUT_DIR = Path("data/processed/atmosphere_soil")


def main():
    log.info("Reading raw atmosphere+soil file: %s", RAW_FILE)
    raw = read_atmosphere_soil_raw(RAW_FILE)
    log.info("Loaded %d rows, %d columns", len(raw), len(raw.columns))

    meteo = extract_meteorology(raw)
    log.info("Meteorology: %d rows, columns: %s", len(meteo), list(meteo.columns))
    save_processed(meteo, OUT_DIR / "meteo_dehar_30min.csv")

    fluxes = extract_fluxes(raw)
    log.info("Fluxes: %d rows, columns: %s", len(fluxes), list(fluxes.columns))
    save_processed(fluxes, OUT_DIR / "fluxes_dehar_30min.csv")

    soil = extract_soil_moisture(raw)
    log.info("Soil moisture: %d rows, columns: %s", len(soil), list(soil.columns))
    save_processed(soil, OUT_DIR / "soil_moisture_dehar_30min.csv")

    log.info("Done. Outputs written to %s", OUT_DIR)


if __name__ == "__main__":
    main()
