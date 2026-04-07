"""Process raw stem water potential to cleaned time series."""

import logging
from pathlib import Path

from dehar.physiology.water_potential import read_stemwater_potential
from dehar.utils.io import save_processed

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

RAW_FILE = Path(
    "data/raw/physiology/stemwater_potential/"
    "20251006_Stemwater_potential_hornbeam_2025_top_sensor.csv"
)
OUT_DIR = Path("data/processed/physiology/stemwater_potential")


def main():
    log.info("Reading stem water potential: %s", RAW_FILE)
    swp = read_stemwater_potential(RAW_FILE)
    log.info(
        "Loaded %d rows, columns: %s, range: %s to %s",
        len(swp),
        list(swp.columns),
        swp.index.min(),
        swp.index.max(),
    )
    save_processed(swp, OUT_DIR / "swp_dehar_15min.csv")
    log.info("Done. Output written to %s", OUT_DIR)


if __name__ == "__main__":
    main()
