"""Process raw tree water deficit (TWD) dendrometer data."""

import logging
from pathlib import Path

from dehar.physiology.twd import read_twd
from dehar.utils.io import save_processed

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

RAW_FILE = Path(
    "data/raw/physiology/twd/20260202_twd_Hartheim_C-betulus_2025_raw_final.csv"
)
OUT_DIR = Path("data/processed/physiology/twd")


def main():
    log.info("Reading TWD data: %s", RAW_FILE)
    twd = read_twd(RAW_FILE)
    log.info(
        "Loaded %d rows, %d trees: %s",
        len(twd),
        len(twd.columns),
        list(twd.columns),
    )
    save_processed(twd, OUT_DIR / "twd_dehar_30min.csv")
    log.info("Done. Output written to %s", OUT_DIR)


if __name__ == "__main__":
    main()
