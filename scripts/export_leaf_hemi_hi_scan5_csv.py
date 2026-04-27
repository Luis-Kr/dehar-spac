"""Export 05:00 UTC scans from leaf hemi-hi parquet to a small CSV."""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

PARQUET = Path("data/processed/proximal_rs/leaf/leaf_hemi_hi_2025.parquet")
# Same directory; one row per height layer per scan (05:00 UTC only)
OUT_CSV = PARQUET.with_name("leaf_hemi_hi_2025_scan05utc.csv")
SCAN_HOUR = 5  # UTC, matches `scan_hour` in batch pipeline


def main() -> None:
    if not PARQUET.is_file():
        log.error("Missing parquet: %s", PARQUET.resolve())
        return

    df = pd.read_parquet(PARQUET)
    if "scan_hour" not in df.columns:
        log.error("Expected column 'scan_hour' in parquet")
        return

    sel = df.loc[df["scan_hour"] == SCAN_HOUR].copy()
    sel = sel.sort_values(["datetime", "height"], ignore_index=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    sel.to_csv(OUT_CSV, index=False)

    n_scan = sel["datetime"].nunique()
    log.info(
        "Wrote %s  (%d rows, %d unique scans @ %02d:00 UTC)",
        OUT_CSV,
        len(sel),
        n_scan,
        SCAN_HOUR,
    )


if __name__ == "__main__":
    main()
