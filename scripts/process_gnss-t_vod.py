"""Process GNSS-transmissometry VOD to filtered 30-min and daily metrics."""

import logging
from pathlib import Path

from dehar.proximal_rs.gnss_vod import (
    build_half_hourly,
    compute_daily_vod_metrics,
    read_all_receivers,
)
from dehar.utils.io import save_processed

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/proximal_rs/gnss-t")
OUT_DIR = Path("data/processed/proximal_rs/gnss_vod")


def main():
    log.info("Reading GNSS-T VOD files from %s", RAW_DIR)
    receiver_data = read_all_receivers(RAW_DIR, min_theta=30.0)

    for rid, df in sorted(receiver_data.items()):
        log.info(
            "  %s: %d obs after theta filter, %s to %s",
            rid,
            len(df),
            df.index.min().date(),
            df.index.max().date(),
        )

    half_hourly = build_half_hourly(receiver_data)
    log.info(
        "30-min merged: %d rows, columns: %s",
        len(half_hourly),
        list(half_hourly.columns),
    )
    save_processed(half_hourly, OUT_DIR / "gnss_vod_dehar_30min.csv")

    daily = compute_daily_vod_metrics(receiver_data)
    log.info(
        "Daily metrics: %d days, columns: %s",
        len(daily),
        list(daily.columns),
    )
    save_processed(daily, OUT_DIR / "gnss_vod_dehar_daily.csv")

    log.info("Done. Outputs written to %s", OUT_DIR)


if __name__ == "__main__":
    main()
