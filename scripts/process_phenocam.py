"""Process raw network-PhenoCam summary products into one daily parquet.

Reads the four standard PhenoCam CSVs (GCC / NDVI x 1-day / 3-day) for the
DE-Har ROI and writes the FULL multi-year daily record to

    data/processed/proximal_rs/phenocam/phenocam_daily.parquet

The full record (2018 -> present) is kept deliberately: it captures the
post-2018-drought legacy that frames the site. The 2025 analysis slice is taken
downstream by ``scripts/aggregate_daily_streams_2025.py`` (it joins only 2025).

Canonical greenness = ``gcc_phenocam_{agg}_p90``; canonical proximal NDVI =
``ndvi_phenocam_{agg}_p90`` (Sonnentag et al. 2012; Richardson et al. 2018).
3-day products keep their native cadence (NaN between window dates — no fill).

Run from the repo root (needs the ``dehar-spac`` env):
    python scripts/process_phenocam.py
"""
from __future__ import annotations

import logging
from pathlib import Path

from dehar.proximal_rs.phenocam import load_phenocam_daily

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("process_phenocam")

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "proximal_rs" / "phenocam"
OUT_DIR = ROOT / "data" / "processed" / "proximal_rs" / "phenocam"
OUT_PATH = OUT_DIR / "phenocam_daily.parquet"


def main() -> None:
    log.info("Reading PhenoCam products from %s", RAW_DIR)
    daily = load_phenocam_daily(RAW_DIR)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(OUT_PATH, index=True)

    canon = [c for c in daily.columns if c.endswith("_p90") and "smooth" not in c]
    log.info("Columns    : %d  (%d canonical p90)", daily.shape[1], len(canon))
    log.info("Range      : %s -> %s  (%d days)",
             daily.index.min().date(), daily.index.max().date(), len(daily))
    for c in canon:
        log.info("  %-28s valid=%d", c, int(daily[c].notna().sum()))
    log.info("Saved      : %s (%.2f MB)", OUT_PATH, OUT_PATH.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
