"""Build a LEAF hemi_hi variant: up-drift re-fold on the DAILY MEDIAN "up".

Off-pipeline exploration (other/). The canonical up-drift correction re-folds
each hemi scan about the *smoothed seasonal* "up" curve (up_smooth_deg, ADR
0005). This variant instead re-folds about the *raw daily median* "up"
(up_median_deg) -- the black dots in leaf_up_daily_2025.png -- to see how the
un-smoothed day-to-day "up" propagates into total PAI.

Recipe: tilt=rotation (canonical), up_drift=True, but the up lookup points at a
median CSV whose `up_smooth_deg` column is overwritten with the daily median
(scan-less days dropped; up_on_date snaps to the nearest measured day).

Writes:
- data/processed/proximal_rs/leaf/leaf_up_dailymedian_2025.csv   (median lookup)
- data/processed/proximal_rs/leaf/variants/leaf_hemi_hi_2025_dailymedian_updrift.parquet

Run:
    /home/lk1167/miniconda3/envs/dehar-spac/bin/python \
        other/build_leaf_variant_dailymedian_updrift.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import process_leaf as pl  # noqa: E402

UP_DAILY = REPO / "data/processed/proximal_rs/leaf/leaf_up_daily_2025.csv"
UP_MEDIAN = REPO / "data/processed/proximal_rs/leaf/leaf_up_dailymedian_2025.csv"
SCAN_TYPE = "hemi_hi"


def write_median_lookup(src: Path, dst: Path) -> int:
    """Median 'up' lookup: put up_median_deg into the up_smooth_deg column."""
    df = pd.read_csv(src, parse_dates=["date"])
    df = df.dropna(subset=["up_median_deg"]).copy()
    df["up_smooth_deg"] = df["up_median_deg"]   # column load_up_lookup() reads
    df.to_csv(dst, index=False)
    return len(df)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    n = write_median_lookup(UP_DAILY, UP_MEDIAN)
    logging.info("median up lookup: %d measured days -> %s", n, UP_MEDIAN.name)

    cfg = pl.load_config(pl.DEFAULT_CONFIG)
    cfg["transform"] = {
        "tilt": "rotation",
        "up_drift": True,
        "up_lookup_csv": str(UP_MEDIAN.relative_to(REPO)),
    }
    cfg["paths"] = dict(cfg["paths"])
    cfg["paths"]["out_dir"] = "data/processed/proximal_rs/leaf/variants"
    cfg["paths"]["out_pattern"] = "leaf_{scan_type}_{year}_dailymedian_updrift.parquet"

    pl.run_process(cfg, SCAN_TYPE, None, None)
    pl.run_temporal(cfg, SCAN_TYPE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
