#!/usr/bin/env python3
"""Build the seasonal LEAF "up"-drift daily lookup table (GitHub issue #11).

Separate precompute step for the self-calibration. The per-scan seam fit
(:func:`dehar.proximal_rs.leaf.calibrate_up`) is occasionally unreliable for a
single scan (a weak seam can return a spurious ~180 deg, e.g. 2025-05-19), but
the true "up" drifts smoothly over the season. So we:

1. calibrate "up" for **every good-quality hemi scan** (hemi_hi + hemi_low),
   in parallel across cores;
2. take a robust per-day **median**;
3. robust-LOWESS across days, evaluated on **every** day (gaps filled).

The smoothed daily "up" is what downstream code should fold each scan about
(no single scan can escape correction).

Note: the lower-resolution ``hemi_low`` seam fit is systematically biased high
and unreliable in leaf-off (16x fewer shots -> sparse seam), so the smoothed
**trend** is built from ``hemi_hi`` only by default (TREND_SCAN_TYPES). All
scan types are still calibrated and written to the per-scan table for the record.

Run:
  /home/lk1167/miniconda3/envs/dehar-spac/bin/python scripts/build_leaf_up_lookup.py

Outputs (data/processed/proximal_rs/leaf/):
  leaf_up_per_scan_2025.csv   datetime, scan_type, filename, quality_all, up_deg
  leaf_up_daily_2025.csv      date, n_scans, up_median_deg, up_smooth_deg
  leaf_up_daily_2025.png      QC plot
"""

from __future__ import annotations

import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from dehar.proximal_rs.leaf import daily_up_lookup, up_for_scan  # noqa: E402

# ===========================================================================
# PARAMETERS
# ===========================================================================
SCAN_TYPES = ("hemi_hi", "hemi_low")     # calibrate "up" for these (the record)
TREND_SCAN_TYPES = ("hemi_hi",)          # build the smoothed trend from these only
QUALITY_ONLY = True                      # use quality_all scans only
UP_LO, UP_HI = 176.0, 215.0              # calibrate_up search bounds (deg)
LOWESS_FRAC = 0.20                       # seasonal smoothing span
ROBUST_IT = 2                            # LOWESS robustifying iterations
WORKERS = 56
YEAR = 2025

LEAF_DIR = REPO_ROOT / "data" / "processed" / "proximal_rs" / "leaf"
RAW_DIR = REPO_ROOT / "data" / "raw" / "proximal_rs" / "leaf"
# ===========================================================================

log = logging.getLogger("build_leaf_up_lookup")


def collect_scans() -> pd.DataFrame:
    """One row per good-quality hemi scan across the configured scan types."""
    frames = []
    for st in SCAN_TYPES:
        pq = LEAF_DIR / f"leaf_{st}_{YEAR}.parquet"
        df = pd.read_parquet(pq, columns=["datetime", "filename", "quality_all"])
        df = df.drop_duplicates("datetime").assign(scan_type=st)
        frames.append(df)
    scans = pd.concat(frames, ignore_index=True)
    scans["datetime"] = pd.to_datetime(scans["datetime"], utc=True)
    if QUALITY_ONLY:
        scans = scans[scans["quality_all"]]
    return scans.sort_values("datetime").reset_index(drop=True)


def calibrate_all(scans: pd.DataFrame) -> pd.DataFrame:
    """Parallel per-scan calibrated "up" for every row of ``scans``."""
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futures = {
            ex.submit(up_for_scan, RAW_DIR / r.filename, UP_LO, UP_HI): r
            for r in scans.itertuples(index=False)
        }
        for i, fut in enumerate(as_completed(futures), start=1):
            r = futures[fut]
            rows.append(
                {
                    "datetime": r.datetime,
                    "scan_type": r.scan_type,
                    "filename": r.filename,
                    "quality_all": bool(r.quality_all),
                    "up_deg": fut.result(),
                }
            )
            if i % 250 == 0:
                rate = i / (time.time() - t0)
                left = (len(scans) - i) / rate
                log.info("  %d/%d  (~%.0fs left)", i, len(scans), left)
    out = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
    log.info("calibrated %d scans in %.0fs", len(out), time.time() - t0)
    return out


def qc_plot(per_scan: pd.DataFrame, daily: pd.DataFrame, path: Path) -> None:
    """Per-scan ups (by type) + daily median + smoothed seasonal trend."""
    fig, ax = plt.subplots(figsize=(13, 5.5))
    colors = {"hemi_hi": "#1f77b4", "hemi_low": "#d9a23a"}
    for st, c in colors.items():
        sub = per_scan[per_scan.scan_type == st]
        ax.plot(sub.datetime, sub.up_deg, ".", ms=3, alpha=0.35, color=c,
                label=f"{st} per scan")
    d = daily.assign(date=pd.to_datetime(daily.date, utc=True))
    ax.plot(d.date, d.up_median_deg, "o", ms=4, color="black", alpha=0.6,
            label="hemi_hi daily median")
    ax.plot(d.date, d.up_smooth_deg, "-", color="crimson", lw=2.6,
            label="smoothed seasonal up")
    ax.axhline(180, color="0.6", ls="--", lw=1)
    ax.axvline(pd.Timestamp("2025-05-19", tz="UTC"), color="0.6", ls=":", lw=1)
    ax.set(xlabel="date", ylabel='calibrated "up" (deg)',
           title='LEAF seasonal "up"-drift  (dotted = 2025-05-19)')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    LEAF_DIR.mkdir(parents=True, exist_ok=True)

    scans = collect_scans()
    log.info("calibrating %d good hemi scans (%s) on %d workers",
             len(scans), "+".join(SCAN_TYPES), WORKERS)
    per_scan = calibrate_all(scans)
    per_scan.to_csv(LEAF_DIR / f"leaf_up_per_scan_{YEAR}.csv", index=False)

    trend_src = per_scan[per_scan.scan_type.isin(TREND_SCAN_TYPES)]
    daily = daily_up_lookup(trend_src, frac=LOWESS_FRAC, robust_it=ROBUST_IT)
    daily.to_csv(LEAF_DIR / f"leaf_up_daily_{YEAR}.csv", index=False)
    qc_plot(per_scan, daily, LEAF_DIR / f"leaf_up_daily_{YEAR}.png")

    # quick report
    d = daily.assign(date=pd.to_datetime(daily.date, utc=True)).set_index("date")
    log.info("daily lookup: %d days, smoothed up %.0f..%.0f deg",
             len(daily), daily.up_smooth_deg.min(), daily.up_smooth_deg.max())
    for probe in ("2025-05-19", "2025-04-20", "2025-09-15"):
        row = d.loc[pd.Timestamp(probe, tz="UTC")]
        log.info("  %s: n=%d  median=%.1f  smoothed=%.1f deg",
                 probe, int(row.n_scans),
                 row.up_median_deg if pd.notna(row.up_median_deg) else float("nan"),
                 row.up_smooth_deg)
    log.info("wrote leaf_up_per_scan / leaf_up_daily (.csv) + QC .png to %s", LEAF_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
