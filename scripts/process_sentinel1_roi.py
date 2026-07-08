"""Build the canonical Sentinel-1 GRD_FLOAT product (ADR 0011).

Transparent, in-repo S1: one relative orbit per direction (descending 139 = morning,
ascending 15 = evening), all processing in **linear power** (dB only for display),
from ``data/raw/satellite/sentinel1_grdfloat/``. One row per scene; the full matrix of
ROI mean over **50/100/200 m** x temporal moving-mean **raw + w3..w13** for bands
**VV, VH, CR, SPAN, RVI**. The headline daily columns pick 100 m / w5 downstream
(``export_all_streams``); the rest is the sensitivity space (paper/90).

Science lives in ``dehar.satellite.sentinel1_roi``; this only orchestrates.

Run
---
    /home/lk1167/miniconda3/envs/dehar-spac/bin/python scripts/process_sentinel1_roi.py

Output
------
    data/processed/satellite/sentinel1/s1_grdfloat_by_scene.parquet
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from dehar.satellite import sentinel1_roi as s1

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

OUT = Path("data/processed/satellite/sentinel1/s1_grdfloat_by_scene.parquet")
ORBITS = {"a": 15, "d": 139}            # asc 15 (evening), desc 139 (morning)
RINGS = (50, 100, 200)
WINDOWS = {"raw": 1, "w3": 3, "w5": 5, "w7": 7,
           "w9": 9, "w11": 11, "w13": 13}
BANDS = ("VV", "VH", "CR", "SPAN", "RVI")
CANON_RING, CANON_WIN = 100, "w5"       # headline pick (also carries a std column)
K_DB = 10.0 / np.log(10.0)              # linear-std -> dB-std (first-order)


def _direction_frame(tag: str, orbit: int) -> pd.DataFrame:
    sc = s1.scene_index(tag)
    sco = sc[sc.rel_orbit == orbit].reset_index(drop=True)
    if sco.empty:
        log.warning("no scenes for %s orbit %d", tag, orbit)
        return pd.DataFrame()
    win = s1.window_geometry(tag)
    arr = s1.load_window(tag, sco, win)
    masks = s1.roi_masks(win, RINGS)
    cols: dict = {
        "datetime": pd.to_datetime(sco.datetime.values),
        "direction": s1.DIRECTIONS[tag], "orbit": orbit,
        "platform": sco.platform.values,
    }
    for wname, wsize in WINDOWS.items():
        vv = arr["VV"] if wsize == 1 else s1.temporal_mean(arr["VV"], wsize)
        vh = arr["VH"] if wsize == 1 else s1.temporal_mean(arr["VH"], wsize)
        for band in BANDS:
            blin = s1.compute_band(vv, vh, band)
            rm = s1.ring_means(blin, masks)                 # linear mean per ring
            for r in RINGS:
                cols[f"{band}_r{r}_{wname}"] = s1.to_display(rm[r], band)
            if wname == CANON_WIN:                          # headline std at 100 m
                mlin, slin = s1.ring_stat(blin, masks[CANON_RING])
                cols[f"{band}_r{CANON_RING}_{wname}_std"] = (
                    slin if band == "RVI" else K_DB * slin / mlin
                )
    for r in RINGS:
        cols[f"n_r{r}"] = int(masks[r].sum())
    log.info("  %s orbit %d: %d scenes", s1.DIRECTIONS[tag], orbit, len(sco))
    return pd.DataFrame(cols)


def _collapse_slice_overlap(prod: pd.DataFrame) -> pd.DataFrame:
    """Average GRD slice-overlap scenes into one row per acquisition.

    Consecutive Sentinel-1 GRD granules overlap along-track, so a fixed ROI in
    the overlap zone is captured twice at the same acquisition time (both fully
    covering the ROI) with slightly different edge-processed sigma0. Collapse
    each ``(datetime, direction, orbit)`` to a single scene by averaging the ROI
    statistics — the standard slice-overlap mitigation — so the product carries
    one row per acquisition.
    """
    key = ["datetime", "direction", "orbit"]
    if not prod.duplicated(subset=key).any():
        return prod
    num = [c for c in prod.columns
           if c not in key and pd.api.types.is_numeric_dtype(prod[c])]
    agg = {c: "mean" for c in num}
    agg.update({c: "first" for c in prod.columns if c not in key and c not in num})
    n_before = len(prod)
    prod = prod.groupby(key, as_index=False).agg(agg)
    log.info("collapsed %d slice-overlap scene(s) by averaging",
             n_before - len(prod))
    return prod


def main() -> None:
    frames = [_direction_frame(tag, orb) for tag, orb in ORBITS.items()]
    prod = pd.concat([f for f in frames if not f.empty]).sort_values("datetime")
    prod = _collapse_slice_overlap(prod).sort_values("datetime")
    prod = prod.reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prod.to_parquet(OUT)
    log.info("wrote %s (%d scenes, %d cols)", OUT, *prod.shape)
    log.info("bands %s | rings %s | windows %s | canonical %d m/%s",
             list(BANDS), list(RINGS), list(WINDOWS), CANON_RING, CANON_WIN)


if __name__ == "__main__":
    main()
