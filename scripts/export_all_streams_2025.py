"""
Export ALL DE-Har 2025 data streams into one central file for analysis.

Brings together every in-situ and satellite stream at NATIVE resolution
(no resampling, no filtering, no means), outer-joined on a UTC datetime
index, and writes a single parquet:

    data/processed/dehar_all_streams_2025.parquet

Design rules
------------
* **No filters** — every scene / observation is kept (no cloud mask, no rain /
  humidity / clear-sky drop, no smoothing).
* **All raw columns** — every soil-moisture depth, every per-tree sensor, every
  GNSS receiver / camera is kept verbatim. No collapsed mean/std columns for the
  in-situ streams. (The only exception is leaf angle, whose native ~13 s cadence is
  subsampled to 30-min so the master table is not dominated by it; each anglecam frame
  is itself a leaf-angle *distribution*, so per camera we keep the 30-min mean, median
  and std — see ``load_leaf_angle``.)
* **Sentinel-1** — 100 m / w5 VV/VH/SPAN/CR (dB) + RVI mean+std from the transparent
  GRD_FLOAT product (desc 139 + asc 15; ADR 0011).
* **LEAF** — canopy-total PAI per scan for hemi_hi and hinge (NOT hemi_low). The
  full height profiles remain in the leaf parquet files.

Sentinel-2 is **not** exported here anymore: the canonical S2 ROI product is the
hand-audited per-scene table built by ``scripts/process_sentinel2_roi.py`` and
consumed directly by ``aggregate_daily_streams_2025.build_sentinel2`` (ADR 0012).

Run from the repo root (needs the `dehar-spac` env with xarray):

    python scripts/export_all_streams_2025.py
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("export_all_streams")

# ── Config ──────────────────────────────────────────────────────────────────
TARGET_YEAR = 2025
# Canopy-layer split for the hemi profile (PAVD gap at 8-10 m): understory 1.5-9 m
# (deciduous, the varying/well-sampled layer), overstory 9-18 m (evergreen,
# occlusion-limited). App C uses the understory band as its structural regressor (ADR 0008).
LEAF_LAYER_EDGES = (1.5, 9.0, 18.0)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT_PATH = DATA / "processed" / f"dehar_all_streams_{TARGET_YEAR}.parquet"

# Sentinel-1 canonical = the GRD_FLOAT product; headline pick 100 m / w5 (ADR 0011).
S1_CANON = "r100_w5"

# In-situ stream sources (all under data/processed/).
PROC = DATA / "processed"
PATHS = {
    "meteo":      PROC / "atmosphere_soil/meteo_dehar_30min.csv",
    "fluxes":     PROC / "atmosphere_soil/fluxes_dehar_30min.csv",
    "sm":         PROC / "atmosphere_soil/soil_moisture_dehar_30min.csv",
    "precip":     PROC / "atmosphere_soil/HARTHM_2025_Precipitation_30min_UTC.csv",
    "sapflow":    PROC / "physiology/sap_flux_density/sapflow_dehar_30min.csv",
    "swp":        PROC / "physiology/stemwater_potential/swp_dehar_15min.csv",
    "twd":        PROC / "physiology/twd/twd_dehar_30min.csv",
    "vod":        PROC / "proximal_rs/gnss_vod/gnss_vod_dehar_30min.csv",
    "anglecam_pred": DATA / "raw/proximal_rs/anglecam/leaf-angle-predictions-2025",
    "leaf_hemi_hi": PROC / "proximal_rs/leaf/leaf_hemi_hi_2025.parquet",
    "leaf_hinge":   PROC / "proximal_rs/leaf/leaf_hinge_2025.parquet",
    # uncorrected (rotation, no up-drift) hemi_hi -> _uncorr comparison columns (ADR 0005)
    "leaf_hemi_hi_uncorr": PROC
    / "proximal_rs/leaf_pre_updrift_fix_20260626/leaf_hemi_hi_2025.parquet",
    "s1_product": PROC / "satellite/sentinel1/s1_grdfloat_by_scene.parquet",
    "pheno_scratch": PROC / "proximal_rs/phenology/scratch",
}


# ── Generic helpers ───────────────────────────────────────────────────────────
def _to_year_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, make the index UTC-aware, drop duplicate timestamps, clip to year."""
    if df.index.tz is None:
        df.index = pd.DatetimeIndex(df.index).tz_localize("UTC")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.index.name = "datetime"
    return df[df.index.year == TARGET_YEAR].copy()


def load_insitu(path: Path, datetime_col: str = "datetime", prefix: str = "") -> pd.DataFrame:
    """Load a processed in-situ CSV keeping every raw column (optionally prefixed)."""
    df = pd.read_csv(path, parse_dates=[datetime_col])
    df = df.rename(columns={datetime_col: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime")
    df = _to_year_utc(df)
    return df.add_prefix(prefix) if prefix else df


def load_precip(path: Path) -> pd.DataFrame:
    """Precipitation depth per 30-min step."""
    raw = pd.read_csv(path)
    out = pd.DataFrame(
        {"precip_mm": raw["Precipitation_Sum_mm"].to_numpy()},
        index=pd.to_datetime(raw["From"], utc=True),
    )
    return _to_year_utc(out)


def leaf_totals(path: Path, prefix: str) -> pd.DataFrame:
    """Per-scan canopy totals (max over height of the cumulative PAI) + quality flag.

    These are canopy-integrated totals, not means. Full height profiles stay in
    the leaf parquet files. For hemi scans, also the WeightedPAI of the two canopy
    layers (understory / overstory, split at ``LEAF_LAYER_EDGES``) as the
    cumulative-PAI difference across each band's edges — App C uses the understory
    band as its structural regressor (ADR 0008).
    """
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df[df["datetime"].dt.year == TARGET_YEAR]
    g = df.groupby("datetime")
    out = pd.DataFrame(
        {
            f"{prefix}HingePAI_total": g["HingePAI"].max(),
            f"{prefix}LinearPAI_total": g["LinearPAI"].max(),
            f"{prefix}WeightedPAI_total": g["WeightedPAI"].max(),
            f"{prefix}quality_all": g["quality_all"].first(),
        }
    )
    if "hemi" in prefix:
        lo, mid, hi = LEAF_LAYER_EDGES
        piv = df.pivot_table(index="datetime", columns="height",
                             values="WeightedPAI", aggfunc="first")
        H = np.array(sorted(piv.columns))
        near = lambda h: H[np.abs(H - h).argmin()]              # noqa: E731
        out[f"{prefix}WeightedPAI_understory"] = piv[near(mid)] - piv[near(lo)]
        out[f"{prefix}WeightedPAI_overstory"] = piv[near(hi)] - piv[near(mid)]
    return _to_year_utc(out)


LA_BIN_CENTERS = np.linspace(0.0, 90.0, 43)      # anglecam PMF support (verified vs stored mean)
LA_CAMS = list(range(60, 71))                    # G5Bullet_60 … _70


def pmf_stats(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-row mean, median, std of leaf-angle PMFs (rows sum to 1, 43 bins over 0–90°)."""
    c = LA_BIN_CENTERS
    mean = P @ c
    std = np.sqrt(np.clip(P @ (c ** 2) - mean ** 2, 0.0, None))
    cdf = np.cumsum(P, axis=1)
    # First bin whose CDF reaches 0.5, then linearly interpolate within that bin.
    idx = np.argmax(cdf >= 0.5, axis=1)
    lo = np.maximum(idx - 1, 0)
    c_lo, c_hi = c[lo], c[idx]
    cdf_lo = np.take_along_axis(cdf, lo[:, None], axis=1).ravel()
    cdf_hi = np.take_along_axis(cdf, idx[:, None], axis=1).ravel()
    span = cdf_hi - cdf_lo
    frac = np.where(span > 0, (0.5 - cdf_lo) / span, 0.0)
    median = np.where(idx == 0, c_hi, c_lo + frac * (c_hi - c_lo))
    return mean, median, std


def _parse_cam_predictions(cam_dir: Path, cam: int) -> pd.DataFrame:
    """Per-frame mean/median/std of the leaf-angle distribution for one camera.

    Mirrors notebooks/02_analysis/compound_event.ipynb: timestamps are UTC, frames with a
    "missing" timestamp are skipped, out-of-bounds timestamps dropped, duplicate timestamps
    averaged.
    """
    ts, pmfs = [], []
    for jf in sorted(cam_dir.glob(f"G5Bullet_{cam}_*_leaf-angle-predictions.json")):
        try:
            frames = json.loads(jf.read_text())["frames"]
        except (json.JSONDecodeError, KeyError):
            log.warning("leaf_angle: skipped unreadable %s", jf.name)
            continue
        for fr in frames:
            dist = fr.get("angle_distribution")
            if fr.get("timestamp") == "missing" or not dist or len(dist) != LA_BIN_CENTERS.size:
                continue
            ts.append(fr["timestamp"])
            pmfs.append(dist)
    if not pmfs:
        return pd.DataFrame()

    mean, median, std = pmf_stats(np.asarray(pmfs, dtype="float64"))
    df = pd.DataFrame(
        {
            f"leaf_angle_cam{cam}_mean": mean,
            f"leaf_angle_cam{cam}_median": median,
            f"leaf_angle_cam{cam}_std": std,
        },
        index=pd.to_datetime(pd.Series(ts), utc=True, errors="coerce"),
    )
    df = df[df.index.notna()].sort_index()
    df = df.groupby(level=0).mean()              # average frames sharing a timestamp
    return df


def load_leaf_angle(pred_dir: Path) -> pd.DataFrame:
    """Leaf-angle distribution stats per camera, 30-min mean of the per-frame mean/median/std.

    Each anglecam frame is a 43-bin leaf-angle PMF; we summarise every frame, then average to
    30-min so the stream does not dominate the master table's native ~13 s cadence.
    """
    frames = []
    for cam in LA_CAMS:
        cam_df = _parse_cam_predictions(pred_dir / f"G5Bullet_{cam}", cam)
        if cam_df.empty:
            log.warning("leaf_angle: no usable frames for cam%d", cam)
            continue
        frames.append(cam_df.resample("30min").mean())
    if not frames:
        return pd.DataFrame()
    return _to_year_utc(pd.concat(frames, axis=1))


def load_phenology(scratch_dir: Path) -> pd.DataFrame:
    """GCC per camera (cam60–70), 2-min scratch parquet → 30-min mean, one col per cam."""
    frames = []
    for n in range(60, 71):
        cam = f"G5Bullet_{n}"
        files = sorted(scratch_dir.glob(f"{cam}_*.parquet"))
        if not files:
            log.warning("phenology: no scratch files for %s", cam)
            continue
        daily = [d for f in files if not (d := pd.read_parquet(f)).empty]
        if not daily:
            continue
        cam_df = pd.concat(daily, ignore_index=True)
        # Camera timestamps are local (Europe/Berlin) → convert to UTC.
        cam_df["datetime"] = (
            pd.to_datetime(cam_df["timestamp"])
            .dt.tz_localize("Europe/Berlin", ambiguous="infer", nonexistent="shift_forward")
            .dt.tz_convert("UTC")
        )
        cam_df = cam_df.set_index("datetime").sort_index()
        cam_df = cam_df[cam_df.index.year == TARGET_YEAR]
        frames.append(cam_df["gcc"].resample("30min").mean().rename(f"gcc_cam{n}"))
    if not frames:
        return pd.DataFrame()
    return _to_year_utc(pd.concat(frames, axis=1))


# ── Satellite helpers ─────────────────────────────────────────────────────────
def load_s1(product_path: Path, orbit: str) -> pd.DataFrame:
    """Canonical S1 daily inputs from the transparent GRD_FLOAT product (ADR 0011).

    Reads ``s1_grdfloat_by_scene.parquet`` and takes the headline 100 m / w5 pick for
    one pass direction. All processing (linear-domain moving mean, ROI averaging) is
    upstream in ``scripts/process_sentinel1_roi.py``; dB here is a display transform.
    Emits ``s1_{asc,desc}_{VV,VH,SPAN,CR}_dB_{mean,std}`` + ``RVI_{mean,std}``, indexed
    by acquisition time (so downstream resampling to daily is unchanged).
    """
    direction = "ascending" if orbit == "a" else "descending"
    p = pd.read_parquet(product_path)
    p = p[p["direction"] == direction].copy()
    p["datetime"] = pd.to_datetime(p["datetime"], utc=True)
    p = p.set_index("datetime").sort_index()
    p = p[p.index.year == TARGET_YEAR]                  # daily table is one year
    out = pd.DataFrame(index=p.index)
    for v in ("VV", "VH", "SPAN", "CR"):
        out[f"{v}_dB_mean"] = p[f"{v}_{S1_CANON}"]
        out[f"{v}_dB_std"] = p[f"{v}_{S1_CANON}_std"]
    out["RVI_mean"] = p[f"RVI_{S1_CANON}"]
    out["RVI_std"] = p[f"RVI_{S1_CANON}_std"]
    prefix = "s1_asc_" if orbit == "a" else "s1_desc_"
    return out.add_prefix(prefix)


# ── Orchestration ─────────────────────────────────────────────────────────────
def _safe(label: str, fn, *args, **kwargs) -> pd.DataFrame:
    """Run a loader; on failure log a warning and return an empty frame."""
    try:
        df = fn(*args, **kwargs)
        log.info("%-14s %s → %s  (%d rows × %d cols)",
                 label, df.index.min() if len(df) else "—",
                 df.index.max() if len(df) else "—", len(df), df.shape[1])
        return df
    except Exception as exc:                              # noqa: BLE001
        log.warning("%-14s FAILED: %s", label, exc)
        return pd.DataFrame()


def main() -> None:
    log.info("DE-Har %d all-streams export", TARGET_YEAR)

    streams = {
        "meteo":      _safe("meteo",      load_insitu, PATHS["meteo"], prefix="meteo_"),
        "fluxes":     _safe("fluxes",     load_insitu, PATHS["fluxes"], prefix="flux_"),
        "soil_moist": _safe("soil_moist", load_insitu, PATHS["sm"], prefix="sm_"),
        "precip":     _safe("precip",     load_precip, PATHS["precip"]),
        "sapflow":    _safe("sapflow",    load_insitu, PATHS["sapflow"]),
        "swp":        _safe("swp",        load_insitu, PATHS["swp"]),
        "twd":        _safe("twd",        load_insitu, PATHS["twd"]),
        "vod":        _safe("vod",        load_insitu, PATHS["vod"]),
        "leaf_angle": _safe("leaf_angle", load_leaf_angle, PATHS["anglecam_pred"]),
        "phenology":  _safe("phenology",  load_phenology, PATHS["pheno_scratch"]),
        "leaf_hemi":  _safe("leaf_hemi_hi", leaf_totals, PATHS["leaf_hemi_hi"], "leaf_hemi_hi_"),
        "leaf_hinge": _safe("leaf_hinge", leaf_totals, PATHS["leaf_hinge"], "leaf_hinge_"),
        "leaf_hemi_uncorr": _safe("leaf_hemi_hi_uncorr", leaf_totals,
                                  PATHS["leaf_hemi_hi_uncorr"], "leaf_hemi_hi_uncorr_"),
        # CANONICAL S1 = transparent GRD_FLOAT product, 100 m / w5 (ADR 0011).
        "s1_asc":     _safe("s1_ascending",  load_s1, PATHS["s1_product"], "a"),
        "s1_desc":    _safe("s1_descending", load_s1, PATHS["s1_product"], "d"),
    }

    frames = [df for df in streams.values() if not df.empty]
    combined = pd.concat(frames, axis=1, join="outer").sort_index()
    combined.index.name = "datetime"

    # Collision guard — every column must be unique.
    dupes = combined.columns[combined.columns.duplicated()].tolist()
    if dupes:
        raise ValueError(f"Duplicate columns after join: {dupes}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_PATH, index=True)

    filled = 100 * combined.notna().to_numpy().sum() / combined.size
    log.info("─" * 60)
    for name, df in streams.items():
        log.info("  %-14s %3d cols", name, df.shape[1])
    log.info("─" * 60)
    log.info("Combined : %d rows × %d cols  (%.2f%% filled)",
             combined.shape[0], combined.shape[1], filled)
    log.info("Range    : %s → %s", combined.index.min(), combined.index.max())
    log.info("Saved    : %s (%.1f MB)", OUT_PATH, OUT_PATH.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
