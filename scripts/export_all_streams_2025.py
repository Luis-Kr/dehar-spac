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
* **Sentinel-2** — 100 m ROI mean+std of an expanded `spyndex` index set plus the
  raw bands, computed from the raw raster cube over *all* scenes.
* **Sentinel-1** — 100 m ROI mean+std of angle-normalised VV/VH/SPAN/CR/RVI
  (linear + dB) for both orbits.
* **LEAF** — canopy-total PAI per scan for hemi_hi and hinge (NOT hemi_low). The
  full height profiles remain in the leaf parquet files.

Run from the repo root (needs the `dehar-spac` env with spyndex / xarray):

    python scripts/export_all_streams_2025.py
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import spyndex

from dehar.utils.constants import SITE_CRS, SITE_UTM_X, SITE_UTM_Y

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("export_all_streams")

# ── Config ──────────────────────────────────────────────────────────────────
TARGET_YEAR = 2025
# Canopy-layer split for the hemi profile (PAVD gap at 8-10 m): understory 1.5-9 m
# (deciduous, the varying/well-sampled layer), overstory 9-18 m (evergreen,
# occlusion-limited). App C uses the understory band as its structural regressor (ADR 0008).
LEAF_LAYER_EDGES = (1.5, 9.0, 18.0)
TOWER_X, TOWER_Y = SITE_UTM_X, SITE_UTM_Y          # EPSG:32632 (UTM 32N)
ROI_RADIUS_M = 100.0

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT_PATH = DATA / "processed" / f"dehar_all_streams_{TARGET_YEAR}.parquet"

# Sentinel-2 raw bands kept alongside the indices.
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

# Curated broad common index set (computed from the bands above via spyndex).
# Any index whose required spectral bands are unavailable is skipped at runtime.
S2_INDICES = [
    # greenness / canopy structure
    "NDVI", "kNDVI", "EVI", "EVI2", "SAVI", "OSAVI", "MSAVI", "NIRv", "GNDVI",
    "RDVI", "DVI",
    # red-edge / chlorophyll / pigment
    "NDREI", "CIRE", "CIG", "MTCI", "IRECI", "MCARI", "TCARI", "MCARI1",
    "ARI", "ARI2", "CRI550",
    # water / moisture
    "NDII", "NDMI", "NDWI", "NMDI", "MSI", "GVMI",
    # senescence / burn
    "PSRI", "NBR", "NBR2",
    # atmospheric-resistant / broadband RGB
    "ARVI", "VARI", "GLI", "TGI", "ExG",
]

# Sentinel-1
S1_LINEAR_VARS = ["VV_lin", "VH_lin", "SPAN_lin", "CR_lin", "RVI"]
S1_THETA_REF = 40.0      # reference incidence angle [deg] for cosine normalisation
S1_N = 2                 # cosine exponent

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
    "s2_nc":      PROC / f"satellite/sentinel2/s2_sr_dehar_{TARGET_YEAR}_indices.nc",
    "s1_nc_a":    PROC / f"satellite/sentinel1/s1_grd_dehar_a_{TARGET_YEAR}_indices.nc",
    "s1_nc_d":    PROC / f"satellite/sentinel1/s1_grd_dehar_d_{TARGET_YEAR}_indices.nc",
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
def _norm_dims(ds: xr.Dataset) -> xr.Dataset:
    rn = {}
    if "X" in ds.dims and "x" not in ds.dims:
        rn["X"] = "x"
    if "Y" in ds.dims and "y" not in ds.dims:
        rn["Y"] = "y"
    return ds.rename(rn) if rn else ds


def _roi_reduce(ds: xr.Dataset, keep: list[str]) -> pd.DataFrame:
    """ROI mean+std over (x, y) within ROI_RADIUS_M of the tower, per scene."""
    mask = ((ds.x - TOWER_X) ** 2 + (ds.y - TOWER_Y) ** 2) <= ROI_RADIUS_M ** 2
    keep = [v for v in keep if v in ds and {"time", "x", "y"}.issubset(set(ds[v].dims))]
    roi = ds[keep].where(mask)
    mean = roi.mean(dim=("x", "y"), skipna=True).to_dataframe()[keep].add_suffix("_mean")
    std = roi.std(dim=("x", "y"), skipna=True).to_dataframe()[keep].add_suffix("_std")
    return _to_year_utc(mean.join(std))


def load_s2(path: Path) -> pd.DataFrame:
    """Sentinel-2 ROI mean+std of the expanded index set + raw bands, all scenes."""
    ds = _norm_dims(xr.open_dataset(path))
    if float(ds["B2"].max()) > 2.0:                      # reflectance stored ×10000
        for b in S2_BANDS:
            if b in ds:
                ds[b] = ds[b] / 10000.0

    band_params = {
        "N": ds["B8"], "N2": ds["B8A"], "R": ds["B4"], "G": ds["B3"], "B": ds["B2"],
        "RE1": ds["B5"], "RE2": ds["B6"], "RE3": ds["B7"], "S1": ds["B11"], "S2": ds["B12"],
        "kNN": 1.0,
    }
    band_params = {k: (v.astype("float32") if hasattr(v, "astype") else v)
                   for k, v in band_params.items()}
    band_params["kNR"] = spyndex.computeKernel(
        kernel="RBF",
        params={"a": band_params["N"], "b": band_params["R"],
                "sigma": float(np.abs(ds["B8"] - ds["B4"]).median().item())},
    )

    added, skipped = [], []
    for name in S2_INDICES:
        if name not in spyndex.indices:
            skipped.append(name)
            continue
        params = dict(band_params)
        ok = True
        for p in spyndex.indices[name].bands:
            if p in params:
                continue
            if p in spyndex.constants:                   # L, g, C1, C2, gamma, ...
                params[p] = spyndex.constants[p].default
            else:                                        # a needed spectral band is missing
                ok = False
                break
        if not ok:
            skipped.append(name)
            continue
        ds[name] = spyndex.computeIndex(index=name, params=params, online=False)
        added.append(name)

    log.info("S2 indices computed (%d): %s", len(added), ", ".join(added))
    if skipped:
        log.info("S2 indices skipped (bands unavailable): %s", ", ".join(skipped))

    out = _roi_reduce(ds, added + S2_BANDS)
    return out.add_prefix("s2_")


def _db_to_lin(da: xr.DataArray) -> xr.DataArray:
    return 10 ** (da / 10.0)


def _angle_normalise(vv_db, vh_db, angle_deg, theta_ref=S1_THETA_REF, n=S1_N):
    """Cosine incidence-angle normalisation to a reference angle."""
    corr_db = 10 * n * np.log10(np.cos(np.deg2rad(theta_ref)) / np.cos(np.deg2rad(angle_deg)))
    return vv_db + corr_db, vh_db + corr_db


def load_s1(path: Path, orbit: str) -> pd.DataFrame:
    """Sentinel-1 ROI mean+std of angle-normalised VV/VH/SPAN/CR/RVI (linear + dB)."""
    ds = _norm_dims(xr.open_dataset(path))
    vv_db, vh_db = _angle_normalise(ds["VV"], ds["VH"], ds["angle"])
    vv_lin, vh_lin = _db_to_lin(vv_db), _db_to_lin(vh_db)
    ds["VV_lin"] = vv_lin
    ds["VH_lin"] = vh_lin
    ds["SPAN_lin"] = vv_lin + vh_lin
    ds["CR_lin"] = vh_lin / vv_lin.where(vv_lin > 0)
    ds["RVI"] = 4 * vh_lin / (vv_lin + vh_lin)

    df = _roi_reduce(ds, S1_LINEAR_VARS + ["angle"])

    # Linear → dB with first-order error propagation on the std.
    k = 10 / np.log(10)
    for v in ["VV", "VH", "SPAN", "CR"]:
        df[f"{v}_dB_mean"] = 10 * np.log10(df[f"{v}_lin_mean"])
        df[f"{v}_dB_std"] = k * df[f"{v}_lin_std"] / df[f"{v}_lin_mean"]

    prefix = "s1_asc_" if orbit == "a" else "s1_desc_"
    return df.add_prefix(prefix)


# Sentle weekly composites — CANONICAL S1 source (see docs/adr/0001-sentinel1-sentle-weekly.md).
# All S1 passes composited per ISO week → far less speckle, regular cadence, near-complete coverage.
SENTLE_ZARR = Path(
    "/mnt/data/lk1167/projects/other/data/icos-har/processed/sentinel/sentinel_har_2025.zarr"
)


def load_s1_sentle(zarr_path: Path, orbit: str) -> pd.DataFrame:
    """Sentinel-1 ROI mean+std of VV/VH/SPAN/CR/RVI from the sentle weekly-composite zarr.

    Identical derivation to ``load_s1`` EXCEPT angle-normalisation is dropped (the zarr has no
    incidence-angle band); sentle σ⁰ is in LINEAR units. One value per ISO week (Thursday-anchored).
    """
    suff = "asc" if orbit == "a" else "desc"
    da = xr.open_zarr(zarr_path).sortby("time")["sentle"]
    vv = da.sel(band=f"vv_{suff}").drop_vars("band")   # linear σ⁰
    vh = da.sel(band=f"vh_{suff}").drop_vars("band")
    ds = xr.Dataset(
        {
            "VV_lin": vv,
            "VH_lin": vh,
            "SPAN_lin": vv + vh,
            "CR_lin": vh / vv.where(vv > 0),
            "RVI": 4 * vh / (vv + vh),
        }
    )
    df = _roi_reduce(ds, S1_LINEAR_VARS)               # 100 m ROI mean+std, per week

    k = 10 / np.log(10)
    for v in ["VV", "VH", "SPAN", "CR"]:
        df[f"{v}_dB_mean"] = 10 * np.log10(df[f"{v}_lin_mean"])
        df[f"{v}_dB_std"] = k * df[f"{v}_lin_std"] / df[f"{v}_lin_mean"]

    prefix = "s1_asc_" if orbit == "a" else "s1_desc_"
    return df.add_prefix(prefix)


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
    log.info("ROI centre (%.1f, %.1f) %s, radius %.0f m", TOWER_X, TOWER_Y, SITE_CRS, ROI_RADIUS_M)

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
        "s2":         _safe("sentinel2",  load_s2, PATHS["s2_nc"]),
        # CANONICAL S1 = sentle weekly composites (ADR 0001). GEE per-pass kept for provenance:
        #   "s1_asc":  _safe("s1_ascending",  load_s1, PATHS["s1_nc_a"], "a"),
        #   "s1_desc": _safe("s1_descending", load_s1, PATHS["s1_nc_d"], "d"),
        "s1_asc":     _safe("s1_asc_sentle",  load_s1_sentle, SENTLE_ZARR, "a"),
        "s1_desc":    _safe("s1_desc_sentle", load_s1_sentle, SENTLE_ZARR, "d"),
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
