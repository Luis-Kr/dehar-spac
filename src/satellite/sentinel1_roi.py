"""Reusable Sentinel-1 GRD_FLOAT processing for the S1 playground (ADR 0010).

The **clean start**: everything operates on the raw linear-power σ⁰ stacks in
``data/raw/satellite/sentinel1_grdfloat/`` (VV, VH, angle), and **all averaging /
speckle filtering happens in linear power** — dB (10·log₁₀) is only a display
transform applied last (``to_display``). This is the fix for the old GEE export's
dB-domain focal median (see `CONTEXT.md` → *S1 backscatter*, ADR 0010).

A *processing config* chains, in this order and all in linear:
``[angle-norm] → [spatial speckle] → [temporal filter] → band → ROI mean → dB``.
Orbits stay separate; ascending and descending are never blended.

The tower window / ring geometry is shared with the S2 tool (identical grid).
Side-effect-free except the file *loaders* (``scene_index`` / ``load_window``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from dehar.satellite import (
    sentinel2_roi as geo,  # tower_window, ring_masks, RING_RADII_M
)
from dehar.utils.constants import SITE_UTM_X, SITE_UTM_Y
from scipy import ndimage

RAW_DIR = Path("data/raw/satellite/sentinel1_grdfloat")
POL: tuple[str, ...] = ("VV", "VH")
DIRECTIONS = {"a": "ascending", "d": "descending"}
RING_RADII_M = geo.RING_RADII_M
WINDOW_HALF_PX = geo.WINDOW_HALF_PX
BANDS_DB = ("VV", "VH", "CR", "SPAN")            # displayed in dB; RVI stays linear
ENL = 4.9                                        # S1 IW GRD equivalent number of looks


# ── dB <-> linear ─────────────────────────────────────────────────────────────
def to_db(x: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.where(np.asarray(x) > 0, x, np.nan))


def to_lin(x: np.ndarray) -> np.ndarray:
    return 10.0 ** (np.asarray(x) / 10.0)


def to_display(arr: np.ndarray, band: str) -> np.ndarray:
    """dB for VV/VH/CR/SPAN, linear for RVI."""
    return to_db(arr) if band in BANDS_DB else np.asarray(arr, float)


# ── scene index (+ per-scene relative orbit) ──────────────────────────────────
def scene_index(direction: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """One row per scene for a pass direction ('a'/'d'), sorted, with rel_orbit."""
    rows: list[dict] = []
    for f in sorted(raw_dir.glob(f"s1gf_dehar_{direction}_*.nc")):
        with xr.open_dataset(f) as ds:
            times = pd.to_datetime(ds["time"].values)
        for i, t in enumerate(times):
            rows.append({"file": f.name, "i": int(i), "datetime": t})
    if not rows:
        return pd.DataFrame(columns=["file", "i", "datetime", "scene_id", "direction",
                                     "rel_orbit", "platform"])
    df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
    df["scene_id"] = df["datetime"].dt.strftime("%Y%m%dT%H%M%S")
    df["direction"] = direction
    meta = raw_dir / f"s1gf_meta_{direction}.csv"
    if meta.exists():
        m = pd.read_csv(meta)
        m["datetime"] = pd.to_datetime(m["datetime_utc"], format="ISO8601")
        m = m.sort_values("datetime")
        df = pd.merge_asof(df, m[["datetime", "rel_orbit", "platform"]],
                           on="datetime", direction="nearest",
                           tolerance=pd.Timedelta("2h"))
    else:
        df["rel_orbit"] = np.nan
        df["platform"] = ""
    return df


def window_geometry(direction: str, raw_dir: Path = RAW_DIR) -> dict:
    """Tower-window index bounds + coords + ring masks (shared S2 grid)."""
    f = sorted(raw_dir.glob(f"s1gf_dehar_{direction}_*.nc"))[0]
    with xr.open_dataset(f) as ds:
        win = geo.tower_window(ds)
    win["masks"] = geo.ring_masks(win)
    return win


def load_window(direction: str, scenes: pd.DataFrame, win: dict,
                raw_dir: Path = RAW_DIR) -> dict[str, np.ndarray]:
    """Linear VV/VH/angle window stacks ``[T, ny, nx]`` for the given scenes (N-up)."""
    out = {b: [] for b in ("VV", "VH", "angle")}
    xs = slice(win["x0"], win["x1"])
    ys = slice(win["y0"], win["y1"])
    for fname, grp in scenes.groupby("file"):
        with xr.open_dataset(raw_dir / fname) as ds:
            for r in grp.itertuples():
                for b in out:
                    a = ds[b].isel(time=r.i).values[xs, ys]     # (nx, ny)
                    out[b].append(a.T[::-1].astype(float))       # (ny, nx) N-up
    return {b: np.stack(v) for b, v in out.items()}


# ── angle normalisation (linear, multiplicative) ──────────────────────────────
def angle_norm_factor(angle_deg: np.ndarray, ref_deg: float,
                      exp: float = 2.0) -> np.ndarray:
    """Cosine-model correction ``(cos θ_ref / cos θ_local)^exp`` in linear power."""
    return (np.cos(np.deg2rad(ref_deg)) / np.cos(np.deg2rad(angle_deg))) ** exp


# ── spatial speckle filters (2-D, linear, NaN-aware) ──────────────────────────
def _conv(a: np.ndarray, w: np.ndarray) -> np.ndarray:
    return ndimage.convolve(a, w, mode="nearest")


def _local_mean_var(a: np.ndarray, kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fin = np.isfinite(a).astype(float)
    a0 = np.where(np.isfinite(a), a, 0.0)
    wsum = _conv(fin, kernel)
    ok = wsum > 0
    m = np.where(ok, _conv(a0, kernel) / np.where(ok, wsum, 1), np.nan)
    m2 = np.where(ok, _conv(a0 * a0, kernel) / np.where(ok, wsum, 1), np.nan)
    return m, np.maximum(m2 - m * m, 0.0)


def boxcar(a: np.ndarray, size: int = 5) -> np.ndarray:
    """NaN-aware uniform mean over a size×size window (linear power)."""
    return _local_mean_var(a, np.ones((size, size)))[0]


def _oriented_kernels(size: int) -> list[np.ndarray]:
    """Eight half-plane windows — edge-aligned sub-regions for Refined Lee."""
    r = size // 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    dirs = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    return [((yy * dy + xx * dx) >= 0).astype(float) for dy, dx in dirs]


def refined_lee(a: np.ndarray, size: int = 7, enl: float = ENL) -> np.ndarray:
    """Edge-directional Lee MMSE speckle filter (linear power).

    For each pixel, pick the oriented half-window with the smallest coefficient of
    variation (the homogeneous side of any edge), then apply the Lee minimum-mean-
    square-error weighting with that window's local mean/variance. Smooths flat
    speckle while preserving edges (Lee 1981; edge-directional refinement).
    """
    sigv2 = 1.0 / enl
    ms, vs = zip(*(_local_mean_var(a, k) for k in _oriented_kernels(size)))
    M, V = np.stack(ms), np.stack(vs)
    cv = np.sqrt(V) / np.where(M > 0, M, np.nan)
    idx = np.nanargmin(np.where(np.isfinite(cv), cv, np.inf), axis=0)
    lm = np.take_along_axis(M, idx[None], 0)[0]
    lv = np.take_along_axis(V, idx[None], 0)[0]
    ci2 = lv / np.where(lm > 0, lm * lm, np.nan)
    b = np.clip((ci2 - sigv2) / (ci2 * (1.0 + sigv2)), 0.0, 1.0)
    out = lm + b * (a - lm)
    return np.where(np.isfinite(a), out, np.nan)


def frost(a: np.ndarray, size: int = 7, damping: float = 2.0) -> np.ndarray:
    """Frost adaptive exponential filter (linear power).

    Per-pixel weights decay with distance and with the local coefficient of
    variation: homogeneous pixels get near-uniform (strong) smoothing, edge/high-
    texture pixels concentrate weight on the centre (edge-preserving). NaN-aware.
    """
    m, v = _local_mean_var(a, np.ones((size, size)))
    ci2 = np.nan_to_num(v / np.where(m > 0, m * m, np.nan), nan=0.0)
    fin = np.isfinite(a).astype(float)
    a0 = np.where(np.isfinite(a), a, 0.0)
    r = size // 2
    num = np.zeros_like(a0)
    den = np.zeros_like(a0)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            w = np.exp(-damping * ci2 * float(np.hypot(dy, dx)))
            sh = np.roll(np.roll(a0, dy, 0), dx, 1)
            sf = np.roll(np.roll(fin, dy, 0), dx, 1)
            num += w * sh * sf
            den += w * sf
    out = np.where(den > 0, num / den, np.nan)
    return np.where(np.isfinite(a), out, np.nan)


def gamma_map(a: np.ndarray, size: int = 7, enl: float = ENL) -> np.ndarray:
    """Gamma-MAP speckle filter (Lopes et al. 1990; linear power).

    Assumes Gamma-distributed scene and speckle. Homogeneous pixels (local CV
    <= speckle CV) -> local mean; otherwise the MAP estimate that blends mean and
    centre. Strong smoothing with good edge preservation — usually the cleanest
    single-image SAR result. NaN-aware.
    """
    m, v = _local_mean_var(a, np.ones((size, size)))
    cu2 = 1.0 / enl
    ci2 = v / np.where(m > 0, m * m, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        alpha = (1.0 + cu2) / np.maximum(ci2 - cu2, 1e-9)
        b = alpha - enl - 1.0
        disc = m * m * b * b + 4.0 * alpha * enl * m * a
        rmap = (b * m + np.sqrt(np.maximum(disc, 0.0))) / (2.0 * alpha)
    out = np.where(ci2 <= cu2, m, rmap)
    return np.where(np.isfinite(a), out, np.nan)


# ── temporal filters (over axis 0, linear, NaN-aware) ─────────────────────────
def temporal_mean(stack: np.ndarray, window: int = 5) -> np.ndarray:
    """Centred moving mean over ``window`` nearest passes, per pixel."""
    fin = np.isfinite(stack).astype(float)
    s0 = np.where(np.isfinite(stack), stack, 0.0)
    num = ndimage.uniform_filter1d(s0, window, axis=0, mode="nearest") * window
    den = ndimage.uniform_filter1d(fin, window, axis=0, mode="nearest") * window
    return np.where(den > 0, num / den, np.nan)


def multitemporal_quegan(stack: np.ndarray, size: int = 5) -> np.ndarray:
    """Quegan & Yu (2001) multi-temporal filter: ``J_k = m_k · mean_i(I_i / m_i)``.

    ``m`` is a spatial local mean (boxcar ``size``). Exploits the whole stack to cut
    speckle while preserving each scene's local mean level.
    """
    m = np.stack([boxcar(stack[t], size) for t in range(stack.shape[0])])
    ratio = np.where(m > 0, stack / m, np.nan)
    rbar = np.nanmean(ratio, axis=0)
    return m * rbar[None]


# ── config application + bands ────────────────────────────────────────────────
SPATIAL_FILTERS = {
    "boxcar": boxcar, "refined_lee": refined_lee,
    "frost": frost, "gamma_map": gamma_map,
}


def apply_config(vv: np.ndarray, vh: np.ndarray, angle: np.ndarray, cfg: dict
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Run a processing config over linear VV/VH stacks; returns processed VV, VH."""
    if cfg.get("angle_norm"):
        f = angle_norm_factor(angle, cfg.get("ref_angle", 40.0),
                              cfg.get("angle_exp", 2.0))
        vv, vh = vv * f, vh * f
    sp = cfg.get("spatial", "none")
    if sp in SPATIAL_FILTERS:
        sz = cfg.get("spatial_size", 7)
        fn = SPATIAL_FILTERS[sp]
        vv = np.stack([fn(vv[t], sz) for t in range(vv.shape[0])])
        vh = np.stack([fn(vh[t], sz) for t in range(vh.shape[0])])
    tp = cfg.get("temporal", "none")
    if tp == "mean":
        w = cfg.get("temporal_window", 5)
        vv, vh = temporal_mean(vv, w), temporal_mean(vh, w)
    elif tp == "quegan":
        s = cfg.get("temporal_size", 5)
        vv, vh = multitemporal_quegan(vv, s), multitemporal_quegan(vh, s)
    return vv, vh


def compute_band(vv: np.ndarray, vh: np.ndarray, band: str) -> np.ndarray:
    """Linear-domain band from processed VV/VH."""
    if band == "VV":
        return vv
    if band == "VH":
        return vh
    if band == "SPAN":
        return vv + vh
    if band == "CR":
        return np.where(vv > 0, vh / vv, np.nan)
    if band == "RVI":
        s = vv + vh
        return np.where(s > 0, 4.0 * vh / s, np.nan)
    raise ValueError(f"unknown band {band}")


def roi_masks(win: dict, radii) -> dict[int, np.ndarray]:
    """Boolean ROI masks (row=y, col=x) for arbitrary radii [m] about the tower."""
    xx, yy = np.meshgrid(win["xs"], win["ys"], indexing="xy")
    dist = np.sqrt((xx - SITE_UTM_X) ** 2 + (yy - SITE_UTM_Y) ** 2)
    return {int(r): dist <= r for r in radii}


def ring_stat(band_lin: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-scene ROI (mean, std) **in linear** over one mask; caller converts to dB."""
    means, stds = [], []
    for t in range(band_lin.shape[0]):
        v = band_lin[t][mask]
        v = v[np.isfinite(v)]
        means.append(float(v.mean()) if v.size else np.nan)
        stds.append(float(v.std(ddof=1)) if v.size > 1 else np.nan)
    return np.array(means), np.array(stds)


def ring_means(band_lin: np.ndarray, masks: dict[int, np.ndarray]
               ) -> dict[int, np.ndarray]:
    """Per-scene ROI mean **in linear** for each ring; caller applies ``to_display``."""
    out: dict[int, np.ndarray] = {}
    n = band_lin.shape[0]
    for r, mask in masks.items():
        vals = []
        for t in range(n):
            v = band_lin[t][mask]
            v = v[np.isfinite(v)]
            vals.append(float(v.mean()) if v.size else np.nan)
        out[r] = np.array(vals)
    return out


# ── default named configs (seed the compare list) ─────────────────────────────
DEFAULT_CONFIGS: list[dict] = [
    {"name": "raw", "spatial": "none", "temporal": "none"},
    {"name": "boxcar-5", "spatial": "boxcar", "spatial_size": 5, "temporal": "none"},
    {"name": "refined-lee", "spatial": "refined_lee", "spatial_size": 7},
    {"name": "temporal-mean-5", "temporal": "mean", "temporal_window": 5},
    {"name": "temporal-mean-17", "temporal": "mean", "temporal_window": 17},
    {"name": "temporal-mean-19", "temporal": "mean", "temporal_window": 19},
    {"name": "gamma-map", "spatial": "gamma_map", "spatial_size": 7},
    {"name": "multitemporal", "temporal": "quegan", "temporal_size": 5},
    {"name": "RL+multitemporal", "spatial": "refined_lee", "spatial_size": 7,
     "temporal": "quegan", "temporal_size": 5},
    {"name": "refined-lee+anglenorm", "spatial": "refined_lee", "spatial_size": 7,
     "temporal": "none", "angle_norm": True, "ref_angle": 40.0},
]
