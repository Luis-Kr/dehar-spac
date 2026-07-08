"""Reusable Sentinel-2 ROI + vegetation-index helpers for the clear-sky audit.

Shared by the off-pipeline inspector (``other/s2_inspector/server.py``) and the
processing orchestrator (``scripts/process_sentinel2_roi.py``) so the tool the eye
audits and the table the analysis reads are built by the *same* code (issue #8).

Everything here operates on the **raw** yearly stacks in
``data/raw/satellite/sentinel2/`` which, after ADR 0009, are exported UNMASKED
(clouds present, all pixels finite). Side-effect-free: no file writes, no globals
mutated. Reflectance is on the GEE SR scale (0-10000).

Geometry note: the raw stacks carry dims ``(time, X, Y)`` with ``X`` = easting and
``Y`` = northing (both EPSG:32632, 10 m). Band windows here are returned as
``(row=y, col=x)`` arrays; rendering (north-up) is the caller's concern.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import spyndex
import xarray as xr
from dehar.utils.constants import SITE_UTM_X, SITE_UTM_Y

log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/satellite/sentinel2")
BANDS: tuple[str, ...] = (
    "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12",
)
RING_RADII_M: tuple[int, ...] = (30, 50, 100, 500)
PIXEL_M = 10.0
WINDOW_HALF_PX = 60  # 600 m half-window — covers the 500 m ring with margin


# ── vegetation indices (operate on dicts of 2-D band arrays) ──────────────────
def _nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Normalised difference (a - b) / (a + b); 0/0 -> NaN."""
    denom = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (a - b) / denom
    return np.where(denom == 0, np.nan, out)


def _evi(b: dict[str, np.ndarray]) -> np.ndarray:
    nir, red, blue = b["B8"], b["B4"], b["B2"]
    denom = nir + 6 * red - 7.5 * blue + 1e4
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom == 0, np.nan, 2.5 * (nir - red) / denom)


def _nirv(b: dict[str, np.ndarray]) -> np.ndarray:
    return _nd(b["B8"], b["B4"]) * (b["B8"] / 1e4)


def _mcari(b: dict[str, np.ndarray]) -> np.ndarray:
    re1, red, green = b["B5"], b["B4"], b["B3"]
    safe = np.where(red == 0, np.nan, red)
    return ((re1 - red) - 0.2 * (re1 - green)) * (re1 / safe)


INDEX_DEFS: dict[str, Callable[[dict[str, np.ndarray]], np.ndarray]] = {
    "NDVI": lambda b: _nd(b["B8"], b["B4"]),
    "EVI": _evi,
    "NDWI": lambda b: _nd(b["B8"], b["B11"]),   # NIR-SWIR1 (Gao 1996); == NDII here
    "NBR": lambda b: _nd(b["B8"], b["B12"]),
    "NDII": lambda b: _nd(b["B8"], b["B11"]),
    "CCI": lambda b: _nd(b["B3"], b["B5"]),
    "NIRv": _nirv,
    "MCARI": _mcari,
}

# Full science-index menu computed via ``spyndex`` (Awesome Spectral Indices).
# Mirrors the retired ``export_all_streams`` S2 set verbatim so the audited ROI
# product is a drop-in superset for every headline consumer (App B/C). These are
# the *science* columns the analysis reads; ``INDEX_DEFS`` above stays the
# inspector's lightweight dropdown (ADR 0012).
SCIENCE_INDICES: tuple[str, ...] = (
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
)


def science_index_arrays(bands: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-pixel science indices via ``spyndex``, matching the retired GEE path.

    Parameters
    ----------
    bands
        ROI-window band arrays on the GEE SR scale (0-10000). ``spyndex``
        expects reflectance in ``[0, 1]``, so bands are divided by ``1e4``.

    Returns
    -------
    dict[str, np.ndarray]
        ``{index_name: 2-D array}`` for every entry of :data:`SCIENCE_INDICES`
        computable from the available bands. kNDVI uses the canonical per-pixel
        RBF kernel with ``sigma = 0.5*(N + R)`` (Camps-Valls et al. 2021), so
        ``kNDVI = tanh(NDVI**2)`` — monotonic in NDVI. (The retired GEE export
        used a single nonstandard global ``median|N - R|`` sigma; ADR 0012.)
    """
    b = {k: v.astype("float32") / 1e4 for k, v in bands.items()}
    params: dict[str, object] = {
        "N": b["B8"], "N2": b["B8A"], "R": b["B4"], "G": b["B3"], "B": b["B2"],
        "RE1": b["B5"], "RE2": b["B6"], "RE3": b["B7"], "S1": b["B11"],
        "S2": b["B12"], "kNN": 1.0,
    }
    params["kNR"] = spyndex.computeKernel(
        kernel="RBF",
        params={"a": b["B8"], "b": b["B4"], "sigma": 0.5 * (b["B8"] + b["B4"])},
    )
    out: dict[str, np.ndarray] = {}
    for name in SCIENCE_INDICES:
        if name not in spyndex.indices:
            continue
        p = dict(params)
        ok = True
        for band in spyndex.indices[name].bands:
            if band in p:
                continue
            if band in spyndex.constants:            # L, g, C1, C2, gamma, ...
                p[band] = spyndex.constants[band].default
            else:                                    # a needed band is absent
                ok = False
                break
        if not ok:
            continue
        arr = np.asarray(
            spyndex.computeIndex(index=name, params=p, online=False), dtype=float
        )
        out[name] = np.where(np.isfinite(arr), arr, np.nan)   # inf (0-denom) -> NaN
    return out


def index_names() -> list[str]:
    """Vegetation indices offered by the tool's dropdown."""
    return list(INDEX_DEFS)


def series_names() -> list[str]:
    """Everything the analysis/pixel time series can plot: indices + raw bands."""
    return list(INDEX_DEFS) + list(BANDS)


def needed_bands(var: str) -> list[str]:
    """Raw bands required to compute ``var`` (a band name or an index)."""
    if var in BANDS:
        return [var]
    req = {
        "NDVI": ["B8", "B4"], "EVI": ["B8", "B4", "B2"], "NDWI": ["B8", "B11"],
        "NBR": ["B8", "B12"], "NDII": ["B8", "B11"], "CCI": ["B3", "B5"],
        "NIRv": ["B8", "B4"], "MCARI": ["B5", "B4", "B3"],
    }
    return req.get(var, list(BANDS))


def grid_coords(raw_dir: Path = RAW_DIR) -> tuple[np.ndarray, np.ndarray]:
    """1-D (X, Y) coords of the canonical grid — ascending (X east, Y north)."""
    sidx = scene_index(raw_dir)
    with xr.open_dataset(raw_dir / sidx.iloc[0]["file"]) as ds:
        return ds["X"].values.copy(), ds["Y"].values.copy()


def pixel_map(
    raw_dir: Path, var: str, scenes: pd.DataFrame, reducer: str = "median",
) -> np.ndarray | None:
    """North-up ``(nY, nX)`` per-pixel reduction of ``var`` over ``scenes``.

    Reads the full tile for every scene (grouped by file), computes the index,
    and reduces over time with ``median`` (cloud-robust) or ``mean``. Row 0 is
    north; column 0 is west. Returns ``None`` if ``scenes`` is empty.
    """
    need = needed_bands(var)
    frames: list[np.ndarray] = []
    for fname, grp in scenes.groupby("file"):
        with xr.open_dataset(raw_dir / fname) as ds:
            for r in grp.itertuples():
                b = {bd: ds[bd].isel(time=r.i).values.T[::-1].astype(float)
                     for bd in need}
                frames.append(INDEX_DEFS[var](b) if var in INDEX_DEFS else b[var])
    if not frames:
        return None
    stack = np.stack(frames)
    with np.errstate(invalid="ignore"):
        return np.nanmedian(stack, 0) if reducer == "median" else np.nanmean(stack, 0)


# ── scene index ───────────────────────────────────────────────────────────────
def _grid_key(ds: xr.Dataset) -> tuple[float, float, int, int]:
    """Identity of a file's pixel grid: (X[0], Y[0], nX, nY)."""
    x, y = ds["X"].values, ds["Y"].values
    return (round(float(x[0]), 1), round(float(y[0]), 1), int(x.size), int(y.size))


def _tower_offcenter(ds: xr.Dataset) -> float:
    """How far (px, L1) the tower pixel sits from the tile centre in this grid."""
    x, y = ds["X"].values, ds["Y"].values
    cx = int(np.argmin(np.abs(x - SITE_UTM_X)))
    cy = int(np.argmin(np.abs(y - SITE_UTM_Y)))
    return abs(cx - x.size / 2) + abs(cy - y.size / 2)


def scene_index(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """One row per S2 scene on the canonical tower-centred grid, sorted in time.

    Files whose pixel grid is not the canonical one (the grid on which the tower
    is best centred — e.g. years exported before the tower coordinate was fixed)
    are **skipped with a warning**, so the tool never mixes grids (which would
    misplace the map, rings, and pixel picks). Columns: ``file, year, i`` (time
    index within the file), ``datetime``, ``scene_id`` (``YYYYmmddTHHMMSS``).
    """
    files = sorted(raw_dir.glob("s2_sr_dehar_*.nc"))
    grids: dict[tuple, list[float]] = {}
    file_key: dict[Path, tuple] = {}
    for f in files:
        with xr.open_dataset(f) as ds:
            key = _grid_key(ds)
            grids.setdefault(key, []).append(_tower_offcenter(ds))
            file_key[f] = key
    if not files:
        return pd.DataFrame(columns=["file", "year", "i", "datetime", "scene_id"])

    canon = min(grids, key=lambda k: min(grids[k]))       # grid with tower most centred
    skipped = sorted({f.name for f, k in file_key.items() if k != canon})
    if skipped:
        log.warning(
            "scene_index: skipping %d file(s) not on the tower-centred grid %s: %s",
            len(skipped), canon, ", ".join(skipped),
        )

    rows: list[dict] = []
    for f in files:
        if file_key[f] != canon:
            continue
        with xr.open_dataset(f) as ds:
            times = pd.to_datetime(ds["time"].values)
        year = int(f.stem.split("_")[-1])
        for i, t in enumerate(times):
            rows.append({"file": f.name, "year": year, "i": int(i), "datetime": t})
    df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
    df["scene_id"] = df["datetime"].dt.strftime("%Y%m%dT%H%M%S")
    return df


# ── tower window + ring masks ─────────────────────────────────────────────────
def tower_window(ds: xr.Dataset, half: int = WINDOW_HALF_PX) -> dict:
    """Index bounds + coords of a square window centred on the tower pixel."""
    x = ds["X"].values
    y = ds["Y"].values
    cx = int(np.argmin(np.abs(x - SITE_UTM_X)))
    cy = int(np.argmin(np.abs(y - SITE_UTM_Y)))
    x0, x1 = max(0, cx - half), min(x.size, cx + half + 1)
    y0, y1 = max(0, cy - half), min(y.size, cy + half + 1)
    return {"x0": x0, "x1": x1, "y0": y0, "y1": y1,
            "xs": x[x0:x1], "ys": y[y0:y1], "cx": cx, "cy": cy}


def ring_masks(win: dict) -> dict[int, np.ndarray]:
    """Boolean ROI masks (row=y, col=x) for each ring radius, from window coords."""
    xx, yy = np.meshgrid(win["xs"], win["ys"], indexing="xy")   # (ny, nx)
    dist = np.sqrt((xx - SITE_UTM_X) ** 2 + (yy - SITE_UTM_Y) ** 2)
    return {int(r): dist <= r for r in RING_RADII_M}


def _band_window(ds: xr.Dataset, i: int, win: dict, band: str) -> np.ndarray:
    """Window of one band for scene ``i`` as a ``(row=y, col=x)`` float array."""
    a = ds[band].isel(time=i).values                       # (nX, nY)
    a = a[win["x0"]:win["x1"], win["y0"]:win["y1"]]        # (nx, ny)
    return a.T.astype(float)                                # (ny, nx)


def cloud_score(bands: dict[str, np.ndarray], mask: np.ndarray) -> float:
    """ROI whiteness pre-sort hint in ~[0, 1.5]: mean visible reflectance / 3000.

    Bright (cloud/haze/snow) -> high; dark vegetation -> low. A *hint* only; it
    never sets the verdict (ADR 0009).
    """
    vis = (bands["B2"] + bands["B3"] + bands["B4"]) / 3.0
    sel = vis[mask]
    return float(np.nanmean(sel) / 3000.0) if np.isfinite(sel).any() else np.nan


# ── ROI-mean table (the shared product) ───────────────────────────────────────
def build_roi_table(
    raw_dir: Path = RAW_DIR,
    progress: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    """Per-scene ROI means for every ring x (band+index), plus QA hints.

    Wide table, one row per scene: ``{var}_r{radius}`` columns (e.g.
    ``NDVI_r100``), ``n_r{radius}`` valid-pixel counts, ``cloud_score`` and
    ``roi_nan_frac`` (over the 500 m ring). Downstream joins the audit verdict.
    """
    idx = scene_index(raw_dir)
    big = max(RING_RADII_M)
    rows: list[dict] = []
    done, total = 0, len(idx)
    for fname, grp in idx.groupby("file"):
        with xr.open_dataset(raw_dir / fname) as ds:
            win = tower_window(ds)
            masks = ring_masks(win)
            for r in grp.itertuples():
                bands = {b: _band_window(ds, r.i, win, b) for b in BANDS}
                allvars = {**bands, **{n: f(bands) for n, f in INDEX_DEFS.items()}}
                allvars.update(science_index_arrays(bands))   # spyndex wins overlaps
                rec = {"datetime": r.datetime, "scene_id": r.scene_id, "year": r.year}
                for rm, mask in masks.items():
                    for name, arr in allvars.items():
                        vals = arr[mask]
                        ok = np.isfinite(vals).any()
                        rec[f"{name}_r{rm}"] = float(np.nanmean(vals)) if ok else np.nan
                    rec[f"n_r{rm}"] = int(np.isfinite(allvars["NDVI"][mask]).sum())
                rec["cloud_score"] = cloud_score(bands, masks[big])
                ndvi_big = allvars["NDVI"][masks[big]]
                rec["roi_nan_frac"] = float(np.mean(~np.isfinite(ndvi_big)))
                rows.append(rec)
                done += 1
                if progress is not None:
                    progress(done, total, r.scene_id)
    return pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)


# ── baseline climatology ──────────────────────────────────────────────────────
def baseline_climatology(
    table: pd.DataFrame,
    col: str,
    years: range = range(2018, 2025),
    window_days: int = 15,
    min_n: int = 3,
) -> pd.DataFrame:
    """Day-of-year median + IQR band for ``col`` pooled over ``years``.

    Rolling circular ``+/- window_days`` window; a DOY with fewer than ``min_n``
    scenes yields NaN (a gap rather than a spurious point). Feeds the analysis
    panel's shaded baseline against which the focus year is overplotted.
    """
    sub = table[table["year"].isin(list(years))][["datetime", col]].dropna()
    doy = sub["datetime"].dt.dayofyear.to_numpy()
    vals = sub[col].to_numpy(float)
    grid = np.arange(1, 366)
    med = np.full(grid.size, np.nan)
    p25 = np.full(grid.size, np.nan)
    p75 = np.full(grid.size, np.nan)
    nwin = np.zeros(grid.size, dtype=int)
    for k, d in enumerate(grid):
        dd = np.abs(doy - d)
        dd = np.minimum(dd, 365 - dd)
        sel = vals[dd <= window_days]
        nwin[k] = sel.size
        if sel.size >= min_n:
            med[k], p25[k], p75[k] = np.nanpercentile(sel, [50, 25, 75])
    return pd.DataFrame({"doy": grid, "median": med, "p25": p25, "p75": p75, "n": nwin})
