"""Loaders for network-PhenoCam GCC and camera-NDVI summary products.

Reads the standard PhenoCam-network 1-day / 3-day summary CSVs delivered for the
DE-Har ROI (``hartheim2_UN_1000``) and returns tidy daily frames with a UTC
``DatetimeIndex``. These are the gold-standard, citable canopy greenness/NDVI
products (Sonnentag et al. 2012; Richardson et al. 2018) and supersede the
in-house anglecam GCC as the project's default greenness (see ADR 0002).

Side-effect free: parsing only, no plotting, no I/O beyond reading the raw CSV.
Canonical statistic = the 90th percentile (``gcc_90`` / ``ndvi_90``) — robust to
shadow/illumination noise and the direct analog of the anglecam ``_p90``
convention. The Gaussian-smoothed ``smooth_gcc_90`` is carried but NEVER used as
the canonical value (it would bias changepoint timing in Part A).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Canonical ROI for the site (PhenoCam veg-type UN = understory, top-down view).
ROI_ID = "hartheim2_UN_1000"

# Raw-column -> output-stat for each product. Output columns are later prefixed
# with ``{index}_phenocam_{agg}_`` by ``load_product``. Canonical stat first.
_GCC_COLS = {
    "gcc_90": "p90",                 # canonical
    "gcc_mean": "mean",
    "gcc_std": "std",
    "smooth_gcc_90": "smooth_p90",   # carried, non-canonical
    "image_count": "image_count",
}
_NDVI_COLS = {
    "ndvi_90": "p90",                # canonical
    "ndvi_mean": "mean",
    "ndvi_std": "std",
    "image_count": "image_count",
}

_PRODUCTS: dict[str, dict] = {
    "gcc": {"cols": _GCC_COLS, "stem": ""},          # files: <roi>_<agg>.csv
    "ndvi": {"cols": _NDVI_COLS, "stem": "ndvi_"},   # files: <roi>_ndvi_<agg>.csv
}


def _raw_filename(index: str, agg: str) -> str:
    """PhenoCam summary-product filename for an index ('gcc'/'ndvi') and
    aggregation ('1day'/'3day')."""
    if index not in _PRODUCTS:
        raise ValueError(f"index must be one of {list(_PRODUCTS)}, got {index!r}")
    if agg not in ("1day", "3day"):
        raise ValueError(f"agg must be '1day' or '3day', got {agg!r}")
    stem = _PRODUCTS[index]["stem"]
    return f"{ROI_ID}_{stem}{agg}.csv"


def load_product(raw_dir: Path, index: str, agg: str) -> pd.DataFrame:
    """Load one PhenoCam summary product as a daily UTC-indexed frame.

    Parameters
    ----------
    raw_dir : Path
        Directory holding the raw PhenoCam CSVs.
    index : {'gcc', 'ndvi'}
        Which camera index to load.
    agg : {'1day', '3day'}
        PhenoCam temporal aggregation product.

    Returns
    -------
    pd.DataFrame
        Columns ``{index}_phenocam_{agg}_{stat}`` (e.g. ``gcc_phenocam_1day_p90``)
        on a daily UTC ``DatetimeIndex`` named ``date``. The native cadence is
        preserved: off-window days in the 3-day products stay ``NaN`` (no fill).

    Raises
    ------
    FileNotFoundError
        If the expected raw CSV is missing.
    KeyError
        If an expected column is absent from the file.
    """
    path = raw_dir / _raw_filename(index, agg)
    if not path.exists():
        raise FileNotFoundError(f"PhenoCam product missing: {path}")

    spec = _PRODUCTS[index]["cols"]
    raw = pd.read_csv(path, comment="#", na_values=["NA"])
    if "date" not in raw.columns:
        raise KeyError(f"{path} has no 'date' column")
    missing = [c for c in spec if c not in raw.columns]
    if missing:
        raise KeyError(f"{path} missing expected columns: {missing}")

    idx = pd.to_datetime(raw["date"]).dt.tz_localize("UTC")
    out = (
        raw[list(spec)]
        .set_axis([f"{index}_phenocam_{agg}_{s}" for s in spec.values()], axis=1)
        .set_index(idx)
        .sort_index()
    )
    out.index.name = "date"
    out = out[~out.index.duplicated(keep="first")]
    return out


def load_phenocam_daily(raw_dir: Path) -> pd.DataFrame:
    """Outer-join all four PhenoCam products (GCC/NDVI x 1-day/3-day) on the daily
    UTC date index — the full multi-year record, unfiltered and un-filled."""
    frames = [
        load_product(raw_dir, index, agg)
        for index in ("gcc", "ndvi")
        for agg in ("1day", "3day")
    ]
    daily = pd.concat(frames, axis=1).sort_index()
    daily.index.name = "date"
    return daily
