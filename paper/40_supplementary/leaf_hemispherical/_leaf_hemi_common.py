"""Shared helpers for the LEAF hemispherical supplementary figures.

Used by ``fig_hemi_corrected_vs_uncorrected.py`` and
``fig_hemi_seasonal_series.py``. Everything here is plumbing: load a LEAF hemi
scan through the canonical reader (``dehar.proximal_rs.leaf``), bin its returns
onto the hemisphere, and style the fisheye polar axes. All *choices* (dates,
fonts, colour scales) stay in the two figure scripts.

The geometry correction is selected with a ``mode`` key of :data:`TRANSFORM_CFG`:

* ``none``            - raw fold about the hard-wired "up" = 180 deg (uncorrected)
* ``rotation``        - canonical tilt re-levelling (issue #10 / ADR 0004)
* ``self_calibrate``  - per-scan "up"-drift fix (issue #11)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_2d

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "lib" / "pylidar-tls-canopy"))
from dehar.proximal_rs.leaf import (  # noqa: E402,F401
    calibrate_up,
    get_scan_datetime,
    read_leaf_scan,
    recenter_leaf_data,
)
from pylidar_tls_canopy import leaf_io  # noqa: E402,F401  (re-exported for callers)

RAW_DIR = ROOT / "data" / "raw" / "proximal_rs" / "leaf"
PROC_DIR = ROOT / "data" / "processed" / "proximal_rs" / "leaf"
LEAF_PARQUET = PROC_DIR / "leaf_hemi_hi_2025.parquet"
UP_LOOKUP_CSV = PROC_DIR / "leaf_up_daily_2025.csv"  # scripts/build_leaf_up_lookup.py

#: Geometry modes. ``none``/``rotation``/``self_calibrate`` go through
#: ``read_leaf_scan``; ``seasonal_up`` re-folds about the smoothed daily "up"
#: from the lookup table (robust to single-scan seam failures, issue #11).
TRANSFORM_CFG: dict[str, dict] = {
    "none": {"enabled": False, "method": "leaf_tilt_offset"},
    "rotation": {"enabled": True, "method": "leaf_tilt_rotation"},
    "self_calibrate": {"enabled": True, "method": "leaf_self_calibrate"},
}


def load_up_lookup() -> pd.Series:
    """Smoothed daily "up" (deg) indexed by UTC date (the lookup table)."""
    df = pd.read_csv(UP_LOOKUP_CSV, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.set_index("date")["up_smooth_deg"]


def seasonal_up(filename: str, lookup: pd.Series) -> float:
    """Smoothed "up" (deg) for a scan's date (nearest day if exact missing)."""
    day = get_scan_datetime(filename).replace(hour=0, minute=0, second=0)
    day = pd.Timestamp(day, tz="UTC")
    if day in lookup.index:
        return float(lookup.loc[day])
    pos = lookup.index.get_indexer([day], method="nearest")[0]
    return float(lookup.iloc[pos])

ZENITH_RINGS = (30.0, 57.5, 85.0)        # 57.5 deg = Jupp hinge angle
AZIMUTH_TICKS = (0, 180)                  # only N/S labelled (keeps panels packable)


def scan_table() -> pd.DataFrame:
    """One row per hemi_hi scan from the canonical processed parquet.

    Returns
    -------
    pandas.DataFrame
        Columns ``datetime`` (UTC), ``filename``, ``scan_hour``, ``quality_all``,
        ``weighted_pai`` (canopy total = max over height, the pipeline reduction)
        and ``pgap_hinge`` (canopy-total gap at the 57.5 deg ring).
    """
    df = pd.read_parquet(LEAF_PARQUET)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    g = df.groupby("datetime")
    return pd.DataFrame(
        {
            "filename": g["filename"].first(),
            "scan_hour": g["scan_hour"].first(),
            "quality_all": g["quality_all"].first(),
            "weighted_pai": g["WeightedPAI"].max(),
            "pgap_hinge": g["Pgap_Z057.5"].min(),
        }
    ).reset_index()


def pick_scan(
    meta: pd.DataFrame,
    target_date: str,
    hour: int | None = None,
    require_quality: bool = True,
) -> pd.Series:
    """Nearest scan to ``target_date``, preferring good quality and a given hour.

    Parameters
    ----------
    meta : pandas.DataFrame
        Output of :func:`scan_table`.
    target_date : str
        ``YYYY-MM-DD`` to aim for.
    hour : int, optional
        Preferred scan hour (UTC); ignored if no good scan that hour exists.
    require_quality : bool
        Restrict to ``quality_all`` scans when any are available.
    """
    s = meta.copy()
    if require_quality and bool(s["quality_all"].any()):
        s = s[s["quality_all"]]
    s = s.assign(dist=(s["datetime"] - pd.Timestamp(target_date, tz="UTC")).abs())
    if hour is not None:
        hh = s[s["scan_hour"] == hour]
        if not hh.empty:
            s = hh
    return s.sort_values("dist").iloc[0]


def load_returns(
    filename: str,
    mode: str,
    sensor_height: float,
    max_range: float = 120.0,
    up_lookup: pd.Series | None = None,
) -> tuple[object, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read one scan and return per-shot fisheye arrays.

    ``mode`` is ``none`` / ``rotation`` / ``self_calibrate`` (via
    ``read_leaf_scan``) or ``seasonal_up`` (re-fold about the smoothed daily
    "up" from ``up_lookup``; loaded on demand if not passed).

    Returns
    -------
    leaf : LeafScanFile
        The loaded scan (for callers that need range2 / header / second return).
    az_deg, zen_deg : numpy.ndarray
        Per-shot azimuth and zenith (degrees), after the selected ``mode``.
    h1 : numpy.ndarray
        First-return height (m); ``NaN`` where the beam recorded no return.
    gap : numpy.ndarray
        Boolean per-shot gap mask (no valid return at all -> reached the sky).
    """
    if mode == "seasonal_up":
        if up_lookup is None:
            up_lookup = load_up_lookup()
        leaf = leaf_io.LeafScanFile(
            str(RAW_DIR / filename),
            sensor_height=sensor_height,
            transform=False,
            max_range=max_range,
        )
        recenter_leaf_data(leaf, seasonal_up(filename, up_lookup))
    else:
        leaf = read_leaf_scan(
            str(RAW_DIR / filename),
            sensor_height=sensor_height,
            transform_cfg=TRANSFORM_CFG[mode],
            max_range=max_range,
        )
    d = leaf.data
    az = np.degrees(d["azimuth"].to_numpy())
    zen = np.degrees(d["zenith"].to_numpy())
    h1 = d["h1"].to_numpy()
    gap = d["range1"].isna().to_numpy() & d["range2"].isna().to_numpy()
    return leaf, az, zen, h1, gap


def fisheye_grid(
    az_deg: np.ndarray,
    zen_deg: np.ndarray,
    values: np.ndarray,
    statistic: str,
    abin: float,
    zbin: float,
    zmax: float = 90.0,
    min_count: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    """Bin (azimuth, zenith) onto the hemisphere for a polar ``pcolormesh``.

    Non-finite ``values`` are dropped before binning; cells with fewer than
    ``min_count`` samples are masked. Returns ``theta_edges`` (rad), ``r_edges``
    (zenith deg) and the masked grid shaped ``(n_zenith, n_azimuth)``.
    """
    finite = np.isfinite(values)
    az_f, zen_f, val_f = az_deg[finite], zen_deg[finite], values[finite]
    abins = np.arange(0.0, 360.0 + abin, abin)
    zbins = np.arange(0.0, zmax + zbin, zbin)
    grid, _, _, _ = binned_statistic_2d(
        az_f, zen_f, val_f, statistic=statistic, bins=[abins, zbins]
    )
    cnt, _, _, _ = binned_statistic_2d(
        az_f, zen_f, val_f, statistic="count", bins=[abins, zbins]
    )
    grid = np.where(cnt >= min_count, grid, np.nan)
    return np.radians(abins), zbins, np.ma.masked_invalid(grid.T)


def style_fisheye(
    ax,
    title: str = "",
    title_fs: int = 11,
    zen_label_fs: int = 8,
    az_label_fs: int = 8,
    zenith_rings: tuple[float, ...] = ZENITH_RINGS,
    azimuth_ticks: tuple[float, ...] = AZIMUTH_TICKS,
    show_az_labels: bool = True,
    rlabel_deg: float = 135.0,
) -> None:
    """LEAF fisheye look: N at top, clockwise, prominent zenith rings, faint
    azimuth spokes with labels around the rim.

    Zenith rings (circles) at ``zenith_rings`` are drawn **on top** of the data
    and labelled (white-boxed for legibility); azimuth spokes at
    ``azimuth_ticks`` are faint with degree labels outside the circle.
    """
    ax.set_ylim(0, 90)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_axisbelow(False)  # rings/spokes over the scatter so they stay visible

    # zenith rings - prominent, labelled with a white box
    ax.set_yticks(list(zenith_rings))
    labels = ax.set_yticklabels(
        [f"{r:g}°" for r in zenith_rings], fontsize=zen_label_fs, color="0.12"
    )
    box = {"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.7}
    for t in labels:
        t.set_bbox(box)
    ax.set_rlabel_position(rlabel_deg)
    ax.yaxis.grid(True, color="0.20", lw=1.1, alpha=1.0)

    # azimuth spokes - faint, degree labels around the rim (only 0/180 by default)
    ax.set_xticks(np.radians(list(azimuth_ticks)))
    ax.set_xticklabels(
        [f"{a:g}°" for a in azimuth_ticks] if show_az_labels else [],
        fontsize=az_label_fs, color="0.35",
    )
    ax.xaxis.grid(True, color="0.6", lw=0.5, alpha=0.4)
    ax.tick_params(axis="x", pad=1)

    if title:
        ax.set_title(title, fontsize=title_fs)
