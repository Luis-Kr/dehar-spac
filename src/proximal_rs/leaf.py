"""LEAF scan processing: scan CSV -> gap fraction -> PAVD / PAI profiles.

Reusable, side-effect-free functions behind ``scripts/process_leaf.py``.
They wrap the vendored ``pylidar_tls_canopy`` library (a pinned git submodule;
see ``docs/adr/0003-pai-hinge-hemi-not-sampling-artifact.md``) and add the two
things the notebook could not do:

* expose the **tilt transform** as an explicit, config-driven knob
  (``add_leaf_scan_to_profile``); upstream ``add_leaf_scan_position`` hard-wires
  ``transform=True`` inside the submodule, so the tilt could neither be seen nor
  switched. This function is the seam where the GitHub issue #10 tilt fix plugs
  in;
* run the per-scan inversion as a picklable worker for multiprocessing.

With ``transform.method = leaf_tilt_offset`` (the default) every function here
is numerically identical to the upstream notebook pipeline.
"""

from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pylidar_tls_canopy import leaf_io, plant_profile
from pylidar_tls_canopy.rsmooth import rsmooth

#: Tilt-correction algorithms understood by :func:`transform_mode`.
#: ``leaf_tilt_offset``    - upstream scalar zenith/azimuth offset (leaf_io.py).
#: ``leaf_tilt_rotation``  - proper per-beam re-levelling (issue #10).
#: ``leaf_self_calibrate`` - per-scan re-fold about the drifting "up" (issue #11).
SUPPORTED_TRANSFORM_METHODS = (
    "leaf_tilt_offset",
    "leaf_tilt_rotation",
    "leaf_self_calibrate",
)

_DT_RE = re.compile(r"(\d{8})-(\d{6})Z")


# --------------------------------------------------------------------------
# Identity & transform
# --------------------------------------------------------------------------
def get_scan_datetime(filename: str) -> _dt.datetime | None:
    """Extract the UTC scan datetime from a LEAF filename.

    Parameters
    ----------
    filename : str
        Basename like ``ESS00320_0012_hinge_20250416-191027Z_0005_8500.csv``.

    Returns
    -------
    datetime.datetime or None
        Naive UTC datetime, or ``None`` if the pattern is absent.
    """
    m = _DT_RE.search(filename)
    if m is None:
        return None
    return _dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


def transform_mode(transform_cfg: dict) -> str:
    """Resolve the ``transform`` config block to a processing mode.

    Parameters
    ----------
    transform_cfg : dict
        ``{enabled: bool, method: str}`` from the config.

    Returns
    -------
    str
        ``"off"`` (no tilt correction), ``"offset"`` (upstream scalar offset),
        or ``"rotation"`` (proper per-beam re-levelling).

    Raises
    ------
    ValueError
        For an unrecognised method.
    """
    if not bool(transform_cfg.get("enabled", True)):
        return "off"
    method = transform_cfg.get("method", "leaf_tilt_offset")
    if method == "leaf_tilt_offset":
        return "offset"
    if method == "leaf_tilt_rotation":
        return "rotation"
    if method == "leaf_self_calibrate":
        return "self_calibrate"
    raise ValueError(
        f"Unknown transform.method {method!r}; "
        f"expected one of {list(SUPPORTED_TRANSFORM_METHODS)}."
    )


def resolve_transform(transform_cfg: dict) -> dict:
    """Resolve a transform config into the two orthogonal knobs (issue #11, ADR 0005).

    Accepts either the **2-knob** form ``{tilt: none|offset|rotation, up_drift: bool}``
    or the legacy ``{enabled, method}`` form (``leaf_tilt_offset`` /
    ``leaf_tilt_rotation`` / ``leaf_seasonal_up``). ``leaf_seasonal_up`` is the alias
    for ``{tilt: rotation, up_drift: true}`` (the canonical corrected geometry).

    Returns
    -------
    dict
        ``{"tilt": "none"|"offset"|"rotation", "up_drift": bool}``.
    """
    if "tilt" in transform_cfg:
        tilt = transform_cfg["tilt"]
        if tilt not in ("none", "offset", "rotation"):
            raise ValueError(
                f"transform.tilt must be none|offset|rotation, got {tilt!r}"
            )
        return {"tilt": tilt, "up_drift": bool(transform_cfg.get("up_drift", False))}
    if not bool(transform_cfg.get("enabled", True)):
        return {"tilt": "none", "up_drift": False}
    method = transform_cfg.get("method", "leaf_tilt_offset")
    alias = {
        "leaf_tilt_offset": {"tilt": "offset", "up_drift": False},
        "leaf_tilt_rotation": {"tilt": "rotation", "up_drift": False},
        "leaf_seasonal_up": {"tilt": "rotation", "up_drift": True},
    }
    if method not in alias:
        raise ValueError(
            f"transform.method {method!r} not supported by the pipeline; use the "
            f"2-knob form {{tilt, up_drift}} or one of {list(alias)}."
        )
    return alias[method]


def _apply_tilt_offset(leaf: leaf_io.LeafScanFile) -> None:
    """Upstream scalar tilt offset (``leaf_io.py:131-135``) on an already-folded scan.

    Adds the tilt header's ``theta``/``phi`` to every beam's zenith/azimuth and
    recomputes the Cartesian/height fields. Used to compose the **published**
    offset tilt on top of an up-drift re-fold (the offset + up-drift sensitivity);
    the byte-stable no-up-drift offset still uses the library's ``transform=True``.
    """
    dx, dy, dz = (d / 1024 for d in leaf.header["Tilt"])
    _, theta, phi = leaf_io.xyz2rza(dx, dy, dz)
    zen = leaf.data["zenith"].to_numpy() + theta
    azi = leaf.data["azimuth"].to_numpy() + phi
    leaf.data["zenith"] = zen
    leaf.data["azimuth"] = azi
    for n in (1, 2):
        rng = leaf.data[f"range{n}"].to_numpy()
        x, y, z = leaf_io.rza2xyz(rng, zen, azi)
        leaf.data[f"x{n}"], leaf.data[f"y{n}"], leaf.data[f"z{n}"] = x, y, z
        if leaf.sensor_height:
            leaf.data[f"h{n}"] = z + leaf.sensor_height


def load_up_lookup(path: str | Path) -> pd.Series:
    """Smoothed daily "up" (deg) indexed by UTC date (the lookup table, ADR 0005)."""
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.set_index("date")["up_smooth_deg"]


def up_on_date(lookup: pd.Series, when: _dt.datetime) -> float:
    """Smoothed "up" (deg) for a scan datetime (nearest day if exact missing)."""
    ts = pd.Timestamp(when)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    day = ts.normalize()
    if day in lookup.index:
        return float(lookup.loc[day])
    pos = lookup.index.get_indexer([day], method="nearest")[0]
    return float(lookup.iloc[pos])


def load_up_lookup_perscan(path: str | Path) -> pd.Series:
    """Per-scan "up" (deg) indexed by UTC scan datetime (manual reference).

    Companion to :func:`load_up_lookup` (one smoothed value per day). Reads a
    lookup with one row per hand-inspected scan (columns ``datetime`` and
    ``up_deg``) and returns a UTC datetime-indexed, time-sorted series so that
    :func:`up_on_scan` can interpolate the "up" for scans nobody inspected.
    Duplicate timestamps keep the last (latest submit wins).

    Parameters
    ----------
    path : str or pathlib.Path
        CSV with ``datetime`` and ``up_deg`` columns.

    Returns
    -------
    pandas.Series
        ``up_deg`` indexed by tz-aware UTC datetime, ascending, unique.
    """
    df = pd.read_csv(path, parse_dates=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    s = df.set_index("datetime")["up_deg"].astype(float).sort_index()
    return s[~s.index.duplicated(keep="last")]


def up_on_scan(lookup: pd.Series, when: _dt.datetime) -> float:
    """Manual "up" (deg) for a scan datetime, linearly interpolated in time.

    Returns the exact manual value when ``when`` coincides with an inspected
    scan, otherwise a linear interpolation between the two nearest-in-time
    manual points. Held constant (no extrapolation) outside the inspected
    range, so head/tail scans take the first/last inspected "up".

    Parameters
    ----------
    lookup : pandas.Series
        Per-scan "up" from :func:`load_up_lookup_perscan` (sorted UTC index).
    when : datetime.datetime
        Scan datetime (naive treated as UTC).

    Returns
    -------
    float
        Interpolated "up" in degrees.
    """
    ts = pd.Timestamp(when)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return float(np.interp(ts.value, lookup.index.asi8, lookup.to_numpy(float)))


def leveling_rotation(tilt_vec) -> np.ndarray:
    """Minimal rotation matrix that re-levels the instrument frame (issue #10).

    The LEAF ``Tilt`` header ``(dx, dy, dz)`` is the world-vertical "up"
    direction expressed in the instrument frame (it points to ``+z`` when the
    instrument is level; here it leans ~2 deg). This returns the unique minimal
    rotation ``R`` (about the horizontal axis ``up x z``) that maps that
    measured up-vector onto ``+z``, so a beam direction is re-levelled by
    ``v_world = R @ v_inst``.

    Parameters
    ----------
    tilt_vec : array-like of float, shape (3,)
        The raw ``Tilt`` header vector (need not be normalised).

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        Orthonormal rotation matrix (``det == +1``).
    """
    a = np.asarray(tilt_vec, dtype=float)
    a = a / np.linalg.norm(a)
    b = np.array([0.0, 0.0, 1.0])
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(np.dot(a, b))
    if s < 1e-12:  # already level (c~+1) or inverted (c~-1, not expected)
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    kmat = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + kmat + kmat @ kmat * ((1.0 - c) / s**2)


def level_leaf_data(leaf: leaf_io.LeafScanFile) -> None:
    """Re-level a LEAF scan in place by a true 3-D rotation (issue #10).

    Replaces the upstream scalar zenith/azimuth offset (which adds the same
    ~2 deg to *every* beam, biasing the hinge rings up to ~59 deg) with the
    minimal rotation that brings the instrument's measured up-vector to vertical
    (an azimuth-dependent shift of ``+tilt*cos(az - tilt_az)`` that averages to
    zero over a full azimuth sweep). Operates on a scan read with
    ``transform=False`` and recomputes ``zenith``, ``azimuth``, the per-return
    Cartesian coordinates ``x1/y1/z1``/``x2/y2/z2`` and heights ``h1``/``h2``
    from the rotated per-beam directions (``h`` is unchanged from the angle-only
    form, so the PAI parquet is byte-stable; the Cartesian fields make the
    leveled geometry usable for point-cloud / hemispherical visualisation).
    """
    rot = leveling_rotation(leaf.header["Tilt"])
    zen = leaf.data["zenith"].to_numpy()
    azi = leaf.data["azimuth"].to_numpy()
    x, y, z = leaf_io.rza2xyz(1.0, zen, azi)  # instrument-frame unit beams
    xl, yl, zl = rot @ np.vstack([x, y, z])  # re-levelled
    _, theta, phi = leaf_io.xyz2rza(xl, yl, zl)
    leaf.data["zenith"] = theta
    leaf.data["azimuth"] = phi
    for n in (1, 2):
        rng = leaf.data[f"range{n}"].to_numpy()
        cx, cy, cz = leaf_io.rza2xyz(rng, theta, phi)
        leaf.data[f"x{n}"] = cx
        leaf.data[f"y{n}"] = cy
        leaf.data[f"z{n}"] = cz
        if leaf.sensor_height:
            leaf.data[f"h{n}"] = cz + leaf.sensor_height


def recenter_leaf_data(leaf: leaf_io.LeafScanFile, up_deg: float) -> None:
    """Re-fold a LEAF hemi scan about a corrected "up" encoder angle (issue #11).

    The vendored ``leaf_io`` folds zenith about the encoder mid-point
    (``s = pi`` = 180 deg), assuming the mirror points straight up there. At
    DE-Har that "up" **drifts over the season** (the instrument gradually settles
    in the sandy soil), so the two half-sweeps mis-align in zenith and downward
    beams punch below the ground. This recomputes ``zenith``, ``azimuth`` and the
    Cartesian/height fields by folding about ``up_deg`` instead, from the raw
    scan/rotary encoders. Operates on a scan read with ``transform=False``; pair
    with :func:`calibrate_up` for a self-calibrating, per-scan correction.
    """
    nsteps = 2.56e4 if leaf.header["Firmware ver."] >= 4.11 else 1e4
    s = leaf.data["scan_encoder"].to_numpy() / nsteps * 2 * np.pi + leaf.zenith_offset
    az = leaf.data["rotary_encoder"].to_numpy() / 2e4 * 2 * np.pi
    up = np.radians(up_deg)
    if leaf.scan_type == "hemi":
        az = np.where(s < up, az + np.pi, az)
    az = np.mod(az, 2 * np.pi)
    zen = np.abs(s - up)
    leaf.data["zenith"] = zen
    leaf.data["azimuth"] = az
    for n in (1, 2):
        rng = leaf.data[f"range{n}"].to_numpy()
        x, y, z = leaf_io.rza2xyz(rng, zen, az)
        leaf.data[f"x{n}"] = x
        leaf.data[f"y{n}"] = y
        leaf.data[f"z{n}"] = z
        if leaf.sensor_height:
            leaf.data[f"h{n}"] = z + leaf.sensor_height


def calibrate_up(
    leaf: leaf_io.LeafScanFile,
    lo: float = 176.0,
    hi: float = 206.0,
    step: float = 1.0,
    seam_halfwidth: float = 12.0,
) -> float:
    """Find the "up" encoder angle (deg) where the two half-sweeps agree.

    Searches the fold centre that makes the front and back halves report the same
    nearest structure at the *seam* azimuths (N/S, where both halves image the
    same direction). This tracks the seasonal drift automatically; the upstream
    default is 180 deg. Returns the best ``up`` in degrees (falls back to 180 if
    the seam is too sparse to decide).
    """
    nsteps = 2.56e4 if leaf.header["Firmware ver."] >= 4.11 else 1e4
    s = leaf.data["scan_encoder"].to_numpy() / nsteps * 2 * np.pi + leaf.zenith_offset
    az_raw = leaf.data["rotary_encoder"].to_numpy() / 2e4 * 2 * np.pi
    r1 = leaf.data["range1"].to_numpy(float)
    z_edges = np.arange(15.0, 76.0, 5.0)
    best_up, best_dis = 180.0, np.inf
    for up_deg in np.arange(lo, hi + step, step):
        up = np.radians(up_deg)
        zen = np.degrees(np.abs(s - up))
        front = s < up
        az = np.degrees(np.mod(np.where(front, az_raw + np.pi, az_raw), 2 * np.pi))
        seam = (
            (az < seam_halfwidth)
            | (az > 360 - seam_halfwidth)
            | (np.abs(az - 180) < seam_halfwidth)
        ) & ~np.isnan(r1)
        difs = []
        for k in range(len(z_edges) - 1):
            mf = seam & front & (zen >= z_edges[k]) & (zen < z_edges[k + 1])
            mb = seam & ~front & (zen >= z_edges[k]) & (zen < z_edges[k + 1])
            if mf.sum() > 30 and mb.sum() > 30:
                difs.append(abs(np.nanmedian(r1[mf]) - np.nanmedian(r1[mb])))
        if difs:
            dis = float(np.mean(difs))
            if dis < best_dis:
                best_dis, best_up = dis, float(up_deg)
    return best_up


def up_for_scan(path: str | Path, lo: float = 176.0, hi: float = 212.0) -> float:
    """Calibrated "up" (deg) for one hemi scan file; ``NaN`` if unreadable/empty.

    Thin, picklable wrapper around :func:`calibrate_up` for the seasonal-"up"
    precompute (``scripts/build_leaf_up_lookup.py``): reads the raw scan with no
    transform and returns the per-scan seam-fitted "up". The ``hi`` bound is
    raised to 212 deg (vs the 206 deg default) so the late-season drift is not
    clipped at the search ceiling. Failures return ``NaN`` so one bad file does
    not abort a multiprocessing pool.
    """
    try:
        leaf = leaf_io.LeafScanFile(str(path), transform=False)
    except Exception:  # noqa: BLE001 - skip unreadable scans
        return float("nan")
    if leaf.data.empty:
        return float("nan")
    return float(calibrate_up(leaf, lo=lo, hi=hi))


def daily_up_lookup(
    per_scan: pd.DataFrame,
    frac: float = 0.2,
    robust_it: int = 2,
    clamp: tuple[float, float] = (170.0, 215.0),
) -> pd.DataFrame:
    """Smoothed daily "up" lookup from per-scan calibrated values (issue #11).

    The per-scan seam fit is occasionally unreliable (a weak seam can return a
    spurious ~180 deg, e.g. the 2025-05-19 scan), but the seasonal drift is
    smooth, so we (1) take a robust per-day **median** then (2) robust-LOWESS
    across days, evaluating the fit on **every** day (filling gaps). One bad scan
    can no longer escape correction.

    Parameters
    ----------
    per_scan : pandas.DataFrame
        Columns ``datetime`` and ``up_deg`` (one row per scan).
    frac : float
        LOWESS span (fraction of days). Larger = smoother.
    robust_it : int
        LOWESS robustifying iterations (down-weight outlier days).
    clamp : tuple(float, float)
        Hard bounds the smoothed value is clipped to.

    Returns
    -------
    pandas.DataFrame
        One row per day: ``date`` (UTC), ``n_scans``, ``up_median_deg`` (raw
        daily median, ``NaN`` on scan-less days), ``up_smooth_deg`` (smoothed,
        defined every day).
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess

    s = per_scan.dropna(subset=["up_deg"]).copy()
    s["date"] = pd.to_datetime(s["datetime"], utc=True).dt.normalize()
    grp = s.groupby("date")["up_deg"]
    daily = pd.DataFrame({"n_scans": grp.size(), "up_median_deg": grp.median()})

    full = pd.date_range(daily.index.min(), daily.index.max(), freq="D", tz="UTC")
    daily = daily.reindex(full)
    daily["n_scans"] = daily["n_scans"].fillna(0).astype(int)
    daily.index.name = "date"

    obs = daily["up_median_deg"].dropna()
    t0 = obs.index[0]
    x = (obs.index - t0).days.to_numpy(dtype=float)
    x_all = (daily.index - t0).days.to_numpy(dtype=float)
    y_hat = lowess(obs.to_numpy(), x, frac=frac, it=robust_it, xvals=x_all)
    daily["up_smooth_deg"] = np.clip(y_hat, *clamp)
    return daily.reset_index()


def read_leaf_scan(
    path: str,
    *,
    sensor_height: float | None = None,
    transform_cfg: dict | None = None,
    max_range: float = 120,
    zenith_offset: float = 0,
) -> leaf_io.LeafScanFile:
    """Read a LEAF scan with the configured tilt correction already applied.

    Convenience reader for visualisation/audit code: returns the
    ``LeafScanFile`` with ``leaf.data`` carrying corrected ``zenith``, ``azimuth``,
    ``x*/y*/z*`` and ``h*`` for the selected ``transform_cfg`` (see
    :func:`transform_mode`). Defaults to the per-beam rotation (issue #10 / ADR
    0004).
    """
    if transform_cfg is None:
        transform_cfg = {"enabled": True, "method": "leaf_tilt_rotation"}
    mode = transform_mode(transform_cfg)
    leaf = leaf_io.LeafScanFile(
        str(path),
        sensor_height=sensor_height,
        transform=(mode == "offset"),
        max_range=max_range,
        zenith_offset=zenith_offset,
    )
    if mode == "rotation":
        level_leaf_data(leaf)
    elif mode == "self_calibrate":
        recenter_leaf_data(leaf, calibrate_up(leaf))
    return leaf


def add_leaf_scan_to_profile(
    vpp: plant_profile.Jupp2009,
    leaf_file: str,
    *,
    method: str,
    min_zenith: float,
    max_zenith: float,
    sensor_height: float,
    zenith_offset: float,
    transform_cfg: dict,
    max_range: float,
    up_deg: float | None = None,
) -> bool:
    """Bin one LEAF scan into a Jupp2009 profile, with explicit geometry control.

    Faithful reimplementation of
    ``plant_profile.Jupp2009.add_leaf_scan_position`` (which hard-wires
    ``transform=True`` and ``max_range=120`` inside the pinned submodule), with the
    geometry resolved by :func:`resolve_transform` into ``tilt`` + ``up_drift``
    (ADR 0005). Composition: re-fold **hemi** scans about ``up_deg`` (when
    ``up_drift``) **then** apply the tilt (``rotation`` re-levels per beam,
    ``offset`` adds the scalar tilt). Hinge scans are never re-folded (their single
    ring cannot be). The byte-stable no-up-drift ``offset`` keeps the library's
    ``transform=True`` path.

    Returns
    -------
    bool
        ``True`` if the scan had data and was added, else ``False``.
    """
    min_z = np.radians(min_zenith)
    max_z = np.radians(max_zenith)
    cols = ["zenith", "azimuth", "target_count", "h1", "h2"]
    t = resolve_transform(transform_cfg)
    builtin_offset = t["tilt"] == "offset" and not t["up_drift"]

    with leaf_io.LeafScanFile(
        leaf_file,
        sensor_height=sensor_height,
        transform=builtin_offset,
        max_range=max_range,
        zenith_offset=zenith_offset,
    ) as leaf:
        vpp.datetime = leaf.datetime
        if leaf.data.empty:
            return False
        if t["up_drift"] and leaf.scan_type == "hemi" and up_deg is not None:
            recenter_leaf_data(leaf, up_deg)
        if not builtin_offset:
            if t["tilt"] == "rotation":
                level_leaf_data(leaf)
            elif t["tilt"] == "offset":
                _apply_tilt_offset(leaf)
        data = {c: leaf.data[c].to_numpy() for c in cols}
        for n, height in enumerate(["h1", "h2"], start=1):
            target_index = np.full(data[height].shape, n, dtype=np.uint8)
            idx = (
                (data["zenith"] >= min_z)
                & (data["zenith"] < max_z)
                & ~np.isnan(data[height])
            )
            if np.any(idx):
                vpp.add_targets(
                    data[height][idx],
                    target_index[idx],
                    data["target_count"][idx],
                    data["zenith"][idx],
                    data["azimuth"][idx],
                    method=method,
                )
        idx = (data["zenith"] >= min_z) & (data["zenith"] < max_z)
        if np.any(idx):
            vpp.add_shots(
                data["target_count"][idx],
                data["zenith"][idx],
                data["azimuth"][idx],
                method=method,
            )
        return True


# --------------------------------------------------------------------------
# Per-scan inversion (multiprocessing worker)
# --------------------------------------------------------------------------
def process_single_scan(
    filepath: Path,
    *,
    profile: dict,
    instrument: dict,
    transform: dict,
    up_deg: float | None = None,
) -> pd.DataFrame | None:
    """Invert one LEAF scan to a long-format profile table (no met context).

    Picklable worker for ``ProcessPoolExecutor``: it does the heavy lifting
    (read CSV, tilt transform, Jupp 2009 inversion) and returns one row per
    height bin. Meteorological columns are attached later in the parent
    (:func:`attach_met_context`).

    Returns
    -------
    pandas.DataFrame or None
        Columns ``datetime, scan_hour, filename, height``, the plant profiles
        (``HingePAI`` ...), and ``Pgap_Z*`` columns. ``None`` if the filename
        lacks a datetime or the scan has no usable returns.
    """
    scan_dt = get_scan_datetime(filepath.name)
    if scan_dt is None:
        return None

    vpp = plant_profile.Jupp2009(
        hres=profile["height_resolution_m"],
        zres=profile["zenith_resolution_deg"],
        ares=profile["azimuth_resolution_deg"],
        min_z=profile["min_zenith_deg"],
        max_z=profile["max_zenith_deg"],
        min_h=0,
        max_h=profile["max_height_m"],
    )
    added = add_leaf_scan_to_profile(
        vpp,
        str(filepath),
        method=profile["method"],
        min_zenith=profile["min_zenith_deg"],
        max_zenith=profile["max_zenith_deg"],
        sensor_height=instrument["sensor_height_m"],
        zenith_offset=instrument["zenith_offset_deg"],
        transform_cfg=transform,
        max_range=instrument["max_range_m"],
        up_deg=up_deg,
    )
    if not added:
        return None

    vpp.get_pgap_theta_z()
    df = vpp.exportPlantProfiles().merge(vpp.exportPgapProfiles(), on="Height")
    df = df.rename(columns={"Height": "height"})

    rename = {
        c: f"Pgap_Z{float(c.replace('Zenith', '')) / 10:05.1f}"
        for c in df.columns
        if c.startswith("Zenith")
    }
    df = df.rename(columns=rename)

    df.insert(0, "datetime", scan_dt)
    df.insert(1, "scan_hour", scan_dt.hour)
    df.insert(2, "filename", filepath.name)
    return df


# --------------------------------------------------------------------------
# Meteorological context & quality filter
# --------------------------------------------------------------------------
def load_met(met_file: str | Path, precip_file: str | Path) -> pd.DataFrame:
    """Load and merge the 30-min meteorological and precipitation tables.

    Returns
    -------
    pandas.DataFrame
        UTC ``DatetimeIndex``, met columns plus ``Precipitation_Sum_mm``
        (gaps filled with 0).
    """
    met = pd.read_csv(met_file, parse_dates=["datetime"])
    met["datetime"] = pd.to_datetime(met["datetime"], utc=True)
    met = met.set_index("datetime").sort_index()

    precip = pd.read_csv(precip_file)
    precip["datetime"] = pd.to_datetime(
        precip["From"].str.replace("Z", "+00:00"), utc=True
    )
    precip = precip.set_index("datetime")[["Precipitation_Sum_mm"]].sort_index()

    met_full = met.join(precip, how="outer")
    met_full["Precipitation_Sum_mm"] = met_full["Precipitation_Sum_mm"].fillna(0)
    return met_full


def get_met_context(
    scan_dt: _dt.datetime,
    met_df: pd.DataFrame,
    duration_min: int,
    quality: dict,
) -> dict:
    """Mean met conditions during a scan plus quality flags.

    Returns a dict with ``tair_c, rh_pct, vpd_hpa, ustar_ms, precip_mm`` and the
    flags ``flag_rain, flag_humid, flag_wind, quality_good``. Falls back to the
    nearest 30-min record for short scans that fall between the met grid points.
    """
    scan_start = pd.Timestamp(scan_dt, tz="UTC")
    scan_end = scan_start + pd.Timedelta(minutes=duration_min)
    precip_start = scan_start - pd.Timedelta(hours=quality["precip_window_h"])

    precip_window = met_df.loc[precip_start:scan_end, "Precipitation_Sum_mm"]
    scan_window = met_df.loc[scan_start:scan_end]

    if scan_window.empty:
        pos = met_df.index.get_indexer(
            [scan_start], method="nearest", tolerance=pd.Timedelta("30min")
        )
        if pos[0] != -1:
            scan_window = met_df.iloc[[pos[0]]]

    if scan_window.empty:
        return {
            "tair_c": np.nan,
            "rh_pct": np.nan,
            "vpd_hpa": np.nan,
            "ustar_ms": np.nan,
            "precip_mm": np.nan,
            "flag_rain": False,
            "flag_humid": False,
            "flag_wind": False,
            "quality_good": False,
        }

    precip_total = precip_window.sum()
    rh_mean = scan_window["rh_pct"].mean()
    ustar_max = scan_window["ustar_ms"].max()

    flag_rain = bool(precip_total >= quality["precip_thresh_mm"])
    flag_humid = bool(rh_mean >= quality["rh_thresh_pct"])
    flag_wind = bool(ustar_max >= quality["ustar_thresh_ms"])

    return {
        "tair_c": scan_window["tair_c"].mean(),
        "rh_pct": rh_mean,
        "vpd_hpa": scan_window["vpd_hpa"].mean(),
        "ustar_ms": ustar_max,
        "precip_mm": precip_total,
        "flag_rain": flag_rain,
        "flag_humid": flag_humid,
        "flag_wind": flag_wind,
        "quality_good": not (flag_rain or flag_humid or flag_wind),
    }


def attach_met_context(
    df_profiles: pd.DataFrame,
    met_full: pd.DataFrame,
    duration_min: int,
    quality: dict,
) -> pd.DataFrame:
    """Join per-scan met context onto the concatenated profile table.

    One met lookup per unique scan datetime (cheap), broadcast to every height
    bin of that scan. Equivalent to the per-scan attachment in the notebook.
    """
    dts = pd.DatetimeIndex(sorted(pd.unique(df_profiles["datetime"])))
    rows = []
    for dt in dts:
        ctx = get_met_context(dt.to_pydatetime(), met_full, duration_min, quality)
        ctx["datetime"] = dt
        rows.append(ctx)
    met_df = pd.DataFrame(rows)
    return df_profiles.merge(met_df, on="datetime", how="left")


# --------------------------------------------------------------------------
# Temporal-outlier filter (per scan hour, across the season)
# --------------------------------------------------------------------------
def _segment_bad(
    series: np.ndarray,
    bp: list,
    p: list | float | None,
    max_k: list | int,
    resid_thresh: float,
) -> np.ndarray:
    """Robust-smoothing outlier flag for one daily total-PAI series.

    Generalises the segmented ``rsmooth`` filter from the pylidar LEAF example.
    The season is split by ``bp`` (day indices); each segment is robustly
    smoothed (Garcia 2010 DCT). A day is flagged when missing, declared an
    outlier by the robust fit (weight <= 0), or dropping more than
    ``|resid_thresh|`` below the smooth curve.
    """
    ndays = len(series)
    edges = [ndays if b is None else int(b) for b in bp]
    edges = sorted({min(max(e, 0), ndays) for e in edges})
    if edges[0] != 0:
        edges = [0] + edges
    if edges[-1] != ndays:
        edges = edges + [ndays]

    bad = np.zeros(ndays, dtype=bool)
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        if b <= a:
            continue
        seg = np.where(series[a:b] > 0, series[a:b], np.nan)
        if np.count_nonzero(~np.isnan(seg)) < 3:
            bad[a:b] = np.isnan(seg)
            continue

        p_i = p[i] if isinstance(p, (list, tuple)) else p
        k_i = max_k[i] if isinstance(max_k, (list, tuple)) else max_k

        smooth, weights = rsmooth(seg.copy(), p=p_i, max_k=k_i)
        resid = seg - smooth

        bad_seg = np.isnan(seg)
        bad_seg |= weights <= 0
        bad_seg |= resid < resid_thresh
        bad[a:b] = bad_seg
    return bad


def compute_temporal_flags(df: pd.DataFrame, pai_col: str, temporal: dict) -> dict:
    """Per-scan temporal-outlier verdicts, computed per scan hour.

    For each scan hour, builds a regular daily series of total canopy PAI
    (= max over height of ``pai_col``), runs :func:`_segment_bad`, and maps the
    per-day verdict back onto every scan of that hour.

    Returns
    -------
    dict
        ``{scan datetime -> is_outlier (bool)}`` covering every scan in ``df``.
    """
    bp = temporal["breakpoints"]
    p = temporal["penalty"]
    max_k = temporal["max_k"]
    resid_thresh = temporal["resid_thresh"]

    scan_tp = df.groupby("datetime")[pai_col].max()
    meta = df.drop_duplicates("datetime").set_index("datetime")["scan_hour"]

    flags: dict = {}
    for _hour, dts in meta.groupby(meta).groups.items():
        dts = pd.DatetimeIndex(sorted(dts))
        dates = dts.normalize()
        day0 = dates.min()
        ndays = (dates.max() - day0).days + 1
        day_idx = (dates - day0).days.to_numpy()

        series = np.full(ndays, np.nan)
        series[day_idx] = scan_tp.loc[dts].to_numpy()

        bad = _segment_bad(series, bp, p, max_k, resid_thresh)
        for dt, di in zip(dts, day_idx):
            flags[dt] = bool(bad[di])
    return flags
