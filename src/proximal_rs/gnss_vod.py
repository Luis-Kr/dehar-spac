"""Process GNSS-transmissometry VOD data from multiple receiver pairs.

Each raw CSV contains 30-min aggregated VOD estimates from a
below-canopy GNSS receiver compared to a reference receiver (GPS1).
The normalized VOD (nvod) corrects for signal path length differences
across satellite elevation angles and is the primary variable.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from dehar.utils.constants import CET
from dehar.utils.time import localize_and_convert_to_utc

# Receivers to include in processed output.
# GPS2 and GPS4 are excluded: only ~7 weeks of data (Apr–Jun),
# too short for seasonal drought cascade analysis.
RECEIVERS_INCLUDE = {"GPS1", "GPS3", "GPS5"}

MIN_THETA_DEFAULT = 30.0

# UTC hour windows for diurnal decomposition
PREDAWN_HOURS = (1, 5)    # 01:00–04:59 UTC ≈ 02:00–05:59 CET
AFTERNOON_HOURS = (12, 16)  # 12:00–15:59 UTC ≈ 13:00–16:59 CET


def _parse_receiver_id(filename: str) -> str:
    """Extract the below-canopy receiver ID from a filename.

    Parameters
    ----------
    filename : str
        e.g. ``gnss_vod_Hartheim_GPS2_GPS1_251220180925_#hash.csv``

    Returns
    -------
    str
        Receiver ID, e.g. ``GPS2``.
    """
    match = re.search(r"Hartheim_(GPS\d+)_GPS1", filename)
    if not match:
        raise ValueError(f"Cannot parse receiver ID from {filename}")
    return match.group(1)


def read_single_receiver(
    path: Path,
    min_theta: float = MIN_THETA_DEFAULT,
) -> pd.DataFrame:
    """Read one GNSS-T CSV and apply elevation angle filter.

    Parameters
    ----------
    path : Path
        Path to a single receiver CSV file.
    min_theta : float
        Minimum satellite elevation angle in degrees.
        Observations below this threshold are discarded.

    Returns
    -------
    pd.DataFrame
        Filtered data with UTC DatetimeIndex and columns:
        ``nvod``, ``vod``, ``theta``, ``azimuth``, ``pool``.
    """
    df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")

    # Assume CET (field station in central Europe)
    df.index = localize_and_convert_to_utc(df.index, source_tz=CET)

    df = df[df["theta"] >= min_theta].copy()

    keep = ["nvod", "vod", "theta", "azimuth", "pool"]
    return df[[c for c in keep if c in df.columns]]


def read_all_receivers(
    raw_dir: Path,
    min_theta: float = MIN_THETA_DEFAULT,
    receivers: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Read and filter all selected GNSS-T receiver files.

    Parameters
    ----------
    raw_dir : Path
        Directory containing the CSV and metadata files.
    min_theta : float
        Minimum elevation angle filter.
    receivers : set of str, optional
        Which receivers to include (default: GPS1, GPS3, GPS5).

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from receiver ID (lowercase, e.g. ``gps3``) to
        filtered DataFrames.
    """
    if receivers is None:
        receivers = RECEIVERS_INCLUDE

    csv_files = sorted(raw_dir.glob("gnss_vod_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No GNSS VOD CSV files in {raw_dir}")

    result = {}
    for path in csv_files:
        receiver_id = _parse_receiver_id(path.name)
        if receiver_id not in receivers:
            continue
        df = read_single_receiver(path, min_theta=min_theta)
        result[receiver_id.lower()] = df

    if not result:
        raise ValueError(
            f"No receivers matched {receivers}. "
            f"Found: {[_parse_receiver_id(f.name) for f in csv_files]}"
        )
    return result


def build_half_hourly(
    receiver_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Merge filtered receiver data into a wide 30-min DataFrame.

    Parameters
    ----------
    receiver_data : dict[str, pd.DataFrame]
        Output from :func:`read_all_receivers`.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame with columns ``nvod_gps1``, ``nvod_gps3``, etc.
    """
    frames = {}
    for rid, df in receiver_data.items():
        frames[f"nvod_{rid}"] = df["nvod"]

    wide = pd.DataFrame(frames)
    wide = wide.sort_index()
    wide.index.name = "datetime"
    return wide


def _hour_in_window(hour_series: pd.Series, window: tuple[int, int]) -> pd.Series:
    """Boolean mask: True where UTC hour falls in [start, end)."""
    start, end = window
    if start < end:
        return (hour_series >= start) & (hour_series < end)
    # Wraps midnight
    return (hour_series >= start) | (hour_series < end)


def compute_daily_vod_metrics(
    receiver_data: dict[str, pd.DataFrame],
    predawn_hours: tuple[int, int] = PREDAWN_HOURS,
    afternoon_hours: tuple[int, int] = AFTERNOON_HOURS,
) -> pd.DataFrame:
    """Compute daily VOD summary metrics per receiver.

    For each receiver and each day, computes:

    - **predawn** — median nvod during predawn window (nighttime
      maximum, analogous to predawn stem water potential)
    - **afternoon** — median nvod during afternoon window (daytime
      minimum, peak transpiration depletion)
    - **amplitude** — predawn minus afternoon (diurnal recharge
      capacity; collapses under severe drought)
    - **daily_mean** — median nvod across all observations that day

    Parameters
    ----------
    receiver_data : dict[str, pd.DataFrame]
        Output from :func:`read_all_receivers`.
    predawn_hours : tuple of int
        UTC hour window (start, end) for predawn observations.
    afternoon_hours : tuple of int
        UTC hour window (start, end) for afternoon observations.

    Returns
    -------
    pd.DataFrame
        Daily metrics with columns like ``nvod_predawn_gps3``,
        ``nvod_afternoon_gps3``, ``nvod_amplitude_gps3``,
        ``nvod_mean_gps3``.
    """
    daily_frames = []

    for rid, df in receiver_data.items():
        nvod = df["nvod"].copy()
        hours = nvod.index.hour

        predawn_mask = _hour_in_window(hours, predawn_hours)
        afternoon_mask = _hour_in_window(hours, afternoon_hours)

        predawn_daily = nvod[predawn_mask].resample("D").median()
        afternoon_daily = nvod[afternoon_mask].resample("D").median()
        mean_daily = nvod.resample("D").median()

        metrics = pd.DataFrame({
            f"nvod_predawn_{rid}": predawn_daily,
            f"nvod_afternoon_{rid}": afternoon_daily,
            f"nvod_amplitude_{rid}": predawn_daily - afternoon_daily,
            f"nvod_mean_{rid}": mean_daily,
        })
        daily_frames.append(metrics)

    combined = pd.concat(daily_frames, axis=1)
    combined = combined.sort_index()
    combined.index.name = "datetime"

    # Drop days where all values are NaN (no observations)
    combined = combined.dropna(how="all")
    return combined
