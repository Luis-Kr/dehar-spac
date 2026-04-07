"""Timezone conversion and temporal alignment utilities."""

from datetime import timezone, timedelta

import pandas as pd


def localize_and_convert_to_utc(
    index: pd.DatetimeIndex,
    source_tz: timezone,
) -> pd.DatetimeIndex:
    """Localize a naive DatetimeIndex and convert to UTC.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Timezone-naive datetime index.
    source_tz : timezone
        The timezone the timestamps were recorded in.

    Returns
    -------
    pd.DatetimeIndex
        UTC-aware datetime index.
    """
    offset = source_tz.utcoffset(None)
    tz_name = f"Etc/GMT{int(-offset.total_seconds() // 3600):+d}"
    localized = index.tz_localize(tz_name)
    return localized.tz_convert("UTC")


def parse_mixed_datetime(series: pd.Series, dayfirst: bool = False) -> pd.DatetimeIndex:
    """Parse a Series of date strings with inconsistent formatting.

    Handles formats like ``1/1/2025``, ``1/1/2025 0:30``,
    ``5/6/2025 0:15``, etc.

    Parameters
    ----------
    series : pd.Series
        String column containing datetime values.
    dayfirst : bool
        If True, parse day before month (European convention).

    Returns
    -------
    pd.DatetimeIndex
        Parsed datetime index (timezone-naive).
    """
    return pd.to_datetime(series, dayfirst=dayfirst, format="mixed")
