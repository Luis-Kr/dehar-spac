"""Standardized read/write for processed data files."""

from pathlib import Path

import pandas as pd


def read_processed(path: Path) -> pd.DataFrame:
    """Read a processed CSV with datetime index.

    Parameters
    ----------
    path : Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with UTC DatetimeIndex named ``datetime``.
    """
    df = pd.read_csv(path, parse_dates=True, index_col="datetime")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def save_processed(df: pd.DataFrame, path: Path) -> None:
    """Write a processed DataFrame to CSV with datetime index.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose index is a DatetimeIndex named ``datetime``.
    path : Path
        Destination CSV path.  Parent directories are created if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.index.name != "datetime":
        df.index.name = "datetime"
    df.to_csv(path)
