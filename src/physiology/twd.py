"""Process tree water deficit (TWD) dendrometer data."""

from pathlib import Path

import pandas as pd

from dehar.utils.constants import CET
from dehar.utils.time import localize_and_convert_to_utc, parse_mixed_datetime


def read_twd(path: Path) -> pd.DataFrame:
    """Read TWD CSV (long format) and pivot to wide format.

    The raw file has one row per tree per timestep.
    Output has one column per tree: ``twd_um_h10529``, etc.

    Parameters
    ----------
    path : Path
        Path to the raw TWD CSV file.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame with UTC DatetimeIndex, columns named
        ``twd_um_<tree_id>``, values in micrometers.
    """
    df = pd.read_csv(path)

    df["datetime"] = parse_mixed_datetime(df["TS"])
    df["twd"] = pd.to_numeric(df["twd"], errors="coerce")

    tree_ids = sorted(df["ID"].unique())

    pivoted = df.pivot_table(
        index="datetime",
        columns="ID",
        values="twd",
        aggfunc="first",
    )
    pivoted.columns = [f"twd_um_{tid.lower()}" for tid in pivoted.columns]
    pivoted = pivoted.sort_index()

    # Assume CET (UTC+1) — same sensor infrastructure as SWP
    pivoted.index = localize_and_convert_to_utc(pivoted.index, source_tz=CET)
    return pivoted
