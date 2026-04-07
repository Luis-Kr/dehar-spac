"""Process stem water potential (SWP) measurements."""

from pathlib import Path

import pandas as pd

from dehar.utils.constants import CET
from dehar.utils.time import localize_and_convert_to_utc, parse_mixed_datetime


def read_stemwater_potential(path: Path) -> pd.DataFrame:
    """Read stem water potential CSV.

    The file contains 15-minute SWP (MPa) for multiple trees.
    Timestamps are in CET (UTC+1).

    Parameters
    ----------
    path : Path
        Path to the raw SWP CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with UTC DatetimeIndex and one column per
        sensor: ``swp_mpa_h10545``, ``swp_mpa_h10546``, ``swp_mpa_h10560``.
    """
    df = pd.read_csv(path, na_values=["NA"])

    ts_col = [c for c in df.columns if c.startswith("TS")][0]
    df.index = parse_mixed_datetime(df[ts_col])
    df.index.name = "datetime"

    tree_cols = [c for c in df.columns if c.startswith("H")]
    if not tree_cols:
        raise ValueError(
            f"No tree columns (H*) found. Columns: {list(df.columns)}"
        )

    rename_map = {col: f"swp_mpa_{col.lower()}" for col in tree_cols}
    out = df[tree_cols].rename(columns=rename_map).apply(
        pd.to_numeric, errors="coerce"
    )

    # CET → UTC
    out.index = localize_and_convert_to_utc(out.index, source_tz=CET)
    return out
