"""Read the REddyProc atmosphere+soil file and extract meteorological variables."""

from pathlib import Path

import pandas as pd

from dehar.utils.constants import CET, MISSING_VALUE_SENTINEL
from dehar.utils.qc import replace_sentinels
from dehar.utils.time import localize_and_convert_to_utc


def read_atmosphere_soil_raw(path: Path) -> pd.DataFrame:
    """Read the DE-Har REddyProc results file.

    The file is tab-separated with a header row (column names),
    a units row, and data rows.  Timestamps mark the **end** of
    each 30-minute averaging interval.

    Parameters
    ----------
    path : Path
        Path to the raw ``.txt`` file.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with UTC DatetimeIndex (``datetime``),
        sentinel values (−9999) replaced with NaN.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        skiprows=[1],
        na_values=["-9999", "-9.999e+03", "-9.9990e+03"],
    )
    df.index = pd.to_datetime(df["Date Time"])
    df.index.name = "datetime"
    df = df.drop(columns=["Date Time"])

    df = replace_sentinels(df, sentinel=MISSING_VALUE_SENTINEL)

    # Assume CET (UTC+1, no DST) — standard REddyProc convention
    df.index = localize_and_convert_to_utc(df.index, source_tz=CET)

    return df


# Column mapping: raw name → processed name
_METEO_COLUMNS = {
    "Tair": "tair_c",
    "rH": "rh_pct",
    "VPD": "vpd_hpa",
    "Rg": "rg_wm2",
    "PAR": "par_umol_m2s",
    "Ustar": "ustar_ms",
}


def extract_meteorology(df: pd.DataFrame) -> pd.DataFrame:
    """Extract core meteorological variables from the raw DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Full DataFrame returned by :func:`read_atmosphere_soil_raw`.

    Returns
    -------
    pd.DataFrame
        Subset with renamed columns:
        ``tair_c``, ``rh_pct``, ``vpd_hpa``, ``rg_wm2``,
        ``par_umol_m2s``, ``ustar_ms``.
    """
    available = {raw: proc for raw, proc in _METEO_COLUMNS.items() if raw in df.columns}
    if not available:
        raise ValueError(
            f"No expected meteo columns found. "
            f"Expected: {list(_METEO_COLUMNS.keys())}. "
            f"Got: {list(df.columns[:20])}"
        )
    return df[list(available.keys())].rename(columns=available)
