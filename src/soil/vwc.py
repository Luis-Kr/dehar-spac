"""Extract volumetric water content (VWC) from the atmosphere+soil file."""

import re

import pandas as pd


def extract_soil_moisture(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and rename VWC columns from the raw DataFrame.

    Raw column names like ``VWCSoil_A_5cm`` are mapped to
    ``vwc_pct_a_5cm`` for consistency with project naming conventions.

    Parameters
    ----------
    df : pd.DataFrame
        Full DataFrame from :func:`dehar.atmosphere.meteo.read_atmosphere_soil_raw`.

    Returns
    -------
    pd.DataFrame
        Soil moisture time series with standardized column names.
    """
    vwc_cols = [c for c in df.columns if c.startswith("VWCSoil")]
    if not vwc_cols:
        raise ValueError(
            f"No VWCSoil columns found. Available: {list(df.columns[:30])}"
        )

    rename_map = {}
    for col in vwc_cols:
        match = re.match(r"VWCSoil_([A-C])_(\d+cm)", col)
        if match:
            profile, depth = match.group(1).lower(), match.group(2)
            rename_map[col] = f"vwc_pct_{profile}_{depth}"
        else:
            rename_map[col] = col.lower().replace("vwcsoil", "vwc_pct")

    return df[vwc_cols].rename(columns=rename_map)
