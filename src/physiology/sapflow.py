"""Process sap flux density: read per-tree files, average depths, aggregate."""

from pathlib import Path

import pandas as pd

from dehar.utils.time import parse_mixed_datetime


def read_single_tree(path: Path) -> pd.DataFrame:
    """Read one sap flux density CSV and average inner/outer depths.

    Each file contains 30-minute sap flux density measured at two
    sapwood depths: outer (12.5 mm) and inner (27.5 mm) from the bark.
    Units are cm³ cm⁻² per 30 min.

    Parameters
    ----------
    path : Path
        Path to a single-tree CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``tree_id``, ``species``,
        ``js_outer``, ``js_inner``, ``js_mean``,
        indexed by a timezone-naive datetime.
    """
    df = pd.read_csv(path)
    df.index = parse_mixed_datetime(df["Ts"])
    df.index.name = "datetime"

    tree_id = df["TreeID"].iloc[0].lower()
    species = df["Species"].iloc[0]

    out = pd.DataFrame(index=df.index)
    out["tree_id"] = tree_id
    out["species"] = species
    out["js_outer"] = pd.to_numeric(df["Js_outer_cor"], errors="coerce")
    out["js_inner"] = pd.to_numeric(df["Js_inner_cor"], errors="coerce")
    out["js_mean"] = out[["js_outer", "js_inner"]].mean(axis=1)
    return out


def read_all_trees(raw_dir: Path) -> pd.DataFrame:
    """Read all sap flux density files and merge into wide format.

    Parameters
    ----------
    raw_dir : Path
        Directory containing per-tree CSV files.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame with one column per tree (``js_h10545``, etc.),
        indexed by UTC datetime.  Values are the depth-averaged sap
        flux density in cm³ cm⁻² per 30 min.
    """
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    frames = {}
    for path in csv_files:
        single = read_single_tree(path)
        tree_id = single["tree_id"].iloc[0]
        frames[tree_id] = single["js_mean"]

    wide = pd.DataFrame(frames)
    wide.columns = [f"js_{col}" for col in wide.columns]

    # Timestamps are already UTC per data provider
    wide.index = wide.index.tz_localize("UTC")
    return wide


def compute_daily_sapflow(half_hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 30-minute sap flux density to daily sums.

    Since the data units are cm³ cm⁻² per 30 min, the daily sum
    gives cm³ cm⁻² per day.

    Parameters
    ----------
    half_hourly : pd.DataFrame
        Half-hourly wide DataFrame from :func:`read_all_trees`.

    Returns
    -------
    pd.DataFrame
        Daily sums.  Days with fewer than 40 of 48 half-hours are
        set to NaN to avoid underestimation.
    """
    daily_sum = half_hourly.resample("D").sum(min_count=40)
    return daily_sum
