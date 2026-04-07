"""Quality control utilities: sentinel replacement, gap flagging."""

import numpy as np
import pandas as pd

from dehar.utils.constants import MISSING_VALUE_SENTINEL


def replace_sentinels(
    df: pd.DataFrame,
    sentinel: float = MISSING_VALUE_SENTINEL,
    tolerance: float = 1.0,
) -> pd.DataFrame:
    """Replace sentinel missing-value markers with NaN.

    Handles both exact matches and values close to the sentinel
    (e.g. -9999.0, -9.999e+03).

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    sentinel : float
        The sentinel value that encodes missing data.
    tolerance : float
        Absolute tolerance around the sentinel.

    Returns
    -------
    pd.DataFrame
        Copy with sentinels replaced by NaN.
    """
    numeric = df.select_dtypes(include="number")
    mask = np.abs(numeric - sentinel) < tolerance
    out = df.copy()
    out[mask] = np.nan
    return out
