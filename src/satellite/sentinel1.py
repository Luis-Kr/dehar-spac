"""
Sentinel-1 radar index computation.

All functions operate on xarray Datasets with S1 GRD band names
(VV, VH in dB scale, plus optional angle in degrees).
"""

import numpy as np
import xarray as xr


# ── Helpers ───────────────────────────────────────────────────

def _db_to_linear(da: xr.DataArray) -> xr.DataArray:
    """Convert decibel values to linear power scale."""
    return 10 ** (da / 10)


def _linear_to_db(da: xr.DataArray) -> xr.DataArray:
    """Convert linear power to decibels."""
    return 10 * np.log10(da.where(da > 0, np.nan))


# ── Index definitions ─────────────────────────────────────────

def compute_vv(ds: xr.Dataset) -> xr.DataArray:
    """VV backscatter (passthrough, kept for pipeline symmetry)."""
    return ds["VV"].rename("VV")


def compute_vh(ds: xr.Dataset) -> xr.DataArray:
    """VH backscatter (passthrough)."""
    return ds["VH"].rename("VH")


def compute_vh_vv_ratio(ds: xr.Dataset) -> xr.DataArray:
    """Cross-pol ratio VH/VV in dB = VH_dB - VV_dB."""
    return (ds["VH"] - ds["VV"]).rename("VH_VV_ratio")


def compute_span(ds: xr.Dataset) -> xr.DataArray:
    """Total power (SPAN) = VV_lin + VH_lin, returned in dB."""
    vv_lin = _db_to_linear(ds["VV"])
    vh_lin = _db_to_linear(ds["VH"])
    return _linear_to_db(vv_lin + vh_lin).rename("SPAN")


def compute_rvi(ds: xr.Dataset) -> xr.DataArray:
    """Radar Vegetation Index = 4·VH_lin / (VV_lin + VH_lin).
    Dimensionless, range [0, 1] for typical vegetation."""
    vv_lin = _db_to_linear(ds["VV"])
    vh_lin = _db_to_linear(ds["VH"])
    return (4 * vh_lin / (vv_lin + vh_lin)).rename("RVI")


# ── Convenience ───────────────────────────────────────────────
ALL_INDICES = {
    "VV": compute_vv,
    "VH": compute_vh,
    "VH_VV_ratio": compute_vh_vv_ratio,
    "SPAN": compute_span,
    "RVI": compute_rvi,
}


def add_all_indices(ds: xr.Dataset) -> xr.Dataset:
    """Compute and attach all S1 radar indices to the dataset.

    Existing variables with the same name are overwritten for consistency.
    """
    for name, func in ALL_INDICES.items():
        idx = func(ds)
        if name in ds:
            ds = ds.drop_vars(name)
        ds[name] = idx
    return ds
