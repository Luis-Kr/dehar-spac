"""
Sentinel-2 vegetation index computation.

All functions operate on xarray Datasets with S2 SR band names
(B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12).

Reflectance values are expected in GEE SR scale (0–10000).
"""

import numpy as np
import xarray as xr


# ── Index definitions ─────────────────────────────────────────
# Sentinel-2 band central wavelengths for reference:
#   B2=490nm  B3=560nm  B4=665nm  B5=705nm  B6=740nm
#   B7=783nm  B8=842nm  B8A=865nm B11=1610nm B12=2190nm

def _ndiff(ds: xr.Dataset, a: str, b: str) -> xr.DataArray:
    """Normalised difference: (a - b) / (a + b)."""
    return (ds[a] - ds[b]) / (ds[a] + ds[b])


def compute_ndvi(ds: xr.Dataset) -> xr.DataArray:
    """NDVI = (NIR - Red) / (NIR + Red)."""
    return _ndiff(ds, "B8", "B4").rename("NDVI")


def compute_evi(ds: xr.Dataset) -> xr.DataArray:
    """Enhanced Vegetation Index (Huete et al., 2002)."""
    nir, red, blue = ds["B8"], ds["B4"], ds["B2"]
    return (2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1e4)).rename("EVI")


def compute_ndwi(ds: xr.Dataset) -> xr.DataArray:
    """NDWI-SWIR = (NIR - SWIR1) / (NIR + SWIR1) (Gao, 1996)."""
    return _ndiff(ds, "B8", "B11").rename("NDWI")


def compute_ndii(ds: xr.Dataset) -> xr.DataArray:
    """Normalised Difference Infrared Index (same formula as NDWI-SWIR,
    kept as separate variable for clarity in downstream analysis)."""
    return _ndiff(ds, "B8", "B11").rename("NDII")


def compute_cci(ds: xr.Dataset) -> xr.DataArray:
    """Chlorophyll/Carotenoid Index = (Green - RedEdge1) / (Green + RedEdge1).
    Proxy for pigment ratio using S2 bands (Gitelson et al.)."""
    return _ndiff(ds, "B3", "B5").rename("CCI")


def compute_nirv(ds: xr.Dataset) -> xr.DataArray:
    """Near-infrared reflectance of vegetation (Badgley et al., 2017).
    NIRv = NDVI × NIR reflectance."""
    ndvi = _ndiff(ds, "B8", "B4")
    return (ndvi * ds["B8"]).rename("NIRv")


def compute_mcari(ds: xr.Dataset) -> xr.DataArray:
    """Modified Chlorophyll Absorption Ratio Index (Daughtry et al., 2000).
    MCARI = [(RE1 - Red) - 0.2·(RE1 - Green)] × (RE1 / Red)
    Using S2: RE1=B5, Red=B4, Green=B3."""
    re1, red, green = ds["B5"], ds["B4"], ds["B3"]
    safe_red = red.where(red != 0, 1e-6)
    return (((re1 - red) - 0.2 * (re1 - green)) * (re1 / safe_red)).rename("MCARI")


# ── Convenience: add all indices at once ──────────────────────
ALL_INDICES = {
    # "NDVI": compute_ndvi,
    # "EVI": compute_evi,
    # "NDWI": compute_ndwi,
    # "NDII": compute_ndii,
    # "CCI": compute_cci,
    # "NIRv": compute_nirv,
    # "MCARI": compute_mcari,
}


def add_all_indices(ds: xr.Dataset) -> xr.Dataset:
    """Compute and attach all S2 vegetation indices to the dataset.

    If an index already exists (e.g. NDVI from GEE), it is overwritten
    with the locally computed version for consistency.
    """
    for name, func in ALL_INDICES.items():
        idx = func(ds)
        if name in ds:
            ds = ds.drop_vars(name)
        ds[name] = idx
    return ds
