"""
Load yearly Sentinel-1 / Sentinel-2 NetCDF stacks into unified xarray Datasets.

Produces:
  - ds_s2:  all S2 scenes with time dimension, cloud-masked bands + indices
  - ds_s1a: all S1 ascending scenes
  - ds_s1d: all S1 descending scenes

Usage:
  from dehar.satellite.load_satellite_stacks import load_s2, load_s1

  ds_s2 = load_s2()
  ds_s1a, ds_s1d = load_s1()

  # quick pixel time series at tower location
  tower_s2 = ds_s2.sel(x=TOWER_X, y=TOWER_Y, method="nearest")
  tower_s2["NDVI"].plot()
"""

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from dehar.utils.constants import SITE_CRS, SITE_NAME, SITE_UTM_X, SITE_UTM_Y

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Default data directories — match download scripts
# ──────────────────────────────────────────────────────────────
S2_DIR = Path("data/raw/satellite/sentinel2")
S1_DIR = Path("data/raw/satellite/sentinel1")

TOWER_X = SITE_UTM_X
TOWER_Y = SITE_UTM_Y


def _normalize_dims(ds: xr.Dataset) -> xr.Dataset:
    """Rename spatial dims to lowercase x/y if needed (xee uses X/Y)."""
    rename = {}
    if "X" in ds.dims and "x" not in ds.dims:
        rename["X"] = "x"
    if "Y" in ds.dims and "y" not in ds.dims:
        rename["Y"] = "y"
    return ds.rename(rename) if rename else ds


def _load_yearly_ncs(directory: Path, pattern: str) -> xr.Dataset:
    """Concatenate yearly NetCDFs along time dimension."""
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {directory}/{pattern}")

    datasets = []
    for f in files:
        ds = _normalize_dims(xr.open_dataset(f))
        datasets.append(ds)
        log.info("  loaded %s: %s scenes", f.name, ds.sizes.get("time", "?"))

    ds = xr.concat(datasets, dim="time")
    ds = ds.sortby("time")

    _, idx = np.unique(ds.time.values, return_index=True)
    ds = ds.isel(time=idx)

    log.info(
        "  → %d total scenes, %s to %s",
        ds.sizes["time"],
        str(ds.time.values[0])[:10],
        str(ds.time.values[-1])[:10],
    )
    return ds


# ──────────────────────────────────────────────────────────────
# Sentinel-2
# ──────────────────────────────────────────────────────────────
def load_s2(data_dir: Path = S2_DIR) -> xr.Dataset:
    """Load full S2 stack."""
    log.info("Loading Sentinel-2 from %s", data_dir)
    ds = _load_yearly_ncs(data_dir, "s2_sr_dehar_*.nc")
    ds.attrs["description"] = "Sentinel-2 SR Harmonized, cloud-masked"
    ds.attrs["site"] = SITE_NAME
    ds.attrs["crs"] = SITE_CRS
    ds.attrs["scale_m"] = 10
    return ds


# ──────────────────────────────────────────────────────────────
# Sentinel-1
# ──────────────────────────────────────────────────────────────
def load_s1(data_dir: Path = S1_DIR) -> tuple[xr.Dataset, xr.Dataset]:
    """Load S1 ascending and descending stacks separately."""
    log.info("Loading Sentinel-1 ascending from %s", data_dir)
    ds_a = _load_yearly_ncs(data_dir, "s1_grd_dehar_a_*.nc")
    ds_a.attrs["orbit"] = "ASCENDING"

    log.info("Loading Sentinel-1 descending from %s", data_dir)
    ds_d = _load_yearly_ncs(data_dir, "s1_grd_dehar_d_*.nc")
    ds_d.attrs["orbit"] = "DESCENDING"

    for ds in [ds_a, ds_d]:
        ds.attrs["description"] = (
            "Sentinel-1 GRD, angle-normalised, speckle-filtered"
        )
        ds.attrs["site"] = SITE_NAME
        ds.attrs["crs"] = SITE_CRS
        ds.attrs["scale_m"] = 10

    return ds_a, ds_d


# ──────────────────────────────────────────────────────────────
# Convenience: tower pixel extraction
# ──────────────────────────────────────────────────────────────
def extract_tower_pixel(ds: xr.Dataset, x=TOWER_X, y=TOWER_Y) -> xr.Dataset:
    """Extract the nearest pixel to the tower as a 1-D time series Dataset."""
    return ds.sel(x=x, y=y, method="nearest")


def extract_footprint_mean(
    ds: xr.Dataset,
    x=TOWER_X, y=TOWER_Y,
    radius_m: float = 200,
) -> xr.Dataset:
    """
    Average pixels within `radius_m` of the tower.
    Simple circular mask — replace with actual footprint weights later.
    """
    dx = ds.x - x
    dy = ds.y - y
    dist = np.sqrt(dx**2 + dy**2)
    mask = dist <= radius_m
    return ds.where(mask).mean(dim=["x", "y"])


# ──────────────────────────────────────────────────────────────
# CLI quick check
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ds_s2 = load_s2()
    log.info("S2 bands: %s", list(ds_s2.data_vars))

    ds_s1a, ds_s1d = load_s1()
    log.info("S1 asc bands: %s", list(ds_s1a.data_vars))
    log.info("S1 desc bands: %s", list(ds_s1d.data_vars))

    tower = extract_tower_pixel(ds_s2)
    log.info(
        "Tower NDVI range: %.3f – %.3f",
        float(tower["NDVI"].min()),
        float(tower["NDVI"].max()),
    )
