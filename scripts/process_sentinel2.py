"""
Process raw Sentinel-2 NetCDF stacks → enriched stacks + ROI-mean CSV.

Step 1: Load each yearly raw NC, compute all vegetation indices locally,
        save an enriched NC (original bands + indices).  This preserves
        per-pixel time series capability.

Step 2: Average all pixels inside the ROI for every scene and write a
        single CSV with one row per scene and one column per index/band.

The ROI defaults to the tower location ± BUFFER_M but can be changed
via constants at the top of this file (or CLI args in the future).

Gap-filling / smoothing: **none by default** — raw cloud-masked values
with NaN where masked.  All smoothing should be applied downstream so
the processing chain stays transparent.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from dehar.utils.constants import SITE_CRS, SITE_NAME, SITE_UTM_X, SITE_UTM_Y
from dehar.satellite.sentinel2 import add_all_indices, ALL_INDICES

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────
RAW_DIR = Path("data/raw/satellite/sentinel2")
OUT_STACK_DIR = Path("data/processed/satellite/sentinel2")
OUT_CSV = Path("data/processed/satellite/sentinel2_roi_mean.csv")

# ROI for spatial averaging — tower + circular buffer
ROI_X = SITE_UTM_X
ROI_Y = SITE_UTM_Y
BUFFER_M = 100  # metres around the tower

# Bands / indices to include in the CSV output
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
INDEX_NAMES = list(ALL_INDICES.keys())
CSV_COLUMNS = S2_BANDS + INDEX_NAMES


def _normalize_dims(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    if "X" in ds.dims and "x" not in ds.dims:
        rename["X"] = "x"
    if "Y" in ds.dims and "y" not in ds.dims:
        rename["Y"] = "y"
    return ds.rename(rename) if rename else ds


def _roi_mask(ds: xr.Dataset) -> xr.DataArray:
    """Boolean mask: pixels within BUFFER_M of (ROI_X, ROI_Y)."""
    dx = ds.x - ROI_X
    dy = ds.y - ROI_Y
    return np.sqrt(dx ** 2 + dy ** 2) <= BUFFER_M


def _sanitize_attrs(ds: xr.Dataset) -> xr.Dataset:
    """Remove non-string 'bounds' attrs that break xarray's CF encoder."""
    for var in list(ds.coords) + list(ds.data_vars):
        obj = ds[var]
        if "bounds" in obj.attrs and not isinstance(obj.attrs["bounds"], str):
            del obj.attrs["bounds"]
    return ds


def _process_one_file(nc_path: Path) -> xr.Dataset | None:
    """Load, add indices, return enriched dataset (in memory)."""
    try:
        ds = _normalize_dims(xr.open_dataset(nc_path))
    except Exception:
        log.exception("Cannot open %s", nc_path)
        return None

    n_time = ds.sizes.get("time", 0)
    if n_time == 0:
        log.warning("  %s: no time steps, skipping", nc_path.name)
        return None

    ds = add_all_indices(ds)
    ds = _sanitize_attrs(ds)
    log.info("  %s: %d scenes, %d vars", nc_path.name, n_time, len(ds.data_vars))
    return ds


def _roi_mean_timeseries(ds: xr.Dataset, mask: xr.DataArray) -> pd.DataFrame:
    """Spatial average within ROI for each time step."""
    ds_roi = ds.where(mask)
    cols = [c for c in CSV_COLUMNS if c in ds_roi]
    mean = ds_roi[cols].mean(dim=["x", "y"]).compute()
    df = mean.to_dataframe().reset_index()
    if "time" in df.columns:
        df = df.rename(columns={"time": "datetime"}).set_index("datetime")
    return df


# ── Main pipeline ─────────────────────────────────────────────
def main():
    nc_files = sorted(RAW_DIR.glob("s2_sr_dehar_*.nc"))
    if not nc_files:
        log.error("No raw S2 NetCDFs in %s", RAW_DIR)
        return

    log.info("Found %d raw S2 yearly files", len(nc_files))
    log.info("ROI centre: (%.1f, %.1f) %s, buffer %d m", ROI_X, ROI_Y, SITE_CRS, BUFFER_M)

    OUT_STACK_DIR.mkdir(parents=True, exist_ok=True)
    all_dfs: list[pd.DataFrame] = []

    for nc in nc_files:
        ds = _process_one_file(nc)
        if ds is None:
            continue

        # Step 1: save enriched stack
        out_nc = OUT_STACK_DIR / nc.name.replace(".nc", "_indices.nc")
        ds.to_netcdf(out_nc)
        size_mb = out_nc.stat().st_size / 1e6
        log.info("    → %s (%.1f MB)", out_nc, size_mb)

        # Step 2: ROI mean
        mask = _roi_mask(ds)
        n_pixels = int(mask.sum())
        log.info("    ROI pixels: %d", n_pixels)
        df = _roi_mean_timeseries(ds, mask)
        all_dfs.append(df)

        ds.close()

    if not all_dfs:
        log.warning("No data processed, skipping CSV output")
        return

    df_all = pd.concat(all_dfs).sort_index()
    df_all = df_all[~df_all.index.duplicated(keep="first")]
    df_all.index.name = "datetime"

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(OUT_CSV)
    log.info("ROI-mean CSV: %s (%d scenes, %d columns)", OUT_CSV, len(df_all), len(df_all.columns))
    log.info("Columns: %s", list(df_all.columns))
    log.info(
        "Done. Site: %s | enriched stacks in %s | CSV at %s",
        SITE_NAME, OUT_STACK_DIR, OUT_CSV,
    )


if __name__ == "__main__":
    main()
