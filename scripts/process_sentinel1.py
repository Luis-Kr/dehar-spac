"""
Process raw Sentinel-1 NetCDF stacks → enriched stacks + ROI-mean CSV.

**Spatial extent**

- **NetCDF**: Full GEE export grid (~1 km buffer). Kept for spatial analysis.

- **CSV**: Mean **only** over pixels within ``BUFFER_M`` metres of the tower
  (UTM). The full scene is **not** averaged into the CSV.

Coordinates must be projected metres (same ``SITE_CRS`` as GEE export); see
:func:`_assert_projected_coords`.

Optional ``STACK_CROP_MARGIN_M`` crops saved stacks to tower ± margin.

Gap-filling / smoothing: **none by default**.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from dehar.utils.constants import SITE_CRS, SITE_NAME, SITE_UTM_X, SITE_UTM_Y
from dehar.satellite.sentinel1 import add_all_indices, ALL_INDICES

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────
RAW_DIR = Path("data/raw/satellite/sentinel1")
OUT_STACK_DIR = Path("data/processed/satellite/sentinel1")
OUT_CSV = Path("data/processed/satellite/sentinel1_roi_mean.csv")

ROI_X = SITE_UTM_X
ROI_Y = SITE_UTM_Y
BUFFER_M = 100

STACK_CROP_MARGIN_M: float | None = None

# CSV output columns (angle is metadata, kept for QC)
INDEX_NAMES = list(ALL_INDICES.keys())
CSV_COLUMNS = INDEX_NAMES + ["SPAN", "angle"]
# Deduplicate while preserving order
CSV_COLUMNS = list(dict.fromkeys(CSV_COLUMNS))


def _normalize_dims(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    if "X" in ds.dims and "x" not in ds.dims:
        rename["X"] = "x"
    if "Y" in ds.dims and "y" not in ds.dims:
        rename["Y"] = "y"
    return ds.rename(rename) if rename else ds


def _assert_projected_coords(ds: xr.Dataset) -> None:
    x = ds.x.values
    y = ds.y.values
    xm, ym = float(np.nanmean(x)), float(np.nanmean(y))
    if np.nanmax(np.abs(x)) < 400 and np.nanmax(np.abs(y)) < 90:
        raise ValueError(
            "Dataset x/y look like geographic degrees, not UTM metres. "
            "ROI buffer would be invalid."
        )
    if not (1e5 < abs(xm) < 1e6 and 1e6 < abs(ym) < 1e7):
        log.warning(
            "Unexpected x/y (mean x=%.1f, y=%.1f) — verify CRS %s",
            xm,
            ym,
            SITE_CRS,
        )


def _crop_to_margin(ds: xr.Dataset, cx: float, cy: float, margin_m: float) -> xr.Dataset:
    x0, x1 = cx - margin_m, cx + margin_m
    y_lo, y_hi = cy - margin_m, cy + margin_m
    y_coord = ds.y.values
    if y_coord[0] > y_coord[-1]:
        y_slice = slice(y_hi, y_lo)
    else:
        y_slice = slice(y_lo, y_hi)
    return ds.sel(x=slice(x0, x1), y=y_slice)


def _roi_mask(ds: xr.Dataset) -> xr.DataArray:
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
    _assert_projected_coords(ds)
    log.info("  %s: %d scenes, %d vars", nc_path.name, n_time, len(ds.data_vars))
    log.info(
        "    grid: x [%.0f … %.0f] m, y [%.0f … %.0f] m",
        float(ds.x.min()),
        float(ds.x.max()),
        float(ds.y.min()),
        float(ds.y.max()),
    )
    return ds


def _roi_mean_timeseries(ds: xr.Dataset, mask: xr.DataArray) -> pd.DataFrame:
    ds_roi = ds.where(mask)
    cols = [c for c in CSV_COLUMNS if c in ds_roi]
    mean = ds_roi[cols].mean(dim=["x", "y"]).compute()
    df = mean.to_dataframe().reset_index()
    if "time" in df.columns:
        df = df.rename(columns={"time": "datetime"}).set_index("datetime")
    return df


# ── Main pipeline ─────────────────────────────────────────────
def main():
    log.info("ROI centre: (%.1f, %.1f) %s, buffer %d m", ROI_X, ROI_Y, SITE_CRS, BUFFER_M)

    OUT_STACK_DIR.mkdir(parents=True, exist_ok=True)
    all_dfs: list[pd.DataFrame] = []

    for orbit_tag, orbit_label in [("a", "ASCENDING"), ("d", "DESCENDING")]:
        pattern = f"s1_grd_dehar_{orbit_tag}_*.nc"
        nc_files = sorted(RAW_DIR.glob(pattern))
        if not nc_files:
            log.warning("No %s files matching %s/%s", orbit_label, RAW_DIR, pattern)
            continue

        log.info("%s: %d yearly files", orbit_label, len(nc_files))

        for nc in nc_files:
            ds = _process_one_file(nc)
            if ds is None:
                continue

            # Step 1: enriched stack
            out_nc = OUT_STACK_DIR / nc.name.replace(".nc", "_indices.nc")
            ds.to_netcdf(out_nc)
            size_mb = out_nc.stat().st_size / 1e6
            log.info("    → %s (%.1f MB)", out_nc, size_mb)

            # Step 2: ROI mean
            mask = _roi_mask(ds)
            n_pixels = int(mask.sum())
            log.info("    ROI pixels: %d", n_pixels)
            df = _roi_mean_timeseries(ds, mask)
            df["orbit"] = orbit_label
            all_dfs.append(df)

            ds.close()

    if not all_dfs:
        log.warning("No data processed, skipping CSV output")
        return

    df_all = pd.concat(all_dfs).sort_index()
    df_all.index.name = "datetime"

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(OUT_CSV)
    log.info(
        "ROI-mean CSV: %s (%d scenes, %d columns)",
        OUT_CSV, len(df_all), len(df_all.columns),
    )
    log.info("Columns: %s", list(df_all.columns))

    n_asc = (df_all["orbit"] == "ASCENDING").sum()
    n_desc = (df_all["orbit"] == "DESCENDING").sum()
    log.info("  ASCENDING: %d scenes, DESCENDING: %d scenes", n_asc, n_desc)

    log.info(
        "Done. Site: %s | enriched stacks in %s | CSV at %s",
        SITE_NAME, OUT_STACK_DIR, OUT_CSV,
    )


if __name__ == "__main__":
    main()
