"""
Download Sentinel-1 GRD stack from Google Earth Engine.

Site:   DE-Har flux tower
Buffer: 1 km around tower → ~2×2 km bounding box
Period: Full Sentinel-1 archive (2014-10-03 → present)

Preprocessing (GEE applies by default for S1_GRD):
  - Thermal noise removal
  - Radiometric calibration (σ⁰ backscatter)
  - Terrain correction (SRTM 30 m)
  → values are in dB (10·log10 σ⁰)

Additional filtering:
  - IW mode only (standard over land in Europe)
  - VV + VH dual-pol
  - Single relative orbit per pass direction (consistent geometry)
  - Incidence angle normalisation to reference angle
  - Light speckle filter (focal median, 15 m)

Output: one NetCDF per year × orbit direction with VV, VH, VH/VV, angle
        at 10 m resolution

Requirements:
  pip install earthengine-api geemap xarray netcdf4 numpy
  ee.Authenticate()  # run once interactively
"""

import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

import ee
import numpy as np
import xarray as xr
import xee  # noqa: F401 — registers the 'ee' xarray backend

from dehar.utils.constants import SITE_LAT, SITE_LON, SITE_CRS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────
GEE_PROJECT = "hartheim" 

BUFFER_M = 1000
SCALE = 10
REF_ANGLE = 38.0  # reference incidence angle for normalisation [°]
START_DATE = "2014-10-03"
END_DATE = datetime.now().strftime("%Y-%m-%d")
OUT_DIR = Path("data/raw/satellite/sentinel1")

EXPORT_BANDS = ["VV", "VH", "VH_VV_ratio", "angle"]


# ── GEE helpers ───────────────────────────────────────────────
def _init_gee():
    """Initialise GEE, authenticating only if needed.

    If credentials already exist (from a previous ``earthengine
    authenticate`` or ``ee.Authenticate()``), skip the browser flow.
    Uses only the earthengine + cloud-platform scopes to avoid the
    "app blocked" error triggered by the deprecated drive scope.
    """
    _SCOPES = [
        "https://www.googleapis.com/auth/earthengine",
        "https://www.googleapis.com/auth/cloud-platform",
    ]
    try:
        ee.Initialize(project=GEE_PROJECT, opt_url=None)
        return
    except Exception:
        pass
    ee.Authenticate(
        auth_mode="notebook",
        scopes=_SCOPES,
    )
    ee.Initialize(project=GEE_PROJECT)


def _build_roi():
    tower = ee.Geometry.Point([SITE_LON, SITE_LAT])
    return tower.buffer(BUFFER_M).bounds()


def _speckle_filter(img):
    """Focal median (15 m circle) — preserves per-pixel signal."""
    vv = img.select("VV").focalMedian(15, "circle", "meters").rename("VV")
    vh = img.select("VH").focalMedian(15, "circle", "meters").rename("VH")
    return img.addBands(vv, overwrite=True).addBands(vh, overwrite=True)


def _angle_normalise(img):
    """
    Normalise σ⁰ to a reference incidence angle (cosine-ratio model).
    In dB: σ⁰_norm = σ⁰ + 20·log10(cos θ_ref / cos θ_local)

    Both ref and local angle are converted to Image operations so GEE
    can compute the per-pixel correction.
    """
    angle_rad = img.select("angle").multiply(np.pi / 180)
    ref_cos = ee.Image.constant(np.cos(REF_ANGLE * np.pi / 180))
    correction = ref_cos.divide(angle_rad.cos()).log10().multiply(20)
    vv_norm = img.select("VV").add(correction).rename("VV")
    vh_norm = img.select("VH").add(correction).rename("VH")
    return (
        img.addBands(vv_norm, overwrite=True)
        .addBands(vh_norm, overwrite=True)
    )


def _add_ratio(img):
    """VH/VV ratio in dB (= VH_dB − VV_dB). Sensitive to vegetation."""
    ratio = (
        img.select("VH").subtract(img.select("VV")).rename("VH_VV_ratio")
    )
    return img.addBands(ratio)


def _get_dominant_orbit(roi, orbit_pass):
    """Find the most frequent relative orbit number for this pass."""
    col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(roi)
        .filterDate(START_DATE, END_DATE)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
    )
    orbits = col.aggregate_array("relativeOrbitNumber_start").getInfo()
    counts = Counter(orbits)
    dominant, n_scenes = counts.most_common(1)[0]
    log.info(
        "  %s: orbit %d selected (%d scenes, all orbits: %s)",
        orbit_pass, dominant, n_scenes, dict(counts),
    )
    return dominant


def _build_s1_collection(roi, orbit_pass, rel_orbit):
    """Build filtered + preprocessed S1 collection for one orbit."""
    col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(roi)
        .filterDate(START_DATE, END_DATE)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains(
            "transmitterReceiverPolarisation", "VV",
        ))
        .filter(ee.Filter.listContains(
            "transmitterReceiverPolarisation", "VH",
        ))
        .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
        .filter(ee.Filter.eq("relativeOrbitNumber_start", rel_orbit))
        .filter(ee.Filter.eq("resolution_meters", 10))
    )
    return col.map(_angle_normalise).map(_speckle_filter).map(_add_ratio)


# ── Main pipeline ─────────────────────────────────────────────
def main():
    log.info("Initialising Google Earth Engine")
    _init_gee()

    roi = _build_roi()
    log.info(
        "ROI: %.6f°N %.6f°E (WGS84), %d m buffer, output CRS %s",
        SITE_LAT, SITE_LON, BUFFER_M, SITE_CRS,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for orbit_pass in ("ASCENDING", "DESCENDING"):
        rel_orbit = _get_dominant_orbit(roi, orbit_pass)
        col = _build_s1_collection(roi, orbit_pass, rel_orbit)
        n_total = col.size().getInfo()
        log.info(
            "%s (orbit %d): %d scenes total",
            orbit_pass, rel_orbit, n_total,
        )

        tag = orbit_pass[0].lower()  # 'a' or 'd'

        for yr in range(2014, datetime.now().year + 1):
            t0 = f"{yr}-01-01"
            t1 = f"{yr + 1}-01-01"
            col_yr = col.filterDate(t0, t1).select(EXPORT_BANDS)
            n_yr = col_yr.size().getInfo()
            if n_yr == 0:
                continue

            out_path = OUT_DIR / f"s1_grd_dehar_{tag}_{yr}.nc"
            log.info("  %s %d: %d scenes → %s", orbit_pass, yr, n_yr, out_path)

            try:
                ds = xr.open_dataset(
                    col_yr,
                    engine="ee",
                    scale=SCALE,
                    crs=SITE_CRS,
                    geometry=roi,
                )
                ds.to_netcdf(out_path)
                size_mb = out_path.stat().st_size / 1e6
                log.info("    saved (%.1f MB)", size_mb)
            except Exception:
                log.exception("    download failed for %s %d", orbit_pass, yr)

    log.info("Done. Outputs in %s", OUT_DIR)


if __name__ == "__main__":
    main()
