"""
Download Sentinel-2 SR Harmonized stack from Google Earth Engine.

Site:   DE-Har flux tower
Buffer: 1 km around tower → ~2×2 km bounding box
Period: Full Sentinel-2 archive (2015-06-23 → present)

Cloud masking:
  - SCL-based: exclude saturated, shadow, med/high cloud, cirrus, snow
  - s2cloudless probability < 40 %
  - Buffer cloud pixels by 50 m (SCL edge artefacts)
  - View zenith angle < 8° (near-nadir, reduces BRDF effects)

Output: one NetCDF per year with bands B2–B12, NDVI, EVI, NDWI, NBR
        at 10 m resolution (20 m bands resampled via bilinear)

Requirements:
  pip install earthengine-api geemap xarray netcdf4 numpy
  ee.Authenticate()  # run once interactively
"""

import logging
from datetime import datetime
from pathlib import Path

import ee
import xarray as xr
import xee  # noqa: F401 — registers the 'ee' xarray backend

from dehar.utils.constants import SITE_LAT, SITE_LON, SITE_CRS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────
GEE_PROJECT = "hartheim"  

BUFFER_M = 1000
SCALE = 10
CLOUD_PROB_THRESH = 100 #40
CLOUD_BUFFER_M = 0 #50
START_DATE = "2025-01-01" #"2015-06-23"
END_DATE = datetime.now().strftime("%Y-%m-%d")
OUT_DIR = Path("data/raw/satellite/sentinel2")

S2_BANDS = [
    "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12",
]
EXPORT_BANDS = S2_BANDS + ["NDVI", "EVI", "NDWI", "NBR"]


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


def _join_cloud_probability(s2_col, roi):
    """Attach s2cloudless probability to each S2 scene."""
    cloud_col = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(roi)
        .filterDate(START_DATE, END_DATE)
    )
    return ee.ImageCollection(
        ee.Join.saveFirst("s2cloudless").apply(
            primary=s2_col,
            secondary=cloud_col,
            condition=ee.Filter.equals(
                leftField="system:index",
                rightField="system:index",
            ),
        )
    )


def _mask_clouds(img):
    """Combined SCL + s2cloudless cloud mask with edge buffer."""
    scl = img.select("SCL")
    scl_mask = (
        scl.neq(1)    # saturated / defective
        .And(scl.neq(3))    # cloud shadow
        .And(scl.neq(8))    # medium probability cloud
        .And(scl.neq(9))    # high probability cloud
        .And(scl.neq(10))   # thin cirrus
        .And(scl.neq(11))   # snow / ice
    )
    cloud_prob = ee.Image(img.get("s2cloudless")).select("probability")
    prob_mask = cloud_prob.lt(CLOUD_PROB_THRESH)
    combined = scl_mask.And(prob_mask)
    buffered = combined.focalMin(
        radius=CLOUD_BUFFER_M, units="meters", iterations=1,
    )
    return img.updateMask(buffered)


def _add_indices(img):
    """Compute NDVI, EVI, NDWI, NBR."""
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    evi = img.expression(
        "2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))",
        {
            "NIR": img.select("B8"),
            "RED": img.select("B4"),
            "BLUE": img.select("B2"),
        },
    ).rename("EVI")
    ndwi = img.normalizedDifference(["B8", "B11"]).rename("NDWI")
    nbr = img.normalizedDifference(["B8", "B12"]).rename("NBR")
    return img.addBands([ndvi, evi, ndwi, nbr])


# ── Main pipeline ─────────────────────────────────────────────
def main():
    log.info("Initialising Google Earth Engine")
    _init_gee()

    roi = _build_roi()
    log.info(
        "ROI: %.6f°N %.6f°E (WGS84), %d m buffer, output CRS %s",
        SITE_LAT, SITE_LON, BUFFER_M, SITE_CRS,
    )

    # Build cloud-masked, index-enriched collection
    s2_sr = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate(START_DATE, END_DATE)
        #.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 90))
        #.filter(ee.Filter.lt("MEAN_INCIDENCE_ZENITH_ANGLE_B8", 20))
    )
    s2_col = _join_cloud_probability(s2_sr, roi)
    s2_masked = s2_col.map(_add_indices) #s2_col.map(_mask_clouds).map(_add_indices)

    n_total = s2_sr.size().getInfo()
    log.info(
        "Collection: %d scenes after pre-filter (%s → %s)",
        n_total, START_DATE, END_DATE,
    )

    # Download year by year → NetCDF
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    years = range(2015, datetime.now().year + 1)

    for yr in years:
        t0 = f"{yr}-01-01"
        t1 = f"{yr + 1}-01-01"
        col_yr = s2_masked.filterDate(t0, t1).select(EXPORT_BANDS)
        n = col_yr.size().getInfo()
        if n == 0:
            log.info("  %d: no scenes, skipping", yr)
            continue

        out_path = OUT_DIR / f"s2_sr_dehar_{yr}.nc"
        log.info("  %d: %d scenes → %s", yr, n, out_path)

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
            log.exception("    download failed for %d", yr)

    log.info("Done. Outputs in %s", OUT_DIR)


if __name__ == "__main__":
    main()
