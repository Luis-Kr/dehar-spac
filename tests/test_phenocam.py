"""Data-integrity tests for the PhenoCam summary-product loader.

Focus (per the project testing rules): the raw PhenoCam CSV is parsed correctly,
canonical columns are selected/renamed without corruption, missing values stay
``NaN`` (never fabricated), the index is daily UTC, and malformed inputs fail loud.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dehar.proximal_rs import phenocam

# Minimal PhenoCam-format fixtures: a ``#`` comment header, then the columns the
# loader selects, plus extra columns it must ignore. Row 2 carries an ``NA`` to
# verify missing values are preserved as NaN.
_GCC_CSV = """\
# 1-day summary product timeseries for hartheim2
# Veg Type: UN
date,year,doy,image_count,gcc_mean,gcc_std,gcc_90,smooth_gcc_90,rcc_90,snow_flag
2025-08-06,2025,218,12,0.380,0.004,0.386,0.385,0.40,0
2025-08-07,2025,219,NA,NA,NA,NA,0.384,0.41,0
2025-08-08,2025,220,11,0.382,0.005,0.388,0.386,0.40,0
"""

_NDVI_CSV = """\
# 3-day NDVI summary timeseries for hartheim2
date,year,doy,image_count,ndvi_mean,ndvi_std,ndvi_90,midday_ndvi
2025-08-06,2025,218,9,0.70,0.02,0.74,0.71
2025-08-09,2025,221,8,0.69,0.02,0.73,0.70
"""


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    (tmp_path / f"{phenocam.ROI_ID}_1day.csv").write_text(_GCC_CSV)
    (tmp_path / f"{phenocam.ROI_ID}_ndvi_3day.csv").write_text(_NDVI_CSV)
    return tmp_path


def test_gcc_canonical_columns_and_rename(raw_dir: Path) -> None:
    out = phenocam.load_product(raw_dir, "gcc", "1day")
    assert list(out.columns) == [
        "gcc_phenocam_1day_p90", "gcc_phenocam_1day_mean", "gcc_phenocam_1day_std",
        "gcc_phenocam_1day_smooth_p90", "gcc_phenocam_1day_image_count",
    ]
    # Canonical p90 maps to the raw gcc_90 column, value-exact.
    assert out["gcc_phenocam_1day_p90"].iloc[0] == pytest.approx(0.386)
    # Ignored raw columns (rcc_90, snow_flag) are dropped.
    assert not any("rcc" in c or "snow" in c for c in out.columns)


def test_missing_values_preserved_not_filled(raw_dir: Path) -> None:
    out = phenocam.load_product(raw_dir, "gcc", "1day")
    # The NA row stays NaN (no interpolation/fabrication at the loader boundary).
    assert np.isnan(out["gcc_phenocam_1day_p90"].iloc[1])
    assert out["gcc_phenocam_1day_p90"].notna().sum() == 2


def test_index_is_daily_utc(raw_dir: Path) -> None:
    out = phenocam.load_product(raw_dir, "gcc", "1day")
    assert str(out.index.tz) == "UTC"
    assert out.index.name == "date"
    assert (out.index.normalize() == out.index).all()   # midnight-aligned
    assert out.index.is_monotonic_increasing


def test_3day_native_cadence_not_densified(raw_dir: Path) -> None:
    out = phenocam.load_product(raw_dir, "ndvi", "3day")
    # Two window dates 3 days apart — loader must NOT insert the gap day.
    assert len(out) == 2
    assert out.index[1] - out.index[0] == pd.Timedelta(days=3)
    assert out["ndvi_phenocam_3day_p90"].iloc[0] == pytest.approx(0.74)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        phenocam.load_product(tmp_path, "gcc", "1day")


def test_bad_arguments_raise(raw_dir: Path) -> None:
    with pytest.raises(ValueError):
        phenocam.load_product(raw_dir, "evi", "1day")
    with pytest.raises(ValueError):
        phenocam.load_product(raw_dir, "gcc", "7day")
