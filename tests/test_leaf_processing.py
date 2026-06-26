"""Tests for the LEAF CLI pipeline (scripts/process_leaf.py + dehar.proximal_rs.leaf).

Two layers:

* Fixture-free unit tests for the pure helpers (datetime parsing, transform
  resolution, met context, the temporal-outlier flag).
* A faithfulness test proving the transform-explicit wrapper
  ``add_leaf_scan_to_profile(transform=True)`` is numerically identical to the
  upstream ``Jupp2009.add_leaf_scan_position`` it replaces. Skipped when no raw
  LEAF scan is present (``data/`` is gitignored), so CI stays green.

Run: pytest tests/test_leaf_processing.py
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dehar.proximal_rs import leaf
from pylidar_tls_canopy import plant_profile

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "proximal_rs" / "leaf"


# ── get_scan_datetime ───────────────────────────────────────────────────────
def test_get_scan_datetime_parses_filename():
    name = "ESS00320_0012_hinge_20250416-191027Z_0005_8500.csv"
    assert leaf.get_scan_datetime(name) == dt.datetime(2025, 4, 16, 19, 10, 27)


def test_get_scan_datetime_none_on_garbage():
    assert leaf.get_scan_datetime("not_a_scan.csv") is None


# ── transform_mode (the "transform command") ────────────────────────────────
def test_transform_mode_offset():
    assert (
        leaf.transform_mode({"enabled": True, "method": "leaf_tilt_offset"}) == "offset"
    )


def test_transform_mode_rotation():
    cfg = {"enabled": True, "method": "leaf_tilt_rotation"}
    assert leaf.transform_mode(cfg) == "rotation"


def test_transform_mode_disabled():
    assert (
        leaf.transform_mode({"enabled": False, "method": "leaf_tilt_offset"}) == "off"
    )


def test_transform_mode_unknown_method():
    with pytest.raises(ValueError):
        leaf.transform_mode({"enabled": True, "method": "bogus"})


# ── resolve_transform (2-knob tilt + up_drift; ADR 0005) ────────────────────
def test_resolve_transform_2knob():
    assert leaf.resolve_transform({"tilt": "rotation", "up_drift": True}) == {
        "tilt": "rotation",
        "up_drift": True,
    }
    assert leaf.resolve_transform({"tilt": "offset"}) == {
        "tilt": "offset",
        "up_drift": False,
    }


def test_resolve_transform_legacy_aliases():
    r = leaf.resolve_transform
    assert r({"enabled": True, "method": "leaf_tilt_offset"}) == {
        "tilt": "offset", "up_drift": False
    }
    assert r({"enabled": True, "method": "leaf_tilt_rotation"}) == {
        "tilt": "rotation", "up_drift": False
    }
    assert r({"enabled": True, "method": "leaf_seasonal_up"}) == {
        "tilt": "rotation", "up_drift": True
    }
    assert r({"enabled": False}) == {"tilt": "none", "up_drift": False}


def test_resolve_transform_bad_tilt():
    with pytest.raises(ValueError):
        leaf.resolve_transform({"tilt": "bogus"})


# ── leveling_rotation (issue #10 re-levelling) ──────────────────────────────
def test_leveling_rotation_identity_when_level():
    rot = leaf.leveling_rotation([0.0, 0.0, 1000.0])
    np.testing.assert_allclose(rot, np.eye(3), atol=1e-12)


def test_leveling_rotation_brings_up_vector_to_vertical():
    tilt = [28.9, -24.0, 1058.6]  # a real DE-Har Tilt header (~2 deg lean)
    rot = leaf.leveling_rotation(tilt)
    up = np.asarray(tilt) / np.linalg.norm(tilt)
    np.testing.assert_allclose(rot @ up, [0, 0, 1], atol=1e-12)
    # proper rotation: orthonormal, det +1
    np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rot) == pytest.approx(1.0)


# ── get_met_context ─────────────────────────────────────────────────────────
def _met_frame() -> pd.DataFrame:
    idx = pd.date_range("2025-06-01", periods=8, freq="30min", tz="UTC")
    return pd.DataFrame(
        {
            "tair_c": np.linspace(15, 22, 8),
            "rh_pct": [50, 50, 99, 99, 50, 50, 50, 50],
            "vpd_hpa": np.linspace(5, 9, 8),
            "ustar_ms": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1],
            "Precipitation_Sum_mm": [0, 0, 0, 0, 0, 0, 0, 0.0],
        },
        index=idx,
    )


QUALITY = {
    "precip_window_h": 3,
    "precip_thresh_mm": 0.1,
    "rh_thresh_pct": 95,
    "ustar_thresh_ms": 0.7,
}


def test_met_context_clean_scan_is_good():
    met = _met_frame()
    ctx = leaf.get_met_context(dt.datetime(2025, 6, 1, 0, 0), met, 30, QUALITY)
    assert ctx["quality_good"] is True
    assert ctx["tair_c"] == pytest.approx(met["tair_c"].iloc[:2].mean())


def test_met_context_flags_humidity():
    met = _met_frame()
    ctx = leaf.get_met_context(dt.datetime(2025, 6, 1, 1, 0), met, 30, QUALITY)
    assert ctx["flag_humid"] is True
    assert ctx["quality_good"] is False


def test_met_context_missing_returns_not_good():
    met = _met_frame()
    ctx = leaf.get_met_context(dt.datetime(2030, 1, 1, 0, 0), met, 30, QUALITY)
    assert ctx["quality_good"] is False
    assert np.isnan(ctx["tair_c"])


# ── temporal-outlier flag ───────────────────────────────────────────────────
def test_segment_bad_flags_downward_spike():
    # A gently varying baseline (non-degenerate MAD) with one sharp dip.
    series = 4.0 + 0.3 * np.sin(np.linspace(0, 2 * np.pi, 60))
    series[30] = 1.0  # sharp drop below the smooth trend
    bad = leaf._segment_bad(series, bp=[None], p=[None], max_k=[6], resid_thresh=-1.0)
    assert bad[30]
    assert not bad[10]


def test_segment_bad_flags_missing_days():
    series = np.full(60, 4.0)
    series[5] = np.nan
    bad = leaf._segment_bad(series, bp=[None], p=[None], max_k=[6], resid_thresh=-1.0)
    assert bad[5]


# ── faithfulness: wrapper ≡ upstream add_leaf_scan_position ──────────────────
def _a_scan() -> Path | None:
    files = sorted(RAW_DIR.glob("ESS?????_*_hinge_*.csv"))
    return files[0] if files else None


@pytest.mark.skipif(_a_scan() is None, reason="no raw LEAF scan available")
def test_wrapper_matches_upstream_when_transform_true():
    """transform=True must reproduce the pinned upstream result exactly."""
    scan = str(_a_scan())
    kw = dict(method="FIRSTLAST", min_zenith=5, max_zenith=70, sensor_height=1.5)

    ref = plant_profile.Jupp2009(
        hres=0.5, zres=5, ares=45, min_z=5, max_z=70, min_h=0, max_h=25
    )
    ref.add_leaf_scan_position(scan, zenith_offset=0, **kw)
    ref.get_pgap_theta_z()

    ours = plant_profile.Jupp2009(
        hres=0.5, zres=5, ares=45, min_z=5, max_z=70, min_h=0, max_h=25
    )
    leaf.add_leaf_scan_to_profile(
        ours,
        scan,
        transform_cfg={"enabled": True, "method": "leaf_tilt_offset"},
        zenith_offset=0,
        max_range=120,
        **kw,
    )
    ours.get_pgap_theta_z()

    np.testing.assert_array_equal(ours.pgap_theta_z, ref.pgap_theta_z)
    pd.testing.assert_frame_equal(ours.exportPlantProfiles(), ref.exportPlantProfiles())


@pytest.mark.skipif(_a_scan() is None, reason="no raw LEAF scan available")
def test_rotation_recenters_hinge_zenith_vs_offset():
    """On the hinge scan, rotation must re-level the measured up-vector and pull
    the mean effective zenith back below the offset's uniformly biased value."""
    from pylidar_tls_canopy import leaf_io

    scan = str(_a_scan())

    off = leaf_io.LeafScanFile(scan, sensor_height=1.5, transform=True)
    rot = leaf_io.LeafScanFile(scan, sensor_height=1.5, transform=False)
    leaf.level_leaf_data(rot)

    band = lambda d: (np.degrees(d) >= 55) & (np.degrees(d) < 60)  # noqa: E731
    z_off = off.data["zenith"].to_numpy()
    z_rot = rot.data["zenith"].to_numpy()
    mean_off = np.degrees(z_off[band(z_off)]).mean()
    mean_rot = np.degrees(z_rot[band(z_rot)]).mean()

    # The uniform offset biases the hinge ring up; rotation re-centres it lower.
    assert mean_rot < mean_off
    # ... by roughly the ~2 deg tilt, not a tiny rounding difference.
    assert (mean_off - mean_rot) > 0.5


# ── up-drift composition (ADR 0005) ─────────────────────────────────────────
def _a_hemi_scan() -> Path | None:
    files = sorted(RAW_DIR.glob("ESS?????_*_hemi_*_0800_0400.csv"))
    return files[len(files) // 2] if files else None


def _hinge_pai(scan: str, transform_cfg: dict, up_deg: float | None) -> float:
    v = plant_profile.Jupp2009(
        hres=0.5, zres=5, ares=45, min_z=5, max_z=70, min_h=0, max_h=25
    )
    leaf.add_leaf_scan_to_profile(
        v, scan, method="FIRSTLAST", min_zenith=5, max_zenith=70, sensor_height=1.5,
        zenith_offset=0, transform_cfg=transform_cfg, max_range=120, up_deg=up_deg,
    )
    v.get_pgap_theta_z()
    return float(v.exportPlantProfiles()["HingePAI"].max())


@pytest.mark.skipif(_a_scan() is None, reason="no raw LEAF scan available")
def test_hinge_not_refolded_by_up_drift():
    """The hinge scan is tilt-rotated only; up_drift must NOT change it (ADR 0005)."""
    scan = str(_a_scan())  # _a_scan() globs a hinge scan
    rot = _hinge_pai(scan, {"tilt": "rotation", "up_drift": False}, None)
    up = _hinge_pai(scan, {"tilt": "rotation", "up_drift": True}, 204.0)
    assert up == pytest.approx(rot)


@pytest.mark.skipif(_a_hemi_scan() is None, reason="no raw hemi scan available")
def test_hemi_refolded_by_up_drift():
    """A hemi scan IS re-folded by up_drift (re-fold about a non-180 up changes PAI)."""
    scan = str(_a_hemi_scan())
    off = _hinge_pai(scan, {"tilt": "rotation", "up_drift": False}, None)
    on = _hinge_pai(scan, {"tilt": "rotation", "up_drift": True}, 200.0)
    assert abs(on - off) > 1e-3
