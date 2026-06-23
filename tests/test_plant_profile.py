"""Method-correctness tests for the LEAF PAI inversion (pylidar-tls-canopy).

Audit of GitHub issue #6: lock the Jupp (2009) plant-profile maths against silent
regressions. These are pure, fixture-free checks on the vendored library:

* HingePAI = -1.1 * ln(Pgap) at the 57.5 deg ring, plus the log floor at Pgap<=0.
* LinearPAI recovers a known (PAIv, PAIh) model from synthetic Pgap(theta).
* The negative-slope fallback returns a defensible value, never a silent zero.
* WeightedPAI's canopy total equals the HingePAI total *by construction* (the
  solid-angle profile is rescaled to max(HingePAI)) — the invariant the audit relies on.
* The LEAF coordinate transform: zenith fold |zenith - pi| (all scans) and the
  180 deg azimuth flip applied to hemi scans only.

Run: pytest tests/test_plant_profile.py
"""
from __future__ import annotations

import numpy as np
import pytest
from pylidar_tls_canopy import leaf_io, plant_profile

HINGE_K = 1.1
LOG_FLOOR = np.log(1e-5)


def _profile(nh: int = 3) -> plant_profile.Jupp2009:
    """A Jupp2009 whose 57.5 deg ring exists; pgap_theta_z is set per test."""
    p = plant_profile.Jupp2009(min_z=5, max_z=70, zres=5, min_h=0, max_h=nh, hres=1)
    assert len(p.height_bin) == nh
    return p


def _hinge_index(p) -> int:
    return int(np.argmin(np.abs(np.radians(p.zenith_bin) - np.arctan(np.pi / 2))))


def _xtheta(p) -> np.ndarray:
    return np.abs(2 * np.tan(np.radians(p.zenith_bin)) / np.pi)


# ── HingePAI ──────────────────────────────────────────────────────────────────
def test_hinge_index_is_575_bin():
    p = _profile()
    assert p.zenith_bin[_hinge_index(p)] == pytest.approx(57.5)


def test_hinge_pai_known_pgap():
    p = _profile()
    p.pgap_theta_z = np.full((len(p.zenith_bin), len(p.height_bin)), 0.7)
    p.pgap_theta_z[_hinge_index(p), :] = 0.5            # only the hinge ring is read
    pai = p.calcHingePlantProfiles()
    assert pai == pytest.approx(-HINGE_K * np.log(0.5))  # ~0.7625 at every height


def test_hinge_log_floor_on_zero_pgap():
    p = _profile()
    p.pgap_theta_z = np.full((len(p.zenith_bin), len(p.height_bin)), 0.5)
    p.pgap_theta_z[_hinge_index(p), 0] = 0.0     # closed canopy -> clamp, not -inf
    pai = p.calcHingePlantProfiles()
    assert np.isfinite(pai).all()
    assert pai[0] == pytest.approx(-HINGE_K * LOG_FLOOR)


# ── LinearPAI ─────────────────────────────────────────────────────────────────
def test_linear_pai_recovers_known_model():
    p = _profile()
    paiv, paih = 2.0, 0.5
    y = paiv * _xtheta(p) + paih                         # -ln(Pgap) = PAIv*x + PAIh
    pgap = np.exp(-y)[:, None] * np.ones(len(p.height_bin))
    p.pgap_theta_z = pgap
    pai = p.calcLinearPlantProfiles()
    assert pai == pytest.approx(paiv + paih, abs=1e-4)


def test_linear_negative_slope_fallback_is_not_zero():
    p = _profile()
    y = -1.0 * _xtheta(p) + 3.0                          # deliberately negative slope
    pgap = np.exp(-y)[:, None] * np.ones(len(p.height_bin))
    p.pgap_theta_z = pgap
    pai = p.calcLinearPlantProfiles()
    assert (pai > 0).all()                               # not a silent zero
    assert pai == pytest.approx(np.mean(y), abs=1e-4)    # fallback = mean(-ln Pgap)


# ── WeightedPAI invariant ─────────────────────────────────────────────────────
def test_weighted_total_equals_hinge_total():
    p = _profile(nh=4)
    rng = np.linspace(0.9, 0.02, len(p.height_bin))      # gap closes upward
    p.pgap_theta_z = np.tile(rng, (len(p.zenith_bin), 1))
    hinge_total = float(np.max(p.calcHingePlantProfiles()))
    weighted_total = float(np.max(p.calcSolidAnglePlantProfiles()))
    assert weighted_total == pytest.approx(hinge_total, rel=1e-5)


# ── LEAF coordinate transform ────────────────────────────────────────────────
_HDR = "# Firmware ver.: 4.11\n# Tilt: [0, 0, 1024]\n"
# scan_encoder for raw zenith ~100 deg (folds to 80) and ~200 deg (folds to 20);
# rotary_encoder 5000 -> azimuth 90 deg. Columns: sample_count, scan_encoder,
# rotary_encoder, range1, intensity1, range2, intensity2, sample_time.
_ROWS = "1,7111,5000,5.0,100,-1.0,-1,10\n2,14222,5000,5.0,100,-1.0,-1,10\n"


def _write_scan(tmp_path, scan_type: str):
    name = f"ESS00320_0001_{scan_type}_20250101-000000Z_0800_0400.csv"
    fp = tmp_path / name
    fp.write_text(_HDR + _ROWS)
    return fp


def test_zenith_fold_applies_to_all_scans(tmp_path):
    for scan_type in ("hemi", "hinge"):
        leaf = leaf_io.LeafScanFile(str(_write_scan(tmp_path, scan_type)),
                                    sensor_height=1.5, zenith_offset=0)
        zen = np.degrees(leaf.data["zenith"].to_numpy())
        assert zen == pytest.approx([80.0, 20.0], abs=0.2)   # |zenith - 180|


def test_hemi_azimuth_flip_only_for_hemi(tmp_path):
    hemi = leaf_io.LeafScanFile(str(_write_scan(tmp_path, "hemi")),
                                sensor_height=1.5, zenith_offset=0)
    hinge = leaf_io.LeafScanFile(str(_write_scan(tmp_path, "hinge")),
                                 sensor_height=1.5, zenith_offset=0)
    az_hemi = np.degrees(hemi.data["azimuth"].to_numpy())
    az_hinge = np.degrees(hinge.data["azimuth"].to_numpy())
    # raw azimuth 90 deg both; the <pi-hemisphere shot is flipped 180 deg for hemi only
    assert az_hemi == pytest.approx([270.0, 90.0], abs=0.2)
    assert az_hinge == pytest.approx([90.0, 90.0], abs=0.2)


def test_rza_xyz_roundtrip():
    r, theta, phi = 7.3, np.radians(35.0), np.radians(120.0)
    x, y, z = leaf_io.rza2xyz(r, theta, phi)
    r2, theta2, phi2 = leaf_io.xyz2rza(x, y, z)
    assert (r2, theta2) == pytest.approx((r, theta))
