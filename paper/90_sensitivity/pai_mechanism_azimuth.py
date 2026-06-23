"""
Stage 90 — Mechanism of the hinge/hemi residual (issue #6): footprint vs azimuth bias.

The subsample test ruled out shot COUNT. The Pgap(theta) diagnostic showed the scans
sit at different effective zeniths (hinge ~59 deg, hemi ~57.4 deg) AND differ ~2x even
at matched zenith. That residual could be:
  (a) real footprint / leveling geometry, or
  (b) azimuth-sampling bias -- the hemi raster does not fill azimuth uniformly at a
      given zenith, so its azimuth-averaged Pgap weights different compass directions.

Test: restrict BOTH scans to a narrow zenith band, bin azimuth finely, and compare Pgap
per azimuth bin where BOTH scans are well-sampled.
  * If hinge ~= hemi cell-by-cell -> the aggregate 2x was azimuth-sampling bias (b).
  * If hinge systematically > hemi across cells -> real footprint/leveling (a).

Run:
  /home/lk1167/miniconda3/envs/dehar-spac/bin/python paper/90_sensitivity/pai_mechanism_azimuth.py
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parents[0] / "config"))
sys.path.insert(0, str(ROOT / "lib" / "pylidar-tls-canopy"))
from paper_common import paper_style  # noqa: E402
from pylidar_tls_canopy import leaf_io, plant_profile  # noqa: E402

paper_style()

RAW_DIR = ROOT / "data" / "raw" / "proximal_rs" / "leaf"
LEAF_DIR = ROOT / "data" / "processed" / "proximal_rs" / "leaf"
OUT_DIR = HERE / "outputs" / "appendix"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SENSOR_HEIGHT = 1.5
METHOD = "FIRSTLAST"
ZBAND = (58.5, 59.5)          # narrow zenith band where BOTH scans are dense
AZ_RES = 10.0                 # 36 azimuth bins of 10 deg
SEASON = ("2025-04-01", "2025-12-31")
# Jupp2009: one narrow zenith bin, fine azimuth -> per-azimuth cover before averaging.
JAZ = dict(min_z=ZBAND[0], max_z=ZBAND[1], zres=ZBAND[1] - ZBAND[0],
           ares=AZ_RES, min_h=0, max_h=25, hres=0.5)


def scan_az_profile(path: Path):
    """Accumulate one scan into the narrow-zenith, fine-azimuth grid."""
    with leaf_io.LeafScanFile(str(path), sensor_height=SENSOR_HEIGHT,
                              zenith_offset=0) as leaf:
        d = leaf.data
        if d.empty:
            return None
        zen, azi = d["zenith"].to_numpy(), d["azimuth"].to_numpy()
        tc = d["target_count"].to_numpy()
        h1, h2 = d["h1"].to_numpy(), d["h2"].to_numpy()
        zen_deg = np.degrees(zen)
        m = (zen_deg >= ZBAND[0]) & (zen_deg < ZBAND[1])
        if m.sum() == 0:
            return None
        prof = plant_profile.Jupp2009(**JAZ)
        for n, h in enumerate((h1, h2), start=1):
            tidx = np.full(h.shape, n, dtype=np.uint8)
            ok = m & ~np.isnan(h)
            if np.any(ok):
                prof.add_targets(h[ok], tidx[ok], tc[ok], zen[ok], azi[ok],
                                 method=METHOD)
        prof.add_shots(tc[m], zen[m], azi[m], method=METHOD)
        return prof.target_output, prof.shot_output


def predawn_files(scan: str, hour: int) -> list[Path]:
    d = pd.read_parquet(LEAF_DIR / f"leaf_{scan}_2025.parquet")
    d = d[(d.scan_hour == hour) & d.quality_all]
    d["date"] = pd.to_datetime(d["datetime"]).dt.normalize().dt.tz_localize(None)
    d = d[(d.date >= SEASON[0]) & (d.date <= SEASON[1])]
    return [RAW_DIR / f for f in d.groupby("date")["filename"].first()]


def pooled_az_cover(paths):
    """Per-azimuth canopy-top cover and shot count, pooled across scans."""
    with ProcessPoolExecutor() as ex:
        out = [r for r in ex.map(scan_az_profile, paths) if r is not None]
    tgt = np.sum([t for t, _ in out], axis=0)        # (1, nA, nH)
    sht = np.sum([s for _, s in out], axis=0)         # (1, nA, 1)
    cover_top = np.divide(tgt.cumsum(axis=2)[:, :, -1], sht[:, :, 0],
                          out=np.full(sht[:, :, 0].shape, np.nan),
                          where=sht[:, :, 0] > 0)
    return cover_top[0], sht[0, :, 0]                 # (nA,), (nA,)


def main() -> None:
    hinge_p = predawn_files("hinge", 1)
    hemi_p = predawn_files("hemi_hi", 2)
    print(f"pooled scans: hinge={len(hinge_p)} hemi_hi={len(hemi_p)} "
          f"| zenith band {ZBAND} deg, azimuth {AZ_RES} deg bins")

    cov_h, n_h = pooled_az_cover(hinge_p)
    cov_e, n_e = pooled_az_cover(hemi_p)
    az = np.arange(0, 360, AZ_RES) + AZ_RES / 2

    pg_h, pg_e = 1 - cov_h, 1 - cov_e
    both = (n_h > 500) & (n_e > 500) & (pg_h > 0) & (pg_e > 0)
    print(f"azimuth cells well-sampled in BOTH: {both.sum()}/{len(az)}")

    # Aggregate Pgap two ways: azimuth-averaged (pipeline) vs matched-cell mean.
    pipe_h, pipe_e = np.nanmean(pg_h), np.nanmean(pg_e)
    cell_h, cell_e = np.nanmean(pg_h[both]), np.nanmean(pg_e[both])
    print(f"\npipeline-style (all azimuths):   hinge Pgap={pipe_h:.4f}  "
          f"hemi={pipe_e:.4f}  ratio={pipe_e/pipe_h:.2f}")
    print(f"matched azimuth cells only:      hinge Pgap={cell_h:.4f}  "
          f"hemi={cell_e:.4f}  ratio={cell_e/cell_h:.2f}")
    paiH = lambda p: -1.1 * np.log(np.clip(p, 1e-5, None))
    print(f"=> HingePAI matched cells:       hinge={paiH(cell_h):.2f}  "
          f"hemi={paiH(cell_e):.2f}")

    fig, (ax, axn) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                  gridspec_kw=dict(height_ratios=[3, 1]))
    ax.plot(az[both], pg_h[both], "-o", color="#1b7837", lw=2, ms=5, label="hinge scan")
    ax.plot(az[both], pg_e[both], "-o", color="#762a83", lw=2, ms=5, label="hemi_hi scan")
    ax.set(ylabel="Pgap at canopy top", yscale="log",
           title=f"Pgap per azimuth cell at matched zenith {ZBAND} deg "
                 f"(parallel curves => real geometry, not azimuth bias)")
    ax.legend(loc="upper right", fontsize=9)
    axn.semilogy(az, n_h, color="#1b7837")
    axn.semilogy(az, n_e, color="#762a83")
    axn.set(ylabel="shots/cell", xlabel="azimuth (deg)")
    fig.tight_layout()
    out = OUT_DIR / "pai_mechanism_azimuth_2025.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
