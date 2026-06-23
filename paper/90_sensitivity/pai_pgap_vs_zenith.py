"""
Stage 90 — Where does the hinge/hemi difference come from? (issue #6 diagnostic)

Not shot count (the subsample test settled that). This asks: at the SAME zenith,
do the hinge scan and the hemi_hi scan measure the same empirical gap fraction?

For matched predawn days, pool ring-ish shots (50-65 deg) and compute the empirical
Pgap = mean(target_count == 0) in fine 0.5 deg zenith sub-bands, separately per scan.

  * If the two curves overlap at equal zenith -> the headline difference is which
    EFFECTIVE ANGLE each scan occupies inside the 5 deg bin (geometry), not footprint.
  * If they stay apart at equal zenith -> footprint / leveling, not just angle.

Also reports each scan's effective zenith distribution inside the 55-60 deg bin.

Run:
  /home/lk1167/miniconda3/envs/dehar-spac/bin/python paper/90_sensitivity/pai_pgap_vs_zenith.py
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
ZEN_LO, ZEN_HI = 50.0, 65.0          # wide window so the scans overlap in zenith
ZRES = 1.0                           # fine zenith bins (vs the canonical 5 deg)
# Fine-zenith Jupp2009: real cover algorithm, 1 deg bins, full height range.
JFINE = dict(min_z=ZEN_LO, max_z=ZEN_HI, zres=ZRES, ares=45, min_h=0, max_h=25, hres=0.5)
BIN_CENT = np.arange(ZEN_LO, ZEN_HI, ZRES) + ZRES / 2
RING = (55.0, 60.0)                  # the canonical hinge bin (for effective-angle report)
PEAK = ("2025-04-01", "2025-12-31")  # full season pooled (Pgap(theta) shape)


RING_EDGES = np.arange(RING[0], RING[1] + 0.1, 0.1)   # 0.1 deg ring zenith histogram


def scan_profile(path: Path):
    """Accumulate ONE scan into a fine-zenith Jupp2009; return its raw arrays.

    Returns (target_output, shot_output, ring_zenith_hist). The first two are
    summed across days to pool the real cover algorithm; the histogram recovers
    the effective zenith inside the 55-60 deg bin.
    """
    with leaf_io.LeafScanFile(str(path), sensor_height=SENSOR_HEIGHT,
                              zenith_offset=0) as leaf:
        d = leaf.data
        if d.empty:
            return None
        zen = d["zenith"].to_numpy()
        azi = d["azimuth"].to_numpy()
        tc = d["target_count"].to_numpy()
        h1, h2 = d["h1"].to_numpy(), d["h2"].to_numpy()
        zen_deg = np.degrees(zen)
        m = (zen_deg >= ZEN_LO) & (zen_deg < ZEN_HI)
        if m.sum() == 0:
            return None

        prof = plant_profile.Jupp2009(**JFINE)
        for n, h in enumerate((h1, h2), start=1):
            tidx = np.full(h.shape, n, dtype=np.uint8)
            ok = m & ~np.isnan(h)
            if np.any(ok):
                prof.add_targets(h[ok], tidx[ok], tc[ok], zen[ok], azi[ok],
                                 method=METHOD)
        prof.add_shots(tc[m], zen[m], azi[m], method=METHOD)

        ring = (zen_deg >= RING[0]) & (zen_deg < RING[1])
        hist, _ = np.histogram(zen_deg[ring], bins=RING_EDGES)
        return prof.target_output, prof.shot_output, hist


def predawn_files(scan: str, hour: int) -> pd.Series:
    d = pd.read_parquet(LEAF_DIR / f"leaf_{scan}_2025.parquet")
    d = d[(d.scan_hour == hour) & d.quality_all]
    d["date"] = pd.to_datetime(d["datetime"]).dt.normalize().dt.tz_localize(None)
    d = d[(d.date >= PEAK[0]) & (d.date <= PEAK[1])]
    return d.groupby("date")["filename"].first()


def pooled_pgap(scan_files):
    """Pool all scans into one fine-zenith profile; return Pgap(theta) + ring stats."""
    paths = [RAW_DIR / f for f in scan_files]
    with ProcessPoolExecutor() as ex:
        out = [r for r in ex.map(scan_profile, paths) if r is not None]
    tgt = np.sum([t for t, _, _ in out], axis=0)
    sht = np.sum([s for _, s, _ in out], axis=0)
    hist = np.sum([h for _, _, h in out], axis=0)

    prof = plant_profile.Jupp2009(**JFINE)
    prof.target_output, prof.shot_output = tgt, sht
    prof.get_pgap_theta_z()
    pgap_top = prof.pgap_theta_z[:, -1]            # canopy-total Pgap per zenith bin
    shots_per_bin = sht[:, :, 0].sum(axis=1)
    return pgap_top, shots_per_bin, hist


def ring_effective_zenith(hist) -> dict:
    cent = (RING_EDGES[:-1] + RING_EDGES[1:]) / 2
    c = np.cumsum(hist) / hist.sum()
    mean = float((cent * hist).sum() / hist.sum())
    p10 = float(cent[np.searchsorted(c, 0.10)])
    med = float(cent[np.searchsorted(c, 0.50)])
    p90 = float(cent[np.searchsorted(c, 0.90)])
    return dict(n=int(hist.sum()), mean=mean, median=med, p10=p10, p90=p90)


def main() -> None:
    hinge_f = predawn_files("hinge", 1)
    hemi_f = predawn_files("hemi_hi", 2)
    print(f"predawn days pooled: hinge={len(hinge_f)} hemi_hi={len(hemi_f)}")

    pgh, nh, hist_h = pooled_pgap(hinge_f)
    pge, ne, hist_e = pooled_pgap(hemi_f)

    # Effective zenith inside the canonical 55-60 deg bin.
    for name, hist in (("hinge", hist_h), ("hemi_hi", hist_e)):
        s = ring_effective_zenith(hist)
        print(f"{name:8} ring[55,60): n={s['n']:>9}  mean={s['mean']:.2f}  "
              f"median={s['median']:.2f}  p10={s['p10']:.2f}  p90={s['p90']:.2f}")

    # HingePAI implied by each scan at its own effective angle vs at a common angle.
    pai_h = -1.1 * np.log(np.clip(pgh, 1e-5, None))
    pai_e = -1.1 * np.log(np.clip(pge, 1e-5, None))

    fig, (ax, axz) = plt.subplots(2, 1, figsize=(11, 9), sharex=True,
                                  gridspec_kw=dict(height_ratios=[3, 1]))
    ax.plot(BIN_CENT, pgh, "-o", color="#1b7837", lw=2.2, ms=4, label="hinge scan")
    ax.plot(BIN_CENT, pge, "-o", color="#762a83", lw=2.2, ms=4, label="hemi_hi scan")
    ax.axvspan(RING[0], RING[1], color="0.85", zorder=0, label="55-60 deg bin")
    ax.set(ylabel="Pgap at canopy top (real cover algorithm)", yscale="log",
           title="Pgap vs zenith, predawn pooled (overlapping curves => "
                 "same angle gives same Pgap)")
    ax.legend(loc="lower left", fontsize=9)
    axz.semilogy(BIN_CENT, nh, "-", color="#1b7837")
    axz.semilogy(BIN_CENT, ne, "-", color="#762a83")
    axz.set(ylabel="shots / bin", xlabel="zenith (deg)")
    fig.tight_layout()
    out = OUT_DIR / "pai_pgap_vs_zenith_2025.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")

    both = ~np.isnan(pgh) & ~np.isnan(pge) & (nh > 0) & (ne > 0)
    tab = pd.DataFrame(dict(zenith=BIN_CENT[both],
                            pgap_hinge=np.round(pgh[both], 4),
                            pgap_hemi=np.round(pge[both], 4),
                            paiH_hinge=np.round(pai_h[both], 3),
                            paiH_hemi=np.round(pai_e[both], 3),
                            hemi_over_hinge=np.round(pge[both] / pgh[both], 2)))
    print("\nPgap at equal zenith (ratio ~1 => footprint-independent, angle-only):")
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
