"""
Stage 90 — Ring sensitivity & angle-matched hinge/hemi (issue #6 follow-up).

(a) Per-ring spread: the hinge scan fires 5 zenith rings (~58.4-60.2 deg). Compute
    HingePAI from EACH ring alone and bootstrap over rings -> how much does the hinge
    number wobble purely from which ring is used (i.e. from leveling-dependent placement)?

(b) Angle-matched: restrict BOTH hinge and hemi_hi to the SAME narrow zenith band
    (no ring offset), recompute HingePAI per day -> does the divergence survive once the
    effective angle is identical? Any residual is pure footprint/geometry.

HingePAI total here = -1.1*log(Pgap), Pgap = 1 - mean_azimuth(0.5*returns/shots), the
pipeline's FIRSTLAST cover at canopy top. Validated against the Jupp2009 class in main().

Run:
  /home/lk1167/miniconda3/envs/dehar-spac/bin/python paper/90_sensitivity/pai_ring_sensitivity.py
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
ARES = 45.0                       # azimuth bins (= pipeline)
MATCH_BAND = (58.5, 59.5)         # common zenith band for part (b)
N_BOOT = 1000
SEED = 7


def total_pgap(zen_rad, azi_rad, h1, h2) -> float:
    """Pipeline FIRSTLAST canopy-top Pgap for a set of shots (azimuth-averaged)."""
    a_idx = (np.degrees(azi_rad) // ARES).astype(int) % int(360 // ARES)
    ret = (~np.isnan(h1)).astype(float) + (~np.isnan(h2)).astype(float)
    cover = []
    for a in range(int(360 // ARES)):
        m = a_idx == a
        ns = int(m.sum())
        if ns > 0:
            cover.append(0.5 * ret[m].sum() / ns)
    if not cover:
        return 1.0
    return float(np.clip(1 - np.mean(cover), 1e-5, 1.0))


def hingepai(pgap: float) -> float:
    return -1.1 * np.log(pgap)


def _load(path: Path):
    leaf = leaf_io.LeafScanFile(str(path), sensor_height=SENSOR_HEIGHT, zenith_offset=0)
    return leaf.data


def per_scan(task: dict) -> dict | None:
    """Per-ring HingePAI (hinge) and angle-matched HingePAI (hinge vs hemi)."""
    dh = _load(RAW_DIR / task["hinge_file"])
    de = _load(RAW_DIR / task["hemi_file"])
    if dh.empty or de.empty:
        return None

    # (a) split hinge into its 5 rings by scan_encoder.
    rings = []
    for _, g in dh.groupby("scan_encoder"):
        z = np.degrees(g["zenith"].to_numpy())
        pg = total_pgap(g["zenith"].to_numpy(), g["azimuth"].to_numpy(),
                        g["h1"].to_numpy(), g["h2"].to_numpy())
        rings.append((float(z.mean()), hingepai(pg)))
    rings.sort()
    ring_pai = np.array([p for _, p in rings])

    # (b) both scans restricted to the SAME zenith band.
    def band_pai(d):
        z = np.degrees(d["zenith"].to_numpy())
        m = (z >= MATCH_BAND[0]) & (z < MATCH_BAND[1])
        if m.sum() < 500:
            return np.nan
        return hingepai(total_pgap(d["zenith"].to_numpy()[m], d["azimuth"].to_numpy()[m],
                                   d["h1"].to_numpy()[m], d["h2"].to_numpy()[m]))

    return dict(date=task["date"],
                ring_pai_mean=float(ring_pai.mean()),
                ring_pai_std=float(ring_pai.std()),
                ring_pai_min=float(ring_pai.min()),
                ring_pai_max=float(ring_pai.max()),
                n_rings=len(ring_pai),
                hinge_band=band_pai(dh), hemi_band=band_pai(de),
                ring_pai=ring_pai)


def build_tasks() -> list[dict]:
    def predawn(scan, hour):
        d = pd.read_parquet(LEAF_DIR / f"leaf_{scan}_2025.parquet")
        d = d[(d.scan_hour == hour) & d.quality_all]
        d["date"] = pd.to_datetime(d["datetime"]).dt.normalize().dt.tz_localize(None)
        return d.groupby("date")["filename"].first()

    h, e = predawn("hinge", 1), predawn("hemi_hi", 2)
    common = h.index.intersection(e.index)
    return [dict(date=dt, hinge_file=h[dt], hemi_file=e[dt]) for dt in common]


def validate(tasks):
    """Direct total_pgap must match the Jupp2009 class on the 57.5 deg bin."""
    JP = dict(min_z=5, max_z=70, zres=5, ares=45, min_h=0, max_h=25, hres=0.5)
    print(f"{'date':12} {'direct':>8} {'Jupp2009':>9} {'diff':>8}")
    for t in tasks[:5]:
        d = _load(RAW_DIR / t["hinge_file"])
        z = np.degrees(d["zenith"].to_numpy())
        m = (z >= 55) & (z < 60)
        direct = hingepai(total_pgap(d["zenith"].to_numpy()[m], d["azimuth"].to_numpy()[m],
                                     d["h1"].to_numpy()[m], d["h2"].to_numpy()[m]))
        prof = plant_profile.Jupp2009(**JP)
        zr, ar = d["zenith"].to_numpy()[m], d["azimuth"].to_numpy()[m]
        tc = d["target_count"].to_numpy()[m]
        h1, h2 = d["h1"].to_numpy()[m], d["h2"].to_numpy()[m]
        for n, h in enumerate((h1, h2), start=1):
            ok = ~np.isnan(h)
            prof.add_targets(h[ok], np.full(ok.sum(), n, np.uint8), tc[ok], zr[ok], ar[ok],
                             method="FIRSTLAST")
        prof.add_shots(tc, zr, ar, method="FIRSTLAST")
        prof.get_pgap_theta_z()
        ref = float(np.max(prof.calcHingePlantProfiles()))
        print(f"{str(t['date'].date()):12} {direct:8.4f} {ref:9.4f} {abs(direct-ref):8.4f}")


def main() -> None:
    tasks = build_tasks()
    print(f"{len(tasks)} matched predawn days")
    print("\n[validation] direct total_pgap vs Jupp2009 class:")
    validate(tasks)

    with ProcessPoolExecutor() as ex:
        res = [r for r in ex.map(per_scan, tasks) if r is not None]
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "ring_pai"} for r in res])
    df = df.sort_values("date").set_index("date")
    df.to_csv(OUT_DIR / "pai_ring_sensitivity.csv")

    # (a) ring bootstrap: resample the per-day ring spreads.
    all_rings = np.concatenate([r["ring_pai"] for r in res])
    rng = np.random.default_rng(SEED)
    boot = [rng.choice(all_rings, all_rings.size, replace=True).mean() for _ in range(N_BOOT)]
    print(f"\n(a) per-ring HingePAI: pooled mean={all_rings.mean():.3f}  "
          f"within-day std (median)={df.ring_pai_std.median():.3f}  "
          f"within-day range (median)={ (df.ring_pai_max-df.ring_pai_min).median():.3f}")
    print(f"    ring-bootstrap 95% CI of the mean = "
          f"[{np.percentile(boot,2.5):.3f}, {np.percentile(boot,97.5):.3f}]")

    # (b) angle-matched residual.
    d2 = df.dropna(subset=["hinge_band", "hemi_band"])
    gap = (d2.hinge_band - d2.hemi_band)
    print(f"\n(b) angle-matched band {MATCH_BAND} deg ({len(d2)} days): "
          f"hinge={d2.hinge_band.median():.3f}  hemi={d2.hemi_band.median():.3f}  "
          f"median(hinge-hemi)={gap.median():.3f}")

    fig, (axa, axb) = plt.subplots(2, 1, figsize=(13, 9))
    axa.plot(df.index, df.ring_pai_mean, color="#1b7837", lw=2.2, label="hinge ring mean")
    axa.fill_between(df.index, df.ring_pai_min, df.ring_pai_max, color="#1b7837",
                     alpha=0.25, lw=0, label="ring min-max (58.4-60.2 deg)")
    axa.set(ylabel=r"HingePAI ($m^2\,m^{-2}$)", title="(a) Per-ring spread within the hinge scan")
    axa.legend(loc="upper right", fontsize=9)

    axb.plot(d2.index, d2.hinge_band, color="#1b7837", lw=2.2,
             label=f"hinge @ {MATCH_BAND} deg")
    axb.plot(d2.index, d2.hemi_band, color="#762a83", lw=2.2,
             label=f"hemi_hi @ {MATCH_BAND} deg")
    axb.set(ylabel=r"HingePAI ($m^2\,m^{-2}$)", xlabel="2025 (predawn)",
            title="(b) Same zenith band for both -> residual is pure footprint")
    axb.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / "pai_ring_sensitivity_2025.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
