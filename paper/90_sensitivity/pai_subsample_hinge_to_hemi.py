"""
Stage 90 — Subsample the hinge scan to hemi_hi density (issue #6, the dangerous test).

Question: is the hinge scan's earlier/stronger PAI response a *shot-count* artifact?
If it were, throwing away hinge ring shots until only as many remain as the matched
hemi_hi scan has at 57.5 deg should pull the hinge HingePAI toward the hemi_hi value.

Method (faithful to the pipeline):
  * Predawn only: hinge @ hour 1 paired with same-date hemi_hi @ hour 2 (both pre-sunrise,
    ~1 h apart -- the scans interleave, never share an hour). quality_all scans only.
  * HingePAI total is recomputed from RAW shots via the unmodified Jupp2009 class
    (FIRSTLAST), feeding only the 55-60 deg ring (other zeniths do not touch the hinge bin).
    Validated against the stored parquet HingePAI before trusting the subsample.
  * For each day: N = hemi_hi ring shot count. Draw N hinge ring shots WITHOUT replacement,
    recompute HingePAI total, repeat N_BOOT times -> mean +/- band.

Run:
  /home/lk1167/miniconda3/envs/dehar-spac/bin/python \
      paper/90_sensitivity/pai_subsample_hinge_to_hemi.py --validate   # check replication
  /home/lk1167/miniconda3/envs/dehar-spac/bin/python \
      paper/90_sensitivity/pai_subsample_hinge_to_hemi.py              # full run + figure
"""
from __future__ import annotations

import argparse
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

# Jupp2009 binning = the batch-notebook processing parameters.
JPARAMS = dict(min_z=5, max_z=70, zres=5, ares=45, min_h=0, max_h=25, hres=0.5)
SENSOR_HEIGHT = 1.5
METHOD = "FIRSTLAST"
RING_DEG = (55.0, 60.0)          # the 57.5 deg hinge bin
N_BOOT = 50
SEED = 12345


def _profile_from_rows(zen_r, azi_r, tcount, h1, h2) -> float:
    """HingePAI canopy total from a set of shots, via the unmodified Jupp2009."""
    prof = plant_profile.Jupp2009(**JPARAMS)
    for n, h in enumerate((h1, h2), start=1):
        tidx = np.full(h.shape, n, dtype=np.uint8)
        ok = ~np.isnan(h)
        if np.any(ok):
            prof.add_targets(h[ok], tidx[ok], tcount[ok], zen_r[ok], azi_r[ok],
                             method=METHOD)
    prof.add_shots(tcount, zen_r, azi_r, method=METHOD)
    prof.get_pgap_theta_z()
    return float(np.max(prof.calcHingePlantProfiles()))


def _ring_arrays(path: Path):
    """Load a raw scan, return (zenith_rad, azimuth_rad, target_count, h1, h2) in ring."""
    with leaf_io.LeafScanFile(str(path), sensor_height=SENSOR_HEIGHT,
                              zenith_offset=0) as leaf:
        d = leaf.data
        if d.empty:
            return None
        zen_deg = np.degrees(d["zenith"].to_numpy())
        ring = (zen_deg >= RING_DEG[0]) & (zen_deg < RING_DEG[1])
        if ring.sum() == 0:
            return None
        return (d["zenith"].to_numpy()[ring], d["azimuth"].to_numpy()[ring],
                d["target_count"].to_numpy()[ring],
                d["h1"].to_numpy()[ring], d["h2"].to_numpy()[ring])


def process_day(task: dict) -> dict | None:
    """One predawn day: full hinge, subsampled hinge (to hemi N), true hemi_hi."""
    hinge = _ring_arrays(RAW_DIR / task["hinge_file"])
    hemi = _ring_arrays(RAW_DIR / task["hemi_file"])
    if hinge is None or hemi is None:
        return None
    zr, ar, tc, h1, h2 = hinge
    n_hinge, n_hemi = zr.shape[0], hemi[0].shape[0]

    pai_hinge_full = _profile_from_rows(zr, ar, tc, h1, h2)
    pai_hemi_full = _profile_from_rows(*hemi)

    rng = np.random.default_rng(SEED + task["seed_off"])
    n_draw = min(n_hemi, n_hinge)
    boots = np.empty(N_BOOT)
    for b in range(N_BOOT):
        s = rng.choice(n_hinge, size=n_draw, replace=False)
        boots[b] = _profile_from_rows(zr[s], ar[s], tc[s], h1[s], h2[s])
    return dict(date=task["date"], n_hinge=n_hinge, n_hemi=n_hemi,
                pai_hinge_full=pai_hinge_full, pai_hemi_full=pai_hemi_full,
                pai_hinge_sub_mean=float(boots.mean()),
                pai_hinge_sub_std=float(boots.std()))


def build_tasks() -> list[dict]:
    """Predawn quality_all days present in BOTH hinge(h1) and hemi_hi(h2)."""
    def predawn(scan, hour):
        d = pd.read_parquet(LEAF_DIR / f"leaf_{scan}_2025.parquet")
        d = d[(d.scan_hour == hour) & d.quality_all]
        d["date"] = pd.to_datetime(d["datetime"]).dt.normalize().dt.tz_localize(None)
        return d.groupby("date")["filename"].first()

    hinge = predawn("hinge", 1)
    hemi = predawn("hemi_hi", 2)
    common = hinge.index.intersection(hemi.index)
    return [dict(date=dt, hinge_file=hinge[dt], hemi_file=hemi[dt], seed_off=i)
            for i, dt in enumerate(common)]


def validate(tasks: list[dict]) -> None:
    """Confirm raw->HingePAI replication matches the stored parquet totals."""
    hq = pd.read_parquet(LEAF_DIR / "leaf_hinge_2025.parquet")
    hq["date"] = pd.to_datetime(hq["datetime"]).dt.normalize().dt.tz_localize(None)
    stored = hq[hq.scan_hour == 1].groupby("date")["HingePAI"].max()
    print(f"{'date':12} {'replicated':>11} {'parquet':>9} {'abs_diff':>9}")
    for t in tasks[:8]:
        arr = _ring_arrays(RAW_DIR / t["hinge_file"])
        rep = _profile_from_rows(*arr)
        ref = float(stored.get(t["date"], np.nan))
        print(f"{str(t['date'].date()):12} {rep:11.4f} {ref:9.4f} {abs(rep-ref):9.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    tasks = build_tasks()
    print(f"{len(tasks)} predawn days matched (hinge h1 <-> hemi_hi h2)")
    if args.validate:
        validate(tasks)
        return

    with ProcessPoolExecutor() as ex:
        res = [r for r in ex.map(process_day, tasks) if r is not None]
    df = pd.DataFrame(res).sort_values("date").set_index("date")
    df.to_csv(OUT_DIR / "pai_subsample_hinge_to_hemi.csv")
    print(f"processed {len(df)} days; median ring shots hinge={df.n_hinge.median():.0f} "
          f"hemi={df.n_hemi.median():.0f}")

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(df.index, df.pai_hinge_full, color="#1b7837", lw=2.4,
            label=f"hinge full (~{df.n_hinge.median():.0f} ring shots)")
    ax.plot(df.index, df.pai_hinge_sub_mean, color="#d95f0e", lw=2.0,
            label=f"hinge subsampled to hemi_hi (~{df.n_hemi.median():.0f} shots)")
    ax.fill_between(df.index, df.pai_hinge_sub_mean - df.pai_hinge_sub_std,
                    df.pai_hinge_sub_mean + df.pai_hinge_sub_std,
                    color="#d95f0e", alpha=0.25, lw=0)
    ax.plot(df.index, df.pai_hemi_full, color="#762a83", lw=2.4,
            label="hemi_hi full")
    ax.set(ylabel=r"HingePAI total ($m^2\,m^{-2}$)", xlabel="2025 (predawn)",
           title="Subsampling hinge to hemi_hi shot density (HingePAI @ 57.5 deg)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / "pai_subsample_hinge_to_hemi_2025.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")

    gap = (df.pai_hinge_full - df.pai_hemi_full)
    closed = (df.pai_hinge_full - df.pai_hinge_sub_mean)
    frac = (closed / gap).replace([np.inf, -np.inf], np.nan)
    print(f"median hinge-hemi gap   = {gap.median():.3f}")
    print(f"median gap closed by subsampling = {closed.median():.3f} "
          f"({100*frac.median():.1f}% of the gap)")


if __name__ == "__main__":
    main()
