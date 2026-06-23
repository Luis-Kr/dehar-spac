"""
Stage 90 — How zenith angle drives the PAI signal (issue #6 appendix).

The hinge-angle inversion PAI(theta) = -1.1*ln(Pgap(theta)) is strongly zenith-dependent:
near nadir the beam path through the canopy is short (low PAI); near the horizon it is long
(high PAI). The dedicated hinge scan samples ONE angle (~57.5-59 deg); the hemispherical scan
samples the whole 7.5-67.5 deg range, so it alone can show the curve.

Per-zenith canopy-total Pgap is read directly from the leaf parquet Pgap_Z* columns (min over
height = fully accumulated gap); -1.1*ln of that reproduces HingePAI exactly. quality_all scans,
daily mean (= the canonical aggregation).

  Figure A: PAI vs zenith, averaged over three phenophases (hemi curve + hinge points).
  Figure B: season time series of PAI at selected fixed zeniths (hemi) + the hinge 57.5 deg bin.

Run:
  /home/lk1167/miniconda3/envs/dehar-spac/bin/python paper/90_sensitivity/pai_zenith_sensitivity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "config"))
from paper_common import paper_style  # noqa: E402

paper_style()

ROOT = HERE.parents[1]
LEAF_DIR = ROOT / "data" / "processed" / "proximal_rs" / "leaf"
OUT_DIR = HERE / "outputs" / "appendix"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PHENO = {                                   # representative windows
    "leaf-out (May)": ("2025-05-01", "2025-05-20"),
    "peak canopy (Jul)": ("2025-07-01", "2025-07-31"),
    "senescence (Oct)": ("2025-10-01", "2025-10-31"),
}
TS_ZENITHS = [32.5, 47.5, 57.5, 62.5]       # fixed zeniths for the time-series figure


def per_zenith_pai(scan: str) -> pd.DataFrame:
    """Daily PAI(theta) = -1.1*ln(canopy-total Pgap) per zenith bin, quality_all scans."""
    d = pd.read_parquet(LEAF_DIR / f"leaf_{scan}_2025.parquet")
    d["datetime"] = pd.to_datetime(d["datetime"], utc=True)
    pcols = sorted([c for c in d.columns if c.startswith("Pgap_Z")])
    qual = d.groupby("datetime")["quality_all"].first()
    pgap = d.groupby("datetime")[pcols].min()           # min over height = canopy total
    pgap = pgap[qual.reindex(pgap.index).fillna(False)]
    pai = -1.1 * np.log(pgap.clip(lower=1e-5))
    pai.columns = [float(c.replace("Pgap_Z", "")) for c in pai.columns]
    return pai.resample("1D").mean()                    # daily mean (canonical)


def main() -> None:
    pai_hemi = per_zenith_pai("hemi_hi")
    pai_hinge = per_zenith_pai("hinge")
    zen = np.array(pai_hemi.columns, dtype=float)

    # ---- Figure A: PAI vs zenith per phenophase --------------------------------
    figA, ax = plt.subplots(figsize=(10, 7))
    colors = cm.viridis(np.linspace(0.1, 0.85, len(PHENO)))
    for (name, (a, b)), col in zip(PHENO.items(), colors):
        win = pai_hemi.loc[a:b]
        ax.plot(zen, win.mean(axis=0), "-o", color=col, lw=2.4, ms=5, label=f"hemi_hi — {name}")
        hwin = pai_hinge.loc[a:b]
        hz = np.array(hwin.columns, dtype=float)
        hm = hwin.mean(axis=0)
        good = hm.notna().to_numpy() & (np.array([hwin[c].notna().sum() for c in hwin.columns]) > 0)
        ax.plot(hz[good], hm.to_numpy()[good], "X", color=col, ms=13, mec="k", mew=0.6,
                label=f"hinge scan — {name}")
    ax.axvspan(55, 60, color="0.9", zorder=0)
    ax.axvline(57.5, color="0.5", ls="--", lw=1)
    ax.annotate("57.5 deg\nhinge angle", xy=(57.5, ax.get_ylim()[1]), xytext=(0, -28),
                textcoords="offset points", ha="center", fontsize=8, color="0.4")
    ax.set(xlabel="zenith angle (deg)", ylabel=r"PAI = -1.1 ln(Pgap)  ($m^2\,m^{-2}$)",
           title="PAI is strongly zenith-dependent — hemi samples the whole curve, "
                 "hinge only ~57.5-62.5 deg")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    figA.tight_layout()
    outA = OUT_DIR / "pai_vs_zenith_phenophase.png"
    figA.savefig(outA, dpi=200, bbox_inches="tight")
    print(f"wrote {outA}")

    # ---- Figure B: season time series at fixed zeniths -------------------------
    figB, ax = plt.subplots(figsize=(13, 7))
    cols = cm.plasma(np.linspace(0.05, 0.85, len(TS_ZENITHS)))
    for z, col in zip(TS_ZENITHS, cols):
        s = pai_hemi[z].dropna()
        ax.plot(s.index, s.values, color=col, lw=2.2, label=f"hemi_hi @ {z:.1f} deg")
    if 57.5 in pai_hinge.columns:
        s = pai_hinge[57.5].dropna()
        ax.plot(s.index, s.values, color="#1b7837", lw=2.6, ls="--",
                label="hinge scan @ 57.5 deg bin (~59 deg)")
    ax.set(xlabel="2025", ylabel=r"PAI = -1.1 ln(Pgap)  ($m^2\,m^{-2}$)",
           title="Same canopy, different zenith -> different level AND different seasonal signal")
    ax.legend(fontsize=9, ncol=2, loc="upper right")
    figB.tight_layout()
    outB = OUT_DIR / "pai_zenith_timeseries.png"
    figB.savefig(outB, dpi=200, bbox_inches="tight")
    print(f"wrote {outB}")

    # numeric: how much does the zenith choice move the season-median PAI?
    med = pai_hemi.median(axis=0)
    print("\nhemi_hi season-median PAI by zenith:")
    print(med.round(2).to_string())
    print(f"\n=> choosing zenith moves the canopy PAI from {med.min():.2f} "
          f"(@{med.idxmin():.1f} deg) to {med.max():.2f} (@{med.idxmax():.1f} deg)")


if __name__ == "__main__":
    main()
