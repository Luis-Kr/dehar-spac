"""
Stage 90 — Season PAI by inversion, one panel per scan (issue #6 appendix).

Canopy-total PAI (max over height, quality_all, daily mean = the canonical aggregation)
under the three Jupp (2009) inversions, shown separately for the hemi_hi scan and the
hinge scan:
  * Hinge and Weighted TOTALS coincide by construction (Weighted is rescaled to the hinge
    total, plant_profile.py:270) -> the two lines overlap.
  * Linear differs; for the hinge scan it is DEGENERATE (single-angle regression is
    ill-posed) and must never be consumed.

Run:
  /home/lk1167/miniconda3/envs/dehar-spac/bin/python paper/90_sensitivity/pai_inversion_by_scan.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "config"))
from paper_common import paper_style  # noqa: E402

paper_style()

ROOT = HERE.parents[1]
LEAF_DIR = ROOT / "data" / "processed" / "proximal_rs" / "leaf"
OUT_DIR = HERE / "outputs" / "appendix"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INVERSIONS = [
    ("WeightedPAI", "Weighted inversion (all angles)", "#d95f0e", 3.4, "-"),
    ("HingePAI", "Hinge inversion (57.5 deg ring)", "#1b7837", 1.8, "--"),
    ("LinearPAI", "Linear inversion (multi-angle regression)", "#3b78c2", 2.0, "-"),
]
SCANS = [("hemi_hi", "hemi_hi scan (samples all angles)"),
         ("hinge", "hinge scan (single ~59 deg ring)")]


def daily_total(scan: str, metric: str) -> pd.Series:
    """max over height (canopy total), quality_all, daily mean."""
    df = pd.read_parquet(LEAF_DIR / f"leaf_{scan}_2025.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df[df["quality_all"]]
    return df.groupby("datetime")[metric].max().resample("1D").mean()


def main() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for ax, (scan, title) in zip(axes, SCANS):
        for metric, label, color, lw, ls in INVERSIONS:
            s = daily_total(scan, metric).dropna()
            ax.plot(s.index, s.values, color=color, lw=lw, ls=ls, label=label)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylabel(r"PAI total ($m^2\,m^{-2}$)")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    axes[1].set_xlabel("2025")
    # annotate the hinge=weighted overlap on the hemi panel
    axes[0].annotate("Hinge & Weighted totals overlap exactly\n"
                     "(Weighted is rescaled to the hinge total)",
                     xy=(0.015, 0.06), xycoords="axes fraction", fontsize=8,
                     color="0.35", va="bottom")
    axes[1].annotate("Linear is DEGENERATE for the single-angle hinge scan",
                     xy=(0.015, 0.92), xycoords="axes fraction", fontsize=8,
                     color="#3b78c2", va="top")
    fig.suptitle("Season PAI by inversion, per scan (canopy total, quality_all, daily mean)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = OUT_DIR / "pai_inversion_by_scan_2025.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
