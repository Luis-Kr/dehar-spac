"""
Stage 90 — PAI hinge/hemi processing audit (appendix / verification, NOT selection).

Supports GitHub issue #6: verify the PAI inversion is implemented as intended and
expose the hinge-scan vs hemi-scan behaviour under each Jupp (2009) inversion.

Vocabulary (see CONTEXT.md): "hinge" rides on two axes.
  * scan mode    : hinge scan (dense 57.5 deg sweep) vs hemi scan (hemi_hi / hemi_low).
  * inversion    : Hinge-angle PAI (-1.1*log Pgap at 57.5 deg), Linear (multi-angle
                   regression), Weighted (solid-angle weighted).
Both headline columns use the *Hinge-angle inversion*; they differ only in which
*scan* fed the 57.5 deg ring.

Figure 1 (this script): per-scan canopy-total PAI over the season, one panel per
inversion, hinge vs hemi_hi vs hemi_low. Total = max over height (the pipeline
reduction), quality_all only, daily mean -- identical to build_pai().

Run:
  /home/lk1167/miniconda3/envs/dehar-spac/bin/python paper/90_sensitivity/pai_hinge_hemi_audit.py
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

SCANS = {
    "hinge": dict(color="#1b7837", label="hinge scan (dense 57.5 deg)"),
    "hemi_hi": dict(color="#762a83", label="hemi_hi scan"),
    "hemi_low": dict(color="#b8b8b8", label="hemi_low scan"),
}
INVERSIONS = [
    ("HingePAI", "Hinge-angle inversion  (-1.1 log Pgap @ 57.5 deg)"),
    ("LinearPAI", "Linear inversion  (multi-angle regression)"),
    ("WeightedPAI", "Weighted inversion  (solid-angle)"),
]


def daily_total(scan: str, metric: str) -> pd.Series:
    """Canopy-total PAI per day: max over height, quality_all, daily mean.

    Mirrors leaf_totals() (max over height) then build_pai() (quality_all
    mask + daily mean) so the numbers match the canonical daily table.
    """
    df = pd.read_parquet(LEAF_DIR / f"leaf_{scan}_2025.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df[df["quality_all"]]
    total = df.groupby("datetime")[metric].max()            # per-scan canopy total
    return total.resample("1D").mean().rename(scan)         # daily mean of good scans


def main() -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    for ax, (metric, title) in zip(axes, INVERSIONS):
        for scan, sty in SCANS.items():
            s = daily_total(scan, metric).dropna()
            if s.empty:
                continue
            ax.plot(s.index, s.values, color=sty["color"], lw=2.2,
                    label=sty["label"])
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylabel(r"PAI total  ($m^2\,m^{-2}$)")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    axes[-1].set_xlabel("2025")
    fig.suptitle(
        "PAI by scan mode and inversion (canopy total = max over height, "
        "quality_all, daily mean)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    out = OUT_DIR / "pai_scan_x_inversion_2025.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")

    # Tabular sanity: season-median total per scan x inversion.
    rows = []
    for metric, _ in INVERSIONS:
        for scan in SCANS:
            s = daily_total(scan, metric).dropna()
            rows.append(dict(inversion=metric, scan=scan,
                             n_days=int(s.shape[0]),
                             median=round(float(s.median()), 3) if s.size else None))
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))
    tab.to_csv(OUT_DIR / "pai_scan_x_inversion_median.csv", index=False)


if __name__ == "__main__":
    main()
