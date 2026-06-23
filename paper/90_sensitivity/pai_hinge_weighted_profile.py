"""
Stage 90 — HingePAI vs WeightedPAI: same total, different vertical shape (issue #6).

Jupp (2009): HingePAI takes BOTH the canopy total and the vertical distribution from the
single 57.5 deg ring. WeightedPAI takes the SAME total (it is rescaled to max(HingePAI),
plant_profile.py:270) but redistributes it over height using ALL zenith angles, solid-angle
weighted. So the canopy total is identical; only the height profile / PAVD differs.

Shown on one representative peak-canopy hemi_hi scan (a scan that samples all angles).

Run:
  /home/lk1167/miniconda3/envs/dehar-spac/bin/python paper/90_sensitivity/pai_hinge_weighted_profile.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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

C_HINGE, C_WEIGHT = "#1b7837", "#d95f0e"


def representative_scan() -> pd.DataFrame:
    """A peak-canopy predawn hemi_hi scan (all angles sampled -> Weighted meaningful)."""
    d = pd.read_parquet(LEAF_DIR / "leaf_hemi_hi_2025.parquet")
    d["datetime"] = pd.to_datetime(d["datetime"])
    peak = d[(d.datetime.dt.month == 7) & (d.scan_hour == 2) & d.quality_all]
    dt = sorted(peak["datetime"].unique())[len(peak["datetime"].unique()) // 2]
    return d[d.datetime == dt].sort_values("height"), pd.Timestamp(dt)


def main() -> None:
    g, dt = representative_scan()
    h = g["height"].to_numpy()
    total_h, total_w = g["HingePAI"].max(), g["WeightedPAI"].max()

    fig, (axp, axd) = plt.subplots(1, 2, figsize=(12, 7), sharey=True)

    # Left: cumulative PAI profile.
    axp.plot(g["HingePAI"], h, color=C_HINGE, lw=2.6, label="HingePAI (57.5 deg ring)")
    axp.plot(g["WeightedPAI"], h, color=C_WEIGHT, lw=2.6, label="WeightedPAI (all angles)")
    axp.axvline(total_h, color="0.6", ls="--", lw=1.2)
    axp.annotate(f"identical total = {total_h:.2f}", xy=(total_h, h.max() * 0.5),
                 xytext=(-6, 0), textcoords="offset points", rotation=90,
                 va="center", ha="right", fontsize=9, color="0.4")
    axp.set(xlabel=r"cumulative PAI ($m^2\,m^{-2}$)", ylabel="height (m)",
            title="Cumulative profile")
    axp.legend(loc="lower right", fontsize=9)

    # Right: PAVD (per-layer density).
    axd.plot(g["HingePAVD"], h, color=C_HINGE, lw=2.6, label="HingePAVD")
    axd.plot(g["WeightedPAVD"], h, color=C_WEIGHT, lw=2.6, label="WeightedPAVD")
    axd.set(xlabel=r"PAVD ($m^2\,m^{-3}$)", title="Per-layer density (where foliage sits)")
    axd.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        f"HingePAI vs WeightedPAI — same total, different vertical shape "
        f"(hemi_hi predawn, {dt.date()})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = OUT_DIR / "pai_hinge_vs_weighted_profile.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"total HingePAI={total_h:.4f}  WeightedPAI={total_w:.4f}  "
          f"identical={np.isclose(total_h, total_w)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
