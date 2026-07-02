"""Supplementary figure: height-resolved LEAF canopy dynamics 2025.

Now that the hemi "up" is the hand-verified manual reference (ADR 0006), the
vertical Jupp (2009) profiles are trustworthy, so the two DE-Har canopy layers
separate cleanly by height:

  (A) PAVD(height, time)   - plant-area volume density; the DECIDUOUS understory
      band (~4-6 m) waxes and wanes while the EVERGREEN overstory (~12 m) persists.
  (B) Pgap(height, time)   - gap fraction at the 57.5 deg ring; the understory
      opens up through autumn (leaf fall), the overstory barely changes.
  (C) per-layer PAI        - the whole seasonal + August-drought PAI signal is the
      understory; the overstory is a near-constant offset (its slight autumn rise
      is reduced understory occlusion seen from below, not real growth).

Reads the canonical processed profiles
(``data/processed/proximal_rs/leaf/leaf_hemi_hi_2025.parquet``, quality_all scans).

Run:
  /home/lk1167/miniconda3/envs/dehar-spac/bin/python \
      paper/40_supplementary/leaf_hemispherical/fig_height_resolved_dynamics.py

Outputs:
  outputs/height_resolved_dynamics.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "paper" / "config"))
import _leaf_hemi_common as hemi  # noqa: E402
from paper_common import paper_style  # noqa: E402

# ── PARAMETERS ───────────────────────────────────────────────────────────────
HMAX = 18.0                                        # crop the empty profile top (m)
SPLIT = 9.0                                        # understory | overstory boundary (m)
UNDER = (1.5, SPLIT)                               # understory band (m)
OVER = (SPLIT, HMAX)                               # overstory band (m)
EVENT = (pd.Timestamp("2025-08-07"), pd.Timestamp("2025-08-22"))
GAPCOL = "Pgap_Z057.5"                             # gap fraction at the 57.5 deg ring
TIME_INTERP_LIMIT = 7                              # bridge <=7 d gaps in the heatmaps
SMOOTH_D = 7                                       # per-layer PAI smoothing (days)
OUT_NAME = "height_resolved_dynamics.png"
DPI = 150


def daily_grids():
    """Daily median height profiles (quality_all scans), time-interpolated."""
    df = pd.read_parquet(hemi.LEAF_PARQUET)
    df = df[df["quality_all"] == True].copy()                      # noqa: E712
    df["date"] = pd.to_datetime(df["datetime"]).dt.normalize()
    g = (df.groupby(["date", "height"])[["WeightedPAVD", "WeightedPAI", GAPCOL]]
         .median().reset_index())
    full = pd.date_range(g["date"].min(), g["date"].max(), freq="D")

    def grid(col, crop):
        p = g.pivot(index="date", columns="height", values=col).sort_index()
        p = p.reindex(full).interpolate(axis=0, limit=TIME_INTERP_LIMIT)
        cols = [h for h in p.columns if h <= HMAX] if crop else sorted(p.columns)
        return p[sorted(cols)]

    return grid("WeightedPAVD", True), grid(GAPCOL, True), grid("WeightedPAI", False)


def band_pai(pai_grid, h1, h2):
    """Layer PAI = cumulative WeightedPAI at h2 minus at h1 (nearest bins), smoothed."""
    H = np.array(sorted(pai_grid.columns))
    at = lambda h: pai_grid[H[np.abs(H - h).argmin()]]            # noqa: E731
    return (at(h2) - at(h1)).rolling(SMOOTH_D, center=True, min_periods=2).mean()


def _heatmap(ax, grid, cmap, label, title):
    X = mdates.date2num(grid.index.to_pydatetime())
    Y = np.array(sorted(grid.columns))
    pcm = ax.pcolormesh(X, Y, grid[Y].to_numpy().T, cmap=cmap, shading="auto")
    ax.axhline(SPLIT, color="k", ls="--", lw=1.0, alpha=0.7)
    ax.axvspan(mdates.date2num(EVENT[0]), mdates.date2num(EVENT[1]), color="k", alpha=0.12)
    ax.xaxis_date(); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.annotate("understory", (X[3], 5.0), color="0.12", fontsize=8)
    ax.annotate("overstory", (X[3], 12.5), color="0.12", fontsize=8)
    ax.set(ylabel="height above ground (m)", title=title)
    ax.figure.colorbar(pcm, ax=ax, label=label, pad=0.01)


def main() -> int:
    paper_style()
    pavd, pgap, pai = daily_grids()
    under = band_pai(pai, *UNDER)
    over = band_pai(pai, *OVER)
    total = band_pai(pai, UNDER[0], OVER[1])

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 0.85])
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])

    _heatmap(axA, pavd, "YlGn", "PAVD (m² m⁻³)",
             "(A) plant-area volume density by height — understory waxes/wanes, overstory persists")
    _heatmap(axB, pgap, "Blues_r", "gap fraction  Pgap(57.5°)",
             "(B) gap fraction by height — understory opens up in autumn (leaf fall)")

    axC.plot(under.index, under, color="#238b45", lw=2.6,
             label=f"understory {UNDER[0]:.0f}–{UNDER[1]:.0f} m (deciduous)")
    axC.plot(over.index, over, color="#8c6d31", lw=2.6,
             label=f"overstory {OVER[0]:.0f}–{OVER[1]:.0f} m (evergreen)")
    axC.plot(total.index, total, color="0.4", lw=1.4, ls="--", label="total canopy")
    axC.axvspan(EVENT[0], EVENT[1], color="k", alpha=0.12)
    axC.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axC.set(ylabel="layer PAI (m² m⁻²)",
            title="(C) per-layer PAI (7 d smooth) — the seasonal + August-drought signal is the understory")
    axC.legend(loc="best")

    fig.suptitle("Height-resolved LEAF dynamics 2025 (canonical up-corrected hemi) — "
                 "deciduous understory vs evergreen overstory", fontsize=13)
    out = HERE / "outputs" / OUT_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    def stat(s, lab):
        peak = s.max(); nov = s.loc["2025-11-01":"2025-11-30"].mean()
        b = s.loc["2025-06-01":"2025-06-30"].mean(); e = s.loc[EVENT[0]:EVENT[1]].mean()
        print(f"{lab:20s} peak={peak:.2f}  Nov={nov:.2f}  leaf-off={peak - nov:+.2f}  "
              f"Aug-event dip={e - b:+.2f}")
    print("=== per-layer PAI summary ===")
    stat(under, "understory 1.5-9 m"); stat(over, "overstory 9-18 m")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
