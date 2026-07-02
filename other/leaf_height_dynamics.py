"""Height-resolved LEAF canopy dynamics 2025 (up-corrected hemi_hi; other/, off-pipeline).

Now that the hemi "up" is corrected (ADR 0006) the vertical Jupp profiles are
trustworthy, so we can ask: do different canopy heights show different dynamics?

They do, sharply. DE-Har is a two-layer canopy — a DECIDUOUS understory (PAVD peak
~4-6 m) under a persistent EVERGREEN/dying overstory (PAVD peak ~12 m). The figure
shows it three ways, all resolved by height:
  (A) PAVD(height, time)   — where plant area sits and how it moves over the season.
  (B) Pgap(height, time)   — gap fraction at the 57.5 deg ring (canopy opening up).
  (C) per-layer PAI        — understory (big seasonal + drought signal) vs overstory (flat).

Run:
  /home/lk1167/miniconda3/envs/dehar-spac/bin/python other/leaf_height_dynamics.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
LEAF = REPO / "data/processed/proximal_rs/leaf/leaf_hemi_hi_2025.parquet"
OUT = REPO / "other" / "leaf_height_dynamics.png"
HMAX = 18.0                                   # crop the empty top of the profile
SPLIT = 9.0                                   # understory | overstory boundary (PAVD gap)
EVENT = (pd.Timestamp("2025-08-07"), pd.Timestamp("2025-08-22"))
GAPCOL = "Pgap_Z057.5"                        # gap fraction at the 57.5 deg hinge ring


def daily_grids():
    """Daily median height profiles (quality_all scans), time-interpolated."""
    df = pd.read_parquet(LEAF)
    df = df[df["quality_all"] == True].copy()            # noqa: E712
    df["date"] = pd.to_datetime(df["datetime"]).dt.normalize()
    g = (df.groupby(["date", "height"])[["WeightedPAVD", "WeightedPAI", GAPCOL]]
         .median().reset_index())
    full = pd.date_range(g["date"].min(), g["date"].max(), freq="D")

    def grid(col, crop):
        p = g.pivot(index="date", columns="height", values=col).sort_index()
        p = p.reindex(full).interpolate(axis=0, limit=7)
        cols = [h for h in p.columns if h <= HMAX] if crop else sorted(p.columns)
        return p[sorted(cols)]

    return grid("WeightedPAVD", True), grid(GAPCOL, True), grid("WeightedPAI", False), full


def band_pai(pai_grid, h1, h2):
    """Layer PAI = cumulative WeightedPAI at h2 minus at h1 (nearest bins), 7 d smooth."""
    H = np.array(sorted(pai_grid.columns))
    at = lambda h: pai_grid[H[np.abs(H - h).argmin()]]   # noqa: E731
    return (at(h2) - at(h1)).rolling(7, center=True, min_periods=2).mean()


def heatmap(ax, grid, cmap, label, title):
    X = mdates.date2num(grid.index.to_pydatetime())
    Y = np.array(sorted(grid.columns))
    pcm = ax.pcolormesh(X, Y, grid[Y].to_numpy().T, cmap=cmap, shading="auto")
    ax.axhline(SPLIT, color="k", ls="--", lw=1.0, alpha=0.7)
    ax.axvspan(mdates.date2num(EVENT[0]), mdates.date2num(EVENT[1]), color="k", alpha=0.12)
    ax.xaxis_date(); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set(ylabel="height above ground (m)", title=title)
    ax.figure.colorbar(pcm, ax=ax, label=label, pad=0.01)


def main() -> int:
    import sys
    sys.path.insert(0, str(REPO / "paper" / "config"))
    from paper_common import paper_style
    paper_style()

    pavd, pgap, pai, full = daily_grids()
    under = band_pai(pai, 1.5, SPLIT)
    over = band_pai(pai, SPLIT, HMAX)
    total = band_pai(pai, 1.5, HMAX)

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 0.85], hspace=0.30, wspace=0.20)
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])

    heatmap(axA, pavd, "YlGn", "PAVD (m² m⁻³)",
            "(A) plant-area volume density by height — understory waxes/wanes, overstory persists")
    heatmap(axB, pgap, "Blues_r", "gap fraction Pgap(57.5°)",
            "(B) gap fraction by height — understory opens up in autumn (leaf fall)")
    for ax in (axA, axB):
        ax.annotate("understory", (mdates.date2num(full[3]), 5), color="0.15", fontsize=8)
        ax.annotate("overstory", (mdates.date2num(full[3]), 12.5), color="0.15", fontsize=8)

    axC.plot(under.index, under, color="#238b45", lw=2.6, label="understory 1.5–9 m (deciduous)")
    axC.plot(over.index, over, color="#8c6d31", lw=2.6, label="overstory 9–18 m (evergreen)")
    axC.plot(total.index, total, color="0.4", lw=1.4, ls="--", label="total canopy")
    axC.axvspan(EVENT[0], EVENT[1], color="k", alpha=0.12)
    axC.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axC.set(ylabel="layer PAI (m² m⁻²)",
            title="(C) per-layer PAI (7 d smooth) — the whole seasonal + drought signal is the understory")
    axC.legend(loc="best")

    fig.suptitle("Height-resolved LEAF dynamics 2025 (up-corrected hemi_hi) — "
                 "deciduous understory vs evergreen overstory", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT, dpi=150); plt.close(fig)

    print(f"understory PAI  peak≈{under.max():.2f}  Nov≈{under.loc['2025-11-01':'2025-11-30'].mean():.2f}")
    print(f"overstory PAI   peak≈{over.max():.2f}  Nov≈{over.loc['2025-11-01':'2025-11-30'].mean():.2f}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
