"""Supplementary figure: seasonal LEAF hemispherical series (self-calibrated).

Wide layout: one good-quality night scan per month as a column (Apr - Oct, July
dropped), three fisheye rows (r = zenith, N up), all on the ``seasonal_up``
geometry (smoothed daily "up" lookup, GitHub issue #11):

  row 1 - canopy height        (per-return scatter, colour = height)
  row 2 - gap fraction, binned (az/zen cells, continuous P_gap, log scale)
  row 3 - gap fraction, boolean (per-shot scatter: sky = gap, canopy = hit)

Each column header carries the canonical canopy-total **WeightedPAI** read from
``leaf_hemi_hi_2025.parquet`` (the adopted ``leaf_tilt_rotation`` pipeline), so
the figure doubles as a check on the canonical data; a validation table
(per-cell gap at 57.5 deg vs the canonical Pgap) is printed to stdout.

Prerequisite: the seasonal-"up" lookup (run scripts/build_leaf_up_lookup.py once).

Run:
  /home/lk1167/miniconda3/envs/dehar-spac/bin/python \
      paper/40_supplementary/leaf_hemispherical/fig_hemi_seasonal_series.py

Outputs:
  outputs/hemi_seasonal_series.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "paper" / "config"))
import _leaf_hemi_common as hemi  # noqa: E402
from paper_common import paper_style  # noqa: E402

# ===========================================================================
# PARAMETERS  (edit these)
# ===========================================================================
# One scene per COLUMN: aim for the ~20th of each month, good quality, at night.
MONTH_DATES = [
    "2025-04-20",
    "2025-05-20",
    #"2025-06-20",
    "2025-08-20",   # July dropped
    "2025-09-20",
    "2025-10-20",
    "2025-11-20",
]
TARGET_HOUR = 20         # night UTC (~21:00 local); None = nearest of any hour
MODE = "seasonal_up"     # smoothed daily "up" lookup (robust; issue #11)
SENSOR_HEIGHT = 1.5      # m

POINT_SIZE = 0.7         # scatter marker size (raise to see more)
POINT_ALPHA = 0.85
SUBSAMPLE = 1            # plot every Nth point (1 = all)

# row 1: canopy height
HEIGHT_CMAP = "turbo"
HEIGHT_VLIM = (0.0, 20.0)   # fixed; SAME scale as the corrected-vs-uncorrected figure

# row 2: real (binned) gap fraction -- az/zen cells, continuous P_gap (log scale)
GAP_ABIN = 3.0              # azimuth bin (deg)
GAP_ZBIN = 3.0              # zenith bin (deg)
GAP_ZMAX = 88.0            # outer radius (deg zenith)
GAP_MIN_COUNT = 20         # cells with fewer shots are masked (grey)
REALGAP_CMAP = "bone"      # dark = closed, white = open (sky)
REALGAP_VLIM = (1e-3, 1.0)  # log scale (canopy gap is tiny)

# row 3: boolean gap fraction (per shot)
CANOPY_COLOR = "#1b5e20"    # gap = 0  (beam hit canopy)
SKY_COLOR = "#bfe6ff"       # gap = 1  (beam reached the sky)

# fonts / label sizes
FS_COL_HEADER = 19          # month + PAI above each column
FS_ROW_LABEL = 18           # row labels on the left
FS_RING_TICK = 11           # zenith ring labels
FS_AZ_TICK = 14              # azimuth labels around the rim
FS_CBAR_LABEL = 16
FS_CBAR_TICK = 14

FIG_W_PER_COL = 4.0
FIG_H_PER_ROW = 4.0
WSPACE = 0.01           # horizontal gap between panels (0 = touching)
HSPACE = 0.08           # vertical gap between rows
OUT_NAME = "hemi_seasonal_series.png"
DPI = 200
# ===========================================================================

OUT_DIR = HERE / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _cmap(name: str):
    """Colormap with a fixed grey for masked (no-data) cells."""
    c = plt.get_cmap(name).copy()
    c.set_bad("0.85")
    return c


def _style(ax, title: str = "") -> None:
    hemi.style_fisheye(
        ax, title=title, title_fs=FS_COL_HEADER,
        zen_label_fs=FS_RING_TICK, az_label_fs=FS_AZ_TICK,
    )


def scene_arrays(filename: str, up_lookup=None):
    """Height scatter, boolean-gap scatter, and binned gap grid for one scan.

    Returns ``(height, boolean_gap, binned_gap)`` where ``height`` /
    ``boolean_gap`` are ``(az_rad, zen_deg, value)`` tuples and ``binned_gap`` is
    the ``(theta_edges, r_edges, P_gap_grid)`` from :func:`hemi.fisheye_grid`.
    """
    leaf, az, zen, h1, gap = hemi.load_returns(
        filename, MODE, SENSOR_HEIGHT, up_lookup=up_lookup
    )
    h2 = leaf.data["h2"].to_numpy()
    az_h = np.radians(np.concatenate([az, az]))      # both returns (notebook style)
    zen_h = np.concatenate([zen, zen])
    h = np.concatenate([h1, h2])
    ok = np.isfinite(h)
    s = slice(None, None, SUBSAMPLE)
    height = (az_h[ok][s], zen_h[ok][s], h[ok][s])
    boolean = (np.radians(az)[s], zen[s], gap.astype(float)[s])
    binned = hemi.fisheye_grid(
        az, zen, gap.astype(float), "mean", GAP_ABIN, GAP_ZBIN, GAP_ZMAX, GAP_MIN_COUNT
    )
    return height, boolean, binned


def main() -> int:
    paper_style()
    meta = hemi.scan_table()
    up_lookup = hemi.load_up_lookup() if MODE == "seasonal_up" else None

    scenes = []
    print(f"{'date':12s} {'picked':>10s} {'h':>2s} {'q':>1s} {'|d|':>4s} {'up':>4s} "
          f"{'PAI_canon':>9s} {'gap57_calc':>10s} {'gap57_canon':>11s}")
    for date in MONTH_DATES:
        pick = hemi.pick_scan(meta, date, hour=TARGET_HOUR)
        height, boolean, binned = scene_arrays(pick.filename, up_lookup=up_lookup)
        up = (
            hemi.seasonal_up(pick.filename, up_lookup)
            if up_lookup is not None
            else np.nan
        )
        zen_g, g = boolean[1], boolean[2]
        ring = (zen_g >= 55) & (zen_g < 60)
        gap57 = float(np.mean(g[ring])) if ring.any() else np.nan
        days_off = abs((pick["datetime"] - pd.Timestamp(date, tz="UTC")).days)
        scenes.append((pick, height, boolean, binned))
        print(
            f"{date:12s} {pick.filename.split('_')[1]:>10s} {pick.scan_hour:02d} "
            f"{int(pick.quality_all):>1d} {days_off:>4d} {up:>4.0f} "
            f"{pick.weighted_pai:>9.2f} {gap57:>10.3f} {pick.pgap_hinge:>11.3f}"
        )

    ncol = len(scenes)
    fig, axes = plt.subplots(
        3,
        ncol,
        figsize=(FIG_W_PER_COL * ncol, FIG_H_PER_ROW * 3),
        subplot_kw={"projection": "polar"},
        layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=WSPACE, hspace=HSPACE)

    gap_cmap = ListedColormap([CANOPY_COLOR, SKY_COLOR])
    gap_norm = BoundaryNorm([0.0, 0.5, 1.0], gap_cmap.N)

    m_h = m_b = m_bool = None
    for j, (pick, (azh, zenh, h), (azg, zeng, g), (th, r, ggrid)) in enumerate(scenes):
        m_h = axes[0, j].scatter(
            azh, zenh, c=np.clip(h, *HEIGHT_VLIM), s=POINT_SIZE, cmap=HEIGHT_CMAP,
            vmin=HEIGHT_VLIM[0], vmax=HEIGHT_VLIM[1], alpha=POINT_ALPHA,
            rasterized=True,
        )
        m_b = axes[1, j].pcolormesh(
            th, r, np.ma.clip(ggrid, *REALGAP_VLIM), cmap=_cmap(REALGAP_CMAP),
            norm=LogNorm(*REALGAP_VLIM), shading="flat",
        )
        m_bool = axes[2, j].scatter(
            azg, zeng, c=g, s=POINT_SIZE, cmap=gap_cmap, norm=gap_norm,
            alpha=POINT_ALPHA, rasterized=True,
        )
        for i in range(3):
            _style(axes[i, j])
        month = pick["datetime"].strftime("%b %d")
        # axes[0, j].set_title(
        #     f"{month}\nPAI = {pick.weighted_pai:.2f}", fontsize=FS_COL_HEADER, pad=12
        # )

    row_labels = ["canopy height", "gap fraction\n(binned)", "gap fraction\n(boolean)"]
    for row, label in enumerate(row_labels):
        axes[row, 0].text(
            -0.32, 0.5, label, transform=axes[row, 0].transAxes,
            rotation=90, va="center", ha="center", fontsize=FS_ROW_LABEL,
        )

    cb_h = fig.colorbar(
        m_h, ax=axes[0, :].tolist(), location="right", shrink=0.8, pad=0.01
    )
    cb_h.set_label("height (m)", fontsize=FS_CBAR_LABEL)
    cb_h.ax.tick_params(labelsize=FS_CBAR_TICK)
    cb_b = fig.colorbar(
        m_b, ax=axes[1, :].tolist(), location="right", shrink=0.8, pad=0.01
    )
    cb_b.set_label("$P_{gap}$", fontsize=FS_CBAR_LABEL)
    cb_b.ax.tick_params(labelsize=FS_CBAR_TICK)
    cb_bool = fig.colorbar(
        m_bool, ax=axes[2, :].tolist(), location="right", shrink=0.55, pad=0.01,
        ticks=[0.25, 0.75],
    )
    cb_bool.set_ticklabels(["canopy", "sky"], fontsize=FS_CBAR_TICK)

    out = OUT_DIR / OUT_NAME
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
