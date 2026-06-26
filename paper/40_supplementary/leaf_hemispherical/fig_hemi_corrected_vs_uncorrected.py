"""Supplementary figure: LEAF hemispherical scans, uncorrected vs corrected.

Four scenes across the 2025 season, each shown twice as a fisheye (r = zenith,
colour = canopy height): LEFT the raw fold about the hard-wired "up" = 180 deg,
RIGHT re-folded about the per-scan calibrated "up" (``self_calibrate``, GitHub
issue #11). By autumn the calibrated "up" drifts to ~204 deg; uncorrected, the
dome splits along the N-S seam with a displaced apex, and the correction
re-registers the two mirror half-sweeps into one continuous hemisphere.

This is the paper-supplement version of section 10 of
``notebooks/00_exploration/leaf_transform_playground.ipynb``.

Run:
  /home/lk1167/miniconda3/envs/dehar-spac/bin/python \
      paper/40_supplementary/leaf_hemispherical/fig_hemi_corrected_vs_uncorrected.py

Outputs:
  outputs/hemi_uncorrected_vs_corrected.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "paper" / "config"))
import _leaf_hemi_common as hemi  # noqa: E402
from paper_common import paper_style  # noqa: E402

# ===========================================================================
# PARAMETERS  (edit these)
# ===========================================================================
SCENES = ["2025-04-22", "2025-06-15", "2025-08-15", "2025-10-15"]  # one per row
TARGET_HOUR = 2          # predawn UTC (calmest); None = nearest of any hour
CORRECTED_MODE = "seasonal_up"   # "seasonal_up" (robust lookup) | "self_calibrate"
SENSOR_HEIGHT = 1.5      # m, canonical LEAF height

HEIGHT_CMAP = "turbo"
HEIGHT_VLIM = (0.0, 20.0)   # fixed; SAME scale as the seasonal-series figure
POINT_SIZE = 0.6            # scatter marker size
POINT_ALPHA = 0.8

# fonts / label sizes
FS_SUPTITLE = 16
FS_COL_HEADER = 15          # scene date + up above each column
FS_ROW_LABEL = 15           # "uncorrected" / "corrected" on the left
FS_RING_TICK = 10           # zenith ring labels
FS_AZ_TICK = 8              # azimuth labels around the rim
FS_CBAR_LABEL = 14
FS_CBAR_TICK = 12

FIG_W_PER_COL = 4.8
FIG_H_PER_ROW = 4.8
WSPACE = 0.01            # horizontal gap between panels (0 = touching)
HSPACE = 0.16           # vertical gap (room for the per-row headers)
OUT_NAME = "hemi_uncorrected_vs_corrected.png"
DPI = 300
# ===========================================================================

OUT_DIR = HERE / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def scene_points(
    filename: str, mode: str, up_lookup=None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-return (azimuth rad, zenith deg, height) for both pulses of a scan."""
    leaf, az, zen, _, _ = hemi.load_returns(
        filename, mode, SENSOR_HEIGHT, up_lookup=up_lookup
    )
    d = leaf.data
    az2 = np.radians(np.concatenate([az, az]))
    zen2 = np.concatenate([zen, zen])
    h = np.concatenate([d["h1"].to_numpy(), d["h2"].to_numpy()])
    ok = np.isfinite(h)
    return az2[ok], zen2[ok], h[ok]


def main() -> int:
    paper_style()
    meta = hemi.scan_table()
    up_lookup = hemi.load_up_lookup() if CORRECTED_MODE == "seasonal_up" else None

    # pass 1: load every scene (raw + corrected) and the "up" used for it
    scenes = []
    for date in SCENES:
        pick = hemi.pick_scan(meta, date, hour=TARGET_HOUR)
        if up_lookup is not None:
            up = hemi.seasonal_up(pick.filename, up_lookup)
        else:
            leaf_u, *_ = hemi.load_returns(pick.filename, "none", SENSOR_HEIGHT)
            up = hemi.calibrate_up(leaf_u)
        au, zu, hu = scene_points(pick.filename, "none")
        ac, zc, hc = scene_points(pick.filename, CORRECTED_MODE, up_lookup=up_lookup)
        scenes.append((pick, up, au, zu, hu, ac, zc, hc))
        print(
            f"{date}: {pick.filename}  hour={pick.scan_hour:02d}  "
            f"quality_all={bool(pick.quality_all)}  up={up:.0f} deg"
        )

    # transposed: scenes are COLUMNS, uncorrected (top) / corrected (bottom) are rows
    ncol = len(scenes)
    fig, axes = plt.subplots(
        2,
        ncol,
        figsize=(FIG_W_PER_COL * ncol, FIG_H_PER_ROW * 2),
        subplot_kw={"projection": "polar"},
        layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=WSPACE, hspace=HSPACE)
    sc = None
    for j, (pick, up, au, zu, hu, ac, zc, hc) in enumerate(scenes):
        date = pick["datetime"].strftime("%Y-%m-%d")
        rows = [(au, zu, hu), (ac, zc, hc)]   # row 0 = uncorrected, row 1 = corrected
        for i, (az, zen, h) in enumerate(rows):
            sc = axes[i, j].scatter(
                az, zen, c=np.clip(h, *HEIGHT_VLIM), s=POINT_SIZE, cmap=HEIGHT_CMAP,
                vmin=HEIGHT_VLIM[0], vmax=HEIGHT_VLIM[1], alpha=POINT_ALPHA,
                rasterized=True,
            )
            hemi.style_fisheye(
                axes[i, j], zen_label_fs=FS_RING_TICK, az_label_fs=FS_AZ_TICK
            )
        # date above the uncorrected row; "up" above the CORRECTED row (row 2)
        axes[0, j].set_title(date, fontsize=FS_COL_HEADER, pad=8)
        axes[1, j].set_title(
            f"up = {up:.0f}°  (Δ {up - 180:+.0f}°)", fontsize=FS_COL_HEADER, pad=8
        )

    row_labels = ["uncorrected\n(fold @ 180°)", "corrected\n(seasonal up)"]
    for i, label in enumerate(row_labels):
        axes[i, 0].text(
            -0.32, 0.5, label, transform=axes[i, 0].transAxes,
            rotation=90, va="center", ha="center", fontsize=FS_ROW_LABEL,
        )

    cbar = fig.colorbar(sc, ax=axes, location="right", shrink=0.6, pad=0.02)
    cbar.set_label("canopy height (m)", fontsize=FS_CBAR_LABEL)
    cbar.ax.tick_params(labelsize=FS_CBAR_TICK)

    out = OUT_DIR / OUT_NAME
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
