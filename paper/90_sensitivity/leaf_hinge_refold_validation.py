"""Validation: the dedicated hinge scan, re-folded to its TRUE angle theta(t),
agrees with the corrected hemi ring at theta(t) (issue #11, ADR 0005).

The dedicated hinge scan fires at a fixed encoder setpoint labelled 57.5 deg, but
its *true* zenith is |setpoint - up(t)|, which drifts (~61 deg Apr -> ~30 deg
Dec) as the scanner settles. If that drift (not real canopy change) is what moves
the hinge-scan PAI, then re-folding the hinge about the seasonal "up" and reading
its directional gap at theta(t) should match the corrected hemi's gap at the SAME
zenith theta(t). This script measures both (raw directional gap = beams that reach
the sky) on matched-day hinge/hemi_hi pairs and plots them against the 1:1 line.

Run:
  /home/lk1167/miniconda3/envs/dehar-spac/bin/python \
      paper/90_sensitivity/leaf_hinge_refold_validation.py

Outputs:
  outputs/leaf_hinge_refold/hinge_refold_vs_hemi.png   (+ .csv)
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
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "lib" / "pylidar-tls-canopy"))
sys.path.insert(0, str(ROOT / "src"))
from dehar.proximal_rs.leaf import (  # noqa: E402
    load_up_lookup,
    recenter_leaf_data,
    up_on_date,
)
from pylidar_tls_canopy import leaf_io  # noqa: E402

# ── parameters ──────────────────────────────────────────────────────────────
SAMPLE_EVERY = 5          # use every Nth good hinge scan (runtime vs density)
RING_HALFWIDTH = 2.5      # deg; hemi beams within theta +/- this define the ring
SENSOR_HEIGHT = 1.5
LEAF = ROOT / "data" / "processed" / "proximal_rs" / "leaf"
RAW = ROOT / "data" / "raw" / "proximal_rs" / "leaf"
LOOKUP = LEAF / "leaf_up_daily_2025.csv"
OUT = HERE / "outputs" / "leaf_hinge_refold"
# ────────────────────────────────────────────────────────────────────────────


def _scans(parquet: str) -> pd.DataFrame:
    cols = ["datetime", "filename", "quality_all"]
    df = pd.read_parquet(LEAF / parquet, columns=cols)
    df = df.drop_duplicates("datetime")
    df = df[df["quality_all"]].copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.sort_values("datetime").reset_index(drop=True)


def _gap_and_zenith(filename: str, up: float):
    """Re-fold a scan about ``up`` and return (zenith_deg, gap_mask)."""
    leaf = leaf_io.LeafScanFile(str(RAW / filename), sensor_height=SENSOR_HEIGHT,
                                transform=False)
    recenter_leaf_data(leaf, up)
    zen = np.degrees(leaf.data["zenith"].to_numpy())
    gap = leaf.data["range1"].isna().to_numpy() & leaf.data["range2"].isna().to_numpy()
    return zen, gap


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    lookup = load_up_lookup(LOOKUP)
    hinge = _scans("leaf_hinge_2025.parquet").iloc[::SAMPLE_EVERY]
    hemi = _scans("leaf_hemi_hi_2025.parquet")

    rows = []
    for r in hinge.itertuples(index=False):
        up = up_on_date(lookup, r.datetime.to_pydatetime())
        zh, gh = _gap_and_zenith(r.filename, up)
        theta = float(np.median(zh))
        hinge_gap = float(np.mean(gh))
        # nearest good hemi_hi in time -> directional gap at the SAME ring
        j = int((hemi["datetime"] - r.datetime).abs().values.argmin())
        ze, ge = _gap_and_zenith(hemi.iloc[j]["filename"], up)
        sel = np.abs(ze - theta) <= RING_HALFWIDTH
        if sel.sum() < 200:
            continue
        rows.append({
            "date": r.datetime.date(), "theta_deg": theta,
            "hinge_gap": hinge_gap, "hemi_gap": float(np.mean(ge[sel])),
            "n_hemi": int(sel.sum()),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "hinge_refold_vs_hemi.csv", index=False)

    r = np.corrcoef(df.hinge_gap, df.hemi_gap)[0, 1]
    mae = float(np.mean(np.abs(df.hinge_gap - df.hemi_gap)))
    print(f"n={len(df)} pairs  "
          f"theta {df.theta_deg.min():.0f}-{df.theta_deg.max():.0f} deg")
    print(f"hinge-refold gap vs hemi gap @theta:  r={r:.3f}  MAE={mae:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
    sc = ax[0].scatter(df.hemi_gap, df.hinge_gap, c=df.theta_deg, cmap="viridis",
                       s=36, edgecolor="k", lw=0.3)
    lim = [0, max(df.hinge_gap.max(), df.hemi_gap.max()) * 1.05]
    ax[0].plot(lim, lim, "k--", lw=1, label="1:1")
    ax[0].set(xlim=lim, ylim=lim, xlabel="hemi directional gap @ θ(t)",
              ylabel="hinge-scan gap (re-folded to θ(t))",
              title=f"Hinge re-fold vs hemi at the same angle  (r={r:.2f})")
    ax[0].legend(loc="upper left")
    fig.colorbar(sc, ax=ax[0], label="hinge true zenith θ (deg)")

    d = pd.to_datetime(df.date)
    ax[1].plot(d, df.theta_deg, "o-", color="#762a83", ms=4)
    ax[1].axhline(57.5, color="0.5", ls="--", lw=1, label="labelled 57.5°")
    ax[1].set(ylabel="hinge true zenith θ(t) (deg)", xlabel="date",
              title="Hinge scan's true angle drifts off 57.5° over the season")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "hinge_refold_vs_hemi.png", dpi=130)
    print(f"wrote {OUT / 'hinge_refold_vs_hemi.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
