"""Plot DHP LAI (5 VOD positions) and compare its plot-average to canonical PAI 2025.

Off-pipeline diagnostic (lives in other/, never feeds a headline number).

Inputs
------
- data/raw/proximal_rs/dhp/LAI_processed_VOD_Hartheim2_20260415.csv
    Wide DHP table: one row per VOD position (vod1..vod5), one column per
    acquisition date (YYYYMMDD). Values are true LAI (m2 m-2) from digital
    hemispherical photography. Spans 2025-03-26 .. 2026-04-02.
- data/processed/dehar_daily_2025.parquet
    Canonical daily table. Canonical proximal "bulk biomass" PAI = up-corrected
    hemi_hi == pai_hemi_hi_hinge_mean_m2m2 (analysis_config streams.pai, ADR 0005).

Outputs
-------
- other/dhp_lai_timeseries.png         per-position DHP LAI time series
- other/dhp_lai_vs_canonical_pai.png   plot-mean LAI (+/-1 SD) vs canonical PAI 2025

Run:
    /home/lk1167/miniconda3/envs/dehar-spac/bin/python other/dhp_lai_vs_canonical_pai.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "paper" / "config"))
from paper_common import paper_style  # noqa: E402

DHP_CSV = REPO / "data/raw/proximal_rs/dhp/LAI_processed_VOD_Hartheim2_20260415.csv"
DAILY = REPO / "data/processed/dehar_daily_2025.parquet"
PAI_COL = "pai_hemi_hi_hinge_mean_m2m2"   # canonical up-corrected hemi_hi (== weighted total)
PAI_STD_COL = "pai_hemi_hi_hinge_std_m2m2"
OUT_DIR = REPO / "other"


def load_dhp_lai(path: Path) -> pd.DataFrame:
    """Read the wide DHP table into positions-x-dates LAI (tidy, datetime cols)."""
    raw = pd.read_csv(path)
    date_cols = [c for c in raw.columns if str(c).isdigit()]
    dates = pd.to_datetime(date_cols, format="%Y%m%d")
    lai = raw.set_index("ID")[date_cols]
    lai.columns = dates
    return lai.T.sort_index()   # index = date, columns = vod1..vod5


def load_canonical_pai(path: Path) -> pd.DataFrame:
    """Daily canonical PAI mean + SD, tz-naive index for clean plotting."""
    df = pd.read_parquet(path)[[PAI_COL, PAI_STD_COL]].dropna(subset=[PAI_COL])
    df.index = df.index.tz_localize(None)
    return df


def main() -> None:
    paper_style()
    lai = load_dhp_lai(DHP_CSV)            # rows: dates, cols: positions
    pai = load_canonical_pai(DAILY)

    lai_mean = lai.mean(axis=1)
    lai_sd = lai.std(axis=1, ddof=1)

    # ---- Figure 1: per-position DHP LAI -------------------------------------
    fig1, ax = plt.subplots(figsize=(11, 5))
    palette = plt.cm.viridis(np.linspace(0.05, 0.85, lai.shape[1]))
    for color, pos in zip(palette, lai.columns):
        ax.plot(lai.index, lai[pos], marker="o", color=color, label=pos)
    ax.set_ylabel("DHP LAI (m$^2$ m$^{-2}$)")
    ax.set_title("DHP true LAI per VOD position — DE-Har")
    ax.legend(title="position", ncol=5, fontsize=8, loc="lower center")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    fig1.tight_layout()
    out1 = OUT_DIR / "dhp_lai_timeseries.png"
    fig1.savefig(out1)
    print(f"wrote {out1}")

    # ---- Figure 2: plot-mean LAI vs canonical PAI (shared y-axis) -----------
    fig2, ax = plt.subplots(figsize=(11, 5.5))

    ax.fill_between(lai_mean.index, lai_mean - lai_sd, lai_mean + lai_sd,
                    color="tab:green", alpha=0.18, lw=0,
                    label="DHP LAI plot mean $\\pm$1 SD (n=5 positions)")
    ax.plot(lai_mean.index, lai_mean, marker="o", color="tab:green",
            label="DHP LAI plot mean")

    ax.fill_between(pai.index, pai[PAI_COL] - pai[PAI_STD_COL],
                    pai[PAI_COL] + pai[PAI_STD_COL],
                    color="tab:purple", alpha=0.15, lw=0)
    ax.plot(pai.index, pai[PAI_COL], color="tab:purple",
            label="Canonical PAI 2025 (up-corr. hemi_hi)")

    ax.set_ylabel("Leaf / plant area index (m$^2$ m$^{-2}$)")
    ax.set_title("DHP plot-mean LAI vs canonical lidar PAI — same y-scale")
    ax.legend(fontsize=9, loc="lower center", ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    fig2.tight_layout()
    out2 = OUT_DIR / "dhp_lai_vs_canonical_pai.png"
    fig2.savefig(out2)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
