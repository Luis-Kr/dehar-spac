"""Correlate DHP plot-mean LAI against the lidar PAI processing variants (2025).

Off-pipeline validation (lives in other/, never feeds a headline number).

Question: over the 2025 leaf season (Apr -> end Oct), how well does each
up-drift / tilt PAI processing variant track the independent DHP ground LAI?

PAI variants (daily-mean total weighted PAI, same recipe as the canonical
pipeline: per scan take max-over-height WeightedPAI, keep quality_all scans,
then 24 h daily mean) -- the full 5-line tilt-comparison figure:
- hemi_hi weighted, UNcorrected (rotation, no up-drift)   leaf_pre_updrift_fix_*/leaf_hemi_hi_2025
- hemi_hi weighted, up-drift only (NO tilt)               variants/..._none_updrift
- hemi_hi weighted, up-drift + offset/TLS tilt            variants/..._offset_updrift
- hemi_hi weighted, up-drift + rotation tilt [canonical]  leaf/leaf_hemi_hi_2025
- hemi_hi weighted, up-drift (DAILY MEDIAN) + rotation     variants/..._dailymedian_updrift
- hinge scan weighted, dedicated (drifting)               leaf/leaf_hinge_2025

Note: the "..._none_updrift" parquet is tilt=none + up-drift ON (up-drift only),
NOT the uncorrected geometry; the genuine no-up-drift series is the pre-up-drift
snapshot (rotation, up-drift OFF) the daily table uses for its _uncorr columns.

Ground truth: DHP true LAI at 5 VOD positions, plot mean +/-1 SD.

Outputs (other/):
- dhp_lai_vs_pai_variants_2025.png   time-series overlay + per-variant scatter
- dhp_lai_vs_pai_variants_2025.csv   correlation table

Run:
    /home/lk1167/miniconda3/envs/dehar-spac/bin/python \
        other/dhp_lai_vs_pai_variants_2025.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "paper" / "config"))
from paper_common import paper_style  # noqa: E402

PROC = REPO / "data/processed/proximal_rs"
LEAF = PROC / "leaf"
DHP_CSV = REPO / "data/raw/proximal_rs/dhp/LAI_processed_VOD_Hartheim2_20260415.csv"
OUT_DIR = REPO / "other"

WIN = (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-10-31"))
MATCH_TOL = pd.Timedelta("3D")   # sample PAI at the nearest day to each DHP scene

# label -> (parquet, color, linestyle, linewidth)
VARIANTS = {
    "UNcorrected (no up-drift)": (
        PROC / "leaf_pre_updrift_fix_20260626/leaf_hemi_hi_2025.parquet",
        "#7fb3d5", "--", 1.8),
    "up-drift only (NO tilt)": (
        LEAF / "variants/leaf_hemi_hi_2025_none_updrift.parquet",
        "tab:purple", "-", 1.8),
    "up-drift + offset/TLS tilt": (
        LEAF / "variants/leaf_hemi_hi_2025_offset_updrift.parquet",
        "tab:orange", "-", 1.8),
    "up-drift + rotation tilt [canonical]": (
        LEAF / "leaf_hemi_hi_2025.parquet",
        "tab:blue", "-", 2.6),
    "up-drift (DAILY MEDIAN) + rotation tilt": (
        LEAF / "variants/leaf_hemi_hi_2025_dailymedian_updrift.parquet",
        "tab:brown", "-", 1.8),
    "hinge scan (dedicated, drifting)": (
        LEAF / "leaf_hinge_2025.parquet",
        "tab:red", "-", 1.8),
}


def daily_total_weighted_pai(path: Path) -> pd.Series:
    """Daily-mean canopy-total WeightedPAI from a LEAF parquet (pipeline recipe)."""
    df = pd.read_parquet(path)
    g = df.groupby("datetime")
    total = g["WeightedPAI"].max()                 # canopy-integrated total per scan
    good = g["quality_all"].first() == True        # noqa: E712
    s = total.where(good)
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.resample("1D").mean()


def load_dhp_plot_mean(path: Path) -> tuple[pd.Series, pd.Series]:
    """DHP plot-mean LAI and spatial SD across the 5 VOD positions."""
    raw = pd.read_csv(path)
    date_cols = [c for c in raw.columns if str(c).isdigit()]
    dates = pd.to_datetime(date_cols, format="%Y%m%d")
    vals = raw[date_cols].astype(float)
    mean = pd.Series(vals.mean(axis=0).values, index=dates)
    sd = pd.Series(vals.std(axis=0, ddof=1).values, index=dates)
    return mean, sd


def main() -> None:
    paper_style()
    pai = {lab: daily_total_weighted_pai(p) for lab, (p, *_) in VARIANTS.items()}
    lai_mean, lai_sd = load_dhp_plot_mean(DHP_CSV)

    in_win = (lai_mean.index >= WIN[0]) & (lai_mean.index <= WIN[1])
    lai_mean, lai_sd = lai_mean[in_win], lai_sd[in_win]

    # ---- correlation table: PAI sampled at the DHP scene dates --------------
    rows, matched = [], {}
    for lab, s in pai.items():
        samp = s.reindex(lai_mean.index, method="nearest", tolerance=MATCH_TOL)
        ok = samp.notna() & lai_mean.notna()
        matched[lab] = (samp, ok)
        x, y = samp[ok].values, lai_mean[ok].values
        rp, pp = pearsonr(x, y)
        rs, ps = spearmanr(x, y)
        slope, icpt = np.polyfit(x, y, 1)
        rows.append({
            "variant": lab, "n": int(ok.sum()),
            "pearson_r": round(rp, 3), "pearson_p": round(pp, 4),
            "spearman_r": round(rs, 3), "spearman_p": round(ps, 4),
            "mean_bias_pai_minus_lai": round(float((x - y).mean()), 3),
            "slope": round(slope, 3), "intercept": round(icpt, 3),
        })
    table = pd.DataFrame(rows)
    out_csv = OUT_DIR / "dhp_lai_vs_pai_variants_2025.csv"
    table.to_csv(out_csv, index=False)
    print(table.to_string(index=False))
    print(f"\nwrote {out_csv}")

    # ---- figure: time series (top) + per-variant scatter (bottom row) -------
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, len(VARIANTS), height_ratios=[1.25, 1.0],
                          hspace=0.32, wspace=0.32)

    axts = fig.add_subplot(gs[0, :])
    for lab, (path, color, ls, lw) in VARIANTS.items():
        s = pai[lab]
        s = s[(s.index >= WIN[0]) & (s.index <= WIN[1])]
        axts.plot(s.index, s.values, color=color, ls=ls, lw=lw, label=lab)
    axts.fill_between(lai_mean.index, lai_mean - lai_sd, lai_mean + lai_sd,
                      color="tab:green", alpha=0.15, lw=0)
    axts.plot(lai_mean.index, lai_mean, color="tab:green", marker="o", lw=2.4,
              label="DHP plot-mean LAI $\\pm$1 SD (n=5)")
    axts.set_ylabel("Total PAI / LAI (m$^2$ m$^{-2}$)")
    axts.set_title("DHP plot-mean LAI vs lidar PAI variants — Apr–Oct 2025")
    axts.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axts.legend(fontsize=8, loc="lower left", ncol=2)

    lo = min(lai_mean.min(), min(pai[l].loc[WIN[0]:WIN[1]].min() for l in pai))
    hi = max(lai_mean.max(), max(pai[l].loc[WIN[0]:WIN[1]].max() for l in pai))
    pad = 0.3
    for j, (lab, (path, color, ls, lw)) in enumerate(VARIANTS.items()):
        ax = fig.add_subplot(gs[1, j])
        samp, ok = matched[lab]
        x, y = samp[ok].values, lai_mean[ok].values
        ax.scatter(x, y, color=color, s=42, zorder=3, edgecolor="0.3", lw=0.4)
        xs = np.array([lo - pad, hi + pad])
        slope, icpt = np.polyfit(x, y, 1)
        ax.plot(xs, slope * xs + icpt, color=color, lw=1.8)
        ax.plot(xs, xs, color="0.6", ls=":", lw=1.0)        # 1:1 reference
        rp, _ = pearsonr(x, y)
        ax.set_title(f"{lab}\nr={rp:+.2f}  n={ok.sum()}", fontsize=8)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", "box")
        ax.set_xlabel("PAI variant (m$^2$ m$^{-2}$)", fontsize=8)
        if j == 0:
            ax.set_ylabel("DHP plot-mean LAI", fontsize=8)

    out_png = OUT_DIR / "dhp_lai_vs_pai_variants_2025.png"
    fig.savefig(out_png)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
