"""Validate the MANUAL up reference against DHP LAI (up_manual candidate, 2025).

Off-pipeline validation (other/, never feeds a headline number). Sibling of
``dhp_lai_vs_pai_variants_2025.py`` but built to answer one question:

    Does re-folding the hemi scans about the hand-verified **manual up
    reference** (leaf.up_on_scan, per-scan, interpolated in time) track the
    independent DHP ground LAI *better* than the ADR 0005 smoothed-lookup
    canonical and the other up variants?

Decision rule (agreed): **best full-window Pearson r wins.** Bias and slope are
printed alongside because r is blind to absolute level, and the leaf-off
residuals at the Nov DHP scenes are called out because that is where the
manual and smoothed "up" diverge most.

Window is **Apr -> Dec 15 2025** (the full leaf-scan span) so the leaf-off DHP
scenes (2025-10-09, 10-21, 11-05, 11-20) count -- the existing variants script
stops at Oct 31 and cannot see them.

Ground truth: DHP true LAI at 5 VOD positions, plot mean +/-1 SD.

Outputs (other/):
- dhp_lai_vs_pai_upmanual_2025.png   time-series overlay + per-variant scatter
- dhp_lai_vs_pai_upmanual_2025.csv   correlation table (sorted by pearson_r)

Run:
    /home/lk1167/miniconda3/envs/dehar-spac/bin/python \
        other/dhp_lai_vs_pai_upmanual_2025.py
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

WIN = (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-12-15"))
MATCH_TOL = pd.Timedelta("3D")
LEAF_OFF = pd.to_datetime(["2025-10-09", "2025-10-21", "2025-11-05", "2025-11-20"])

# label -> (parquet, color, linestyle, linewidth)
VARIANTS = {
    "UNcorrected (no up-drift)": (
        PROC / "leaf_pre_updrift_fix_20260626/leaf_hemi_hi_2025.parquet",
        "#7fb3d5", "--", 1.6),
    "smoothed up [ADR 0005 canonical]": (
        LEAF / "leaf_hemi_hi_2025.parquet",
        "tab:blue", "-", 2.2),
    "daily-median up": (
        LEAF / "variants/leaf_hemi_hi_2025_dailymedian_updrift.parquet",
        "tab:brown", "-", 1.6),
    "offset tilt + up-drift": (
        LEAF / "variants/leaf_hemi_hi_2025_offset_updrift.parquet",
        "tab:orange", "-", 1.6),
    "MANUAL up [candidate]": (
        LEAF / "variants/leaf_hemi_hi_2025_upmanual.parquet",
        "black", "-", 2.8),
    "hinge scan (dedicated, drifting)": (
        LEAF / "leaf_hinge_2025.parquet",
        "tab:red", "-", 1.4),
}


def daily_total_weighted_pai(path: Path) -> pd.Series:
    """Daily-mean canopy-total WeightedPAI from a LEAF parquet (pipeline recipe)."""
    df = pd.read_parquet(path)
    g = df.groupby("datetime")
    total = g["WeightedPAI"].max()
    good = g["quality_all"].first() == True  # noqa: E712
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

    rows, matched = [], {}
    for lab, s in pai.items():
        samp = s.reindex(lai_mean.index, method="nearest", tolerance=MATCH_TOL)
        ok = samp.notna() & lai_mean.notna()
        matched[lab] = (samp, ok)
        x, y = samp[ok].values, lai_mean[ok].values
        rp, pp = pearsonr(x, y)
        rs, ps = spearmanr(x, y)
        slope, icpt = np.polyfit(x, y, 1)
        # leaf-off residual: PAI - LAI at the Nov-ish scenes, mean abs
        lo_idx = lai_mean.index.intersection(LEAF_OFF)
        lo_res = float((samp.reindex(lo_idx) - lai_mean.reindex(lo_idx)).abs().mean())
        rows.append({
            "variant": lab, "n": int(ok.sum()),
            "pearson_r": round(rp, 3), "pearson_p": round(pp, 4),
            "spearman_r": round(rs, 3), "spearman_p": round(ps, 4),
            "mean_bias_pai_minus_lai": round(float((x - y).mean()), 3),
            "slope": round(slope, 3), "intercept": round(icpt, 3),
            "leafoff_abs_resid": round(lo_res, 3),
        })
    table = pd.DataFrame(rows).sort_values("pearson_r", ascending=False)
    out_csv = OUT_DIR / "dhp_lai_vs_pai_upmanual_2025.csv"
    table.to_csv(out_csv, index=False)
    print(table.to_string(index=False))
    winner = table.iloc[0]["variant"]
    print(f"\nbest full-window Pearson r: {winner!r}")
    print(f"wrote {out_csv}")

    # ---- figure: time series (top) + per-variant scatter (bottom) -----------
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
    axts.set_title("DHP plot-mean LAI vs lidar PAI up variants — Apr–Dec 2025")
    axts.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axts.legend(fontsize=8, loc="lower left", ncol=2)

    lo = min(lai_mean.min(), min(pai[k].loc[WIN[0]:WIN[1]].min() for k in pai))
    hi = max(lai_mean.max(), max(pai[k].loc[WIN[0]:WIN[1]].max() for k in pai))
    pad = 0.3
    for j, (lab, (path, color, ls, lw)) in enumerate(VARIANTS.items()):
        ax = fig.add_subplot(gs[1, j])
        samp, ok = matched[lab]
        x, y = samp[ok].values, lai_mean[ok].values
        ax.scatter(x, y, color=color, s=42, zorder=3, edgecolor="0.3", lw=0.4)
        xs = np.array([lo - pad, hi + pad])
        slope, icpt = np.polyfit(x, y, 1)
        ax.plot(xs, slope * xs + icpt, color=color, lw=1.8)
        ax.plot(xs, xs, color="0.6", ls=":", lw=1.0)
        rp, _ = pearsonr(x, y)
        ax.set_title(f"{lab}\nr={rp:+.2f}  n={ok.sum()}", fontsize=8)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", "box")
        ax.set_xlabel("PAI variant (m$^2$ m$^{-2}$)", fontsize=8)
        if j == 0:
            ax.set_ylabel("DHP plot-mean LAI", fontsize=8)

    out_png = OUT_DIR / "dhp_lai_vs_pai_upmanual_2025.png"
    fig.savefig(out_png)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
