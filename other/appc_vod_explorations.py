"""Exploratory App C diagnostics (other/, off-pipeline — never a headline number).

Four explorations, each a multi-panel figure in other/:
  1. PAI vs raw VOD: scatter (the biomass-fit basis) + how their correlation
     evolves over the expanding season window.
  2. Sap flow: raw VOD vs sap flow (structural), and water-VOD vs the
     SEASONALLY-DETRENDED sap flow (the water residual), + the two time series.
  3. Correct VOD with a SEASONAL-FUNCTION fit of PAI (deg-4 polynomial in
     day-of-season) instead of the raw daily PAI; effect on water-VOD <-> SWP.
  4. LOESS half-width {10,20,30,40,50 d} on VOD: how water-VOD <-> SWP evolves
     over the expanding window.

All correlations are cumulative from May 1 (reuse vod_pai_correction.window_resolved):
each point = Pearson r over every valid daily pair from May 1 to that date.

Run:
  /home/lk1167/miniconda3/envs/dehar-spac/bin/python other/appc_vod_explorations.py
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
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "paper" / "config"))
sys.path.insert(0, str(REPO / "paper" / "30_appC_vod_correction"))
from paper_common import load_config, load_daily, paper_style  # noqa: E402
import vod_pai_correction as V  # noqa: E402

paper_style()
OUT = REPO / "other"
NO_EXCL = (pd.Timestamp("2099-01-01"), pd.Timestamp("2099-01-02"))  # event-blind fit


def _r(a, b):
    d = pd.concat([pd.Series(a).rename("a"), pd.Series(b).rename("b")], axis=1).dropna()
    return (stats.pearsonr(d["a"], d["b"])[0], len(d)) if len(d) > 3 else (np.nan, len(d))


def setup():
    cfg = load_config()
    df = load_daily(cfg).join(V.build_vod_daily_max(cfg))
    ac = cfg["appC"]
    vod = "vod_daily_max"
    pai = "pai_hemi_hi_weighted_mean_m2m2"
    swp = ac["validation"]["primary"]
    sap = ac["validation"]["corroborate"][1]
    season = [pd.Timestamp(x) for x in ac["window"]]
    event = [pd.Timestamp(x) for x in ac["event_exclude"]]
    dfw = df.loc[season[0]:season[1]].copy()
    corr, fits = V.corrections(dfw, vod, cfg)          # event-blind (ADR 0007)
    ends = pd.date_range(season[0] + pd.Timedelta("75D"), season[1], freq="7D")
    return cfg, ac, dfw, vod, pai, swp, sap, season, event, ends, corr, fits


def shade(ax, event):
    ax.axvspan(pd.Timestamp(event[0]), pd.Timestamp(event[1]), color="0.85", alpha=0.7)


# --------------------------------------------------------------------------- #
def fig1_pai_vs_vod(dfw, vod, pai, fits, season, event, ends):
    a, b, r2 = fits["hemi-PAI"]["a"], fits["hemi-PAI"]["b"], fits["hemi-PAI"]["r2"]
    fig, ax = plt.subplots(2, 1, figsize=(9, 9))
    d = dfw[[pai, vod]].dropna()
    doy = (d.index - season[0]).days
    sc = ax[0].scatter(d[pai], d[vod], c=doy, cmap="viridis", s=22, edgecolor="none")
    xs = np.array([d[pai].min(), d[pai].max()])
    ax[0].plot(xs, a + b * xs, "k--", lw=1.8, label=f"biomass fit b={b:.3f}, R²={r2:.2f}")
    r, n = _r(d[pai], d[vod])
    ax[0].set(xlabel="hemi-PAI (m² m⁻²)", ylabel="raw VOD (daily max)",
              title=f"(1a) raw VOD vs hemi-PAI — r={r:.2f}, n={n} (colour = day of season)")
    ax[0].legend(loc="best")
    fig.colorbar(sc, ax=ax[0], label="days since May 1")
    wr = V.window_resolved(dfw, [("PAI ↔ raw VOD", pai, vod)], season[0], ends)
    ax[1].plot(wr["end"], wr["r"], marker="o", ms=4, lw=2.4, color="#5e3c99")
    ax[1].axhline(0, color="0.5", lw=0.8); shade(ax[1], event)
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax[1].set(xlabel="evaluation window end (from May 1)", ylabel="Pearson r",
              title="(1b) cumulative correlation PAI ↔ raw VOD (the biomass-fit basis)")
    fig.tight_layout(); fig.savefig(OUT / "appc_expl_1_pai_vs_vod.png", dpi=150); plt.close(fig)
    return r


# --------------------------------------------------------------------------- #
def fig2_sapflow(dfw, vod, swp, sap, water, event):
    sap_detr = dfw[sap] - V.loess_baseline(dfw[sap], 30).reindex(dfw.index)
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.9], hspace=0.32, wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1])
    axts = fig.add_subplot(gs[1, :])
    r1, n1 = _r(dfw[sap], dfw[vod])
    ax1.scatter(dfw[sap], dfw[vod], s=20, color="#8c510a", edgecolor="none", alpha=0.7)
    ax1.set(xlabel="sap flux density", ylabel="raw VOD",
            title=f"(2a) raw VOD vs sap flow — r={r1:.2f}, n={n1}\n(structural / seasonal co-decline)")
    r2, n2 = _r(sap_detr, water)
    ax2.scatter(sap_detr, water, s=20, color="#2166ac", edgecolor="none", alpha=0.7)
    ax2.set(xlabel="sap flow, seasonally detrended (LOESS ±30 d residual)", ylabel="water-VOD (hemi-PAI)",
            title=f"water-VOD vs detrended sap flow — r={r2:.2f}, n={n2}")
    axts.plot(dfw.index, V._smooth7(V.z(water)), color="#2166ac", lw=2.2, label="water-VOD (hemi-PAI)")
    axts.plot(dfw.index, V._smooth7(V.z(sap_detr)), color="#8c510a", lw=2.2,
              label="sap flow, seasonally detrended")
    axts.axhline(0, color="0.6", lw=0.6); shade(axts, event)
    axts.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axts.set(ylabel="z-score (7 d smooth)",
             title="(2b) water-VOD vs seasonally-detrended sap flow — time series")
    axts.legend(loc="best")
    fig.suptitle("App C exploration 2 — sap flow: raw is structural; the water residual is what water-VOD should track")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "appc_expl_2_sapflow.png", dpi=150); plt.close(fig)
    return r1, r2


# --------------------------------------------------------------------------- #
def fig3_seasonal_pai(dfw, vod, pai, swp, water, season, event, ends):
    d = dfw[pai].dropna()
    x = (d.index - season[0]).days.values.astype(float)
    coef = np.polyfit(x, d.values, 3)                       # deg-4 seasonal function
    pai_fit = pd.Series(np.polyval(coef, (dfw.index - season[0]).days.astype(float)),
                        index=dfw.index)
    a2, b2, info2 = V.fit_biomass(dfw.assign(paifit=pai_fit), vod, "paifit", NO_EXCL)
    water_fit = dfw[vod] - (a2 + b2 * pai_fit)
    fig, ax = plt.subplots(2, 1, figsize=(10, 9))
    ax[0].plot(dfw.index, dfw[pai], lw=0, marker=".", ms=4, alpha=0.4, color="#238b45")
    ax[0].plot(dfw.index, pai_fit, lw=2.6, color="#238b45", label="deg-4 seasonal fit of PAI")
    ax[0].axhline(0, color="0.8", lw=0.5); shade(ax[0], event)
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax[0].set(ylabel="hemi-PAI", title="(3) top: hemi-PAI (dots) + smooth seasonal function fit")
    ax[0].legend(loc="best")
    dff = dfw.assign(water_raw=water, water_fit=water_fit)
    wr = V.window_resolved(dff, [("raw VOD (ref)", vod, swp),
                                 ("hemi-PAI (raw daily)", "water_raw", swp),
                                 ("hemi-PAI (seasonal fit)", "water_fit", swp)],
                           season[0], ends)
    sty = {"raw VOD (ref)": ("0.55", "--", 1.6), "hemi-PAI (raw daily)": ("#2166ac", "-", 2.6),
           "hemi-PAI (seasonal fit)": ("#d94801", "-", 2.4)}
    for lab, g in wr.groupby("pair"):
        c, ls, lw = sty[lab]; g = g.sort_values("end")
        ax[1].plot(g["end"], g["r"], marker="o", ms=4, ls=ls, lw=lw, color=c, label=lab)
    ax[1].axhline(0, color="0.5", lw=0.8); shade(ax[1], event)
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax[1].set(xlabel="evaluation window end (from May 1)", ylabel="Pearson r",
              title="bottom: water-VOD ↔ SWP — raw-daily PAI vs seasonal-fit PAI correction")
    ax[1].legend(loc="best")
    fig.tight_layout(); fig.savefig(OUT / "appc_expl_3_seasonal_pai.png", dpi=150); plt.close(fig)
    rr, _ = _r(water, dfw[swp]); rf, _ = _r(water_fit, dfw[swp])
    return rr, rf


# --------------------------------------------------------------------------- #
def fig4_loess_windows(dfw, vod, swp, season, event, ends):
    hws = [10, 20, 30, 40, 50, 60]
    dff = dfw.copy()
    for hw in hws:
        dff[f"loess{hw}"] = dfw[vod] - V.loess_baseline(dfw[vod], hw).reindex(dfw.index)
    pairs = [(f"LOESS ±{hw} d", f"loess{hw}", swp) for hw in hws]
    wr = V.window_resolved(dff, pairs, season[0], ends)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    cmap = plt.get_cmap("plasma")
    for i, hw in enumerate(hws):
        g = wr[wr["pair"] == f"LOESS ±{hw} d"].sort_values("end")
        ax.plot(g["end"], g["r"], marker="o", ms=3.5, lw=2.2,
                color=cmap(i / (len(hws) - 1)), label=f"LOESS ±{hw} d")
    ax.axhline(0, color="0.5", lw=0.8); shade(ax, event)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set(xlabel="evaluation window end (from May 1)", ylabel="Pearson r (water-VOD ↔ SWP)",
           title="(4) LOESS-detrended VOD ↔ SWP — sensitivity to the LOESS half-width\n"
                 "(cumulative from May 1; canonical is ±30 d)")
    ax.legend(loc="best", title="half-width")
    fig.tight_layout(); fig.savefig(OUT / "appc_expl_4_loess_windows.png", dpi=150); plt.close(fig)
    return wr


def main() -> int:
    cfg, ac, dfw, vod, pai, swp, sap, season, event, ends, corr, fits = setup()
    water = corr["hemi-PAI"]
    r_pv = fig1_pai_vs_vod(dfw, vod, pai, fits, season, event, ends)
    r_sap_raw, r_sap_detr = fig2_sapflow(dfw, vod, swp, sap, water, event)
    r_raw_pai, r_fit_pai = fig3_seasonal_pai(dfw, vod, pai, swp, water, season, event, ends)
    wr4 = fig4_loess_windows(dfw, vod, swp, season, event, ends)
    print("=== exploration summary (full-season r) ===")
    print(f"1  PAI <-> raw VOD                     r={r_pv:.2f}")
    print(f"2  raw VOD <-> sap flow                r={r_sap_raw:.2f}  (structural)")
    print(f"   water-VOD <-> detrended sap flow    r={r_sap_detr:.2f}  (water residual)")
    print(f"3  water-VOD <-> SWP, raw-daily PAI     r={r_raw_pai:.2f}")
    print(f"   water-VOD <-> SWP, seasonal-fit PAI  r={r_fit_pai:.2f}")
    print("4  LOESS half-width sweep (full-window r vs SWP):")
    last = wr4[wr4["end"] == wr4["end"].max()].set_index("pair")["r"]
    print(last.round(3).to_string())
    print(f"\nwrote 4 figures -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
