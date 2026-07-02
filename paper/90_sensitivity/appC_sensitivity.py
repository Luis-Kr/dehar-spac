"""
Stage 90 — App C sensitivity / robustness (appendix, NOT selection).

Shows the App C conclusion (hemi-PAI corrected VOD tracks SWP and preserves the event) is
insensitive to free choices. Canonical values are NEVER chosen by best-r here — we only report
the spread to support an appendix sentence like "varying the filter changed nothing".

Three sweeps:
  1. Biomass-fit window: INCLUDE all (canonical, event-blind, ADR 0007) vs EXCLUDE August (sensitivity).
  2. VOD rain-lookback {6,12,24,48 h} x RH {90,95}%: rebuild daily-max VOD, hemi-correct,
     report (days retained, r vs SWP) with a min-n guard.
  3. LOESS half-width {15,30,45,60 d} (Humphrey ±30 d canonical): r vs SWP + August-dip retention
     (shows the PAI-free detrend's width trade-off that hemi-PAI sidesteps).

Run:
  /home/lk1167/miniconda3/envs/dehar-spac/bin/python paper/90_sensitivity/appC_sensitivity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "config"))
sys.path.insert(0, str(HERE.parents[0] / "30_appC_vod_correction"))
from paper_common import load_config, load_daily, load_event_windows, paper_style, stars  # noqa: E402
from vod_pai_correction import build_vod_daily_max, fit_biomass, loess_baseline  # noqa: E402

paper_style()
MIN_N = 20
NO_EXCL = ("2099-01-01", "2099-01-02")   # exclude nothing => include-August


def _annot_rp(sweep2):
    """Heatmap annotation: 'r' + significance stars, per (rain_lookback × RH) cell."""
    pr = sweep2.pivot(index="rain_lookback_h", columns="rh_max", values="r_season")
    ps = sweep2.pivot(index="rain_lookback_h", columns="rh_max", values="sig")
    return pr.applymap(lambda v: f"{v:.2f}" if pd.notna(v) else "") + ps.fillna("")


def r_vs(series, target, w):
    """Return (r, p, n) over window w."""
    d = pd.concat([series.rename("x"), target.rename("y")], axis=1).loc[w[0]:w[1]].dropna()
    if len(d) < 4:
        return np.nan, np.nan, len(d)
    r, p = stats.pearsonr(d.x, d.y)
    return round(r, 3), p, len(d)


def main() -> int:
    cfg = load_config(); ac = cfg["appC"]
    df = load_daily(cfg); ev = load_event_windows()
    season = [pd.Timestamp(x) for x in ac["window"]]
    event = [pd.Timestamp(x) for x in ac["event_exclude"]]
    base = (ev.loc[ev.stage == "baseline", "start"].iloc[0], ev.loc[ev.stage == "baseline", "end"].iloc[0])
    pai, psi = ac["biomass_col"], ac["validation"]["primary"]
    tdir = HERE / "outputs" / "tables"; tdir.mkdir(parents=True, exist_ok=True)
    fdir = HERE / "outputs" / "figures"; fdir.mkdir(parents=True, exist_ok=True)
    adir = HERE / "outputs" / "appendix"; adir.mkdir(parents=True, exist_ok=True)

    # canonical VOD
    vod = build_vod_daily_max(cfg)
    d0 = df.join(vod).loc[season[0]:season[1]]
    vcol = "vod_daily_max"

    # ---- Sweep 1: include (canonical, event-blind) vs exclude August in the fit (ADR 0007) --- #
    rows = []
    for label, excl in [("include all (canonical)", NO_EXCL), ("exclude Aug (sensitivity)", event)]:
        a, b, info = fit_biomass(d0, vcol, pai, excl)
        water = d0[vcol] - (a + b * d0[pai])
        rs, ps, ns = r_vs(water, d0[psi], season)
        re, pe, ne = r_vs(water, d0[psi], event)
        rows.append(dict(variant=label, slope_b=round(b, 4), r_season=rs, sig_season=stars(ps),
                         n_season=ns, r_event=re, sig_event=stars(pe)))
    sweep1 = pd.DataFrame(rows)
    sweep1.to_csv(tdir / "appC_sens_biomass_window.csv", index=False)

    # ---- Sweep 2: rain-lookback x RH --------------------------------------- #
    rows = []
    for rl in [6, 12, 24, 48]:
        for rh in [90, 95]:
            v = build_vod_daily_max(cfg, rain_lookback_h=rl, rh_max=rh)
            dd = df.join(v).loc[season[0]:season[1]]
            a, b, _ = fit_biomass(dd, "vod_daily_max", pai, event)
            water = dd["vod_daily_max"] - (a + b * dd[pai])
            rs, ps, ns = r_vs(water, dd[psi], season)
            rows.append(dict(rain_lookback_h=rl, rh_max=rh, n_season=ns,
                             r_season=rs if ns >= MIN_N else np.nan,
                             sig=stars(ps) if ns >= MIN_N else "",
                             guard="ok" if ns >= MIN_N else f"n<{MIN_N}"))
    sweep2 = pd.DataFrame(rows)
    sweep2.to_csv(tdir / "appC_sens_rain_rh.csv", index=False)

    # ---- Sweep 3: LOESS half-width (Humphrey ±30 d canonical) --------------- #
    def dip(s):
        return s.loc[base[0]:base[1]].mean() - s.loc[event[0]:event[1]].mean()
    raw_dip = dip(d0[vcol])
    rows = []
    for hw in [15, 30, 45, 60]:
        bl = loess_baseline(d0[vcol], hw).reindex(d0.index)
        water = d0[vcol] - bl
        rs, ps, ns = r_vs(water, d0[psi], season)
        rows.append(dict(loess_halfwidth_days=hw, r_season=rs, sig_season=stars(ps), n_season=ns,
                         event_retention=round(dip(water) / raw_dip, 3) if raw_dip else np.nan))
    sweep3 = pd.DataFrame(rows)
    sweep3.to_csv(tdir / "appC_sens_loess_halfwidth.csv", index=False)

    # ---- figure ------------------------------------------------------------ #
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    sns.barplot(data=sweep1, x="variant", y="r_season", ax=ax[0], color="#4292c6")
    ax[0].set(title="(1) biomass-fit window\n(r vs SWP, season)", ylabel="r", xlabel="")
    ax[0].tick_params(axis="x", rotation=15)
    piv = sweep2.pivot(index="rain_lookback_h", columns="rh_max", values="r_season")
    sns.heatmap(piv, annot=_annot_rp(sweep2), fmt="", cmap="viridis", ax=ax[1],
                cbar_kws={"label": "r vs SWP"})
    ax[1].set(title="(2) rain-lookback × RH\n(r vs SWP, season; * sig.)")
    ax2b = ax[2].twinx()
    sns.lineplot(data=sweep3, x="loess_halfwidth_days", y="r_season", marker="o", ax=ax[2], color="#2166ac")
    sns.lineplot(data=sweep3, x="loess_halfwidth_days", y="event_retention", marker="s", ax=ax2b, color="#b2182b")
    ax[2].set(title="(3) LOESS half-width (Humphrey ±30 d)", ylabel="r vs SWP (blue)", xlabel="LOESS ±half-width (days)")
    ax2b.set_ylabel("event retention (red)"); ax2b.axhline(1, color="#b2182b", ls="--", lw=0.8, alpha=0.5)
    fig.suptitle("App C robustness — canonical choices are pre-set, not best-r")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(fdir / "appC_sensitivity.png", dpi=150); plt.close(fig)

    # ---- dedicated rain/RH lockout figure (appendix) ----------------------- #
    fig2, axr = plt.subplots(1, 2, figsize=(11, 4.4))
    piv_r = sweep2.pivot(index="rain_lookback_h", columns="rh_max", values="r_season")
    piv_n = sweep2.pivot(index="rain_lookback_h", columns="rh_max", values="n_season")
    sns.heatmap(piv_r, annot=_annot_rp(sweep2), fmt="", cmap="viridis", ax=axr[0],
                linewidths=0.4, linecolor="white", cbar_kws={"label": "r vs predawn SWP"})
    axr[0].set(title="(a) hemi-PAI water-VOD ↔ SWP (* sig.)", xlabel="RH max (%)", ylabel="rain lockout (h)")
    sns.heatmap(piv_n, annot=True, fmt="d", cmap="rocket_r", ax=axr[1],
                linewidths=0.4, linecolor="white", cbar_kws={"label": "days retained"})
    axr[1].set(title="(b) sample size retained", xlabel="RH max (%)", ylabel="rain lockout (h)")
    fig2.suptitle("Appendix — VOD meteorological filter robustness (canonical: 12 h, RH 95 %)")
    fig2.tight_layout(rect=(0, 0, 1, 0.93))
    fig2.savefig(adir / "rain_rh_lockout.png"); plt.close(fig2)

    print("=== (1) biomass-fit window ===\n", sweep1.to_string(index=False))
    print("\n=== (2) rain-lookback x RH (r vs SWP, season) ===\n", sweep2.to_string(index=False))
    print("\n=== (3) LOESS half-width (Humphrey ±30 d) ===\n", sweep3.to_string(index=False))
    print(f"\nwrote tables -> {tdir}\nwrote figures -> {fdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
