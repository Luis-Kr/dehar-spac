"""
Stage 30 / Application C — recover the water signal in GNSS-T VOD via PAI correction.

Claim: raw GNSS-T VOD is biomass-dominated and ~uncorrelated with water status over the season;
subtracting the independently measured (lidar) hemi-PAI biomass term yields a water-VOD that
tracks predawn stem Psi. Within the August event biomass is static, so PAI-correction preserves
the event water dip. We compare three corrections — hemi-PAI (canonical), hinge-PAI, and a
PAI-free LOESS detrend (±30 d, Humphrey & Frankenberg 2023) — on (a) seasonal VOD<->physiology gain and (b) preservation of
the August water dip ("is concurrent high-res PAI worth it vs a cheap detrend?").

Note on the event exclusion: only the *regression* corrections (hemi-PAI, hinge-PAI) exclude the
August window from the biomass fit, so the water drop cannot bias the slope. The rolling-median is
a filter (no fit), so there is nothing to exclude. The include-August variant is reported as a
sensitivity (paper/90_sensitivity/appC_sensitivity.py).

Engine ported/adapted from analysis/satellite_comparison/vod_biomass_water.py (frozen prototype).

Run:
  /home/lk1167/miniconda3/envs/dehar-spac/bin/python paper/30_appC_vod_correction/vod_pai_correction.py
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
import seaborn as sns
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "config"))
from paper_common import REPO, load_config, load_daily, load_event_windows, paper_style, stars, hornbeam_qc_cams  # noqa: E402

paper_style()


# --------------------------------------------------------------------------- #
# Build canonical daily-max VOD from the 30-min nvod ensemble + QC             #
# --------------------------------------------------------------------------- #
def build_vod_daily_max(cfg: dict, receivers=("gps1", "gps3", "gps5"),
                        rain_lookback_h=None, rh_max=None, rain_thresh=None,
                        name="vod_daily_max") -> pd.Series:
    """QC'd daily-max VOD from the 30-min nvod stream. Params overridable for the sweep;
    `receivers` selects which below-canopy receivers to ensemble (single-item list = per-receiver)."""
    qc = cfg["appC"]["qc"]
    rl = int(rain_lookback_h if rain_lookback_h is not None else qc["rain_lookback_h"])
    rh = rh_max if rh_max is not None else qc["rh_max_pct"]
    rt = rain_thresh if rain_thresh is not None else qc["rain_thresh_mm"]
    raw = pd.read_parquet(REPO / cfg["meta"]["streams_parquet"])
    raw.index = pd.to_datetime(raw.index)
    ens = raw[[f"nvod_{r}" for r in receivers]].mean(axis=1, skipna=True)
    precip = raw["precip_mm"].fillna(0).rolling(f"{rl}h").sum()
    ens = ens.where((raw["meteo_rh_pct"] <= rh) & (precip <= rt))
    hourly = ens.resample("1h").mean()                 # kill 30-min spikes before taking the max
    daily_max = hourly.resample("1D").max()
    daily_max.index = daily_max.index.tz_convert("UTC").tz_localize(None).normalize()
    return daily_max.rename(name)


# --------------------------------------------------------------------------- #
# Decomposition                                                                #
# --------------------------------------------------------------------------- #
def fit_biomass(df, vod, pai, exclude):
    d = df[[vod, pai]].dropna()
    d = d.loc[(d.index < exclude[0]) | (d.index > exclude[1])]
    lr = stats.linregress(d[pai], d[vod])
    return float(lr.intercept), float(lr.slope), {"r2": float(lr.rvalue ** 2), "n": len(d),
                                                  "p": float(lr.pvalue)}


def loess_baseline(s: pd.Series, half_width_days: float = 30) -> pd.Series:
    """Tricube-weighted local-linear LOESS with a fixed DAY bandwidth — replicates
    Humphrey & Frankenberg 2023 ("a local regression filter (LOESS) with ±30 d width" for the
    biomass v_veg low-pass). Day-bandwidth (not statsmodels' fraction-of-points `frac`) so gaps in
    the VOD series don't distort the window."""
    s = s.dropna()
    if len(s) < 3:
        return s
    x = s.index.values.astype("datetime64[D]").astype(float)   # day numbers
    y = s.to_numpy(float)
    out = np.empty(len(x))
    for i in range(len(x)):
        d = np.abs(x - x[i])
        w = np.where(d <= half_width_days, (1 - (d / half_width_days) ** 3) ** 3, 0.0)
        sw = w.sum()
        if sw <= 0:
            out[i] = y[i]; continue
        xm, ym = (w * x).sum() / sw, (w * y).sum() / sw
        sxx, sxy = (w * (x - xm) ** 2).sum(), (w * (x - xm) * (y - ym)).sum()
        b = sxy / sxx if sxx > 0 else 0.0
        out[i] = ym + b * (x[i] - xm)
    return pd.Series(out, index=s.index)


def corrections(df, vod, cfg):
    """Return dict label -> water-VOD series for the three corrections."""
    c = cfg["appC"]["corrections"]
    excl = [pd.Timestamp(x) for x in cfg["appC"]["event_exclude"]]
    out, fits = {}, {}
    for label, pcol in c["regressors"].items():
        a, b, info = fit_biomass(df, vod, pcol, excl)
        out[label] = df[vod] - (a + b * df[pcol])
        fits[label] = dict(a=a, b=b, **info, pai=pcol)
    hw = c["loess_halfwidth_days"]                              # Humphrey&Frankenberg 2023: ±30 d
    out["LOESS detrend"] = df[vod] - loess_baseline(df[vod], hw).reindex(df.index)
    fits["LOESS detrend"] = dict(loess_halfwidth_days=hw)
    return out, fits


def z(s):
    s = pd.Series(s)
    return (s - s.mean()) / s.std()


def _rsq(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))


def momen(df, vod, pai, psi):
    d = df[[vod, pai, psi]].dropna()
    y, p, s = z(d[vod]).to_numpy(), z(d[pai]).to_numpy(), z(d[psi]).to_numpy()
    designs = {"M0: VOD~PAI": p[:, None],
               "M1: +Psi": np.column_stack([p, s]),
               "M2: +Psi*PAI": np.column_stack([p, s, p * s])}
    rows, prev, n = [], 0.0, len(d)
    for name, X in designs.items():
        r2 = _rsq(y, X); k = X.shape[1]
        F = ((r2 - prev) / 1) / ((1 - r2) / (n - k - 1)) if r2 < 1 else np.inf
        pval = float(stats.f.sf(F, 1, n - k - 1)) if name != "M0: VOD~PAI" else np.nan
        rows.append({"model": name, "n": n, "r2": round(r2, 3),
                     "delta_r2": round(r2 - prev, 3), "F_p": pval})
        prev = r2
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Validation + event preservation                                              #
# --------------------------------------------------------------------------- #
def validate(df, vod, corr, targets, season, event):
    work = df.assign(**{f"water::{k}": v for k, v in corr.items()})
    signals = [("raw VOD", vod)] + [(k, f"water::{k}") for k in corr]
    rows = []
    for wlabel, w in [("full season", season), ("event", event)]:
        sub = work.loc[w[0]:w[1]]
        for tlabel, tcol in targets:
            for slabel, scol in signals:
                d = sub[[scol, tcol]].dropna()
                if len(d) < 4:
                    rows.append(dict(window=wlabel, target=tlabel, signal=slabel, n=len(d),
                                     r=np.nan, r2=np.nan, p=np.nan, sig="")); continue
                r, p = stats.pearsonr(d[scol], d[tcol])
                rows.append(dict(window=wlabel, target=tlabel, signal=slabel, n=len(d),
                                 r=round(r, 3), r2=round(r ** 2, 3), p=round(p, 4), sig=stars(p)))
    return pd.DataFrame(rows)


def event_preservation(df, vod, corr, baseline, event):
    """Retention of the August water dip: dip amplitude (baseline - event) after each correction,
    normalized by the raw-VOD dip. ~1 = preserved, <1 = attenuated/smeared."""
    def dip(s):
        b = s.loc[baseline[0]:baseline[1]].mean()
        e = s.loc[event[0]:event[1]].mean()
        return b - e
    raw_dip = dip(df[vod])
    rows = [dict(signal="raw VOD", dip=round(raw_dip, 4), retention=1.0)]
    for k, v in corr.items():
        d = dip(v)
        rows.append(dict(signal=k, dip=round(d, 4),
                         retention=round(d / raw_dip, 3) if raw_dip else np.nan))
    return pd.DataFrame(rows)


def regime(df, vod, corr, psi, thr):
    work = df.assign(**{f"water::{k}": v for k, v in corr.items()})
    signals = [("raw VOD", vod)] + [(k, f"water::{k}") for k in corr]
    rows = []
    for slabel, scol in signals:
        d = work[[scol, psi]].dropna()
        for rlabel, mask in [(f"Psi<{thr}", d[psi] < thr), (f"Psi>={thr}", d[psi] >= thr)]:
            dd = d[mask]
            r = stats.pearsonr(dd[scol], dd[psi])[0] if len(dd) >= 3 else np.nan
            rows.append(dict(signal=slabel, regime=rlabel, n=len(dd),
                             r=round(r, 3) if np.isfinite(r) else np.nan))
    return pd.DataFrame(rows)


def per_receiver_appendix(cfg, df, season, event, psi):
    """Appendix: hemi-PAI correction + SWP validation for each receiver separately vs the ensemble
    (the three receivers sample partly different canopy)."""
    recs = {"ensemble": ("gps1", "gps3", "gps5"), "gps1": ("gps1",),
            "gps3": ("gps3",), "gps5": ("gps5",)}
    rows = []
    for label, rcv in recs.items():
        v = build_vod_daily_max(cfg, receivers=rcv, name=f"vod_{label}")
        dw = df.join(v).loc[season[0]:season[1]]
        corr, _ = corrections(dw, f"vod_{label}", cfg)
        work = dw.assign(water=corr["hemi-PAI"])

        def rr(col, w):
            d = work.loc[w[0]:w[1]][[col, psi]].dropna()
            if len(d) < 4:
                return np.nan, np.nan, len(d)
            r, p = stats.pearsonr(d[col], d[psi])
            return round(r, 3), p, len(d)
        r_raw, _, _ = rr(f"vod_{label}", season)
        r_hs, p_hs, n_s = rr("water", season)
        r_he, p_he, n_e = rr("water", event)
        rows.append(dict(receiver=label, n_season=n_s, r_raw_season=r_raw,
                         r_hemi_season=r_hs, sig_season=stars(p_hs),
                         r_hemi_event=r_he, sig_event=stars(p_he), n_event=n_e))
    return pd.DataFrame(rows)


def fig_per_receiver(tbl, path):
    m = tbl.melt(id_vars="receiver", value_vars=["r_raw_season", "r_hemi_season", "r_hemi_event"],
                 var_name="metric", value_name="r")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.barplot(data=m, x="receiver", y="r", hue="metric", ax=ax)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set(title="Appendix — VOD↔SWP per GNSS-T receiver (hemi-PAI corrected)", ylabel="Pearson r", xlabel="")
    ax.legend(fontsize=8, title="")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# --------------------------------------------------------------------------- #
# Figures (seaborn-styled)                                                      #
# --------------------------------------------------------------------------- #
def fig_decomposition(df, vod, fits, corr, psi, event, path):
    a, b = fits["hemi-PAI"]["a"], fits["hemi-PAI"]["b"]
    pcol = fits["hemi-PAI"]["pai"]
    fig, ax = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    ax[0].plot(df.index, df[vod], color="#2166ac", lw=1.4, label="raw VOD (daily max)")
    ax[0].plot(df.index, a + b * df[pcol], "--", color="#8c510a", lw=1.6,
               label=f"biomass fit a+b·hemiPAI (b={b:.3f}, R²={fits['hemi-PAI']['r2']:.2f})")
    ax[0].set_ylabel("VOD"); ax[0].legend(loc="upper right")
    ax[1].axhline(0, color="0.6", lw=0.6)
    for k, c in [("hemi-PAI", "#2166ac"), ("hinge-PAI", "#5aae61"), ("LOESS detrend", "#b2182b")]:
        ax[1].plot(df.index, corr[k], color=c, lw=1.2, alpha=0.85, label=f"water-VOD ({k})")
    ax[1].set_ylabel("water-VOD"); ax[1].legend(loc="upper right")
    ax[2].plot(df.index, df[psi], color="#1b7837", lw=1.5)
    ax[2].set_ylabel("predawn Ψ (MPa)")
    for a_ in ax:
        a_.axvspan(pd.Timestamp(event[0]), pd.Timestamp(event[1]), color="0.8", alpha=0.5)
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.suptitle("App C — VOD biomass/water decomposition (August event shaded)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150); plt.close(fig)


def fig_validation_scatter(df, vod, corr, psi, season, event, path):
    work = df.assign(**{f"water::{k}": v for k, v in corr.items()})
    panels = [("raw VOD", vod), ("water-VOD (hemi-PAI)", "water::hemi-PAI"),
              ("water-VOD (LOESS ±30 d)", "water::LOESS detrend")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, (lab, col) in zip(axes, panels):
        for wlab, w, c in [("full season", season, "#9ecae1"), ("event", event, "#08519c")]:
            d = work.loc[w[0]:w[1]][[col, psi]].dropna()
            if len(d) < 4:
                continue
            r, p = stats.pearsonr(d[col], d[psi])
            ax.scatter(d[psi], z(d[col]), s=22, color=c, alpha=0.7,
                       edgecolor="none", label=f"{wlab} (r={r:.2f}{stars(p)}, n={len(d)})")
        ax.set(xlabel="predawn stem Ψ (MPa)", ylabel="z(VOD signal)", title=lab)
        ax.legend(loc="best")
    fig.suptitle("Does removing biomass make VOD track stem Ψ? (raw vs PAI vs detrend)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150); plt.close(fig)


def fig_correction_comparison(val, evp, path):
    fs = val[(val.window == "full season") & (val.target == "Predawn SWP")].copy()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    sns.barplot(data=fs, x="signal", y="r", ax=ax[0], color="#4292c6")
    ax[0].axhline(0, color="0.5", lw=0.8)
    for patch, (_, row) in zip(ax[0].patches, fs.iterrows()):   # R-style sig stars over bars
        h = patch.get_height()
        ax[0].annotate(f"{row['sig']}\nn={row['n']}", (patch.get_x() + patch.get_width() / 2, h),
                       ha="center", va="bottom" if h >= 0 else "top", fontsize=8,
                       xytext=(0, 3 if h >= 0 else -3), textcoords="offset points")
    ax[0].set(title="(a) Seasonal VOD↔predawn SWP correlation", ylabel="Pearson r", xlabel="")
    ax[0].tick_params(axis="x", rotation=20)
    ax[0].margins(y=0.18)
    sns.barplot(data=evp, x="signal", y="retention", ax=ax[1], color="#41ab5d")
    ax[1].axhline(1.0, color="0.5", ls="--", lw=1)
    ax[1].set(title="(b) August water-dip retention (1 = preserved)", ylabel="retained fraction", xlabel="")
    ax[1].tick_params(axis="x", rotation=20)
    fig.suptitle("Is concurrent high-res PAI worth it? seasonal gain (a) vs event preservation (b)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150); plt.close(fig)


def fig_corroboration(df, vod, corr, cfg, season, event, path):
    """Appendix: hemi-PAI water-VOD vs the corroborating physiology (TWD, sap flow, leaf angle)."""
    work = df.assign(water=corr["hemi-PAI"])
    ac = cfg["appC"]
    targets = [("Tree water deficit (µm)", ac["validation"]["corroborate"][0]),
               ("Sap flux density", ac["validation"]["corroborate"][1]),
               ("Leaf angle (°, hornbeam)", ac["validation"]["exploratory"])]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    for ax, (lab, col) in zip(axes, targets):
        for wlab, w, c in [("full season", season, "#9ecae1"), ("event", event, "#08519c")]:
            d = work.loc[w[0]:w[1]][["water", col]].dropna()
            if len(d) < 4:
                continue
            r, p = stats.pearsonr(d["water"], d[col])
            ax.scatter(d[col], z(d["water"]), s=26, color=c, alpha=0.75,
                       edgecolor="none", label=f"{wlab} (r={r:.2f}{stars(p)}, n={len(d)})")
        ax.set(xlabel=lab, ylabel="z(water-VOD, hemi-PAI)")
        ax.legend(loc="best", frameon=True)
    fig.suptitle("Appendix — biomass-corrected water-VOD vs corroborating physiology")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path); plt.close(fig)


def fig_threshold(df, vod, corr, psi, thr, path):
    work = df.assign(water=corr["hemi-PAI"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (col, lab, color) in zip(axes, [(vod, "raw VOD", "#8c510a"),
                                            ("water", "water-VOD (hemi-PAI)", "#2166ac")]):
        d = work[[col, psi]].dropna()
        wet, dry = d[d[psi] >= thr], d[d[psi] < thr]
        ax.scatter(wet[psi], wet[col], s=20, color="0.6", alpha=0.5, label=f"Ψ≥{thr} (n={len(wet)})")
        ax.scatter(dry[psi], dry[col], s=28, color=color, alpha=0.85, label=f"Ψ<{thr} (n={len(dry)})")
        if len(dry) >= 3:
            lr = stats.linregress(dry[psi], dry[col])
            xs = np.array([dry[psi].min(), dry[psi].max()])
            ax.plot(xs, lr.intercept + lr.slope * xs, color=color, lw=2.2)
            r_all, p_all = stats.pearsonr(d[col], d[psi])
            ax.set_title(f"{lab}\nall r={r_all:.2f}{stars(p_all)} | "
                         f"Ψ<{thr} r={lr.rvalue:.2f}{stars(lr.pvalue)}")
        ax.axvline(thr, color="0.5", ls="--", lw=1)
        ax.set(xlabel="predawn stem Ψ (MPa)", ylabel=lab)
        ax.legend(loc="best")
    fig.suptitle("VOD–Ψ relationship under stress (Ψ < threshold)")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150); plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> int:
    cfg = load_config()
    ac = cfg["appC"]
    df = load_daily(cfg)
    ev = load_event_windows()
    base = (ev.loc[ev.stage == "baseline", "start"].iloc[0], ev.loc[ev.stage == "baseline", "end"].iloc[0])
    event = [pd.Timestamp(x) for x in ac["event_exclude"]]
    season = [pd.Timestamp(x) for x in ac["window"]]

    # derived columns — leaf angle = coherence-QC'd hornbeam cams (exploratory target)
    qc_cams, _ = hornbeam_qc_cams(df, cfg)
    df[ac["validation"]["exploratory"]] = df[[f"leaf_angle_cam{c}_daylight_mean_deg" for c in qc_cams
                                              if f"leaf_angle_cam{c}_daylight_mean_deg" in df.columns]].mean(axis=1)
    # canonical VOD = daily-max (built from 30-min); fallback if too sparse
    vdm = build_vod_daily_max(cfg)
    df = df.join(vdm)
    vod = "vod_daily_max"
    if df.loc[season[0]:season[1], vod].notna().sum() < 60:
        vod = ac["vod_fallback_col"]
        print(f"[warn] daily-max VOD sparse; falling back to {vod}")

    dfw = df.loc[season[0]:season[1]].copy()
    psi = ac["validation"]["primary"]

    corr, fits = corrections(dfw, vod, cfg)
    targets = [("Predawn SWP", psi)] + [("TWD", ac["validation"]["corroborate"][0]),
                                        ("Sap flow", ac["validation"]["corroborate"][1]),
                                        ("Leaf angle", ac["validation"]["exploratory"])]
    val = validate(dfw, vod, corr, targets, season, event)
    evp = event_preservation(dfw, vod, corr, base, event)
    mom = momen(dfw, vod, fits["hemi-PAI"]["pai"], psi)
    reg = regime(dfw, vod, corr, psi, ac["regime_threshold_MPa"])

    tdir = HERE / "outputs" / "tables"; tdir.mkdir(parents=True, exist_ok=True)
    fdir = HERE / "outputs" / "figures"; fdir.mkdir(parents=True, exist_ok=True)
    adir = HERE / "outputs" / "appendix"; adir.mkdir(parents=True, exist_ok=True)
    val.to_csv(tdir / "vod_correction_validation.csv", index=False)
    evp.to_csv(tdir / "vod_event_preservation.csv", index=False)
    mom.to_csv(tdir / "vod_momen.csv", index=False)
    reg.to_csv(tdir / "vod_regime.csv", index=False)
    pd.DataFrame(fits).T.to_csv(tdir / "vod_decomposition_fits.csv")

    fig_decomposition(dfw, vod, fits, corr, psi, event, fdir / "decomposition.png")
    fig_validation_scatter(dfw, vod, corr, psi, season, event, fdir / "validation_scatter.png")
    fig_correction_comparison(val, evp, fdir / "correction_comparison.png")
    fig_threshold(dfw, vod, corr, psi, ac["regime_threshold_MPa"], fdir / "threshold_regime.png")
    fig_corroboration(dfw, vod, corr, cfg, season, event, adir / "corroboration_targets.png")

    # appendix: per-receiver (build off the daily frame WITHOUT the ensemble vod col)
    prx = per_receiver_appendix(cfg, df.drop(columns=[vod]) if vod in df else df,
                                season, event, psi)
    prx.to_csv(tdir / "vod_per_receiver.csv", index=False)
    fig_per_receiver(prx, adir / "per_receiver.png")

    print(f"VOD = {vod}  (n={dfw[vod].notna().sum()} days in window)\n")
    print("=== Validation vs predawn SWP (full season) ===")
    print(val[(val.window == "full season") & (val.target == "Predawn SWP")]
          .to_string(index=False, na_rep="—"))
    print("\n=== Validation: water-VOD (hemi-PAI) across targets ===")
    print(val[val.signal == "hemi-PAI"].to_string(index=False, na_rep="—"))
    print("\n=== Event water-dip retention ===")
    print(evp.to_string(index=False))
    print("\n=== Momen nested model ===")
    print(mom.to_string(index=False))
    print("\n=== Per-receiver (appendix) — hemi-PAI corrected VOD vs SWP ===")
    print(prx.to_string(index=False, na_rep="—"))
    print(f"\nwrote tables -> {tdir}\nwrote figures -> {fdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
