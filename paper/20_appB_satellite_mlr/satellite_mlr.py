"""
Stage 20 / Application B — what in-situ driver does each satellite signal track DURING STRESS?

CRITICAL: raw full-season LEVEL correlations are dominated by the shared phenology/biomass trend
(everything trending correlates ~0.9) and miss the stress signal. So attribution here is done on
EVENT-PRESERVING signals (see [[stress-attribution-detrend-principle]]):
  PRIMARY  event-change delta: standardized change from a recovered pre-event baseline to the acute
           peak, per variable. PAI~0 across the span -> no trend model, cannot diminish the event.
  SUPPORT  smooth-seasonal anomaly attribution: subtract a low-order (3rd-deg) seasonal polynomial
           (captures spring-up/autumn-senescence, KEEPS the 2-3 week event), then driver x index
           correlation matrix + 3 commonality contrasts ON ANOMALIES, with an event-retention check.
  APPENDIX raw-level matrix, explicitly caveated as seasonal climatology (phenology), not stress.

Run:
  /home/lk1167/miniconda3/envs/dehar-spac/bin/python paper/20_appB_satellite_mlr/satellite_mlr.py
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "config"))
sys.path.insert(0, str(HERE.parents[0] / "30_appC_vod_correction"))
from paper_common import load_config, load_daily, paper_style, stars, hornbeam_qc_cams, greenness_col  # noqa: E402
from vod_pai_correction import build_vod_daily_max  # noqa: E402

paper_style()

TARGETS = [  # (label, column, group)
    ("NDVI", "s2_ndvi_mean", "S2 greenness"), ("NIRv", "s2_nirv_mean", "S2 greenness"),
    ("kNDVI", "s2_kndvi_mean", "S2 greenness"), ("EVI", "s2_evi_mean", "S2 greenness"),
    ("CIre", "s2_cire_mean", "S2 red-edge"), ("MTCI", "s2_mtci_mean", "S2 red-edge"),
    ("NDREI", "s2_ndrei_mean", "S2 red-edge"),
    ("NDII", "s2_ndii_mean", "S2 water"), ("NDMI", "s2_ndmi_mean", "S2 water"),
    ("NMDI", "s2_nmdi_mean", "S2 water"),
    ("PSRI", "s2_psri_mean", "S2 senescence"), ("CIg", "s2_cig_mean", "S2 senescence"),
    ("S1 VH", "s1_asc_vh_mean_dB", "S1"), ("S1 VV", "s1_asc_vv_mean_dB", "S1"),
    ("S1 span", "s1_asc_span_mean_dB", "S1"), ("S1 CR", "s1_asc_cr_mean_dB", "S1"),
    ("S1 RVI", "s1_asc_rvi_mean", "S1"),
]
# colour = which kind of variable; one row per sensor stream
GROUP_COLOR = {"forcing": "#8c8c8c", "flux": "#762a83", "biomass": "#8c510a",
               "structure": "#2166ac", "water": "#1b7837", "greenness": "#5aae61",
               "S2": "#b2182b", "S1": "#ef8a62"}
GROUP_LABEL = {"forcing": "forcing (soil, VPD)", "flux": "C/water flux (GPP, ET)",
               "biomass": "biomass / PAI (lidar)", "structure": "canopy structure (leaf angle)",
               "water": "plant water status", "greenness": "phenocam greenness",
               "S2": "Sentinel-2 optical", "S1": "Sentinel-1 radar"}


def build(cfg):
    df = load_daily(cfg)
    qc, _ = hornbeam_qc_cams(df, cfg)
    df["leaf_angle_qc"] = df[[f"leaf_angle_cam{c}_daylight_mean_deg" for c in qc
                              if f"leaf_angle_cam{c}_daylight_mean_deg" in df.columns]].mean(axis=1)
    df["gcc_ens"] = df[greenness_col(cfg)]   # PhenoCam GCC by default (ADR 0002)
    df = df.join(build_vod_daily_max(cfg))
    drivers = [
        ("Soil moisture", "sm_mean_pct", "forcing"), ("VPD", "vpd_max_hPa", "forcing"),
        ("GNSS-T VOD", "vod_daily_max", "water"), ("PAI hemi", "pai_hemi_hi_hinge_mean_m2m2", "biomass"),
        ("PAI hinge", "pai_hinge_hinge_mean_m2m2", "biomass"), ("Leaf angle", "leaf_angle_qc", "structure"),
        ("GCC", "gcc_ens", "greenness"), ("Predawn SWP", "swp_pd_mean_MPa", "water"),
        ("TWD", "twd_pd_mean_um", "water"), ("Sap flow", "sapflow_sum_mean", "water"),
        ("GPP", "gpp_mean_umol_m2_s", "flux"), ("ET", "et_sum_mm", "flux"),
    ]
    return df, drivers


def z(s):
    sd = s.std()
    return (s - s.mean()) / sd if sd else s * 0.0


def poly_detrend(s, order=3, exclude=None):
    """Residual after a low-order seasonal polynomial in day-of-year. The trend is fit on data
    OUTSIDE the event window (like App C's biomass fit) so the event is preserved in full."""
    fit_src = s.dropna()
    if exclude is not None:
        fit_src = fit_src.loc[(fit_src.index < exclude[0]) | (fit_src.index > exclude[1])]
    if len(fit_src) < order + 2:
        return pd.Series(np.nan, index=s.index)
    doy = fit_src.index.dayofyear.to_numpy(float)
    mu, sd = doy.mean(), doy.std() or 1.0
    coef = np.polyfit((doy - mu) / sd, fit_src.to_numpy(float), order)
    fit_all = np.polyval(coef, (s.index.dayofyear.to_numpy(float) - mu) / sd)
    return s - pd.Series(fit_all, index=s.index)


def _rsq(y, X):
    if X.shape[1] == 0:
        return 0.0
    X = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - (r @ r) / ss) if ss > 0 else 0.0


def commonality(d, ycol, drivers):
    d = d[[ycol] + drivers].dropna()
    if len(d) < len(drivers) + 2:
        return None
    y = z(d[ycol]).to_numpy()
    cols = {k: z(d[k]).to_numpy() for k in drivers}
    full = _rsq(y, np.column_stack([cols[k] for k in drivers]))
    uniq = {k: max(0.0, full - _rsq(y, np.column_stack([cols[j] for j in drivers if j != k])
                                    if len(drivers) > 1 else np.empty((len(y), 0)))) for k in drivers}
    return dict(n=len(d), r2=full, unique=uniq, shared=max(0.0, full - sum(uniq.values())))


# --------------------------------------------------------------------------- #
def _items(drivers):
    sat = [(l, c, "S1" if c.startswith("s1_") else "S2") for l, c, _ in TARGETS]
    return [(l, c, g) for l, c, g in drivers] + sat


def _peak_for(col, peak, peak_sat):
    """Satellite streams (sparse) use the satellite peak window; in-situ use the acute peak."""
    return peak_sat if (col.startswith("s2_") or col.startswith("s1_")) else peak


def event_change(df, drivers, baseline, peak, peak_sat):
    """Single hand-set baseline -> peak z-delta per variable (kept for the reference dot)."""
    rows = []
    for lab, col, grp in _items(drivers):
        if col not in df.columns:
            continue
        zz = z(df[col])
        pk = _peak_for(col, peak, peak_sat)
        b = zz.loc[baseline[0]:baseline[1]].mean()
        e = zz.loc[pk[0]:pk[1]].mean()
        if np.isfinite(b) and np.isfinite(e):
            rows.append(dict(var=lab, group=grp, handset_delta=round(e - b, 2)))
    return pd.DataFrame(rows)


def event_change_ensemble(df, drivers, ens, peak, peak_sat):
    """PRIMARY: z-change baseline->peak as a DISTRIBUTION over many defensible baselines.

    Slide a baseline window of each width in `ens['widths']` across `ens['envelope']`; for every
    placement and variable compute delta = peak_mean - baseline_mean on the season-z series. A
    placement contributes to a variable only if that variable has enough baseline observations in
    the window (per-var guard): in-situ >= min_insitu_days; S2 >= min_s2_scenes. So satellite vars
    naturally draw their band from the Jun9-Jul2 cluster (5-week July cloud gap), in-situ vars from
    the full slide. Report median + [2.5, 97.5] pct + n_placements."""
    env0, env1 = pd.Timestamp(ens["envelope"][0]), pd.Timestamp(ens["envelope"][1])
    starts = pd.date_range(env0, env1, freq="D")
    rows = []
    for lab, col, grp in _items(drivers):
        if col not in df.columns:
            continue
        zz = z(df[col])
        pk = _peak_for(col, peak, peak_sat)
        peak_mean = zz.loc[pk[0]:pk[1]].mean()
        if not np.isfinite(peak_mean):
            continue
        is_sat = col.startswith("s2_") or col.startswith("s1_")  # sparse acquisition-date streams
        min_obs = ens["min_s2_scenes"] if is_sat else ens["min_insitu_days"]
        deltas = []
        for w in ens["widths"]:
            for s0 in starts:
                s1 = s0 + pd.Timedelta(days=int(w) - 1)
                if s1 > env1:
                    continue
                seg = zz.loc[s0:s1].dropna()
                if len(seg) < min_obs:
                    continue
                deltas.append(peak_mean - seg.mean())
        if not deltas:
            continue
        d = np.asarray(deltas, float)
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append(dict(var=lab, group=grp, n_placements=len(d),
                         delta_median=round(float(np.median(d)), 2),
                         delta_lo=round(float(lo), 2), delta_hi=round(float(hi), 2),
                         excludes_zero=bool(lo > 0 or hi < 0)))
    return pd.DataFrame(rows)


def corr_matrix(df, drivers, ysuffix, xsuffix):
    rows = []
    for lab, col, grp in TARGETS:
        yc = col + ysuffix
        if yc not in df.columns:
            continue
        for dlab, dcol, _ in drivers:
            xc = dcol + xsuffix
            d = df[[yc, xc]].dropna()
            if len(d) >= 6:
                r, p = stats.pearsonr(d[yc], d[xc])
                rows.append(dict(group=grp, target=lab, driver=dlab, r=round(r, 2), sig=stars(p), n=len(d)))
            else:
                rows.append(dict(group=grp, target=lab, driver=dlab, r=np.nan, sig="", n=len(d)))
    return pd.DataFrame(rows)


def fig_event_change(ec, ens, peak, peak_sat, path, mode="signed", show_handset=False):
    """Per-sensor standardized change from the pre-event baseline ENSEMBLE to the fixed stress event,
    as median + 95% percentile band. `mode='signed'` keeps direction (down vs up); `mode='abs'` ranks
    by response magnitude |z| (all bars one side). Bars whose 95% band excludes 0 are drawn solid;
    bands straddling 0 are faded (a no-response read). `show_handset` overlays the old single hand-set
    baseline as a dot (appendix robustness variant — each dot falls inside its band)."""
    ec = ec.copy()
    if mode == "abs":
        crosses = (ec.delta_lo < 0) & (ec.delta_hi > 0)
        ec["x"] = ec.delta_median.abs()
        ec["lo"] = np.where(crosses, 0.0, np.minimum(ec.delta_lo.abs(), ec.delta_hi.abs()))
        ec["hi"] = np.maximum(ec.delta_lo.abs(), ec.delta_hi.abs())
        ec["hs"] = ec.handset_delta.abs() if "handset_delta" in ec else np.nan
        xlabel = "absolute standardized change |z|  (response magnitude)"
    else:
        ec["x"], ec["lo"], ec["hi"] = ec.delta_median, ec.delta_lo, ec.delta_hi
        ec["hs"] = ec.handset_delta if "handset_delta" in ec else np.nan
        xlabel = "standardized change (z),  baseline → event   (− drop / + rise)"
    ec = ec.sort_values("x").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 8))
    y = np.arange(len(ec))
    xerr = np.vstack([ec.x - ec.lo, ec.hi - ec.x])
    for yi, r in ec.iterrows():
        solid = bool(r.excludes_zero)
        ax.barh(yi, r.x, color=GROUP_COLOR.get(r.group, "#b2182b"),
                edgecolor="#222222" if solid else "none", linewidth=1.4 if solid else 0.0,
                alpha=1.0 if solid else 0.4, zorder=2)
    ax.errorbar(ec.x, y, xerr=xerr, fmt="none", ecolor="0.25", elinewidth=1.4, capsize=3, zorder=3)
    if show_handset and ec.hs.notna().any():
        ax.scatter(ec.hs, y, marker="o", s=26, facecolor="white", edgecolor="black",
                   linewidth=1.0, zorder=4)
    ax.set_yticks(y); ax.set_yticklabels(ec["var"])
    ax.axvline(0, color="0.4", lw=0.8, zorder=1)
    ax.set_xlabel(xlabel)

    what = "magnitude of each sensor's response" if mode == "abs" else "how each sensor changed"
    ax.set_title(f"App B — {what} going into the August stress event\n"
                 f"z-change vs a pre-event baseline ensemble (Jun 1–Aug 5, {'/'.join(map(str, ens['widths']))}-d "
                 "windows); median + 95% band", fontsize=10)

    groups = [g for g in GROUP_LABEL if g in set(ec.group)]
    handles = [Patch(facecolor=GROUP_COLOR[g], label=GROUP_LABEL[g]) for g in groups]
    handles.append(Patch(facecolor="0.7", edgecolor="#222222", linewidth=1.4, label="band excludes 0"))
    handles.append(Patch(facecolor="0.7", alpha=0.4, label="band straddles 0 (no response)"))
    if show_handset and ec.hs.notna().any():
        handles.append(Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white",
                              markeredgecolor="black", label="hand-set baseline (single window)"))
    ax.legend(handles=handles, loc="lower right", fontsize=7.5, framealpha=0.92, title=None)
    sw = f"{pd.Timestamp(peak_sat[0]):%b %-d}–{pd.Timestamp(peak_sat[1]):%-d}"
    iw = f"{pd.Timestamp(peak[0]):%b %-d}–{pd.Timestamp(peak[1]):%-d}"
    fig.text(0.01, 0.005, f"Event window: in-situ = acute {iw} (PAI flattest); satellite = entire August "
             f"{sw} (adds Aug 26/31 → recovery vs persistence). S2 baseline anchored to the "
             "Jun 9–Jul 2 cluster (5-week July gap).", fontsize=6.5, color="0.45")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path); plt.close(fig)


def fig_season_lines(df, panels, peak, ens, path, suptitle, sparse=False, raw=False):
    """Seasonal trajectories, one thematic subpanel per row, event window shaded — a plain 'watch the
    fall' visualization (no stats). Default standardizes (z) over the appB window so panels are
    comparable; `raw=True` plots native index values instead (each panel autoscaled). `sparse=True`
    adds acquisition-date markers (S1/S2 are not daily)."""
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(10, 1.8 * n + 0.6), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (ptitle, series) in zip(axes, panels):
        pal = sns.color_palette("husl", len(series))
        for (lab, col), c in zip(series, pal):
            if col not in df.columns:
                continue
            s = df[col].dropna() if raw else z(df[col]).dropna()
            if s.empty:
                continue
            ax.plot(s.index, s.values, marker="o" if sparse else None, ms=3.5,
                    lw=1.6 if sparse else 2.2, color=c, label=lab)
        ax.axvspan(pd.Timestamp(ens["envelope"][0]), pd.Timestamp(ens["envelope"][1]),
                   color="#1b7837", alpha=0.05, lw=0, zorder=0)
        ax.axvspan(pd.Timestamp(peak[0]), pd.Timestamp(peak[1]), color="#b2182b", alpha=0.12,
                   lw=0, zorder=0)
        if not raw:
            ax.axhline(0, color="0.7", lw=0.6, zorder=0)
        ax.set_title(ptitle, loc="left", fontsize=9.5, color="0.2")
        ax.set_ylabel("value" if raw else "z-score")
        ax.legend(loc="upper left", fontsize=7, ncol=max(1, len(series) // 2),
                  framealpha=0.9, handlelength=1.3, columnspacing=1.0)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[0].annotate("stress\nevent", xy=(pd.Timestamp(peak[0]), 1), xytext=(4, 2),
                     textcoords="offset points", fontsize=7.5, color="#b2182b", va="bottom")
    fig.suptitle(suptitle + "   (red band = stress event; green band = baseline window)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path); plt.close(fig)


def fig_season_grid(df, items, peak, ens, path, suptitle, ncols=2, sparse=True, date_ticks=False):
    """Small multiples — ONE index per panel in its native units (no shared scale, so a large index
    can't flatten a small one). Wide panels, tight spacing. `items` = (label, col, unit, colour).
    `date_ticks=True` labels EVERY acquisition date on the x-axis (+ vertical gridlines) so each
    point's date is readable."""
    n = len(items)
    nrows = int(np.ceil(n / ncols))
    extra_h = 1.1 if date_ticks else 0.0          # room for rotated date labels
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 1.55 * nrows + 0.6 + extra_h),
                             sharex=True, squeeze=False)
    axes = axes.ravel()
    all_dates = sorted({d for _, col, _, _ in items if col in df.columns
                        for d in df[col].dropna().index})
    for ax, (lab, col, unit, color) in zip(axes, items):
        if col in df.columns:
            s = df[col].dropna()
            ax.plot(s.index, s.values, marker="o" if sparse else None, ms=3.5, lw=1.8, color=color)
        ax.axvspan(pd.Timestamp(ens["envelope"][0]), pd.Timestamp(ens["envelope"][1]),
                   color="#1b7837", alpha=0.05, lw=0, zorder=0)
        ax.axvspan(pd.Timestamp(peak[0]), pd.Timestamp(peak[1]), color="#b2182b", alpha=0.12,
                   lw=0, zorder=0)
        ax.set_title(lab, loc="left", fontsize=10, color="0.2", pad=2)
        ax.set_ylabel(unit, fontsize=9)
        ax.tick_params(labelsize=8)
    for ax in axes[n:]:
        ax.axis("off")
    if date_ticks:
        for ax in axes[:n]:
            ax.set_xticks(all_dates)
            ax.set_xticklabels([d.strftime("%b %d") for d in all_dates], rotation=90, fontsize=5.5)
            ax.grid(axis="x", color="0.85", lw=0.4, zorder=0)
    else:
        for ax in axes:
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.suptitle(suptitle + "   (red band = stress event; green band = baseline window)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98), h_pad=0.5, w_pad=1.2)
    fig.savefig(path); plt.close(fig)


def write_index_values_csv(df, items, path):
    """Exact value of every index at every acquisition date (date-indexed wide table)."""
    tbl = pd.DataFrame({lab: df[col] for lab, col, _, _ in items if col in df.columns})
    tbl = tbl.dropna(how="all").round(4)
    tbl.index.name = "date"
    tbl.to_csv(path)


def fig_heatmap(mat, drivers, title, path):
    torder = [t[0] for t in TARGETS if t[0] in mat.target.values]
    dorder = [d[0] for d in drivers]
    pr = mat.pivot(index="target", columns="driver", values="r").reindex(index=torder, columns=dorder)
    ps = mat.pivot(index="target", columns="driver", values="sig").reindex(index=torder, columns=dorder)
    annot = pr.map(lambda v: f"{v:.2f}" if pd.notna(v) else "") + ps.fillna("")
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(pr, annot=annot, fmt="", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                linewidths=0.4, linecolor="white", ax=ax, cbar_kws={"label": "Pearson r"})
    ax.set(xlabel="", ylabel="", title=title)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def run_contrasts(df, suffix):
    dmap = {"NDVI": "s2_ndvi_mean", "NIRv": "s2_nirv_mean", "S1 VH": "s1_asc_vh_mean_dB",
            "S1 CR": "s1_asc_cr_mean_dB", "NDII": "s2_ndii_mean", "NDMI": "s2_ndmi_mean"}
    drv = {"Leaf angle": "leaf_angle_qc", "PAI hemi": "pai_hemi_hi_hinge_mean_m2m2",
           "Soil moisture": "sm_mean_pct", "GNSS-T VOD": "vod_daily_max", "Predawn SWP": "swp_pd_mean_MPa"}
    contrasts = {
        "optical structure": (["NDVI", "NIRv"], ["Leaf angle", "PAI hemi"]),
        "radar": (["S1 VH", "S1 CR"], ["Soil moisture", "GNSS-T VOD", "PAI hemi"]),
        "optical water": (["NDII", "NDMI"], ["GNSS-T VOD", "Predawn SWP", "Soil moisture"]),
    }
    tidy, table = [], []
    for cname, (targs, drvs) in contrasts.items():
        for tl in targs:
            res = commonality(df, dmap[tl] + suffix, [drv[d] + suffix for d in drvs])
            if res is None:
                continue
            lab = {drv[d] + suffix: d for d in drvs}
            for c, v in res["unique"].items():
                tidy.append(dict(contrast=cname, target=tl, component=lab[c], value=max(v, 0)))
                table.append(dict(contrast=cname, target=tl, n=res["n"], component=lab[c], unique_r2=round(v, 3)))
            tidy.append(dict(contrast=cname, target=tl, component="shared", value=max(res["shared"], 0)))
            table.append(dict(contrast=cname, target=tl, n=res["n"], component="shared", unique_r2=round(res["shared"], 3)))
    return pd.DataFrame(tidy), pd.DataFrame(table)


def fig_contrasts(tidy, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, c in zip(axes, ["optical structure", "radar", "optical water"]):
        sns.barplot(data=tidy[tidy.contrast == c], x="component", y="value", hue="target", ax=ax)
        ax.set(title=c, ylabel="unique variance (R², anomalies)", xlabel="")
        ax.tick_params(axis="x", rotation=25); ax.legend(fontsize=8, title="")
    fig.suptitle("App B SUPPORT — commonality on event-preserving ANOMALIES (grey = shared/collinear)")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(path); plt.close(fig)


def event_retention(df, baseline, peak):
    rows = []
    for lab, col, _ in TARGETS:  # all satellite -> peak here is the satellite window
        if col not in df.columns:
            continue
        raw = df[col]; anom = df[col + "_a"]
        rd = raw.loc[peak[0]:peak[1]].mean() - raw.loc[baseline[0]:baseline[1]].mean()
        ad = anom.loc[peak[0]:peak[1]].mean() - anom.loc[baseline[0]:baseline[1]].mean()
        rows.append(dict(index=lab, raw_delta=round(rd, 4), anom_delta=round(ad, 4),
                         retention=round(ad / rd, 2) if rd else np.nan))
    return pd.DataFrame(rows)


def main() -> int:
    cfg = load_config(); ab = cfg["appB"]
    df_all, drivers = build(cfg)
    df = df_all.loc[ab["window"][0]:ab["window"][1]].copy()
    # season-TRAJECTORY plots run past the analysis window, through end of November (autumn senescence)
    df_season = df_all.loc[ab["window"][0]:"2025-11-30"].copy()
    base, peak, ens = ab["baseline"], ab["peak"], ab["baseline_ensemble"]
    peak_sat = ab.get("peak_satellite", peak)   # satellite-only event window (entire August)

    # anomalies: seasonal polynomial fit OUTSIDE the event -> event preserved in the residual.
    # NOTE: SUPPORT is intentionally left UNCHANGED in this pass (primary-figure scope). Decoupling
    # this exclusion to the tighter appB.event_exclude window materially shifts the support results
    # (retention drops; NDII top-driver flips) and is deferred to a dedicated support pass.
    excl = (pd.Timestamp(base[0]), pd.Timestamp(peak[1]))
    for _, col, _ in drivers + TARGETS:
        if col in df.columns:
            df[col + "_a"] = poly_detrend(df[col], ab["poly_order"], exclude=excl)

    tdir = HERE / "outputs" / "tables"; tdir.mkdir(parents=True, exist_ok=True)
    fdir = HERE / "outputs" / "figures"; fdir.mkdir(parents=True, exist_ok=True)
    adir = HERE / "outputs" / "appendix"; adir.mkdir(parents=True, exist_ok=True)

    # PRIMARY: event-change over the baseline ensemble (+ hand-set single-baseline reference dot)
    ec = event_change_ensemble(df, drivers, ens, peak, peak_sat)
    ec = ec.merge(event_change(df, drivers, base, peak, peak_sat)[["var", "handset_delta"]],
                  on="var", how="left")
    ec.to_csv(tdir / "event_change.csv", index=False)
    fig_event_change(ec, ens, peak, peak_sat, fdir / "event_change_delta.png", mode="signed")     # headline
    fig_event_change(ec, ens, peak, peak_sat, fdir / "event_change_delta_abs.png", mode="abs")    # magnitude
    fig_event_change(ec, ens, peak, peak_sat, adir / "event_change_delta_baselinecheck.png",
                     mode="signed", show_handset=True)                                      # appendix: dots inside bands

    # seasonal z-score line plots (plain visualization — watch the fall during the event)
    insitu_panels = [
        ("Forcing", [("Soil moisture", "sm_mean_pct"), ("VPD", "vpd_max_hPa")]),
        ("Plant water status", [("Predawn SWP", "swp_pd_mean_MPa"), ("TWD", "twd_pd_mean_um"),
                                ("GNSS-T VOD", "vod_daily_max")]),
        ("Water / carbon flux", [("Sap flow", "sapflow_sum_mean"), ("GPP", "gpp_mean_umol_m2_s"),
                                 ("ET", "et_sum_mm")]),
        ("Biomass / structure", [("PAI hemi", "pai_hemi_hi_hinge_mean_m2m2"),
                                 ("PAI hinge", "pai_hinge_hinge_mean_m2m2"),
                                 ("Leaf angle", "leaf_angle_qc")]),
        ("Phenocam greenness", [("GCC", "gcc_ens")]),
    ]
    sat_panels = [
        ("Sentinel-2 greenness", [("NDVI", "s2_ndvi_mean"), ("NIRv", "s2_nirv_mean"),
                                  ("kNDVI", "s2_kndvi_mean"), ("EVI", "s2_evi_mean")]),
        ("Sentinel-2 red-edge / chlorophyll", [("CIre", "s2_cire_mean"), ("MTCI", "s2_mtci_mean"),
                                               ("NDREI", "s2_ndrei_mean")]),
        ("Sentinel-2 water", [("NDII", "s2_ndii_mean"), ("NDMI", "s2_ndmi_mean"),
                              ("NMDI", "s2_nmdi_mean")]),
        ("Sentinel-2 senescence", [("PSRI", "s2_psri_mean"), ("CIg", "s2_cig_mean")]),
        ("Sentinel-1 radar", [("VH", "s1_asc_vh_mean_dB"), ("VV", "s1_asc_vv_mean_dB"),
                              ("span", "s1_asc_span_mean_dB"), ("CR", "s1_asc_cr_mean_dB"),
                              ("RVI", "s1_asc_rvi_mean")]),
    ]
    fig_season_lines(df_season, insitu_panels, peak, ens, fdir / "season_lines_insitu.png",
                     "App B — in-situ seasonal z-scores by theme", sparse=False)
    fig_season_lines(df_season, sat_panels, peak, ens, fdir / "season_lines_satellite.png",
                     "App B — satellite seasonal z-scores by theme", sparse=True)
    # native-unit small multiples — one index per panel (no scale squishing), S2 & S1 split
    s2_grid = [  # (label, col, unit, colour-by-theme)
        ("NDVI", "s2_ndvi_mean", "–", "#5aae61"), ("NIRv", "s2_nirv_mean", "–", "#5aae61"),
        ("kNDVI", "s2_kndvi_mean", "–", "#5aae61"), ("EVI", "s2_evi_mean", "–", "#5aae61"),
        ("CIre", "s2_cire_mean", "–", "#762a83"), ("MTCI", "s2_mtci_mean", "–", "#762a83"),
        ("NDREI", "s2_ndrei_mean", "–", "#762a83"),
        ("NDII", "s2_ndii_mean", "–", "#2166ac"), ("NDMI", "s2_ndmi_mean", "–", "#2166ac"),
        ("NMDI", "s2_nmdi_mean", "–", "#2166ac"),
        ("PSRI", "s2_psri_mean", "–", "#d6604d"), ("CIg", "s2_cig_mean", "–", "#d6604d"),
    ]
    s1_grid = [
        ("VH (asc)", "s1_asc_vh_mean_dB", "dB", "#ef8a62"),
        ("VV (asc)", "s1_asc_vv_mean_dB", "dB", "#ef8a62"),
        ("Span = VV+VH", "s1_asc_span_mean_dB", "dB", "#ef8a62"),
        ("CR = VH−VV", "s1_asc_cr_mean_dB", "dB", "#b35806"),
        ("RVI", "s1_asc_rvi_mean", "–", "#b35806"),
    ]
    fig_season_grid(df_season, s2_grid, peak, ens, fdir / "season_grid_s2.png",
                    "App B — Sentinel-2 index trajectories (native units, one per panel)", ncols=2)
    fig_season_grid(df_season, s1_grid, peak, ens, fdir / "season_grid_s1.png",
                    "App B — Sentinel-1 backscatter trajectories (native units, one per panel)", ncols=2)
    # date-labelled variants (every acquisition date on the x-axis) + exact-value CSVs
    fig_season_grid(df_season, s2_grid, peak, ens, fdir / "season_grid_s2_dates.png",
                    "App B — Sentinel-2 index trajectories (acquisition dates labelled)", ncols=2, date_ticks=True)
    fig_season_grid(df_season, s1_grid, peak, ens, fdir / "season_grid_s1_dates.png",
                    "App B — Sentinel-1 backscatter trajectories (acquisition dates labelled)", ncols=2, date_ticks=True)
    write_index_values_csv(df_season, s2_grid, tdir / "season_grid_s2_values.csv")
    write_index_values_csv(df_season, s1_grid, tdir / "season_grid_s1_values.csv")

    mat_a = corr_matrix(df, drivers, "_a", "_a")
    mat_a.to_csv(tdir / "driver_index_corr_anomaly.csv", index=False)
    fig_heatmap(mat_a, drivers, "App B SUPPORT — driver × index on ANOMALIES (stress signal)\n"
                "(event-preserving 3rd-deg seasonal detrend; * p<.05 ** p<.01 *** p<.001)",
                fdir / "driver_index_heatmap_anomaly.png")

    tidy, ctab = run_contrasts(df, "_a")
    ctab.to_csv(tdir / "commonality_anomaly.csv", index=False)
    fig_contrasts(tidy, fdir / "commonality_contrasts_anomaly.png")

    ret = event_retention(df, base, peak_sat)
    ret.to_csv(tdir / "appB_event_retention.csv", index=False)

    # appendix: raw-level matrix (caveated seasonal climatology)
    mat_lvl = corr_matrix(df, drivers, "", "")
    mat_lvl.to_csv(adir / "driver_index_corr_levels.csv", index=False)
    fig_heatmap(mat_lvl, drivers, "APPENDIX — raw LEVELS = seasonal climatology (phenology), "
                "NOT stress attribution", adir / "driver_index_heatmap_levels.png")

    print("=== EVENT-CHANGE (z, baseline ENSEMBLE → peak) — PAI≈0 is the control ===")
    print(ec.sort_values("delta_median").to_string(index=False))
    print("\n=== Anomaly attribution — top driver per satellite signal ===")
    for lab, col, grp in TARGETS:
        s = mat_a[(mat_a.target == lab)].dropna(subset=["r"])
        if s.empty:
            continue
        top = s.iloc[s["r"].abs().argmax()]
        print(f"  {grp:14} {lab:7} best: {top['driver']:13} r={top['r']:+.2f}{top['sig']} (n={top['n']})")
    print("\n=== Event-retention of the seasonal detrend (should be ≳0.8) ===")
    print(ret.to_string(index=False))
    print(f"\nwrote tables -> {tdir}\nfigures -> {fdir}\nappendix -> {adir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
