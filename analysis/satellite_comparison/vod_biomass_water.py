"""Application B: decompose GNSS-T VOD into biomass (PAI) + canopy-water, validate
the water part against predawn stem water potential, and test whether the
corrected water-VOD explains Sentinel-1 cross-ratio better than raw VOD.

Why
---
GNSS-T VOD mixes a biomass/structure term and a canopy-water term. Following
Momen et al. 2017 (VOD = alpha + beta*LAI + gamma*psi + eta*psi*LAI), the
biomass term ~ a + b*PAI, so **regressing VOD on PAI and taking the residual
isolates the water signal** -- and leaves stem water potential (psi) free for an
independent validation. Humphrey & Frankenberg 2023 add the timescale view
(low-frequency VOD = biomass, high-frequency = water) as a PAI-free cross-check.
L-band GNSS-T weights woody/stem water, so predawn *stem* psi is the apt target.

This also tests the user's hypothesis for the S1 CR <-> soil-moisture result:
raw VOD is biomass-contaminated, so once corrected the water-VOD should pick up
more of CR's variance. The L-band(GNSS-T) vs C-band(S1) gap caps this (the test
is one-directional: a clear rise confirms, a null does not refute).

Decomposition is NOT unique -- it is only justified if the psi validation passes.

Run
---
    python analysis/satellite_comparison/vod_biomass_water.py --csv <file>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Reuse helpers from the sibling script WITHOUT editing it (no __init__.py here,
# so add this dir to the path and import; the module is import-safe -- all heavy
# code is guarded by `if __name__ == "__main__"`).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from index_sensitivity import _rsq, commonality, z  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "data" / "processed" / "dehar_daily_season_2025_filtered.csv"
SEASON = ("2025-05-01", "2025-10-31")
EVENT = ("2025-08-07", "2025-08-20")          # the hot-dry event
BASELINE_WINDOW = 45                          # days, for the PAI-free cross-check
PSI_THRESH = -0.5                             # stem Psi (MPa): VOD<->Psi tightens below this

VOD, PAI, PSI, SM = "vod_mean", "pai_total_sg", "swp_mpa_predawn_mean", "sm_pct_mean"


def load(csv: Path, start: str, end: str) -> pd.DataFrame:
    df = pd.read_csv(csv, parse_dates=["date"]).set_index("date").sort_index()
    return df.loc[start:end]


# --------------------------------------------------------------------------- #
# Decomposition                                                               #
# --------------------------------------------------------------------------- #
def fit_biomass(df: pd.DataFrame, exclude: tuple[str, str] | None = None) -> tuple[float, float, dict]:
    """OLS VOD = a + b*PAI. PAI is ~flat through the growing season here and only
    varies in autumn senescence, so the structural slope is identified by the
    autumn co-decline. We exclude the event window so the August water drop does
    not bias the slope, and the event dip is preserved as a (water) residual."""
    d = df[[VOD, PAI]].dropna()
    if exclude is not None:
        d = d.loc[(d.index < exclude[0]) | (d.index > exclude[1])]
    lr = stats.linregress(d[PAI], d[VOD])
    return float(lr.intercept), float(lr.slope), {
        "r2": float(lr.rvalue ** 2), "n": len(d), "slope_p": float(lr.pvalue),
        "pai_cv": float(d[PAI].std() / d[PAI].mean())}


def water_vod(df: pd.DataFrame, a: float, b: float) -> pd.Series:
    """Primary decomposition: VOD minus the PAI-predicted biomass term."""
    return df[VOD] - (a + b * df[PAI])


def water_vod_baseline(df: pd.DataFrame, window: int = BASELINE_WINDOW) -> pd.Series:
    """PAI-free cross-check (Humphrey & Frankenberg): VOD minus a low-frequency
    seasonal baseline (centred rolling median)."""
    base = df[VOD].rolling(window, center=True, min_periods=window // 3).median()
    return df[VOD] - base


def momen_models(df: pd.DataFrame) -> pd.DataFrame:
    """Nested OLS to show psi adds power beyond PAI (Momen 2017 replication)."""
    d = df[[VOD, PAI, PSI]].dropna()
    y = z(d[VOD]).to_numpy()
    pai, psi = z(d[PAI]).to_numpy(), z(d[PSI]).to_numpy()
    designs = {"M0: VOD~PAI": np.column_stack([pai]),
               "M1: +psi": np.column_stack([pai, psi]),
               "M2: +psi*PAI": np.column_stack([pai, psi, pai * psi])}
    rows, prev_r2, n = [], 0.0, len(d)
    for name, X in designs.items():
        r2 = _rsq(y, X)
        k = X.shape[1]
        f = ((r2 - prev_r2) / 1) / ((1 - r2) / (n - k - 1)) if r2 < 1 else np.inf
        p = float(stats.f.sf(f, 1, n - k - 1)) if name != "M0: VOD~PAI" else np.nan
        rows.append({"model": name, "n": n, "r2": round(r2, 3),
                     "delta_r2": round(r2 - prev_r2, 3), "F_p": p})
        prev_r2 = r2
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Validation against predawn stem water potential                            #
# --------------------------------------------------------------------------- #
def validate(df: pd.DataFrame, vw: pd.Series) -> pd.DataFrame:
    work = df.assign(vod_water=vw)
    rows = []
    for wlabel, w in [("full season", SEASON), ("event", EVENT)]:
        sub = work.loc[w[0]:w[1]]
        for label, col in [("raw VOD", VOD), ("water-VOD", "vod_water")]:
            d = sub[[col, PSI]].dropna()
            if len(d) < 4:
                rows.append({"window": wlabel, "signal": label, "n": len(d),
                             "r": np.nan, "r2": np.nan})
                continue
            r, _ = stats.pearsonr(d[col], d[PSI])
            rows.append({"window": wlabel, "signal": label, "n": len(d),
                         "r": round(r, 3), "r2": round(r ** 2, 3)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# CR test: raw VOD vs water-VOD as a predictor                               #
# --------------------------------------------------------------------------- #
def cr_test(df: pd.DataFrame, vw: pd.Series,
            targets=("s1_cr_savgol", "s1_cr_raw", "s1_vh_raw", "s2_nirv_savgol")) -> pd.DataFrame:
    work = df.assign(vod_water=vw)
    rows = []
    for t in targets:
        if t not in work.columns:
            continue
        a = commonality(work, t, [VOD, SM])          # set A: raw VOD + SM
        b = commonality(work, t, ["vod_water", SM])  # set B: water-VOD + SM
        if a is None or b is None:
            continue
        rows.append({
            "target": t, "nA": a["n"], "nB": b["n"],
            "uVOD_raw": round(a["unique"][VOD], 3), "uSM_A": round(a["unique"][SM], 3),
            "uVOD_water": round(b["unique"]["vod_water"], 3), "uSM_B": round(b["unique"][SM], 3),
            "d_uVOD": round(b["unique"]["vod_water"] - a["unique"][VOD], 3),
            "d_uSM": round(b["unique"][SM] - a["unique"][SM], 3),
            "shared_A": round(a["shared"], 3), "shared_B": round(b["shared"], 3)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures                                                                     #
# --------------------------------------------------------------------------- #
def plot_decomposition(df, a, b, vw, vw_base, path):
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    ax[0].plot(df.index, df[VOD], color="#2166ac", label="raw VOD")
    ax[0].plot(df.index, a + b * df[PAI], "--", color="#8c510a",
               label=f"biomass fit a+b*PAI (b={b:.3f})")
    axp = ax[0].twinx()
    axp.plot(df.index, df[PAI], color="#1b7837", alpha=0.5, label="PAI")
    axp.set_ylabel("PAI", color="#1b7837")
    ax[0].set_ylabel("VOD"); ax[0].legend(loc="upper right", fontsize=8)
    ax[1].plot(df.index, vw, color="#2166ac", label="water-VOD (PAI residual)")
    ax[1].plot(df.index, vw_base, color="#b2182b", alpha=0.7,
               label="water-VOD anomaly (baseline cross-check)")
    ax[1].axhline(0, color="0.6", lw=0.6); ax[1].set_ylabel("water-VOD")
    ax[1].legend(loc="upper right", fontsize=8)
    ax[2].plot(df.index, df[PSI], color="#1b7837", label="predawn stem Psi")
    ax[2].set_ylabel("Psi (MPa)"); ax[2].legend(loc="lower right", fontsize=8)
    for a_ in ax:
        a_.axvspan(pd.Timestamp(EVENT[0]), pd.Timestamp(EVENT[1]), color="0.85", alpha=0.6)
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.suptitle("VOD biomass/water decomposition (event shaded)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140); plt.close(fig)


def plot_validation(df, vw, path):
    work = df.assign(vod_water=vw)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (wlabel, w) in zip(axes, [("full season", SEASON), ("event", EVENT)]):
        sub = work.loc[w[0]:w[1]]
        for col, c, lab in [(VOD, "#8c510a", "raw VOD"), ("vod_water", "#2166ac", "water-VOD")]:
            d = sub[[col, PSI]].dropna()
            if len(d) < 4:
                continue
            r, _ = stats.pearsonr(d[col], d[PSI])
            ax.scatter(d[PSI], z(d[col]), s=14, color=c, alpha=0.6,
                       label=f"{lab} (r={r:.2f})")
        ax.set(xlabel="predawn stem Psi (MPa)", ylabel="z-scored VOD", title=wlabel)
        ax.legend(fontsize=8)
    fig.suptitle("Validation: does removing biomass make VOD track stem Psi better?")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140); plt.close(fig)


def regime_table(df: pd.DataFrame, vw: pd.Series, thr: float = PSI_THRESH) -> pd.DataFrame:
    """VOD<->Psi correlation split at the stem-Psi threshold (stressed vs not)."""
    work = df.assign(vod_water=vw)
    rows = []
    for label, col in [("raw VOD", VOD), ("water-VOD", "vod_water")]:
        d = work[[col, PSI]].dropna()
        for rlabel, mask in [(f"Psi<{thr}", d[PSI] < thr), (f"Psi>={thr}", d[PSI] >= thr)]:
            dd = d[mask]
            r = stats.pearsonr(dd[col], dd[PSI])[0] if len(dd) >= 3 else np.nan
            rows.append({"signal": label, "regime": rlabel, "n": len(dd),
                         "r": round(r, 3) if np.isfinite(r) else np.nan,
                         "r2": round(r ** 2, 3) if np.isfinite(r) else np.nan})
    return pd.DataFrame(rows)


def plot_threshold(df: pd.DataFrame, vw: pd.Series, path, thr: float = PSI_THRESH):
    """Scatter VOD vs stem Psi, split at Psi=thr, with a regression line fit on the
    stressed (Psi<thr) points -- showing the relationship tightens under stress."""
    work = df.assign(vod_water=vw)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (col, lab, color) in zip(
            axes, [(VOD, "raw VOD", "#8c510a"), ("vod_water", "water-VOD", "#2166ac")]):
        d = work[[col, PSI]].dropna()
        wet, dry = d[d[PSI] >= thr], d[d[PSI] < thr]
        ax.scatter(wet[PSI], wet[col], s=16, color="0.65", alpha=0.5,
                   label=f"Psi >= {thr} (n={len(wet)})")
        ax.scatter(dry[PSI], dry[col], s=22, color=color, alpha=0.85,
                   label=f"Psi < {thr} (n={len(dry)})")
        r_all = stats.pearsonr(d[col], d[PSI])[0]
        ann = f"all r={r_all:.2f}"
        if len(dry) >= 3:
            lr = stats.linregress(dry[PSI], dry[col])
            xs = np.array([dry[PSI].min(), dry[PSI].max()])
            ax.plot(xs, lr.intercept + lr.slope * xs, "-", color=color, lw=2.2)
            ann += f"  |  Psi<{thr}: r={lr.rvalue:.2f} (slope {lr.slope:.3f})"
        if len(wet) >= 3:
            ax.plot(*_fitline(wet[PSI], wet[col]), ":", color="0.55", lw=1.4)
            ann += f"  |  Psi>={thr}: r={stats.pearsonr(wet[col], wet[PSI])[0]:.2f}"
        ax.axvline(thr, color="0.5", ls="--", lw=1)
        ax.set(xlabel="predawn stem Psi (MPa)", ylabel=lab, title=f"{lab}\n{ann}")
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("VOD-stem Psi relationship tightens under stress (Psi < -0.5 MPa)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140); plt.close(fig)


def _fitline(x, y):
    lr = stats.linregress(x, y)
    xs = np.array([x.min(), x.max()])
    return xs, lr.intercept + lr.slope * xs


def plot_cr(cr: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(cr)); w = 0.35
    ax.bar(x - w / 2, cr.uVOD_raw, w, color="#9ecae1", label="unique VOD (raw)")
    ax.bar(x - w / 2, cr.uSM_A, w, bottom=cr.uVOD_raw, color="#fdae6b", label="unique SM (set A)")
    ax.bar(x + w / 2, cr.uVOD_water, w, color="#2166ac", label="unique VOD (water)")
    ax.bar(x + w / 2, cr.uSM_B, w, bottom=cr.uVOD_water, color="#e6550d", label="unique SM (set B)")
    ax.set_xticks(x); ax.set_xticklabels(cr.target, rotation=20)
    ax.set(ylabel="unique variance", title="CR test: raw VOD+SM (left) vs water-VOD+SM (right)\n"
           "does water-VOD take CR variance from soil moisture?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140); plt.close(fig)


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Application B: PAI-corrected GNSS-T VOD")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--start", default=SEASON[0])
    p.add_argument("--end", default=SEASON[1])
    p.add_argument("--baseline-window", type=int, default=BASELINE_WINDOW)
    a = p.parse_args(argv)

    df = load(a.csv, a.start, a.end)
    a0, b0, info = fit_biomass(df, exclude=EVENT)
    vw = water_vod(df, a0, b0)
    vw_base = water_vod_baseline(df, a.baseline_window)

    summer_cv = df.loc["2025-05-01":"2025-08-01", PAI].std() / df.loc["2025-05-01":"2025-08-01", PAI].mean()
    print("Biomass fit VOD = a + b*PAI (season excluding the event window):")
    print(f"  a={a0:.3f}  b={b0:.4f}  R2={info['r2']:.2f}  n={info['n']}  (slope p={info['slope_p']:.1e})")
    print(f"  PAI is ~flat in summer (CV={summer_cv:.3f}) and varies mainly in autumn,")
    print(f"  so the slope is the autumn senescence co-decline; the event dip stays as residual.\n")

    cross = df.assign(a=vw, b=vw_base)[["a", "b"]].dropna()
    rcc = stats.pearsonr(cross.a, cross.b)[0] if len(cross) > 3 else np.nan
    print(f"Cross-check: r(PAI-residual, baseline-anomaly) = {rcc:.2f} "
          f"(high => the residual is the water signal, not a PAI artefact)\n")

    val = validate(df, vw)
    print("Validation vs predawn stem Psi (success = |r| of water-VOD > raw VOD):")
    print(val.to_string(index=False))
    fs = val[val.window == "full season"].set_index("signal")
    ok = abs(fs.loc["water-VOD", "r"]) > abs(fs.loc["raw VOD", "r"])
    print(f"  -> full-season: water-VOD {'BEATS' if ok else 'does NOT beat'} raw VOD "
          f"against stem Psi.\n")

    reg = regime_table(df, vw)
    print(f"VOD<->stem Psi by regime (relationship tightens when Psi < {PSI_THRESH} MPa):")
    print(reg.to_string(index=False), "\n")

    mom = momen_models(df)
    print("Momen replication (does psi add power beyond PAI?):")
    print(mom.to_string(index=False), "\n")

    cr = cr_test(df, vw)
    print("CR test -- unique variance, set A [raw VOD, SM] vs set B [water-VOD, SM]:")
    print(cr.to_string(index=False))
    crrow = cr[cr.target == "s1_cr_savgol"]
    if not crrow.empty:
        r = crrow.iloc[0]
        if r.d_uVOD > 0 and r.d_uSM < 0:
            print("  -> SUPPORTED: water-VOD takes CR variance from soil moisture.\n")
        else:
            print("  -> NOT supported: the Psi-validated water-VOD does NOT explain CR better")
            print("     (d_uVOD<=0); CR stays tied to soil moisture. So raw VOD was not merely a")
            print("     bad proxy -- CR is a soil/surface-water signal (open canopy) and/or the")
            print("     L-band(VOD) vs C-band(CR) gap prevents canopy-water from matching CR.\n")

    if not ok:
        print("CAVEAT: water-VOD did not beat raw VOD against Psi -> the decomposition is")
        print("not validated here and the CR test should be treated as inconclusive.\n")

    figs = REPO_ROOT / "figures"
    proc = REPO_ROOT / "data" / "processed"
    plot_decomposition(df, a0, b0, vw, vw_base, figs / "vod_biomass_water_decomposition.png")
    plot_validation(df, vw, figs / "vod_biomass_water_validation.png")
    plot_threshold(df, vw, figs / "vod_biomass_water_threshold.png")
    if not cr.empty:
        plot_cr(cr, figs / "vod_biomass_water_cr_commonality.png")
    proc.mkdir(parents=True, exist_ok=True)
    df.assign(vod_biomass_fit=a0 + b0 * df[PAI], vod_water=vw, vod_water_anom=vw_base)[
        [VOD, PAI, "vod_biomass_fit", "vod_water", "vod_water_anom", PSI]
    ].to_csv(proc / "vod_biomass_water_decomposition.csv")
    val.to_csv(proc / "vod_biomass_water_validation.csv", index=False)
    reg.to_csv(proc / "vod_biomass_water_regime.csv", index=False)
    mom.to_csv(proc / "vod_biomass_water_momen.csv", index=False)
    cr.to_csv(proc / "vod_biomass_water_cr_commonality.csv", index=False)
    print(f"Wrote figures to {figs} and CSV summaries to {proc}")


if __name__ == "__main__":
    main()
