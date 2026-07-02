"""App C exploration: VOD daily-aggregation metric sweep (understory-PAI corrected).

App C uses daily-MAX VOD as the predawn water-state proxy (Yao 2024) — predawn is
when predawn SWP is measured. Daily-max is the single most extreme reading, so it
is noise-prone; a high percentile (p90/p95) may be a more robust predawn proxy,
and mean/median/min probe different daily water states.

This sweeps the daily reducer {max, p95, p90, p75, mean, median, min} applied to
the QC'd hourly-mean VOD, corrects each with the canonical understory-PAI
(event-blind fit, ADR 0007/0008), and reports seasonal + event r vs predawn SWP
and the August-dip retention. Robustness check — NOT a best-r metric selection
(max is the literature-principled predawn choice).

Run:
  /home/lk1167/miniconda3/envs/dehar-spac/bin/python other/appc_vod_metric_sweep.py
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
from paper_common import load_config, load_daily, load_event_windows, paper_style  # noqa: E402
import vod_pai_correction as V  # noqa: E402

OUT = REPO / "other" / "appc_vod_metric_sweep.png"
NO_EXCL = (pd.Timestamp("2099-01-01"), pd.Timestamp("2099-01-02"))
PAI = "pai_hemi_hi_weighted_understory_mean_m2m2"
METRICS = {"max (canonical)": "max",
           "p95": lambda s: s.quantile(0.95), "p90": lambda s: s.quantile(0.90),
           "p75": lambda s: s.quantile(0.75), "mean": "mean", "median": "median",
           "min": "min"}


def build_vod_daily(cfg, agg, receivers=("gps1", "gps3", "gps5")):
    """QC'd daily VOD from the 30-min nvod ensemble, reduced per day by `agg`."""
    qc = cfg["appC"]["qc"]
    raw = pd.read_parquet(REPO / cfg["meta"]["streams_parquet"])
    raw.index = pd.to_datetime(raw.index)
    ens = raw[[f"nvod_{r}" for r in receivers]].mean(axis=1, skipna=True)
    precip = raw["precip_mm"].fillna(0).rolling(f"{int(qc['rain_lookback_h'])}h").sum()
    ens = ens.where((raw["meteo_rh_pct"] <= qc["rh_max_pct"]) & (precip <= qc["rain_thresh_mm"]))
    daily = ens.resample("1h").mean().resample("1D").agg(agg)
    daily.index = daily.index.tz_convert("UTC").tz_localize(None).normalize()
    return daily


def r_on(a, b, w0, w1):
    d = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna().loc[w0:w1]
    return (stats.pearsonr(d.a, d.b)[0], len(d)) if len(d) > 3 else (np.nan, len(d))


def main() -> int:
    paper_style()
    cfg = load_config(); ac = cfg["appC"]
    df = load_daily(cfg)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    swp = ac["validation"]["primary"]
    season = [pd.Timestamp(x) for x in ac["window"]]
    event = [pd.Timestamp(x) for x in ac["event_exclude"]]
    ev = load_event_windows()
    base = (ev.loc[ev.stage == "baseline", "start"].iloc[0],
            ev.loc[ev.stage == "baseline", "end"].iloc[0])

    def dip(s):
        return s.loc[base[0]:base[1]].mean() - s.loc[event[0]:event[1]].mean()

    rows, waters = [], {}
    for name, agg in METRICS.items():
        d = df.join(build_vod_daily(cfg, agg).rename("vod")).loc[season[0]:season[1]].copy()
        a, b, info = V.fit_biomass(d, "vod", PAI, NO_EXCL)
        water = d["vod"] - (a + b * d[PAI])
        waters[name] = (d, water)
        rs, ns = r_on(water, d[swp], season[0], season[1])
        re, _ = r_on(water, d[swp], event[0], event[1])
        rraw, _ = r_on(d["vod"], d[swp], season[0], season[1])
        ret = dip(water) / dip(d["vod"]) if dip(d["vod"]) else np.nan
        rows.append(dict(metric=name, n=ns, r_raw_season=round(rraw, 3),
                         r_season=round(rs, 3), r_event=round(re, 3),
                         dip_retention=round(ret, 3), fit_r2=round(info["r2"], 3)))
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))

    # ---- figure: (A) per-metric bars, (B) window-resolved for a few metrics --- #
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.4), gridspec_kw={"width_ratios": [1.15, 1.2]})
    x = np.arange(len(tab))
    ax[0].bar(x - 0.22, tab["r_season"], 0.4, label="seasonal r (SWP)", color="#4292c6")
    ax[0].bar(x + 0.22, tab["r_event"], 0.4, label="event r (SWP)", color="#08519c")
    ax[0].plot(x, tab["dip_retention"], "o-", color="#41ab5d", label="Aug dip retention")
    ax[0].axhline(0, color="0.5", lw=0.8)
    ax[0].set_xticks(x); ax[0].set_xticklabels(tab["metric"], rotation=25, ha="right")
    ax[0].set(ylabel="Pearson r / retention",
              title="(A) VOD daily metric vs understory-PAI water-VOD ↔ SWP")
    ax[0].legend(loc="best", fontsize=8)

    ends = pd.date_range(season[0] + pd.Timedelta("75D"), season[1], freq="7D")
    colors = {"max (canonical)": "#2166ac", "p90": "#d94801", "mean": "#6a51a3", "min": "0.55"}
    for name in ("max (canonical)", "p90", "mean", "min"):
        d, water = waters[name]
        wr = V.window_resolved(d.assign(w=water), [("x", "w", swp)], season[0], ends)
        ax[1].plot(wr["end"], wr["r"], marker="o", ms=3.5, lw=2.4,
                   color=colors[name], label=name)
    ax[1].axhline(0, color="0.5", lw=0.8)
    ax[1].axvspan(event[0], event[1], color="k", alpha=0.12)
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax[1].set(xlabel="evaluation window end (from May 1)", ylabel="Pearson r (water-VOD ↔ SWP)",
              title="(B) window-resolved — daily metric choice")
    ax[1].legend(loc="best", fontsize=9)

    fig.suptitle("App C exploration — VOD daily-aggregation metric sweep (understory-PAI corrected)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT, dpi=150); plt.close(fig)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
