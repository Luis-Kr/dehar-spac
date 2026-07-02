"""App C exploration: correct GNSS-T VOD with the UNDERSTORY PAI only (1.5-9 m).

The height-resolved analysis showed the whole seasonal + drought PAI signal is
the deciduous understory; the overstory is a near-constant (occlusion-limited)
offset. Since the overstory is ~constant, regressing VOD on total-PAI vs
understory-PAI differs only by the intercept — *except* for the overstory's
occlusion-artifact wiggle. So understory-only should match total-PAI or slightly
beat it by dropping that noise. This tests it against predawn SWP.

Compares biomass regressors (all event-blind fits, ADR 0007): total (1.5-18 m),
understory (1.5-9 m), overstory (9-18 m), plus raw VOD and a LOESS +/-30 d
detrend as references. Reports seasonal + event r vs SWP and the August-dip
retention, and draws the window-resolved correlation.

Run:
  /home/lk1167/miniconda3/envs/dehar-spac/bin/python other/appc_understory_correction.py
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
sys.path.insert(0, str(REPO / "other"))
sys.path.insert(0, str(REPO / "paper" / "config"))
sys.path.insert(0, str(REPO / "paper" / "30_appC_vod_correction"))
from paper_common import load_config, load_daily, load_event_windows, paper_style  # noqa: E402
from leaf_height_dynamics import daily_grids, SPLIT, HMAX  # noqa: E402
import vod_pai_correction as V  # noqa: E402

OUT = REPO / "other" / "appc_understory_correction.png"
NO_EXCL = (pd.Timestamp("2099-01-01"), pd.Timestamp("2099-01-02"))   # event-blind fit


def band_raw(pai_grid, h1, h2):
    """Layer PAI = cumulative WeightedPAI at h2 minus at h1 (raw daily, no smooth)."""
    H = np.array(sorted(pai_grid.columns))
    at = lambda h: pai_grid[H[np.abs(H - h).argmin()]]              # noqa: E731
    return at(h2) - at(h1)


def r_on(a, b, w0, w1):
    d = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna().loc[w0:w1]
    return (stats.pearsonr(d.a, d.b)[0], len(d)) if len(d) > 3 else (np.nan, len(d))


def main() -> int:
    paper_style()
    cfg = load_config(); ac = cfg["appC"]
    df = load_daily(cfg)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.join(V.build_vod_daily_max(cfg))
    vod, swp = "vod_daily_max", ac["validation"]["primary"]
    season = [pd.Timestamp(x) for x in ac["window"]]
    event = [pd.Timestamp(x) for x in ac["event_exclude"]]
    ev = load_event_windows()
    base = (ev.loc[ev.stage == "baseline", "start"].iloc[0],
            ev.loc[ev.stage == "baseline", "end"].iloc[0])

    # layer PAI from the up-corrected profiles (raw daily, same aggregation)
    _, _, pai_grid, _ = daily_grids()
    df["pai_total"] = band_raw(pai_grid, 1.5, HMAX).reindex(df.index)
    df["pai_under"] = band_raw(pai_grid, 1.5, SPLIT).reindex(df.index)
    df["pai_over"] = band_raw(pai_grid, SPLIT, HMAX).reindex(df.index)

    dfw = df.loc[season[0]:season[1]].copy()
    regressors = {"total PAI (1.5-18 m)": "pai_total",
                  "understory PAI (1.5-9 m)": "pai_under",
                  "overstory PAI (9-18 m)": "pai_over"}
    corr = {}
    for lab, col in regressors.items():
        a, b, _ = V.fit_biomass(dfw, vod, col, NO_EXCL)
        corr[lab] = dfw[vod] - (a + b * dfw[col])
    corr["LOESS +/-30 d"] = dfw[vod] - V.loess_baseline(dfw[vod], 30).reindex(dfw.index)

    evp = V.event_preservation(dfw, vod, corr, base, event).set_index("signal")["retention"]
    rows = []
    for lab, w in {"raw VOD": dfw[vod], **corr}.items():
        rs, ns = r_on(w, dfw[swp], season[0], season[1])
        re, _ = r_on(w, dfw[swp], event[0], event[1])
        rows.append(dict(signal=lab, r_season=round(rs, 3), n=ns,
                         r_event=round(re, 3), dip_retention=evp.get(lab, np.nan)))
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))

    # ---- figure: window-resolved SWP correlation per regressor ---------------- #
    ends = pd.date_range(season[0] + pd.Timedelta("75D"), season[1], freq="7D")
    dff = dfw.assign(**{f"w_{k}": v for k, v in
                        {"total": corr["total PAI (1.5-18 m)"],
                         "under": corr["understory PAI (1.5-9 m)"],
                         "over": corr["overstory PAI (9-18 m)"]}.items()})
    pairs = [("raw VOD (ref)", vod, swp), ("total PAI", "w_total", swp),
             ("understory PAI", "w_under", swp), ("overstory PAI", "w_over", swp)]
    wr = V.window_resolved(dff, pairs, season[0], ends)
    sty = {"raw VOD (ref)": ("0.55", "--", 1.6), "total PAI": ("#2166ac", "-", 2.4),
           "understory PAI": ("#238b45", "-", 2.8), "overstory PAI": ("#8c6d31", ":", 2.0)}

    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
    for lab, g in wr.groupby("pair"):
        c, ls, lw = sty[lab]; g = g.sort_values("end")
        ax[0].plot(g["end"], g["r"], marker="o", ms=4, ls=ls, lw=lw, color=c, label=lab)
    ax[0].axhline(0, color="0.5", lw=0.8)
    ax[0].axvspan(event[0], event[1], color="k", alpha=0.12)
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax[0].set(xlabel="evaluation window end (from May 1)", ylabel="Pearson r (water-VOD ↔ SWP)",
              title="(A) window-resolved VOD↔SWP — understory-only vs total PAI")
    ax[0].legend(loc="best", fontsize=9)

    sub = tab[tab.signal.isin(["total PAI (1.5-18 m)", "understory PAI (1.5-9 m)",
                               "overstory PAI (9-18 m)", "LOESS +/-30 d"])]
    x = np.arange(len(sub))
    ax[1].bar(x - 0.2, sub["r_season"], 0.38, label="seasonal r (SWP)", color="#4292c6")
    ax[1].bar(x + 0.2, sub["dip_retention"], 0.38, label="Aug dip retention", color="#41ab5d")
    ax[1].axhline(0, color="0.5", lw=0.8)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([s.split(" PAI")[0].split(" +/-")[0] for s in sub.signal], rotation=20)
    ax[1].set(title="(B) seasonal SWP r & event-dip retention", ylabel="")
    ax[1].legend(loc="best", fontsize=9)

    fig.suptitle("App C exploration — correcting VOD with understory-only PAI (1.5-9 m)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT, dpi=150); plt.close(fig)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
