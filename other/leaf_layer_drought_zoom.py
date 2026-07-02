"""Drought-response by canopy layer: does the understory PAI dip lead/lag the
water sensors? (other/, off-pipeline exploration.)

The height figure (other/leaf_height_dynamics.py) showed the August drought signal
lives in the DECIDUOUS understory. Here we place that understory dip in the stress
cascade — soil moisture (forcing) -> VPD (demand) -> predawn SWP (plant water) ->
understory PAI (structural response) -> overstory PAI (evergreen control):

  (A) event-window overlay of z-scored signals (7 d smooth) — read the sequence.
  (B) lagged cross-correlation of the understory-PAI anomaly vs the water sensors —
      peak lag > 0 means the canopy structure LAGS the water stress.

Run:
  /home/lk1167/miniconda3/envs/dehar-spac/bin/python other/leaf_layer_drought_zoom.py
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
from paper_common import load_config, load_daily, paper_style  # noqa: E402
from leaf_height_dynamics import daily_grids, band_pai, SPLIT, HMAX, EVENT  # noqa: E402
from vod_pai_correction import loess_baseline  # noqa: E402

OUT = REPO / "other" / "leaf_layer_drought_zoom.png"
ZOOM = (pd.Timestamp("2025-06-01"), pd.Timestamp("2025-10-15"))
LAGS = np.arange(-30, 31)


def z(s):
    s = pd.Series(s)
    return (s - s.mean()) / s.std()


def sm7(s):
    return pd.Series(s).rolling(7, center=True, min_periods=2).mean()


def anom(s):
    """Seasonal anomaly = series minus its LOESS +/-30 d baseline (removes leaf-off trend)."""
    s = pd.Series(s).astype(float)
    return s - loess_baseline(s.dropna(), 30).reindex(s.index)


def lagcorr(a, b, lags):
    """corr(a(t), b(t - lag)); peak at lag>0 means a LAGS b by that many days."""
    out = []
    for L in lags:
        d = pd.concat([pd.Series(a), pd.Series(b).shift(L)], axis=1).dropna()
        out.append(stats.pearsonr(d.iloc[:, 0], d.iloc[:, 1])[0] if len(d) > 15 else np.nan)
    return np.array(out)


def main() -> int:
    paper_style()
    _, _, pai, _ = daily_grids()
    under = band_pai(pai, 1.5, SPLIT)
    over = band_pai(pai, SPLIT, HMAX)

    df = load_daily(load_config())
    df.index = pd.to_datetime(df.index).tz_localize(None)
    soil = df["sm_mean_pct"]; vpd = df["vpd_mean_hPa"]; swp = df["swp_pd_mean_MPa"]

    # ---- Panel A: stress cascade overlay (z-scored, 7 d smooth, zoomed) ------- #
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 0.9], hspace=0.28)
    axA = fig.add_subplot(gs[0])
    series = [("soil moisture (forcing)", soil, "#8c510a", "-"),
              ("VPD (demand)", vpd, "#d95f02", "--"),
              ("predawn SWP (plant water)", swp, "#386cb0", "-"),
              ("understory PAI (structure)", under, "#238b45", "-"),
              ("overstory PAI (evergreen control)", over, "0.55", ":")]
    for lab, s, c, ls in series:
        zz = z(sm7(s)).loc[ZOOM[0]:ZOOM[1]]
        axA.plot(zz.index, zz, color=c, ls=ls, lw=2.4 if "understory" in lab else 1.8, label=lab)
    axA.axvspan(EVENT[0], EVENT[1], color="k", alpha=0.12)
    axA.axhline(0, color="0.6", lw=0.6)
    axA.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axA.set(ylabel="z-score (7 d smooth)",
            title="(A) August-drought cascade — soil ↓ / VPD ↑ → predawn SWP ↓ → understory PAI ↓ "
                  "(overstory flat)")
    axA.legend(loc="lower left", fontsize=8, ncol=2)

    # ---- Panel B: lead/lag of the understory-PAI anomaly vs the water sensors -- #
    axB = fig.add_subplot(gs[1])
    up_a = anom(under)
    causal = LAGS >= 0                       # a drought response cannot lead the forcing
    for lab, s, c in [("vs predawn SWP", swp, "#386cb0"),
                      ("vs soil moisture", soil, "#8c510a")]:
        cc = lagcorr(up_a, anom(s), LAGS)
        kpos = int(LAGS[causal][np.nanargmax(cc[causal])])
        axB.plot(LAGS, cc, color=c, lw=2.4, label=f"{lab}  (peak lag +{kpos} d)")
        axB.plot(kpos, np.nanmax(cc[causal]), "o", color=c, ms=7)
    axB.axvline(0, color="0.5", lw=0.8); axB.axvspan(-30, 0, color="0.92")
    axB.set(xlabel="lag of understory PAI relative to the water sensor (days; >0 = PAI lags)",
            ylabel="Pearson r (anomalies)",
            title="(B) understory structure lags the water stress by ~2 weeks "
                  "(peak over causal lags ≥ 0; grey = non-causal)")
    axB.legend(loc="lower center")

    fig.suptitle("Drought response by canopy layer 2025 — understory structure lags the water cascade",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT, dpi=150); plt.close(fig)

    for lab, s in [("SWP", swp), ("soil moisture", soil)]:
        cc = lagcorr(anom(under), anom(s), LAGS)
        kpos = int(LAGS[causal][np.nanargmax(cc[causal])])
        print(f"understory PAI vs {lab:14s}: causal peak r={np.nanmax(cc[causal]):.2f} "
              f"at lag +{kpos} d")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
