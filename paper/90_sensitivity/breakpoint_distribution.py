"""
Stage 90 (sensitivity / exploratory) — full-season breakpoint DISTRIBUTION.

The headline Part A (paper/10_partA_sensor_response/onset_bootstrap.py) reduces each
stream to ONE onset (first-departure or acute). This variant keeps the whole picture:
for every stream it shows where ALL qualifying *breakpoints* concentrate over the season,
so the story becomes "clusters of breakpoints in calendar time" rather than a single date.

Method (reuses the headline engine; nothing here feeds a headline number):
  * breakpoint    = a qualifying changepoint: a step in the stream's MEAN, in the stress
                    direction, larger than the stream's own noise (sigma_hat). Reuses
                    onset_bootstrap.qualifying_breakpoints — the SAME filter as an onset,
                    minus the collapse-to-one step. Recovery edges are excluded by the
                    direction filter, so a marker = the START of a drop, never its end.
  * direction     = each stream's stress sign; streams left 'auto' are resolved ONCE to the
                    sign of their largest qualifying step (dominant move), then that fixed
                    sign is used everywhere — so a dip-then-recovery yields one breakpoint.
  * distribution  = pool every qualifying breakpoint from every residual-block-bootstrap
                    replicate into one bag per stream; render as a unit-area Gaussian KDE
                    (Scott bandwidth) over the calendar axis. Concentration = tall peak;
                    spurious cuts smear into the baseline; two real drops = two peaks.
  * penalty       = config onset.breakpoint_dist (looser than headline so multiple
                    breakpoints survive; Killick et al. 2012 penalty form).

Scope: proximal + in-situ streams only (satellite dropped here); PhenoCam NDVI included.
Figure: a single full-season ridgeline, rows ordered by primary peak date, colored by
group. Mean breakpoints-per-replicate is reported in the table, never as strip height
(each stream's density is unit-area, so the eye reads WHERE not HOW MANY).

Run:
  /home/lk1167/miniconda3/envs/dehar-spac/bin/python paper/90_sensitivity/breakpoint_distribution.py
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "config"))
sys.path.insert(0, str(HERE.parents[0] / "10_partA_sensor_response"))
from onset_bootstrap import (  # noqa: E402  (shared headline engine — single source of truth)
    GROUP_COLOR,
    SEED,
    WORKERS,
    _block_resample,
    _changepoints,
    _min_size,
    build_streams_and_frame,
    qualifying_breakpoints,
)
from paper_common import load_config, load_event_windows, paper_style  # noqa: E402

# Visualization parameters (NOT config-contract method params: they change how the pooled
# breakpoint bag is *drawn*, not which breakpoints are detected — so no literature binding).
KDE_BANDWIDTH = "scott"      # data-adaptive Scott's rule; fall back to a fixed day-bw only if needed
PEAK_PROMINENCE_FRAC = 0.25  # a secondary peak must reach >= 25% of the primary peak height
ROW_FILL = 0.85              # tallest density across all streams fills this fraction of a row


# --------------------------------------------------------------------------- #
# Detection helpers                                                            #
# --------------------------------------------------------------------------- #
def resolve_direction(y, model, pen_scale, min_size):
    """Stress sign for an 'auto' stream = the sign of its largest qualifying step.

    Decided ONCE on the observed series, then reused for every bootstrap replicate, so a
    dip-then-recovery contributes only the drop (the dominant move), never the recovery."""
    inc = qualifying_breakpoints(y, "increase", model, pen_scale, min_size)
    dec = qualifying_breakpoints(y, "decrease", model, pen_scale, min_size)
    max_inc = max((sd for _, sd in inc), default=0.0)
    max_dec = max((sd for _, sd in dec), default=0.0)
    return "increase" if max_inc >= max_dec else "decrease"


def pooled_breakpoints(y, direction, model, pen_scale, min_size, n_boot, block, seed):
    """Return (observed_idx, pooled_idx, mean_count_per_replicate).

    observed_idx : breakpoint positions on the observed series (drawn as ticks).
    pooled_idx   : every qualifying breakpoint position from every bootstrap replicate
                   (the bag the KDE is built from).
    """
    observed = [cp for cp, _ in qualifying_breakpoints(y, direction, model, pen_scale, min_size)]
    n = len(y)
    bounds = [0, *_changepoints(y, model, pen_scale, min_size), n]
    fitted = np.empty(n)
    for i in range(len(bounds) - 1):
        fitted[bounds[i]:bounds[i + 1]] = y[bounds[i]:bounds[i + 1]].mean()
    resid = y - fitted

    rng = np.random.default_rng(seed)
    pooled, counts = [], []
    for _ in range(n_boot):
        yb = fitted + _block_resample(resid, rng, block)
        cps = [cp for cp, _ in qualifying_breakpoints(yb, direction, model, pen_scale, min_size)]
        pooled.extend(cps)
        counts.append(len(cps))
    return observed, np.asarray(pooled, dtype=int), float(np.mean(counts)) if counts else 0.0


def _bp_task(args):
    """Pool worker: one stream's resolved direction + bootstrapped breakpoint bag.

    Module-level for picklability (ProcessPoolExecutor)."""
    name, group, y, dates, direction, model, pen_scale, block, n_boot, seed = args
    ms = _min_size(len(y))
    if direction == "auto":
        direction = resolve_direction(y, model, pen_scale, ms)
    obs, pool, mean_cnt = pooled_breakpoints(
        y, direction, model, pen_scale, ms, n_boot, block, seed)
    return {
        "stream": name, "group": group, "direction": direction,
        "obs_dates": dates[obs] if obs else pd.DatetimeIndex([]),
        "pool_num": mdates.date2num(dates[pool]) if len(pool) else np.array([]),
        "n_obs": len(obs), "mean_count": mean_cnt,
    }


# --------------------------------------------------------------------------- #
# Density + peak extraction (calendar axis, in matplotlib date numbers)        #
# --------------------------------------------------------------------------- #
def kde_on_grid(pool_num, grid):
    """Unit-area Gaussian KDE of pooled breakpoint date-numbers, evaluated on `grid`.

    Returns None when the bag is too small/degenerate for a KDE (jitter avoids a
    singular covariance when many replicates land on the exact same day)."""
    if len(pool_num) < 5 or np.ptp(pool_num) == 0:
        return None
    rng = np.random.default_rng(SEED)
    jittered = pool_num + rng.normal(0.0, 0.5, size=len(pool_num))  # +/- half-day, sub-grid
    try:
        kde = gaussian_kde(jittered, bw_method=KDE_BANDWIDTH)
    except np.linalg.LinAlgError:
        return None
    return kde(grid)


def primary_peaks(grid, dens):
    """Top-2 KDE peaks (date-numbers) by height, with a relative-prominence floor."""
    if dens is None:
        return []
    peaks, props = find_peaks(dens, prominence=PEAK_PROMINENCE_FRAC * float(dens.max()))
    if len(peaks) == 0:
        return [grid[int(np.argmax(dens))]]  # monotone/flat: report the global max
    order = np.argsort(props["prominences"])[::-1][:2]  # top-2 peaks by prominence
    top = peaks[order]                                   # their grid indices
    return [grid[gi] for gi in sorted(top)]              # chronological


def build_stream_set(cfg):
    """Part A stream set (already satellite-free, PhenoCam NDVI included — same as headline)."""
    return build_streams_and_frame(cfg)


# --------------------------------------------------------------------------- #
def main() -> int:
    cfg = load_config()
    paper_style()
    plt.rcParams.update({"font.size": 13})  # this figure is read at poster/talk scale
    bp = dict(cfg["onset"]["breakpoint_dist"])
    bp["pen_scale"] = float(os.environ.get("BP_PEN", bp["pen_scale"]))  # exploratory override
    win = [pd.Timestamp(d) for d in cfg["onset"]["search_window"]]

    ev = load_event_windows()
    ref_date = ev.loc[ev.stage == "soil_limit", "start"].iloc[0]
    acute_start = ev.loc[ev.stage == "acute", "start"].iloc[0]
    peak = pd.Timestamp("2025-08-20")  # predawn-Psi minimum (= headline 'Psi min' mark)

    streams, df = build_stream_set(cfg)
    df = df.loc[win[0]:win[1]]

    # --- detect + bootstrap each stream (parallel across streams) ----------- #
    tasks = []
    for i, st in enumerate(streams):
        if st.column not in df.columns:
            continue
        s = df[st.column].dropna()
        if len(s) < 6:
            continue
        tasks.append((st.name, st.group, s.to_numpy(float), s.index, st.direction,
                      bp["model"], bp["pen_scale"], bp["block"], bp["n_boot"], SEED + i))
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        records = list(ex.map(_bp_task, tasks))

    # --- season KDE (the estimator) + peaks; order rows by primary peak ------ #
    season_grid = mdates.date2num(pd.date_range(win[0], win[1], freq="D"))
    for r in records:
        dens = kde_on_grid(r["pool_num"], season_grid)
        r["dens"] = dens
        r["peaks"] = primary_peaks(season_grid, dens)
        r["peak1"] = r["peaks"][0] if r["peaks"] else np.nan
    no_peak = np.nanmax([r["peak1"] for r in records if not np.isnan(r["peak1"])]) + 10
    records.sort(key=lambda r: r["peak1"] if not np.isnan(r["peak1"]) else no_peak)

    gmax = max((float(np.nanmax(r["dens"])) for r in records if r["dens"] is not None),
               default=1.0)
    hscale = ROW_FILL / gmax if gmax > 0 else 1.0
    nrows = len(records)
    marks = [("soil-limit", ref_date, "#8c8c8c"), ("acute start", acute_start, "#b2182b"),
             ("Ψ min", peak, "#1b7837")]

    # --- figure: single full-season ridgeline ------------------------------- #
    fig, ax = plt.subplots(figsize=(15, 0.72 * nrows + 2.2), constrained_layout=True)
    for i, r in enumerate(records):
        c = GROUP_COLOR[r["group"]]
        if r["dens"] is not None:
            ax.fill_between(season_grid, i, i + r["dens"] * hscale, color=c, alpha=0.6,
                            lw=2.2, ec=c, zorder=2)
        for d in r["obs_dates"]:  # observed-series breakpoint ticks on the baseline
            ax.plot([mdates.date2num(d)] * 2, [i, i + 0.20], color=c, lw=3.0, zorder=4)
    for label, d, mc in marks:
        ax.axvline(mdates.date2num(pd.Timestamp(d)), color=mc, ls="--", lw=2.0, alpha=0.8)
        ax.text(mdates.date2num(pd.Timestamp(d)), -0.55, label, rotation=0,
                va="bottom", ha="center", fontsize=12, color=mc, fontweight="bold")

    ticks = pd.date_range(win[0], win[1], freq="14D")  # labels every two weeks
    ax.set_xticks(mdates.date2num(ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_xlim(season_grid[0], season_grid[-1])
    ax.set_ylim(nrows + ROW_FILL, -0.9)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels([r["stream"] for r in records], fontsize=14)
    for lab in ax.get_xticklabels():
        lab.set(rotation=0, ha="center", fontsize=13)

    groups_present = [g for g in GROUP_COLOR if any(r["group"] == g for r in records)]
    handles = [plt.Line2D([], [], color=GROUP_COLOR[g], marker="s", ls="", ms=12, label=g)
               for g in groups_present]
    ax.legend(handles=handles, fontsize=13, loc="lower right", frameon=True)
    ax.set_title("Part A (exploratory) — full-season breakpoint distribution\n"
                 "where each sensor's stress-direction drops concentrate "
                 "(unit-area KDE of bootstrapped breakpoints)", fontsize=15)

    fdir, tdir = HERE / "outputs" / "figures", HERE / "outputs" / "tables"
    fdir.mkdir(parents=True, exist_ok=True)
    tdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fdir / "breakpoint_distribution.png", dpi=140)
    plt.close(fig)

    # --- tables ------------------------------------------------------------- #
    def reldays(num):
        return (mdates.num2date(num).replace(tzinfo=None) - ref_date).days \
            if not (isinstance(num, float) and np.isnan(num)) else np.nan

    def asdate(num):
        return mdates.num2date(num).strftime("%Y-%m-%d") \
            if not (isinstance(num, float) and np.isnan(num)) else ""

    summ = pd.DataFrame([{
        "stream": r["stream"], "group": r["group"], "stress_direction": r["direction"],
        "n_breakpoints_observed": r["n_obs"],
        "mean_breakpoints_per_replicate": round(r["mean_count"], 3),
        "peak1_date": asdate(r["peaks"][0]) if r["peaks"] else "",
        "peak2_date": asdate(r["peaks"][1]) if len(r["peaks"]) > 1 else "",
        "peak1_rel_days": reldays(r["peaks"][0]) if r["peaks"] else np.nan,
    } for r in records])
    summ.to_csv(tdir / "breakpoint_distribution_summary.csv", index=False)

    draws = pd.DataFrame([
        {"stream": r["stream"], "group": r["group"], "replicate_breakpoint": k,
         "breakpoint_date": asdate(num)}
        for r in records for k, num in enumerate(r["pool_num"])])
    draws.to_csv(tdir / "breakpoint_distribution_draws.csv", index=False)

    print(summ.to_string(index=False, na_rep="—"))
    print(f"\nwrote figure  -> {fdir / 'breakpoint_distribution.png'}")
    print(f"wrote tables  -> {tdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
