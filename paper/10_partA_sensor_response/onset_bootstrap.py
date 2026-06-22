"""
Stage 10 / Part A — per-sensor onset detection & cascade ordering.

For each stream: on which day did it depart from its June baseline, and how uncertain is that
day? Then order the streams to reveal the soil->plant lag and the within-event sequence.

Engine ported from analysis/cascade/changepoint_detection.py (frozen EGU prototype):
PELT (ruptures) + direction-aware onset rule + block-bootstrap CI + caterpillar + grid
robustness (Kendall-tau on the ordering). Adapted to be config-driven and read the canonical
dehar_daily_2025.parquet.

Two rules, run together (see paper/config + the plan):
  first-departure : earliest qualifying change -> HEADLINE caterpillar (shows the ~1-month
                    soil->plant lag, because the search spans June->Sept).
  acute-event     : largest stress jump -> COMPANION (within-event ordering of the Aug collapse).
Each onset is also reported as days relative to first_soil_limit_day (2025-07-10) from stage 00.

Run:
  /home/lk1167/miniconda3/envs/dehar-spac/bin/python paper/10_partA_sensor_response/onset_bootstrap.py
"""
from __future__ import annotations

import itertools
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
from scipy.stats import kendalltau

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "config"))
from paper_common import load_config, load_daily, load_event_windows, load_sensors_meta, hornbeam_qc_cams, greenness_col  # noqa: E402

SEED = 20250801
import os
JUMP = int(os.environ.get("ONSET_JUMP", "1"))        # PELT grid step (1=exact; 2/3=faster for dev)
WORKERS = int(os.environ.get("ONSET_WORKERS", str(min(16, os.cpu_count() or 4))))
GRID_MODELS = ["l2", "l1", "rbf", "normal"]
GRID_PENS = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
GROUP_COLOR = {
    "forcing": "#8c8c8c", "physiology": "#1b7837", "proximal": "#2166ac",
    "flux": "#762a83", "satellite": "#b2182b",
}
MODEL_MARKER = {"l2": "o", "l1": "s", "rbf": "^", "normal": "D"}


@dataclass(frozen=True)
class Stream:
    name: str
    column: str
    direction: str  # 'increase' | 'decrease' | 'auto'
    group: str


# --------------------------------------------------------------------------- #
# Stream set: canonical columns in dehar_daily_2025.parquet (see plan)         #
# --------------------------------------------------------------------------- #
def build_streams_and_frame(cfg: dict):
    """Return (streams, dataframe) with derived ensemble columns added."""
    df = load_daily(cfg)
    meta = load_sensors_meta(cfg)

    # hornbeam vs bird-cherry anglecam ids from sensors_meta
    def cam_ids(species_code):
        ids = []
        for _, r in meta.iterrows():
            if r.get("species_code") == species_code and isinstance(r.get("anglecam_ids"), str):
                ids += [int(x) for x in str(r["anglecam_ids"]).split(";") if x.strip().isdigit()]
        return sorted(set(ids))

    horn, _coh = hornbeam_qc_cams(df, cfg)   # coherence-QC'd hornbeam cams (drops poor-viewpoint)
    cherry = cam_ids("prunus")

    def ens(cols):
        cols = [c for c in cols if c in df.columns]
        return df[cols].mean(axis=1) if cols else pd.Series(np.nan, index=df.index)

    df["leaf_angle_hornbeam_daylight_mean"] = ens(
        [f"leaf_angle_cam{c}_daylight_mean_deg" for c in horn])
    df["leaf_angle_birdcherry_daylight_mean"] = ens(
        [f"leaf_angle_cam{c}_daylight_mean_deg" for c in cherry])
    df["greenness_canonical"] = df[greenness_col(cfg)]   # PhenoCam GCC by default (ADR 0002)

    streams = [
        Stream("VPD (max)", "vpd_max_hPa", "increase", "forcing"),
        Stream("Soil moisture", "sm_mean_pct", "decrease", "forcing"),
        Stream("Predawn SWP", "swp_pd_mean_MPa", "decrease", "physiology"),
        Stream("Tree water deficit", "twd_pd_mean_um", "increase", "physiology"),
        Stream("Sap flow", "sapflow_sum_mean", "decrease", "physiology"),
        Stream("GNSS-T VOD", "vod_predawn_mean", "decrease", "proximal"),
        Stream("Leaf angle (hornbeam)", "leaf_angle_hornbeam_daylight_mean", "auto", "proximal"),
        Stream("Leaf angle (bird cherry)", "leaf_angle_birdcherry_daylight_mean", "auto", "proximal"),
        Stream("PAI hinge (early-warning)", "pai_hinge_hinge_mean_m2m2", "decrease", "proximal"),
        Stream("PAI hemi (bulk)", "pai_hemi_hi_hinge_mean_m2m2", "decrease", "proximal"),
        Stream("GCC (greenness)", "greenness_canonical", "decrease", "proximal"),
        Stream("ET", "et_sum_mm", "decrease", "flux"),
        Stream("GPP", "gpp_mean_umol_m2_s", "decrease", "flux"),
        Stream("S2 NDVI", "s2_ndvi_mean", "decrease", "satellite"),
        Stream("S2 NDII (water)", "s2_ndii_mean", "decrease", "satellite"),
        Stream("S1 cross-ratio", "s1_asc_cr_mean_dB", "auto", "satellite"),
    ]
    return streams, df


# --------------------------------------------------------------------------- #
# Onset engine (ported from changepoint_detection.py)                          #
# --------------------------------------------------------------------------- #
def _sigma_hat(y: np.ndarray) -> float:
    if len(y) < 3:
        return float(np.std(y) or 1.0)
    return float(np.std(np.diff(y)) / np.sqrt(2)) or 1.0


def _changepoints(y, model, pen_scale, min_size):
    n = len(y)
    if n < 2 * min_size + 1:
        return []
    algo = rpt.Pelt(model=model, min_size=min_size, jump=JUMP).fit(y)
    unit = abs(algo.cost.error(0, n)) / n
    pen = pen_scale * max(unit, 1e-9) * np.log(n)
    return algo.predict(pen=pen)[:-1]


def detect_onset(y, direction, model, pen_scale, min_size, rule):
    n = len(y)
    cps = _changepoints(y, model, pen_scale, min_size)
    if not cps:
        return None
    bounds = [0, *cps, n]
    seg_mean = [y[bounds[i]:bounds[i + 1]].mean() for i in range(len(bounds) - 1)]
    thresh = _sigma_hat(y)
    qualifying = []
    for i, cp in enumerate(cps):
        delta = seg_mean[i + 1] - seg_mean[i]
        sd = abs(delta) if direction == "auto" else \
            (delta if direction == "increase" else -delta)
        if sd >= thresh:
            qualifying.append((cp, sd))
    if not qualifying:
        return None
    if rule == "first-departure":
        return qualifying[0][0]
    return max(qualifying, key=lambda t: t[1])[0]


def _block_resample(resid, rng, block):
    n = len(resid)
    out, max_start = [], max(0, n - block)
    while sum(len(b) for b in out) < n:
        s = int(rng.integers(0, max_start + 1))
        out.append(resid[s:s + block])
    return np.concatenate(out)[:n]


def bootstrap_onset(y, direction, model, pen_scale, min_size, n_boot, block, seed, rule):
    onset = detect_onset(y, direction, model, pen_scale, min_size, rule)
    if onset is None:
        return None, None, np.array([])
    rng = np.random.default_rng(seed)
    n = len(y)
    bounds = [0, *_changepoints(y, model, pen_scale, min_size), n]
    fitted = np.empty(n)
    for i in range(len(bounds) - 1):
        fitted[bounds[i]:bounds[i + 1]] = y[bounds[i]:bounds[i + 1]].mean()
    resid = y - fitted
    boots = []
    for _ in range(n_boot):
        o = detect_onset(fitted + _block_resample(resid, rng, block),
                         direction, model, pen_scale, min_size, rule)
        if o is not None:
            boots.append(o)
    boots = np.asarray(boots, dtype=int)
    if len(boots) == 0:
        return onset, None, boots
    point = int(round(np.median(boots)))
    if len(boots) < 0.5 * n_boot:
        return point, None, boots
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, (int(round(lo)), int(round(hi))), boots


def _min_size(n):
    return max(2, min(5, n // 4))


# --------------------------------------------------------------------------- #
# Runs                                                                         #
# --------------------------------------------------------------------------- #
def _onset_task(args):
    """Pool worker: one stream's bootstrapped onset. Module-level for picklability."""
    name, group, y, dates, direction, model, pen_scale, n_boot, block, seed, rule = args
    onset, ci, _ = bootstrap_onset(
        y, direction, model, pen_scale, _min_size(len(y)), n_boot, block, seed, rule)
    od = dates[onset] if onset is not None else pd.NaT
    return {"stream": name, "group": group, "n_obs": len(y), "onset_date": od,
            "ci_lo": dates[ci[0]] if ci else pd.NaT,
            "ci_hi": dates[ci[1]] if ci else pd.NaT}


def run_rule(df, streams, rule, ref_date, model, pen_scale, n_boot, block, seed):
    tasks = []
    for i, st in enumerate(streams):
        if st.column not in df.columns:
            continue
        s = df[st.column].dropna()
        if len(s) < 6:
            continue
        tasks.append((st.name, st.group, s.to_numpy(float), s.index, st.direction,
                      model, pen_scale, n_boot, block, seed + i, rule))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        rows = list(ex.map(_onset_task, tasks))
    print(f"  [{rule}] {len(rows)} streams in {time.time()-t0:.1f}s ({WORKERS} workers)", flush=True)
    out = pd.DataFrame(rows)
    out["onset_rel_days"] = [(d - ref_date).days if pd.notna(d) else np.nan for d in out.onset_date]
    out["ci_days"] = (out.ci_hi - out.ci_lo).dt.days
    return out.sort_values("onset_date", na_position="last").reset_index(drop=True)


def plot_caterpillar(res, path, ref_date, marks, title):
    r = res.dropna(subset=["onset_date"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for label, d, c in marks:
        ax.axvline(mdates.date2num(pd.Timestamp(d)), color=c, ls="--", lw=1, alpha=0.7)
        ax.text(mdates.date2num(pd.Timestamp(d)), len(r) - 0.4, label, rotation=90,
                va="top", ha="right", fontsize=7, color=c)
    for i, row in r.iterrows():
        c = GROUP_COLOR[row.group]
        if pd.notna(row.ci_lo):
            ax.plot([mdates.date2num(row.ci_lo), mdates.date2num(row.ci_hi)], [i, i],
                    color=c, lw=6, alpha=0.35, solid_capstyle="round")
        ax.plot(mdates.date2num(row.onset_date), i, "o", color=c, zorder=3)
        rel = row.onset_rel_days
        ax.text(mdates.date2num(row.onset_date), i + 0.28,
                f"{'+' if rel >= 0 else ''}{int(rel)}d", fontsize=7, color=c, ha="center")
    ax.set_yticks(range(len(r)))
    ax.set_yticklabels(r.stream)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set(xlabel=f"onset date (95% CI); labels = days vs soil-limit onset {ref_date.date()}",
           title=title)
    handles = [plt.Line2D([], [], color=v, marker="o", ls="", label=k)
               for k, v in GROUP_COLOR.items()]
    ax.legend(handles=handles, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _grid_task(args):
    """Pool worker: one stream across the cost-model x penalty grid."""
    name, group, y, dates, direction, rule = args
    rows = []
    for model, pen in itertools.product(GRID_MODELS, GRID_PENS):
        o = detect_onset(y, direction, model, pen, _min_size(len(y)), rule)
        rows.append({
            "stream": name, "group": group, "model": model, "pen_scale": pen,
            "onset_date": dates[o] if o is not None else pd.NaT,
            "onset_doy": dates[o].dayofyear if o is not None else np.nan,
        })
    return rows


def grid_search(df, streams, rule, ref_date):
    tasks = []
    for st in streams:
        if st.column not in df.columns:
            continue
        s = df[st.column].dropna()
        if len(s) < 6:
            continue
        tasks.append((st.name, st.group, s.to_numpy(float), s.index, st.direction, rule))
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        nested = list(ex.map(_grid_task, tasks))
    return pd.DataFrame([r for rows in nested for r in rows])


def stability_summary(grid):
    g = grid.dropna(subset=["onset_doy"])
    per = (g.groupby(["stream", "group"])
             .agg(n_cfg=("onset_doy", "size"), median_doy=("onset_doy", "median"),
                  spread_days=("onset_doy", lambda x: x.max() - x.min()))
             .reset_index().sort_values("median_doy"))
    ref = per.set_index("stream")["median_doy"]
    taus = []
    for _, cfg in g.groupby(["model", "pen_scale"]):
        common = cfg.set_index("stream")["onset_doy"]
        shared = ref.index.intersection(common.index)
        if len(shared) >= 4:
            tau, _ = kendalltau(ref[shared], common[shared])
            taus.append(tau)
    per.attrs["mean_tau"] = float(np.nanmean(taus)) if taus else float("nan")
    per.attrs["min_tau"] = float(np.nanmin(taus)) if taus else float("nan")
    per.attrs["n_cfg_total"] = g.groupby(["model", "pen_scale"]).ngroups
    return per


def plot_grid_stability(grid, summary, path, title):
    order = summary.sort_values("median_doy")["stream"].tolist()
    ypos = {s: i for i, s in enumerate(order)}
    base = pd.Timestamp("2025-01-01")
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    g = grid.dropna(subset=["onset_doy"])
    for _, row in g.iterrows():
        c = GROUP_COLOR[row.group]
        x = mdates.date2num(base + pd.Timedelta(days=row.onset_doy - 1))
        ax.plot(x, ypos[row.stream] + rng.uniform(-0.12, 0.12),
                MODEL_MARKER.get(row.model, "o"), color=c, alpha=0.5, ms=5)
    for _, row in summary.iterrows():
        x = mdates.date2num(base + pd.Timedelta(days=row.median_doy - 1))
        ax.plot(x, ypos[row.stream], "|", color="k", ms=18, mew=2, zorder=4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    tau = summary.attrs.get("mean_tau", float("nan"))
    ax.set(xlabel="onset across grid (markers = cost models; | = median)",
           title=f"{title}\nmean Kendall-tau vs median order = {tau:.2f} "
                 f"({summary.attrs.get('n_cfg_total', 0)} configs)")
    handles = [plt.Line2D([], [], color="k", marker=m, ls="", label=k)
               for k, m in MODEL_MARKER.items()]
    ax.legend(handles=handles, fontsize=8, loc="lower right", title="cost model")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> int:
    cfg = load_config()
    ev = load_event_windows()
    ref_date = ev.loc[ev.stage == "soil_limit", "start"].iloc[0]
    acute_start = ev.loc[ev.stage == "acute", "start"].iloc[0]
    peak = pd.Timestamp("2025-08-20")

    streams, df = build_streams_and_frame(cfg)
    win = [pd.Timestamp(d) for d in cfg["onset"]["search_window"]]
    df = df.loc[win[0]:win[1]]
    n_boot = int(os.environ.get("ONSET_NBOOT", cfg["onset"].get("n_boot", 1000)))

    tdir = HERE / "outputs" / "tables"
    fdir = HERE / "outputs" / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    marks = [("soil-limit", ref_date, "#8c8c8c"), ("acute start", acute_start, "#b2182b"),
             ("Psi min", peak, "#1b7837")]

    rule_titles = {
        "first-departure": "Part A onset — FIRST DEPARTURE (headline): who responds first to the drought",
        "acute-event": "Part A onset — ACUTE EVENT (companion): ordering of the mid-Aug collapse",
    }
    fname = {"first-departure": "first_departure", "acute-event": "acute_event"}

    for rule in ["first-departure", "acute-event"]:
        res = run_rule(df, streams, rule, ref_date, model="l2", pen_scale=2.0,
                       n_boot=n_boot, block=5, seed=SEED)
        res.to_csv(tdir / f"onset_{fname[rule]}.csv", index=False)
        plot_caterpillar(res, fdir / f"onset_caterpillar_{fname[rule]}.png",
                         ref_date, marks, rule_titles[rule])

        grid = grid_search(df, streams, rule, ref_date)
        summ = stability_summary(grid)
        summ.to_csv(tdir / f"onset_grid_stability_{fname[rule]}.csv", index=False)
        plot_grid_stability(grid, summ, fdir / f"onset_grid_robustness_{fname[rule]}.png",
                            rule_titles[rule])

        show = res.copy()
        for c in ("onset_date", "ci_lo", "ci_hi"):
            show[c] = show[c].dt.strftime("%Y-%m-%d")
        print(f"\n=== {rule} ===")
        print(show[["stream", "group", "n_obs", "onset_date", "ci_lo", "ci_hi",
                    "ci_days", "onset_rel_days"]].to_string(index=False, na_rep="—"))
        print(f"ordering robustness: mean Kendall-tau = {summ.attrs['mean_tau']:.2f}, "
              f"min = {summ.attrs['min_tau']:.2f}")

    print(f"\nwrote tables -> {tdir}\nwrote figures -> {fdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
