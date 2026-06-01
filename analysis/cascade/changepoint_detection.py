"""Onset detection for the DE-Har hot-and-dry stress event (August 2025).

For each stream we ask: *on which day did it depart from its pre-event state?*
and *how uncertain is that day?* — then rank the streams to reveal the
fast (physiology / proximal) vs. slow (greenness / leaf area) clusters.

Window
------
Default 2025-08-01 .. 2025-09-15. The July dry-down is deliberately excluded:
SWP and sapflow had *recovered* (rain Jul 27-Aug 2, VPD low) before the event.
The hot-and-dry event begins ~Aug 7 (VPD spike, no precip) and ends ~Aug 20
(rain). Aug 1-6 provides a short recovered (non-stressed) baseline so the onset
can be detected; the structural response (PAI) lags into early September.

Method
------
1. PELT (``ruptures``) finds change-points. The penalty is auto-scaled per cost
   model: ``pen = pen_scale * (mean per-sample cost) * log(n)`` so ``pen_scale``
   is comparable across cost functions.
2. **Direction-aware onset rule** (a change-point must shift the segment mean in
   the known stress direction by > ~1 sigma):
     - ``acute-event`` (default): the *largest* stress-direction jump.
     - ``first-departure``: the *earliest* qualifying change-point.
3. **Block bootstrap** of residuals -> 95 % CI on the onset date (median-anchored
   so the point lies inside the CI).

Robustness (``--grid``)
-----------------------
Grid search over cost models {l2, l1, rbf, normal} x penalty scales: re-detects
each onset and checks the *ordering* is stable (Kendall tau vs. the median
ordering). The claim of the paper is the ordering, so this is the key check.

Run
---
    python analysis/cascade/changepoint_detection.py --csv <file>          # single
    python analysis/cascade/changepoint_detection.py --csv <file> --grid   # robustness
"""

from __future__ import annotations

import argparse
import itertools
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

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "data" / "processed" / "dehar_daily_season_2025_filtered.csv"
SEED = 20250801
GRID_MODELS = ["l2", "l1", "rbf", "normal"]
GRID_PENS = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


@dataclass(frozen=True)
class Stream:
    """One time series. ``direction`` is the stress sign:
    'decrease', 'increase', or 'auto' (largest change, any sign)."""

    name: str
    column: str
    direction: str
    group: str  # forcing | physiology | proximal | flux | satellite


def make_streams(leaf_angle: str = "separate") -> list[Stream]:
    """Ordered stream list. ``leaf_angle`` controls the two AngleCams:
    'separate' (default) keeps cam65 and cam67 as distinct streams — they sit at
    different heights and can respond at different speeds; 'mean' averages them.
    """
    leaf = (
        [Stream("Leaf angle (mean)", "leaf_angle", "increase", "proximal")]
        if leaf_angle == "mean"
        else [Stream("Leaf angle cam65", "leaf_angle_cam65_mean", "increase", "proximal"),
              Stream("Leaf angle cam67", "leaf_angle_cam67_mean", "increase", "proximal")]
    )
    return [
        Stream("VPD (max)", "vpd_hpa_max", "increase", "forcing"),
        Stream("Soil moisture", "sm_pct_mean", "decrease", "forcing"),
        Stream("Predawn SWP", "swp_mpa_predawn_mean", "decrease", "physiology"),
        Stream("Tree water deficit", "twd_um_mean", "increase", "physiology"),
        Stream("Sapflow", "sapflow_jscm3cm2d_mean", "decrease", "physiology"),
        Stream("GNSS-T VOD", "vod_mean", "decrease", "proximal"),
        *leaf,
        Stream("PAI", "pai_total_sg", "decrease", "proximal"),
        Stream("GCC", "gcc_p90_mean", "decrease", "proximal"),
        Stream("ET", "et_mmd_mean", "decrease", "flux"),
        Stream("GPP", "gpp_umolm2s_mean", "decrease", "flux"),
        Stream("S2 NDVI", "s2_ndvi_raw", "decrease", "satellite"),
        Stream("S2 NDII (water)", "s2_ndii_raw", "decrease", "satellite"),
        Stream("S1 cross-ratio", "s1_cr_raw", "auto", "satellite"),
    ]

GROUP_COLOR = {
    "forcing": "#8c8c8c", "physiology": "#1b7837", "proximal": "#2166ac",
    "flux": "#762a83", "satellite": "#b2182b",
}
MODEL_MARKER = {"l2": "o", "l1": "s", "rbf": "^", "normal": "D"}


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #
def load_streams(csv: Path, start: str, end: str) -> pd.DataFrame:
    df = pd.read_csv(csv, parse_dates=["date"]).set_index("date").sort_index()
    df["leaf_angle"] = df[["leaf_angle_cam65_mean", "leaf_angle_cam67_mean"]].mean(axis=1)
    return df.loc[start:end]


# --------------------------------------------------------------------------- #
# Onset detection                                                             #
# --------------------------------------------------------------------------- #
def _sigma_hat(y: np.ndarray) -> float:
    """Robust noise SD from first differences (min-jump threshold)."""
    if len(y) < 3:
        return float(np.std(y) or 1.0)
    return float(np.std(np.diff(y)) / np.sqrt(2)) or 1.0


def _changepoints(y: np.ndarray, model: str, pen_scale: float, min_size: int) -> list[int]:
    """PELT change-point indices, with a penalty auto-scaled to the cost model."""
    n = len(y)
    if n < 2 * min_size + 1:
        return []
    algo = rpt.Pelt(model=model, min_size=min_size, jump=1).fit(y)
    unit = abs(algo.cost.error(0, n)) / n  # mean per-sample cost (model-specific)
    pen = pen_scale * max(unit, 1e-9) * np.log(n)
    return algo.predict(pen=pen)[:-1]


def detect_onset(
    y: np.ndarray, direction: str, model: str, pen_scale: float,
    min_size: int, rule: str = "acute-event",
) -> int | None:
    """Index of the onset change-point, or None if no qualifying response."""
    n = len(y)
    cps = _changepoints(y, model, pen_scale, min_size)
    if not cps:
        return None
    bounds = [0, *cps, n]
    seg_mean = [y[bounds[i]:bounds[i + 1]].mean() for i in range(len(bounds) - 1)]
    thresh = _sigma_hat(y)
    qualifying = []  # (cp, stress_magnitude), ordered in time
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
    return max(qualifying, key=lambda t: t[1])[0]  # acute-event: biggest jump


def _block_resample(resid: np.ndarray, rng: np.random.Generator, block: int) -> np.ndarray:
    n = len(resid)
    out, max_start = [], max(0, n - block)
    while sum(len(b) for b in out) < n:
        s = int(rng.integers(0, max_start + 1))
        out.append(resid[s:s + block])
    return np.concatenate(out)[:n]


def bootstrap_onset(
    y: np.ndarray, direction: str, model: str, pen_scale: float, min_size: int,
    n_boot: int, block: int, seed: int, rule: str,
) -> tuple[int | None, tuple[int, int] | None, np.ndarray]:
    """Bootstrap CI for the onset index. Returns (onset, (lo, hi), boots)."""
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


def _min_size(n: int) -> int:
    return max(2, min(5, n // 4))


# --------------------------------------------------------------------------- #
# Single run (with bootstrap CI)                                              #
# --------------------------------------------------------------------------- #
def run(df: pd.DataFrame, streams: list[Stream], model: str, pen_scale: float,
        n_boot: int, block: int, seed: int, rule: str) -> pd.DataFrame:
    rows = []
    for i, st in enumerate(streams):
        if st.column not in df.columns:
            continue
        s = df[st.column].dropna()
        y, dates = s.to_numpy(float), s.index
        onset, ci, _ = bootstrap_onset(
            y, st.direction, model, pen_scale, _min_size(len(y)),
            n_boot, block, seed + i, rule,
        )
        rows.append({
            "stream": st.name, "group": st.group, "n_obs": len(y),
            "onset_date": dates[onset] if onset is not None else pd.NaT,
            "ci_lo": dates[ci[0]] if ci else pd.NaT,
            "ci_hi": dates[ci[1]] if ci else pd.NaT,
        })
    out = pd.DataFrame(rows)
    out["ci_days"] = (out.ci_hi - out.ci_lo).dt.days
    return out.sort_values("onset_date", na_position="last").reset_index(drop=True)


def plot_caterpillar(res: pd.DataFrame, path: Path, subtitle: str = "") -> None:
    r = res.dropna(subset=["onset_date"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, row in r.iterrows():
        c = GROUP_COLOR[row.group]
        if pd.notna(row.ci_lo):
            ax.plot([mdates.date2num(row.ci_lo), mdates.date2num(row.ci_hi)],
                    [i, i], color=c, lw=6, alpha=0.35, solid_capstyle="round")
        ax.plot(mdates.date2num(row.onset_date), i, "o", color=c, zorder=3)
    ax.set_yticks(range(len(r)))
    ax.set_yticklabels(r.stream)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set(xlabel="onset date (95% CI)",
           title=f"DE-Har 2025 stress-cascade onset ordering\n{subtitle}")
    handles = [plt.Line2D([], [], color=v, marker="o", ls="", label=k)
               for k, v in GROUP_COLOR.items()]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Grid search (penalty x cost function) for ordering robustness               #
# --------------------------------------------------------------------------- #
def grid_search(df: pd.DataFrame, streams: list[Stream], rule: str,
                models: list[str], pens: list[float]) -> pd.DataFrame:
    rows = []
    for st in streams:
        if st.column not in df.columns:
            continue
        s = df[st.column].dropna()
        y, dates = s.to_numpy(float), s.index
        for model, pen in itertools.product(models, pens):
            o = detect_onset(y, st.direction, model, pen, _min_size(len(y)), rule)
            rows.append({
                "stream": st.name, "group": st.group, "model": model,
                "pen_scale": pen,
                "onset_date": dates[o] if o is not None else pd.NaT,
                "onset_doy": dates[o].dayofyear if o is not None else np.nan,
            })
    return pd.DataFrame(rows)


def stability_summary(grid: pd.DataFrame) -> pd.DataFrame:
    """Per-stream onset spread + rank stability of the ordering across the grid."""
    g = grid.dropna(subset=["onset_doy"])
    per = (g.groupby(["stream", "group"])
             .agg(n_cfg=("onset_doy", "size"),
                  median_doy=("onset_doy", "median"),
                  spread_days=("onset_doy", lambda x: x.max() - x.min()))
             .reset_index().sort_values("median_doy"))
    ref = per.set_index("stream")["median_doy"]  # reference ordering

    taus = []
    for (_m, _p), cfg in g.groupby(["model", "pen_scale"]):
        common = cfg.set_index("stream")["onset_doy"]
        shared = ref.index.intersection(common.index)
        if len(shared) >= 4:
            tau, _ = kendalltau(ref[shared], common[shared])
            taus.append(tau)
    per.attrs["mean_tau"] = float(np.nanmean(taus)) if taus else float("nan")
    per.attrs["min_tau"] = float(np.nanmin(taus)) if taus else float("nan")
    per.attrs["n_cfg_total"] = g.groupby(["model", "pen_scale"]).ngroups
    return per


def plot_grid_stability(grid: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    order = summary.sort_values("median_doy")["stream"].tolist()
    ypos = {s: i for i, s in enumerate(order)}
    base = pd.Timestamp("2025-01-01")
    fig, ax = plt.subplots(figsize=(9, 5))
    g = grid.dropna(subset=["onset_doy"])
    for _, row in g.iterrows():
        c = GROUP_COLOR[row.group]
        x = mdates.date2num(base + pd.Timedelta(days=row.onset_doy - 1))
        ax.plot(x, ypos[row.stream] + np.random.uniform(-0.12, 0.12),
                MODEL_MARKER.get(row.model, "o"), color=c, alpha=0.5, ms=5)
    for _, row in summary.iterrows():
        x = mdates.date2num(base + pd.Timedelta(days=row.median_doy - 1))
        ax.plot(x, ypos[row.stream], "|", color="k", ms=18, mew=2, zorder=4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    tau = summary.attrs.get("mean_tau", float("nan"))
    ax.set(xlabel="onset date across grid (markers = cost models; | = median)",
           title=f"Onset-ordering robustness across penalty x cost grid\n"
                 f"mean Kendall tau vs median order = {tau:.2f} "
                 f"({summary.attrs.get('n_cfg_total', 0)} configs)")
    handles = [plt.Line2D([], [], color="k", marker=m, ls="", label=k)
               for k, m in MODEL_MARKER.items()]
    ax.legend(handles=handles, fontsize=8, loc="lower right", title="cost model")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="DE-Har cascade onset detection")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--start", default="2025-07-20")
    p.add_argument("--end", default="2025-09-15")
    p.add_argument("--onset-rule", choices=["first-departure", "acute-event"],
                   default="acute-event")
    p.add_argument("--leaf-angle", choices=["separate", "mean"], default="separate",
                   help="keep the two AngleCams separate (default) or average them")
    p.add_argument("--model", default="rbf", help="ruptures cost model (single run)")
    p.add_argument("--pen-scale", type=float, default=1.0)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--block", type=int, default=5)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--grid", action="store_true",
                   help="run the penalty x cost-function robustness grid")
    a = p.parse_args(argv)

    df = load_streams(a.csv, a.start, a.end)
    streams = make_streams(a.leaf_angle)
    print(f"Window {a.start} .. {a.end}  |  rule {a.onset_rule}  |  leaf-angle {a.leaf_angle}")

    if a.grid:
        grid = grid_search(df, streams, a.onset_rule, GRID_MODELS, GRID_PENS)
        summ = stability_summary(grid)
        fig = REPO_ROOT / "figures" / f"cascade_onset_grid_{a.onset_rule}.png"
        out = REPO_ROOT / "data" / "processed" / f"cascade_onset_grid_{a.onset_rule}.csv"
        show = summ.copy()
        show["median_date"] = (pd.Timestamp("2025-01-01")
                               + pd.to_timedelta(show.median_doy - 1, "D")).dt.strftime("%b %d")
        print(f"\nGrid: {GRID_MODELS} x pen {GRID_PENS}  "
              f"({summ.attrs['n_cfg_total']} configs)\n")
        print(show[["stream", "group", "n_cfg", "median_date", "spread_days"]]
              .to_string(index=False))
        print(f"\nOrdering stability: mean Kendall tau = {summ.attrs['mean_tau']:.2f}, "
              f"min = {summ.attrs['min_tau']:.2f}  (1.0 = identical order)")
        out.parent.mkdir(parents=True, exist_ok=True)
        grid.to_csv(out, index=False)
        plot_grid_stability(grid, summ, fig)
        print(f"\nWrote {out}\nWrote {fig}")
        return

    res = run(df, streams, a.model, a.pen_scale, a.n_boot, a.block, a.seed, a.onset_rule)
    show = res.copy()
    for c in ("onset_date", "ci_lo", "ci_hi"):
        show[c] = show[c].dt.strftime("%Y-%m-%d")
    print(f"model {a.model} | pen {a.pen_scale}\n")
    print(show[["stream", "group", "n_obs", "onset_date", "ci_lo", "ci_hi", "ci_days"]]
          .to_string(index=False, na_rep="—"))
    fig = REPO_ROOT / "figures" / f"cascade_onset_caterpillar_{a.onset_rule}.png"
    out = REPO_ROOT / "data" / "processed" / f"cascade_onsets_{a.onset_rule}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    plot_caterpillar(res, fig, subtitle=f"{a.onset_rule} | {a.model} | pen {a.pen_scale}")
    print(f"\nWrote {out}\nWrote {fig}")


if __name__ == "__main__":
    main()
