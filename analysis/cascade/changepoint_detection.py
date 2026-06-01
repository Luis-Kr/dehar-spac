"""Onset detection for the drought-stress cascade at DE-Har (2025).

For each stream we ask: *on which day did it depart from its pre-event state?*
and *how uncertain is that day?* — then rank the streams to reveal the
fast (physiology / proximal) vs. slow (greenness / leaf area) clusters.

Method
------
1. Restrict to the event window (default 2025-07-01 .. 2025-09-15).
2. PELT (``ruptures``, L2 / mean-shift cost) finds change-points.
3. **Direction-aware onset rule**: the onset is the *first* change-point whose
   mean shift goes in the known stress direction and exceeds ~1 sigma. This is
   robust to the August pulse — the September recovery is an opposite-direction
   change-point and is ignored — and to non-responders (no qualifying change
   -> onset = None, reported as "no detected onset").
4. **Block bootstrap** of the residuals around the fitted step model gives a
   95 % CI on the onset date (block resampling preserves day-to-day
   autocorrelation, so the CI stays honest). See ``breakpoint_bootstrap_demo.py``
   in this folder for the same logic on toy data.

Notes on column choices
-----------------------
- Dense in-situ streams: use the raw ``_mean`` (smoothing shifts onsets).
- PAI: use ``_sg`` (raw TLS scans are ~42 % of days); the structural signal is
  slow, so the gap-filled version is appropriate.
- Sentinel-1/-2: use the **raw** per-overpass values (not ``_savgol``) so the
  onset is not given false sub-revisit precision; the ~7-day cadence shows up
  honestly as a wider CI.

Run
---
    python analysis/cascade/changepoint_detection.py \
        --csv data/processed/dehar_daily_season_2025_filtered.csv

Outputs: ``figures/cascade_onset_caterpillar.png`` and
``data/processed/cascade_onsets.csv`` (both git-ignored).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "data" / "processed" / "dehar_daily_season_2025_filtered.csv"
DEFAULT_FIG = REPO_ROOT / "figures" / "cascade_onset_caterpillar.png"
DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "cascade_onsets.csv"
SEED = 20250801


@dataclass(frozen=True)
class Stream:
    """One time series to test. ``direction`` is the stress sign:
    'decrease', 'increase', or 'auto' (first significant change, any sign)."""

    name: str
    column: str
    direction: str
    group: str  # forcing | physiology | proximal | flux | satellite


# Curated streams, ordered by expected cascade position. ``leaf_angle`` is the
# mean of the two AngleCams and is built in load_streams().
STREAMS: list[Stream] = [
    Stream("VPD (max)", "vpd_hpa_max", "increase", "forcing"),
    Stream("Soil moisture", "sm_pct_mean", "decrease", "forcing"),
    Stream("Predawn SWP", "swp_mpa_predawn_mean", "decrease", "physiology"),
    Stream("Tree water deficit", "twd_um_mean", "increase", "physiology"),
    Stream("Sapflow", "sapflow_jscm3cm2d_mean", "decrease", "physiology"),
    Stream("GNSS-T VOD", "vod_mean", "decrease", "proximal"),
    Stream("Leaf angle", "leaf_angle", "increase", "proximal"),
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


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #
def load_streams(csv: Path, start: str, end: str) -> pd.DataFrame:
    """Load the daily file, build ``leaf_angle``, restrict to [start, end]."""
    df = pd.read_csv(csv, parse_dates=["date"]).set_index("date").sort_index()
    df["leaf_angle"] = df[["leaf_angle_cam65_mean", "leaf_angle_cam67_mean"]].mean(axis=1)
    return df.loc[start:end]


# --------------------------------------------------------------------------- #
# Onset detection                                                             #
# --------------------------------------------------------------------------- #
def _sigma_hat(y: np.ndarray) -> float:
    """Robust noise SD from first differences (Var[diff]=2 sigma^2 for noise)."""
    if len(y) < 3:
        return float(np.std(y) or 1.0)
    return float(np.std(np.diff(y)) / np.sqrt(2)) or 1.0


def _changepoints(y: np.ndarray, pen_scale: float, min_size: int) -> list[int]:
    """PELT change-point indices (boundaries), excluding the final endpoint."""
    n = len(y)
    pen = pen_scale * _sigma_hat(y) ** 2 * np.log(n)
    algo = rpt.Pelt(model="l2", min_size=min_size, jump=1).fit(y)
    return algo.predict(pen=pen)[:-1]


def detect_onset(
    y: np.ndarray, direction: str, pen_scale: float, min_size: int,
    rule: str = "acute-event",
) -> int | None:
    """Return the index of the onset change-point, or None if no response.

    A change-point qualifies if its segment-mean shift goes in the stress
    ``direction`` and exceeds ~1 sigma. Among qualifying change-points:
      - ``first-departure``: the earliest one (when the signal first leaves its
        pre-event state — captures a gradual build-up).
      - ``acute-event``: the one with the largest stress-direction jump (the
        dominant shift — centres on the acute event).
    """
    n = len(y)
    cps = _changepoints(y, pen_scale, min_size)
    if not cps:
        return None
    bounds = [0, *cps, n]
    seg_mean = [y[bounds[i]:bounds[i + 1]].mean() for i in range(len(bounds) - 1)]
    thresh = _sigma_hat(y)
    qualifying = []  # (cp, stress_magnitude), cps already ordered in time
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
    y: np.ndarray,
    direction: str,
    pen_scale: float,
    min_size: int,
    n_boot: int,
    block: int,
    seed: int,
    rule: str,
) -> tuple[int | None, tuple[int, int] | None, np.ndarray]:
    """Bootstrap CI for the onset index. Returns (onset, (lo, hi), boot_onsets)."""
    onset = detect_onset(y, direction, pen_scale, min_size, rule)
    if onset is None:
        return None, None, np.array([])
    # Step model = piecewise-constant fit over ALL change-points (keeps recovery).
    rng = np.random.default_rng(seed)
    n = len(y)
    bounds = [0, *_changepoints(y, pen_scale, min_size), n]
    fitted = np.empty(n)
    for i in range(len(bounds) - 1):
        fitted[bounds[i]:bounds[i + 1]] = y[bounds[i]:bounds[i + 1]].mean()
    resid = y - fitted
    boots = []
    for _ in range(n_boot):
        synth = fitted + _block_resample(resid, rng, block)
        o = detect_onset(synth, direction, pen_scale, min_size, rule)
        if o is not None:
            boots.append(o)
    boots = np.asarray(boots, dtype=int)
    if len(boots) == 0:
        return onset, None, boots
    # Point estimate = bootstrap median (guarantees point lies inside the CI and
    # is robust to a fragile single-detection on the raw series).
    point = int(round(np.median(boots)))
    if len(boots) < 0.5 * n_boot:  # mostly "no onset" -> unstable, no CI
        return point, None, boots
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, (int(round(lo)), int(round(hi))), boots


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #
def run(df: pd.DataFrame, pen_scale: float, n_boot: int, block: int, seed: int,
        rule: str) -> pd.DataFrame:
    rows = []
    for i, st in enumerate(STREAMS):
        if st.column not in df.columns:
            continue
        s = df[st.column].dropna()
        y, dates = s.to_numpy(float), s.index
        min_size = max(2, min(5, len(y) // 4))
        onset, ci, _ = bootstrap_onset(
            y, st.direction, pen_scale, min_size, n_boot, block, seed + i, rule
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
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.set(xlabel="onset date (95% CI)",
           title=f"DE-Har 2025 stress-cascade onset ordering\n{subtitle}")
    handles = [plt.Line2D([], [], color=v, marker="o", ls="", label=k)
               for k, v in GROUP_COLOR.items()]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="DE-Har cascade onset detection")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--start", default="2025-07-01")
    p.add_argument("--end", default="2025-09-15")
    p.add_argument("--onset-rule", choices=["first-departure", "acute-event"],
                   default="acute-event",
                   help="first significant departure, or the dominant stress shift")
    p.add_argument("--pen-scale", type=float, default=3.0,
                   help="PELT penalty scale (higher = fewer change-points)")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--block", type=int, default=5, help="bootstrap block length (days)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--fig", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)

    fig = a.fig or REPO_ROOT / "figures" / f"cascade_onset_caterpillar_{a.onset_rule}.png"
    out = a.out or REPO_ROOT / "data" / "processed" / f"cascade_onsets_{a.onset_rule}.csv"

    df = load_streams(a.csv, a.start, a.end)
    res = run(df, a.pen_scale, a.n_boot, a.block, a.seed, a.onset_rule)

    show = res.copy()
    for c in ("onset_date", "ci_lo", "ci_hi"):
        show[c] = show[c].dt.strftime("%Y-%m-%d")
    print(f"Window {a.start} .. {a.end}  |  rule {a.onset_rule}  |  pen {a.pen_scale}\n")
    print(show[["stream", "group", "n_obs", "onset_date", "ci_lo", "ci_hi", "ci_days"]]
          .to_string(index=False, na_rep="—"))

    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    plot_caterpillar(res, fig, subtitle=f"onset rule: {a.onset_rule}")
    print(f"\nWrote {out}\nWrote {fig}")


if __name__ == "__main__":
    main()
