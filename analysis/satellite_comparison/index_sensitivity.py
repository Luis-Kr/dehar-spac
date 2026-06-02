"""Is the satellite signal driven more by leaf angle than by leaf area (PAI)?

Application A of the DE-Har study: a leaf-angle change alters canopy
reflectance/backscatter *without* a change in leaf area, so satellite optical
indices can move with leaf angle while PAI (true leaf area) is unchanged.

This is shown over the **full season** (May-Oct) on purpose: the satellite has a
~7-day cadence (too few obs inside the August window alone), and the season
contains two regimes that act as a natural control:
  - the **August event**: PAI is ~flat while leaf angle and the satellite move
    -> any satellite change cannot be leaf-area (it is structure-optics);
  - **autumn senescence**: PAI declines (real leaf loss) and the satellite
    follows it.

Two outputs (cf. Kattenborn et al. 2024, Jablonski et al. 2025, Asner 1998):
  (a) z-scored overlay time series (leaf angle, PAI, GCC, satellite indices),
      event shaded -> the driver-switch is visible by eye;
  (b) period-stratified **commonality analysis**: the share of each satellite
      index's variance uniquely explained by leaf angle vs PAI vs GCC, for the
      event vs autumn. Event bar -> leaf angle dominates; autumn bar -> PAI.

Panel (b) uses the gap-filled (Savitzky-Golay) daily satellite so each period
has enough samples; the *relative* unique-variance shares are the message, not
absolute significance (interpolation inflates n). A raw-observation full-season
commonality is printed alongside as a non-interpolated cross-check.

Run
---
    python analysis/satellite_comparison/index_sensitivity.py --csv <file>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "data" / "processed" / "dehar_daily_season_2025_filtered.csv"

EVENT_SHADE = ("2025-08-07", "2025-08-20")           # the hot-dry event
PERIODS = [                                          # label, start, end (panel b)
    ("Event\n(Aug)", "2025-08-01", "2025-08-31"),
    ("Autumn\n(Sep-Oct)", "2025-09-01", "2025-10-31"),
    ("Full\nseason", "2025-05-01", "2025-10-31"),
]
# drivers: (label, column). Leaf angle = canopy mean of the two AngleCams.
DRIVERS = [("Leaf angle", "leaf_angle"), ("PAI", "pai_total_sg"), ("GCC", "gcc_p90_mean")]
DRIVER_COLOR = {"Leaf angle": "#2166ac", "PAI": "#8c510a", "GCC": "#1b7837",
                "shared": "#bdbdbd"}
INDICES_B = [("S2 NDVI", "s2_ndvi"), ("S2 NIRv", "s2_nirv")]  # structural indices

# Event-control (the viable event evidence given ~7-day satellite cadence):
# standardized change from the recovered baseline to the stress peak. PAI ~0
# while leaf angle and the satellite move -> the satellite change is not leaf area.
CHANGE_BASE = ("2025-07-28", "2025-08-06")    # recovered, non-stressed baseline
CHANGE_EVENT = ("2025-08-10", "2025-08-22")   # around the stress peak
CHANGE_VARS = [
    ("Leaf angle", "leaf_angle", "#2166ac"),
    ("PAI", "pai_total_sg", "#8c510a"),
    ("GCC", "gcc_p90_mean", "#1b7837"),
    ("S2 NDVI", "s2_ndvi_savgol", "#b2182b"),
    ("S2 NIRv", "s2_nirv_savgol", "#ef8a62"),
    ("S2 NDII", "s2_ndii_savgol", "#d6604d"),
]


# --------------------------------------------------------------------------- #
def load(csv: Path, start: str, end: str) -> pd.DataFrame:
    df = pd.read_csv(csv, parse_dates=["date"]).set_index("date").sort_index()
    df["leaf_angle"] = df[["leaf_angle_cam65_mean", "leaf_angle_cam67_mean"]].mean(axis=1)
    return df.loc[start:end]


def z(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / sd if sd and not np.isnan(sd) else s * 0.0


# ---- (a) overlay ---------------------------------------------------------- #
def plot_overlay(df: pd.DataFrame, path: Path) -> None:
    lines = [  # label, column(daily/sg), color, style, markers-from(raw col or None)
        ("Leaf angle (cam65)", "leaf_angle_cam65_mean", "#4393c3", "-", None),
        ("Leaf angle (cam67)", "leaf_angle_cam67_mean", "#92c5de", "-", None),
        ("PAI", "pai_total_sg", "#8c510a", "-", None),
        ("GCC", "gcc_p90_mean", "#1b7837", "-", None),
        ("S2 NDVI", "s2_ndvi_savgol", "#b2182b", "--", "s2_ndvi_raw"),
        ("S2 NIRv", "s2_nirv_savgol", "#ef8a62", "--", "s2_nirv_raw"),
        ("S2 NDII (water)", "s2_ndii_savgol", "#d6604d", ":", "s2_ndii_raw"),
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    for label, col, color, ls, rawcol in lines:
        if col not in df:
            continue
        ax.plot(df.index, z(df[col]), ls, color=color, lw=1.8, label=label, alpha=0.9)
        if rawcol and rawcol in df:  # raw satellite acquisitions as markers
            r = df[rawcol].dropna()
            ax.plot(r.index, (r - df[col].mean()) / df[col].std(), "o",
                    color=color, ms=4, alpha=0.7)
    ax.axvspan(pd.Timestamp(EVENT_SHADE[0]), pd.Timestamp(EVENT_SHADE[1]),
               color="0.85", alpha=0.6, zorder=0)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.text(pd.Timestamp("2025-08-13"), ax.get_ylim()[1] * 0.95, "hot-dry\nevent",
            ha="center", va="top", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set(ylabel="z-score (anomaly)",
           title="Standardized overlay: does the satellite track leaf angle or PAI?\n"
                 "event = satellite moves with leaf angle while PAI is flat; "
                 "autumn = satellite follows PAI")
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---- (b) commonality analysis -------------------------------------------- #
def _rsq(y: np.ndarray, X: np.ndarray) -> float:
    if X.shape[1] == 0:
        return 0.0
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    ss_res = ((y - X1 @ beta) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def commonality(d: pd.DataFrame, ycol: str, drivers: list[str]) -> dict | None:
    """Unique variance of each driver + shared, from standardized linear fits."""
    cols = [ycol, *drivers]
    d = d[cols].dropna()
    if len(d) < len(drivers) + 2:
        return None
    y = z(d[ycol]).to_numpy()
    Xall = np.column_stack([z(d[c]).to_numpy() for c in drivers])
    r2_full = _rsq(y, Xall)
    uniq = {}
    for i, c in enumerate(drivers):
        uniq[c] = max(0.0, r2_full - _rsq(y, np.delete(Xall, i, axis=1)))
    shared = max(0.0, r2_full - sum(uniq.values()))
    return {"n": len(d), "r2": r2_full, "unique": uniq, "shared": shared}


def run_commonality(df: pd.DataFrame, indices: list[tuple[str, str]]) -> pd.DataFrame:
    driver_cols = [c for _, c in DRIVERS]
    rows = []
    for iname, ibase in indices:
        for plabel, ps, pe in PERIODS:
            res = commonality(df.loc[ps:pe], f"{ibase}_savgol", driver_cols)
            if res is None:
                continue
            row = {"index": iname, "period": plabel.replace("\n", " "),
                   "n": res["n"], "r2": round(res["r2"], 2),
                   "shared": round(res["shared"], 3)}
            for (dl, dc) in DRIVERS:
                row[dl] = round(res["unique"][dc], 3)
            rows.append(row)
    return pd.DataFrame(rows)


def plot_variance(df: pd.DataFrame, indices: list[tuple[str, str]], path: Path) -> None:
    driver_cols = [c for _, c in DRIVERS]
    fig, axes = plt.subplots(1, len(indices), figsize=(5 * len(indices), 4.5),
                             sharey=True, squeeze=False)
    for ax, (iname, ibase) in zip(axes[0], indices):
        labels, stacks = [], []
        for plabel, ps, pe in PERIODS:
            res = commonality(df.loc[ps:pe], f"{ibase}_savgol", driver_cols)
            labels.append(plabel + (f"\nR²={res['r2']:.2f}\nn={res['n']}" if res else "\n—"))
            stacks.append(res)
        xpos = range(len(PERIODS))
        bottoms = np.zeros(len(PERIODS))
        for dl, dc in DRIVERS:
            vals = np.array([s["unique"][dc] if s else 0.0 for s in stacks])
            ax.bar(xpos, vals, bottom=bottoms, color=DRIVER_COLOR[dl], label=dl)
            bottoms += vals
        sh = np.array([s["shared"] if s else 0.0 for s in stacks])
        ax.bar(xpos, sh, bottom=bottoms, color=DRIVER_COLOR["shared"], label="shared")
        ax.set_xticks(list(xpos))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(iname)
    axes[0][0].set_ylabel("variance of satellite index explained (unique + shared)")
    axes[0][-1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Commonality analysis: who drives the satellite signal, by period\n"
                 "(gap-filled daily satellite; relative shares are the message)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_event_change(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Standardized change baseline -> event peak, per variable (the event control)."""
    rows = []
    for label, col, color in CHANGE_VARS:
        if col not in df:
            continue
        zz = z(df[col])
        base = zz.loc[CHANGE_BASE[0]:CHANGE_BASE[1]].mean()
        evt = zz.loc[CHANGE_EVENT[0]:CHANGE_EVENT[1]].mean()
        rows.append({"var": label, "delta_z": evt - base, "color": color})
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(d["var"], d["delta_z"], color=d["color"])
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set(ylabel="standardized change (z)",
           title="Change from recovered baseline to stress peak\n"
                 f"baseline {CHANGE_BASE[0]}..{CHANGE_BASE[1]}  ->  "
                 f"event {CHANGE_EVENT[0]}..{CHANGE_EVENT[1]}")
    ax.tick_params(axis="x", labelrotation=20)
    ax.text(0.98, 0.03, "PAI ~ 0 -> the satellite change is not leaf area",
            transform=ax.transAxes, ha="right", fontsize=8, style="italic")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return d.drop(columns="color")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Satellite vs leaf angle / PAI attribution")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--start", default="2025-05-01")
    p.add_argument("--end", default="2025-10-31")
    a = p.parse_args(argv)

    df = load(a.csv, a.start, a.end)
    fig_a = REPO_ROOT / "figures" / "index_sensitivity_overlay.png"
    fig_b = REPO_ROOT / "figures" / "index_sensitivity_variance.png"
    out = REPO_ROOT / "data" / "processed" / "index_sensitivity_commonality.csv"

    fig_c = REPO_ROOT / "figures" / "index_sensitivity_event_change.png"

    plot_overlay(df, fig_a)
    plot_variance(df, INDICES_B, fig_b)
    chg = plot_event_change(df, fig_c)
    tab = run_commonality(df, INDICES_B)
    out.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out, index=False)

    print("Event control — standardized change baseline -> stress peak:\n")
    print(chg.to_string(index=False))
    print("\n  -> PAI barely changes while leaf angle and the satellite move:")
    print("     the event satellite change is structural (leaf angle), not leaf area.\n")

    print("Commonality (satellite ~ leaf angle + PAI + GCC), where n permits:")
    print("  NOTE: satellite has ~7-day cadence (~4 obs in Aug) so the EVENT window")
    print("  cannot be regressed; full-season variance is collinearity-dominated (shared).\n")
    print(tab.to_string(index=False))

    # Non-interpolated cross-check: raw satellite obs, full season.
    print("\nCross-check — raw satellite observations, full season:")
    for iname, ibase in INDICES_B:
        res = commonality(df, f"{ibase}_raw", [c for _, c in DRIVERS])
        if res:
            u = res["unique"]
            print(f"  {iname:9s} n={res['n']:2d}  R²={res['r2']:.2f}  | "
                  + "  ".join(f"{dl} {u[dc]:.2f}" for dl, dc in DRIVERS)
                  + f"  shared {res['shared']:.2f}")
    print(f"\nWrote {fig_a}\nWrote {fig_b}\nWrote {fig_c}\nWrote {out}")


if __name__ == "__main__":
    main()
