#!/usr/bin/env python3
"""2026 LEAF hemispherical seasonal series (off-pipeline double-check).

A 2026 take on
``paper/40_supplementary/leaf_hemispherical/fig_hemi_seasonal_series.py``, but:

* **2026 data**, one station at a time (``--station Har_01|Har_02|all``), read
  from the gsdata raw tree, with the **rotation** tilt fix (ADR 0004) — the same
  re-levelling that rescues Har_02's strongly-tilted scans.
* **Two rows only** (the third boolean-gap row is dropped):
    row 1 - canopy height        (per-return scatter, colour = height)
    row 2 - gap fraction, binned (az/zen cells, continuous P_gap, log scale)
* **4 representative clean scenes** spread across the season. 2026 has no
  processed meteorology, so rain/fog/wind-disturbed scans are **inferred as
  outliers**: a robust LOWESS trend is fit to the night-scan total-PAI series and
  scans deviating by more than ``--k-mad`` robust SDs are flagged and excluded;
  the 4 scenes are then picked from the clean nights, evenly spread in time.

Self-contained: the night-scan total-PAI series is (re)computed here with the
rotation geometry (so selection and the header PAI both match the rendering),
parallelised across cores. Reuses the path-independent fisheye helpers from the
paper module (``fisheye_grid``, ``style_fisheye``); geometry/inversion come from
the canonical ``dehar.proximal_rs.leaf`` so the rotation matches the pipeline.

Run
---
    /home/lk1167/miniconda3/envs/dehar-spac/bin/python \
        other/fig_hemi_seasonal_2026.py --station all

Outputs (under ``other/``)
    hemi_seasonal_2026_<station>.png         the 2-row fisheye series
    hemi_seasonal_2026_<station>_scenes.csv  the picked scenes + outlier diagnostics
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import LogNorm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
HEMI_DIR = ROOT / "paper" / "40_supplementary" / "leaf_hemispherical"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "paper" / "config"))
sys.path.insert(0, str(HEMI_DIR))

import _leaf_hemi_common as hemi  # noqa: E402
from dehar.proximal_rs.leaf import (  # noqa: E402
    get_scan_datetime,
    process_single_scan,
    read_leaf_scan,
)
from paper_common import paper_style  # noqa: E402

RAW_BASE = Path("/mnt/gsdata/projects/icos_har/strucnet/data/raw")
STATIONS = ("Har_01", "Har_02")
CONFIG = ROOT / "config" / "leaf_processing.yaml"
HEMI_GLOB = "ESS?????_*_hemi_*_0800_0400.csv"        # hemi-high scans
ROTATION_CFG = {"enabled": True, "method": "leaf_tilt_rotation"}  # ADR 0004
SENSOR_HEIGHT = 1.5

# --- rendering (mirrors the paper figure) ----------------------------------
POINT_SIZE = 0.7
POINT_ALPHA = 0.85
HEIGHT_CMAP = "turbo"
HEIGHT_VLIM = (0.0, 20.0)
GAP_ABIN = 3.0
GAP_ZBIN = 3.0
GAP_ZMAX = 88.0
GAP_MIN_COUNT = 20
REALGAP_CMAP = "bone"
REALGAP_VLIM = (1e-3, 1.0)
FS_COL_HEADER = 18
FS_ROW_LABEL = 18
FS_RING_TICK = 11
FS_AZ_TICK = 14
FS_CBAR_LABEL = 16
FS_CBAR_TICK = 14
FIG_W_PER_COL = 4.0
FIG_H_PER_ROW = 4.0
DPI = 200


def _rotation_pai(filepath: Path, profile: dict, instrument: dict) -> dict | None:
    """Worker: total canopy WeightedPAI for one scan under the rotation geometry."""
    df = process_single_scan(
        filepath, profile=profile, instrument=instrument,
        transform=ROTATION_CFG, up_deg=None,
    )
    if df is None:
        return None
    dt = get_scan_datetime(filepath.name)
    return {
        "filename": filepath.name,
        "datetime": pd.Timestamp(dt, tz="UTC"),
        "scan_hour": dt.hour,
        "weighted_pai": float(df["WeightedPAI"].max()),
    }


def night_pai_table(station: str, hour: int, cfg: dict, n_workers: int) -> pd.DataFrame:
    """Rotation total-PAI for every hemi-high night scan of a station's 2026 season."""
    raw_dir = RAW_BASE / station / "2026"
    files = [
        f for f in sorted(raw_dir.glob(HEMI_GLOB))
        if (dt := get_scan_datetime(f.name)) is not None and dt.hour == hour
    ]
    if not files:  # fall back to all hemi-high scans if that hour is absent
        files = sorted(raw_dir.glob(HEMI_GLOB))
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {
            ex.submit(_rotation_pai, f, cfg["profile"], cfg["instrument"]): f
            for f in files
        }
        for fut in as_completed(futs):
            r = fut.result()
            if r is not None:
                rows.append(r)
    return pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)


def flag_outliers(t: pd.DataFrame, k_mad: float, frac: float = 0.3) -> pd.DataFrame:
    """Flag rain/fog/wind-disturbed scans as robust outliers of the PAI trend.

    Fits a robust LOWESS to the total-PAI season (the smooth leaf-out trajectory),
    then flags scans whose residual exceeds ``k_mad`` robust SDs (1.4826*MAD) from
    the residual median. Adds columns ``pai_smooth``, ``resid``, ``z_robust`` and
    boolean ``outlier``.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess

    s = t.sort_values("datetime").copy()
    x = (s["datetime"] - s["datetime"].min()).dt.total_seconds().to_numpy() / 86400.0
    y = s["weighted_pai"].to_numpy()
    s["pai_smooth"] = lowess(y, x, frac=frac, it=3, xvals=x)
    s["resid"] = s["weighted_pai"] - s["pai_smooth"]
    med = float(np.median(s["resid"]))
    mad = float(np.median(np.abs(s["resid"] - med)))
    scale = 1.4826 * mad if mad > 0 else float(np.std(s["resid"]) or 1.0)
    s["z_robust"] = (s["resid"] - med) / scale
    s["outlier"] = s["z_robust"].abs() > k_mad
    return s


def pick_scenes(t: pd.DataFrame, n: int) -> pd.DataFrame:
    """Pick ``n`` clean scenes evenly spread across the season (by date)."""
    clean = t[~t["outlier"]].sort_values("datetime").reset_index(drop=True)
    if len(clean) < n:
        raise RuntimeError(f"only {len(clean)} clean scans, need {n}")
    t0, t1 = clean["datetime"].iloc[0], clean["datetime"].iloc[-1]
    span = (t1 - t0).total_seconds()
    targets = [t0 + pd.Timedelta(seconds=f * span) for f in np.linspace(0.05, 0.95, n)]
    picks, used = [], set()
    for tgt in targets:
        order = (clean["datetime"] - tgt).abs().sort_values().index
        choice = next((i for i in order if i not in used), order[0])
        used.add(choice)
        picks.append(clean.loc[choice])
    return pd.DataFrame(picks).sort_values("datetime").reset_index(drop=True)


def scene_arrays(raw_path: Path):
    """Height scatter and binned-gap grid for one scan (rotation geometry)."""
    leaf = read_leaf_scan(
        str(raw_path), sensor_height=SENSOR_HEIGHT, transform_cfg=ROTATION_CFG
    )
    d = leaf.data
    az = np.degrees(d["azimuth"].to_numpy())
    zen = np.degrees(d["zenith"].to_numpy())
    h1 = d["h1"].to_numpy()
    h2 = d["h2"].to_numpy()
    gap = (d["range1"].isna() & d["range2"].isna()).to_numpy().astype(float)

    az_h = np.radians(np.concatenate([az, az]))
    zen_h = np.concatenate([zen, zen])
    h = np.concatenate([h1, h2])
    ok = np.isfinite(h)
    height = (az_h[ok], zen_h[ok], h[ok])
    binned = hemi.fisheye_grid(
        az, zen, gap, "mean", GAP_ABIN, GAP_ZBIN, GAP_ZMAX, GAP_MIN_COUNT
    )
    return height, binned


def _cmap(name: str):
    c = plt.get_cmap(name).copy()
    c.set_bad("0.85")
    return c


def render(scenes: pd.DataFrame, station: str, out_png: Path) -> None:
    """Two-row fisheye seasonal series (canopy height + binned gap fraction)."""
    paper_style()
    ncol = len(scenes)
    fig, axes = plt.subplots(
        2, ncol, figsize=(FIG_W_PER_COL * ncol, FIG_H_PER_ROW * 2),
        subplot_kw={"projection": "polar"}, layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.01, hspace=0.12)

    m_h = m_b = None
    for j, (_, row) in enumerate(scenes.iterrows()):
        raw_path = RAW_BASE / station / "2026" / row["filename"]
        (azh, zenh, h), (th, r, ggrid) = scene_arrays(raw_path)
        m_h = axes[0, j].scatter(
            azh, zenh, c=np.clip(h, *HEIGHT_VLIM), s=POINT_SIZE, cmap=HEIGHT_CMAP,
            vmin=HEIGHT_VLIM[0], vmax=HEIGHT_VLIM[1], alpha=POINT_ALPHA,
            rasterized=True,
        )
        m_b = axes[1, j].pcolormesh(
            th, r, np.ma.clip(ggrid, *REALGAP_VLIM), cmap=_cmap(REALGAP_CMAP),
            norm=LogNorm(*REALGAP_VLIM), shading="flat",
        )
        for i in range(2):
            hemi.style_fisheye(
                axes[i, j], zen_label_fs=FS_RING_TICK, az_label_fs=FS_AZ_TICK
            )
        date = row["datetime"].strftime("%b %d")
        axes[0, j].set_title(
            f"{date}\nPAI = {row['weighted_pai']:.2f}", fontsize=FS_COL_HEADER, pad=18
        )

    for row_i, label in enumerate(["canopy height", "gap fraction\n(binned)"]):
        axes[row_i, 0].text(
            -0.32, 0.5, label, transform=axes[row_i, 0].transAxes,
            rotation=90, va="center", ha="center", fontsize=FS_ROW_LABEL,
        )

    cb_h = fig.colorbar(
        m_h, ax=axes[0, :].tolist(), location="right", shrink=0.8, pad=0.01
    )
    cb_h.set_label("height (m)", fontsize=FS_CBAR_LABEL)
    cb_h.ax.tick_params(labelsize=FS_CBAR_TICK)
    cb_b = fig.colorbar(
        m_b, ax=axes[1, :].tolist(), location="right", shrink=0.8, pad=0.01
    )
    cb_b.set_label("$P_{gap}$", fontsize=FS_CBAR_LABEL)
    cb_b.ax.tick_params(labelsize=FS_CBAR_TICK)

    fig.suptitle(
        f"DE-Har {station} LEAF hemispherical, 2026 — rotation tilt fix (ADR 0004), "
        "clean night scenes (outlier-screened)",
        fontsize=16,
    )
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png.name}")


def run_station(station: str, hour: int, n_scenes: int, k_mad: float,
                cfg: dict, n_workers: int) -> None:
    """Select clean night scenes for one station and render the series."""
    t = night_pai_table(station, hour, cfg, n_workers)
    flagged = flag_outliers(t, k_mad)
    n_out = int(flagged["outlier"].sum())
    print(f"\n=== {station}: {len(flagged)} night scans (rotation), {n_out} flagged "
          f"as outliers (|z|>{k_mad}) ===")
    scenes = pick_scenes(flagged, n_scenes)

    show = flagged.assign(picked=flagged["datetime"].isin(scenes["datetime"]))
    cols = ["datetime", "filename", "weighted_pai", "pai_smooth", "z_robust",
            "outlier", "picked"]
    print(show[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    scenes_out = HERE / f"hemi_seasonal_2026_{station}_scenes.csv"
    scenes.to_csv(scenes_out, index=False)
    print(f"wrote {scenes_out.name} ({len(scenes)} scenes)")
    render(scenes, station, HERE / f"hemi_seasonal_2026_{station}.png")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--station", default="all",
                   help="Har_01 | Har_02 | all (default: all)")
    p.add_argument("--hour", type=int, default=20,
                   help="Preferred scan hour UTC for the night population (default 20)")
    p.add_argument("--n-scenes", type=int, default=4, help="Scenes per station")
    p.add_argument("--k-mad", type=float, default=3.0,
                   help="Robust-SD threshold for outlier (rain/fog/wind) flagging")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = yaml.safe_load(CONFIG.read_text())
    n_workers = max(1, (os.cpu_count() or 8) - 4)
    stations = STATIONS if args.station == "all" else (args.station,)
    for station in stations:
        if station not in STATIONS:
            raise SystemExit(f"unknown --station {station!r}")
        run_station(station, args.hour, args.n_scenes, args.k_mad, cfg, n_workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
