#!/usr/bin/env python3
"""Double-check: process DE-Har **2026** LEAF scans and plot the PAI time series.

Standalone sibling of ``scripts/process_leaf.py``, deliberately kept OUTSIDE the
canonical pipeline (under ``other/``) because it differs in three ways:

1. **Raw tree** — it reads the 2026 season on gsdata for one LEAF station
   (``/mnt/gsdata/projects/icos_har/strucnet/data/raw/<station>/2026``), not the
   committed 2025 ``data/raw/`` tree. ``--station`` picks ``Har_01`` or ``Har_02``.
2. **Geometry** — ``--tilt`` selects the tilt correction. ``offset`` is the
   **standard pylidar correction** (the upstream scalar offset,
   ``LeafScanFile(transform=True)``); ``rotation`` is the DE-Har per-beam
   re-levelling (ADR 0004). Neither uses the up-drift re-fold (ADR 0005).
   The scalar offset adds the instrument's full lean to *every* beam, so at a
   strongly-tilted station it pushes the dedicated hinge ring off 57.5 deg and the
   hinge inversion saturates — this is why **Har_02** (~5.5 deg lean) needs
   ``rotation`` while **Har_01** (~1.1 deg) is fine on the standard ``offset``.
3. **No met context** — 2026 meteorology is not in the processed pipeline yet, so
   the per-scan quality flags are skipped. The plot uses a robust daily median.

Purpose: confirm the 2025 finding (hinge total PAI < hemi total PAI = the
early-warning/bulk bracket, plus the spring leaf-out rise) reproduces in 2026.

Everything method-bound (profile inversion grid, instrument geometry, scan-type
globs, the per-type total-PAI metric) is loaded from the canonical
``config/leaf_processing.yaml`` so this script never silently diverges from the
2025 contract; only the items above are overridden here, explicitly.

Run
---
    # Har_01 — standard pylidar tilt (the requested correction):
    /home/lk1167/miniconda3/envs/dehar-spac/bin/python other/leaf_2026_doublecheck.py
    # Har_02 — primary uses rotation (standard offset saturates its hinge):
    .../python other/leaf_2026_doublecheck.py --station Har_02 --tilt rotation
    # Har_02 — standard-correction record (kept for comparison):
    .../python other/leaf_2026_doublecheck.py --station Har_02 --tag stdtilt  # offset

Outputs (all under ``other/``; ``<tag>`` = ``_<tag>`` when ``--tag`` is given)
    leaf_hemi_hi_<station>_2026<tag>.parquet  long-format profile (scan x height)
    leaf_hinge_<station>_2026<tag>.parquet    long-format profile
    leaf_pai_daily_<station>_2026<tag>.csv     tidy daily total-PAI summary (plotted)
    pai_timeseries_<station>_2026<tag>.png     the PAI time-series figure
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "paper" / "config"))

from dehar.proximal_rs.leaf import process_single_scan  # noqa: E402

OUT_DIR = REPO_ROOT / "other"
CONFIG = REPO_ROOT / "config" / "leaf_processing.yaml"
RAW_BASE = Path("/mnt/gsdata/projects/icos_har/strucnet/data/raw")
STATIONS = ("Har_01", "Har_02")
TILT_TITLE = {
    "offset": "standard pylidar tilt correction",
    "rotation": "per-beam rotation tilt correction (ADR 0004)",
}

log = logging.getLogger("leaf_2026")


def discover(raw_dir: Path, glob: str) -> list[Path]:
    """Sorted scan files for one scan type under the 2026 raw tree."""
    return sorted(raw_dir.glob(glob))


def process_type(
    scan_type: str, files: list[Path], cfg: dict, transform: dict, n_workers: int
) -> pd.DataFrame:
    """Invert every scan of one type -> long-format profile table."""
    log.info("%s: inverting %d scans (%d workers)", scan_type, len(files), n_workers)
    profiles: list[pd.DataFrame] = []
    n_ok = n_fail = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {
            ex.submit(
                process_single_scan,
                f,
                profile=cfg["profile"],
                instrument=cfg["instrument"],
                transform=transform,
                up_deg=None,
            ): f
            for f in files
        }
        for i, fut in enumerate(as_completed(futures), start=1):
            f = futures[fut]
            try:
                df = fut.result()
            except Exception as exc:  # noqa: BLE001 - log & continue
                log.error("  failed %s: %s", f.name, exc)
                n_fail += 1
            else:
                if df is None:
                    n_fail += 1
                else:
                    profiles.append(df)
                    n_ok += 1
            if i % 100 == 0:
                rate = i / (time.time() - t0)
                eta = (len(files) - i) / rate if rate else 0
                log.info("  %d/%d (%d ok, %d fail) ~%.0fs left", i, len(files),
                         n_ok, n_fail, eta)

    if not profiles:
        raise RuntimeError(f"{scan_type}: no scans produced a profile")
    df_all = pd.concat(profiles, ignore_index=True)
    df_all["datetime"] = pd.to_datetime(df_all["datetime"])
    df_all = df_all.sort_values(["datetime", "height"]).reset_index(drop=True)
    log.info("%s done: %d scans (%d failed), %d rows, %.0fs", scan_type,
             df_all["datetime"].nunique(), n_fail, len(df_all), time.time() - t0)
    return df_all


def total_pai_per_scan(df: pd.DataFrame, metric: str) -> pd.Series:
    """Total canopy PAI per scan = cumulative profile maximum over height."""
    return df.groupby("datetime")[metric].max()


def daily_summary(per_scan: dict[str, pd.Series]) -> pd.DataFrame:
    """Robust daily median (and scan count) of each per-scan total-PAI series."""
    frames = []
    for name, s in per_scan.items():
        d = s.rename("pai").to_frame()
        d["date"] = pd.to_datetime(d.index).normalize()
        g = d.groupby("date")["pai"]
        frames.append(
            pd.DataFrame({f"{name}_median": g.median(), f"{name}_n": g.size()})
        )
    out = pd.concat(frames, axis=1).sort_index()
    out.index.name = "date"
    return out


def make_plot(per_scan: dict[str, pd.Series], daily: pd.DataFrame, out_png: Path,
              station: str, tilt: str) -> None:
    """One PAI time-series figure: raw scans (scatter) + robust daily median."""
    import matplotlib.pyplot as plt
    from paper_common import paper_style

    paper_style()
    # Two headline series (the 2025 comparison): hemi-high bulk total vs hinge
    # total. The hemi HingePAI (ring-matched) is omitted from the plot because it
    # equals hemi WeightedPAI (the solid-angle total is anchored to the hinge
    # angle); it is still written to the daily CSV.
    palette = {
        "hemi_hi_weighted": ("#2e7d32", "hemi-high  (bulk, WeightedPAI)"),
        "hinge_hinge": ("#e07b00", "hinge  (early-warning, HingePAI)"),
    }
    fig, ax = plt.subplots(figsize=(11, 6))
    for key, (color, label) in palette.items():
        if key not in per_scan:
            continue
        s = per_scan[key]
        ax.scatter(s.index, s.values, s=9, color=color, alpha=0.18, linewidths=0)
        med = daily[f"{key}_median"].dropna()
        ax.plot(med.index, med.values, "-", color=color, label=label, zorder=5)

    ax.set_ylabel("Total canopy PAI  (m$^2$ m$^{-2}$)")
    ax.set_xlabel("2026")
    ax.set_title(
        f"DE-Har {station} LEAF PAI, 2026 so far  —  {TILT_TITLE[tilt]}\n"
        "double-check of the 2025 hinge<hemi bracket + spring leaf-out",
        fontsize=11,
    )
    ax.legend(frameon=False, loc="upper left")
    ax.margins(x=0.01)
    fig.autofmt_xdate()
    fig.savefig(out_png)
    plt.close(fig)
    log.info("wrote %s", out_png.name)


def run_station(station: str, cfg: dict, transform: dict, tilt: str,
                tag: str, n_workers: int) -> None:
    """Process one LEAF station's 2026 season and write its (tagged) outputs."""
    raw_dir = RAW_BASE / station / "2026"
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"{station}: raw tree not found: {raw_dir}")
    suffix = f"_{tag}" if tag else ""
    log.info("=== %s  tilt=%s%s  (%s) ===", station, tilt,
             f"  tag={tag}" if tag else "", raw_dir)

    # per-type total metric: hemi_hi -> WeightedPAI, hinge -> HingePAI
    metrics = cfg["quality_temporal"]["pai_metric"]
    # hemi-high glob from the canonical contract; hinge restricted to the standard
    # 5-ring scan (_0005_8500). The handful of single-ring _0001_8500 hinge files
    # (both years, both stations) are a sparse non-standard mode whose 1 ring
    # cannot give a stable hinge inversion (it saturates the Pgap log-floor at
    # ~12.7); they are excluded so the hinge series is canonical.
    jobs = {
        "hemi_hi": cfg["scan_types"]["hemi_hi"]["glob"],
        "hinge": "ESS?????_*_hinge_*_0005_8500.csv",
    }
    tables: dict[str, pd.DataFrame] = {}
    for scan_type, glob in jobs.items():
        files = discover(raw_dir, glob)
        if not files:
            log.warning("%s/%s: no files matched %s", station, scan_type, glob)
            continue
        df = process_type(scan_type, files, cfg, transform, n_workers)
        df.to_parquet(
            OUT_DIR / f"leaf_{scan_type}_{station}_2026{suffix}.parquet", index=False
        )
        tables[scan_type] = df

    # Per-scan totals (canonical per-type metric) + the ring-matched hemi HingePAI.
    per_scan: dict[str, pd.Series] = {}
    if "hemi_hi" in tables:
        per_scan["hemi_hi_weighted"] = total_pai_per_scan(
            tables["hemi_hi"], metrics["hemi_hi"]
        )
        per_scan["hemi_hi_hinge"] = total_pai_per_scan(tables["hemi_hi"], "HingePAI")
    if "hinge" in tables:
        per_scan["hinge_hinge"] = total_pai_per_scan(tables["hinge"], metrics["hinge"])

    daily = daily_summary(per_scan)
    daily.to_csv(OUT_DIR / f"leaf_pai_daily_{station}_2026{suffix}.csv")
    log.info("wrote daily CSV for %s%s (%d days)", station, suffix, len(daily))
    make_plot(per_scan, daily,
              OUT_DIR / f"pai_timeseries_{station}_2026{suffix}.png", station, tilt)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--station", default="Har_01",
        help="LEAF station: Har_01 | Har_02 | all (default: Har_01)",
    )
    p.add_argument(
        "--tilt", default="offset", choices=["offset", "rotation"],
        help="Tilt: offset = standard pylidar (default), rotation = ADR 0004",
    )
    p.add_argument(
        "--tag", default="",
        help="Optional output filename suffix (e.g. 'stdtilt' to keep a variant)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(CONFIG.read_text())
    n_workers = max(1, (os.cpu_count() or 8) - 4)
    transform = {"tilt": args.tilt, "up_drift": False}

    stations = STATIONS if args.station == "all" else (args.station,)
    for station in stations:
        if station not in STATIONS:
            log.error("unknown --station %r; choose from %s or 'all'",
                      station, list(STATIONS))
            return 2
        run_station(station, cfg, transform, args.tilt, args.tag, n_workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
