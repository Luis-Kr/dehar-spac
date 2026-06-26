#!/usr/bin/env python3
"""Process LEAF canopy scans into per-scan-type parquet tables (DE-Har 2025).

Command-line replacement for
``notebooks/00_exploration/leaf_batch_processing_dehar.ipynb``. Every parameter
lives in ``config/leaf_processing.yaml`` (nothing hardcoded here); the heavy
per-scan Jupp (2009) inversion is parallelised across cores.

Pipeline (two steps, both re-runnable independently)
----------------------------------------------------
1. ``process``  raw CSV scans -> invert each (multiprocessing) -> attach met
                context + meteorological quality flag -> write
                ``data/processed/proximal_rs/leaf/leaf_<type>_<year>.parquet``.
2. ``temporal`` second pass over the saved parquet: add ``flag_temporal_outlier``
                and ``quality_all`` (per-scan-hour robust-smoothing outlier on
                the seasonal total-PAI trend). Cheap; no reprocessing.

The tilt **transform** (GitHub issue #10) is surfaced in the config
(``transform.enabled`` / ``transform.method``) and threaded through
``dehar.proximal_rs.leaf.add_leaf_scan_to_profile`` — the seam for the phase-2
re-levelling fix.

Run
---
    /home/lk1167/miniconda3/envs/dehar-spac/bin/python scripts/process_leaf.py
    # subsets / steps:
    .../python scripts/process_leaf.py --scan-type hinge --steps process
    .../python scripts/process_leaf.py --steps temporal
    .../python scripts/process_leaf.py --limit 20 --workers 8   # quick check
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

# Make the editable ``dehar`` package importable when run as a bare script.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dehar.proximal_rs.leaf import (  # noqa: E402
    attach_met_context,
    compute_temporal_flags,
    get_scan_datetime,
    load_met,
    load_up_lookup,
    process_single_scan,
    resolve_transform,
    up_on_date,
)

DEFAULT_CONFIG = REPO_ROOT / "config" / "leaf_processing.yaml"
BOOL_COLS = ["flag_rain", "flag_humid", "flag_wind", "quality_good"]

log = logging.getLogger("process_leaf")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def load_config(path: Path) -> dict:
    """Read the YAML processing contract."""
    with open(path) as fh:
        return yaml.safe_load(fh)


def resolve_workers(runtime: dict, n_tasks: int) -> int:
    """Worker count from config, bounded by task count and one valid core."""
    workers = runtime.get("workers")
    if workers is None:
        workers = max(1, (os.cpu_count() or 8) - int(runtime.get("reserve_cpus", 4)))
    return max(1, min(int(workers), n_tasks))


def discover_scans(raw_dir: Path, glob: str, limit: int | None) -> list[Path]:
    """Sorted list of scan files for one scan type."""
    files = sorted(raw_dir.glob(glob))
    return files[:limit] if limit else files


def outfile_for(cfg: dict, scan_type: str) -> Path:
    """Output parquet path for a scan type from the configured pattern."""
    paths = cfg["paths"]
    name = paths["out_pattern"].format(scan_type=scan_type, year=paths["year"])
    return REPO_ROOT / paths["out_dir"] / name


# --------------------------------------------------------------------------
# Step 1 — process
# --------------------------------------------------------------------------
def run_process(
    cfg: dict, scan_type: str, workers: int | None, limit: int | None
) -> None:
    """Invert every scan of one type and write the parquet (with met context)."""
    paths = cfg["paths"]
    raw_dir = REPO_ROOT / paths["raw_dir"]
    sc = cfg["scan_types"][scan_type]
    files = discover_scans(raw_dir, sc["glob"], limit)
    if not files:
        log.warning("%s: no scan files matched %s", scan_type, sc["glob"])
        return

    outfile = outfile_for(cfg, scan_type)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    n_workers = workers or resolve_workers(cfg["runtime"], len(files))
    progress_every = int(cfg["runtime"].get("progress_every", 100))
    log.info(
        "%s: %d scans -> %s  (%d workers)",
        scan_type,
        len(files),
        outfile.name,
        n_workers,
    )

    met_full = load_met(REPO_ROOT / paths["met_file"], REPO_ROOT / paths["precip_file"])

    # seasonal "up" lookup (only when the up-drift re-fold is on; ADR 0005)
    up_lookup = None
    if resolve_transform(cfg["transform"])["up_drift"]:
        lookup_path = REPO_ROOT / cfg["transform"]["up_lookup_csv"]
        if not lookup_path.exists():
            raise FileNotFoundError(
                f"transform.up_drift is on but the 'up' lookup is missing: "
                f"{lookup_path}. Build it first: python scripts/build_leaf_up_lookup.py"
            )
        up_lookup = load_up_lookup(lookup_path)
        log.info("%s: up-drift on, using %s", scan_type, lookup_path.name)

    def _up_for(f: Path) -> float | None:
        if up_lookup is None:
            return None
        dt = get_scan_datetime(f.name)
        return up_on_date(up_lookup, dt) if dt is not None else None

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
                transform=cfg["transform"],
                up_deg=_up_for(f),
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
            if i % progress_every == 0:
                rate = i / (time.time() - t0)
                eta = (len(files) - i) / rate if rate else 0
                log.info(
                    "  %d/%d (%d ok, %d fail) ~%.0fs remaining",
                    i,
                    len(files),
                    n_ok,
                    n_fail,
                    eta,
                )

    if not profiles:
        log.warning("%s: no valid scans, parquet not written", scan_type)
        return

    df_all = pd.concat(profiles, ignore_index=True)
    df_all = attach_met_context(
        df_all, met_full, sc["duration_min"], cfg["quality_met"]
    )
    df_all["datetime"] = pd.to_datetime(df_all["datetime"])
    for c in BOOL_COLS:
        df_all[c] = df_all[c].astype(bool)
    df_all = df_all.sort_values(["datetime", "height"]).reset_index(drop=True)
    df_all.to_parquet(outfile, index=False)

    n_scans = df_all["datetime"].nunique()
    n_good = int(df_all.drop_duplicates("datetime")["quality_good"].sum())
    log.info(
        "%s done: %d scans (%d met-good), %d rows, %.0fs -> %s (%.1f MB)",
        scan_type,
        n_scans,
        n_good,
        len(df_all),
        time.time() - t0,
        outfile.name,
        outfile.stat().st_size / 1e6,
    )


# --------------------------------------------------------------------------
# Step 2 — temporal filter
# --------------------------------------------------------------------------
def run_temporal(cfg: dict, scan_type: str) -> None:
    """Add ``flag_temporal_outlier`` and ``quality_all`` to a saved parquet."""
    outfile = outfile_for(cfg, scan_type)
    if not outfile.exists():
        log.warning("%s: %s missing, run 'process' first", scan_type, outfile.name)
        return

    temporal = cfg["quality_temporal"]
    pai_col = temporal["pai_metric"][scan_type]

    df = pd.read_parquet(outfile)
    df["datetime"] = pd.to_datetime(df["datetime"])
    flags = compute_temporal_flags(df, pai_col, temporal)

    df["flag_temporal_outlier"] = df["datetime"].map(flags).fillna(True).astype(bool)
    df["quality_all"] = (df["quality_good"] & ~df["flag_temporal_outlier"]).astype(bool)
    df.to_parquet(outfile, index=False)

    scans = df.drop_duplicates("datetime")
    log.info(
        "%s (%s): %d/%d temporal outliers, %d/%d pass quality_all",
        scan_type,
        pai_col,
        int(scans["flag_temporal_outlier"].sum()),
        len(scans),
        int(scans["quality_all"].sum()),
        len(scans),
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Processing config YAML (default: config/leaf_processing.yaml)",
    )
    p.add_argument(
        "--scan-type",
        default="all",
        help="Scan type to run: hemi_hi | hemi_low | hinge | all",
    )
    p.add_argument(
        "--steps",
        default="all",
        choices=["process", "temporal", "all"],
        help="Which step(s) to run",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override worker count (default: from config)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N scans per type (quick check)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = load_config(args.config)

    known = list(cfg["scan_types"])
    if args.scan_type == "all":
        scan_types = known
    elif args.scan_type in known:
        scan_types = [args.scan_type]
    else:
        log.error(
            "unknown --scan-type %r; choose from %s or 'all'",
            args.scan_type,
            known,
        )
        return 2

    for stype in scan_types:
        if args.steps in ("process", "all"):
            run_process(cfg, stype, args.workers, args.limit)
        if args.steps in ("temporal", "all"):
            run_temporal(cfg, stype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
