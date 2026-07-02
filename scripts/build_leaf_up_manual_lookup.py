#!/usr/bin/env python3
"""Build the canonical LEAF "up" lookup from the MANUAL up reference (ADR 0006).

The canonical up-drift re-fold folds each hemi scan about a hand-verified
"up" the observer recorded in the up-inspector app
(``other/up_inspector/up_manual_corrections.csv``, issue #11). That CSV is the
**gold-standard basis**: a human aligned the folded-fisheye seam per scan so the
canopy is coherent. This step turns it into the per-scan lookup the processing
pipeline reads (:func:`dehar.proximal_rs.leaf.load_up_lookup_perscan`).

What it does
------------
1. read every manual correction (one row per inspected scan, latest submit wins);
2. key each by its filename's UTC scan datetime;
3. write ``datetime, up_deg`` (+ provenance) sorted in time.

``process_leaf.py`` (``transform.up_resolve: scan``) then folds each scan about
its exact manual "up" where inspected, and about a linear-in-time interpolation
of the two neighbouring manual points elsewhere (endpoints held). No smoothing:
the hand-verified re-settling steps are preserved (contrast the automatic
``scripts/build_leaf_up_lookup.py``, which LOWESS-smooths and is now a fallback).

Run
---
    /home/lk1167/miniconda3/envs/dehar-spac/bin/python \
        scripts/build_leaf_up_manual_lookup.py

Output (data/processed/proximal_rs/leaf/):
    leaf_up_manual_perscan_2025.csv   datetime, up_deg, filename, note

Re-run this whenever the manual CSV changes (after inspecting more scans), then
re-run the canonical pipeline -- see docs/runbooks/update-canonical-leaf-up.md.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dehar.proximal_rs.leaf import get_scan_datetime  # noqa: E402

MANUAL_CSV = REPO_ROOT / "other/up_inspector/up_manual_corrections.csv"
OUT_CSV = REPO_ROOT / "data/processed/proximal_rs/leaf/leaf_up_manual_perscan_2025.csv"

log = logging.getLogger("build_leaf_up_manual_lookup")


def build_perscan_lookup(src: Path, dst: Path) -> pd.DataFrame:
    """Per-scan "up" lookup from the manual corrections CSV.

    Dedupes by filename (latest submit wins), keys each correction by its
    filename's UTC scan datetime, and writes the ``datetime, up_deg`` shape
    :func:`dehar.proximal_rs.leaf.load_up_lookup_perscan` reads.

    Parameters
    ----------
    src : pathlib.Path
        The up-inspector corrections CSV (``up_manual`` per scan).
    dst : pathlib.Path
        Output lookup CSV path.

    Returns
    -------
    pandas.DataFrame
        The written lookup (``datetime, up_deg, filename, note``).
    """
    if not src.exists():
        raise FileNotFoundError(
            f"manual corrections CSV not found: {src}. Inspect scans in the "
            f"up-inspector app first (other/up_inspector/server.py)."
        )
    raw = pd.read_csv(src)
    for col in ("filename", "up_manual", "saved_at_utc"):
        if col not in raw.columns:
            raise ValueError(f"{src} is missing required column {col!r}")

    m = raw.sort_values("saved_at_utc").drop_duplicates("filename", keep="last")
    m = m.assign(datetime=m["filename"].map(get_scan_datetime))
    bad = m["datetime"].isna() | m["up_manual"].isna()
    if bad.any():
        log.warning("dropping %d rows (no datetime / no up)", int(bad.sum()))
    out = (
        m.loc[~bad]
        .assign(up_deg=lambda d: d["up_manual"].astype(float))
        .loc[:, ["datetime", "up_deg", "filename", "note"]]
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    if out.empty:
        raise ValueError(f"no usable manual corrections in {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manual-csv", type=Path, default=MANUAL_CSV)
    p.add_argument("--out", type=Path, default=OUT_CSV)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    out = build_perscan_lookup(args.manual_csv, args.out)
    span = f"{out['datetime'].min()} -> {out['datetime'].max()}"
    log.info(
        "manual up lookup: %d scans (%s), up %.1f-%.1f deg -> %s",
        len(out),
        span,
        out["up_deg"].min(),
        out["up_deg"].max(),
        args.out.relative_to(REPO_ROOT),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
