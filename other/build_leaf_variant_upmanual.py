"""Build a LEAF hemi_hi variant: up-drift re-fold on the MANUAL "up" reference.

Off-pipeline validation (other/). The ADR 0005 canonical re-folds each hemi
scan about the *smoothed seasonal* "up" curve (up_smooth_deg), one value per
day. This variant instead re-folds about the **manual up reference**: the
hand-verified per-scan "up" a human recorded in the up-inspector app
(other/up_inspector/up_manual_corrections.csv, issue #11). Each inspected scan
uses its exact up_manual; every other scan is interpolated linearly in time
between its two nearest-in-time manual points (leaf.up_on_scan), so abrupt
re-settling steps the LOWESS smoothing loses are preserved.

Recipe: tilt=rotation (canonical), up_drift=True, up_resolve=scan, up lookup =
the per-scan manual CSV built here.

Writes:
- data/processed/proximal_rs/leaf/leaf_up_manual_perscan_2025.csv   (lookup)
- data/processed/proximal_rs/leaf/variants/leaf_hemi_hi_2025_upmanual.parquet

Run:
    /home/lk1167/miniconda3/envs/dehar-spac/bin/python \
        other/build_leaf_variant_upmanual.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import process_leaf as pl  # noqa: E402
from dehar.proximal_rs.leaf import get_scan_datetime  # noqa: E402

MANUAL_CSV = REPO / "other/up_inspector/up_manual_corrections.csv"
UP_PERSCAN = REPO / "data/processed/proximal_rs/leaf/leaf_up_manual_perscan_2025.csv"
SCAN_TYPE = "hemi_hi"


def write_perscan_lookup(src: Path, dst: Path) -> int:
    """Per-scan "up" lookup from the manual corrections (one row per scan).

    Dedupes by filename (latest submit wins), keys each correction by its
    filename's UTC scan datetime, and writes ``datetime, up_deg`` (+ provenance
    columns) sorted in time -- the shape :func:`leaf.load_up_lookup_perscan`
    reads.
    """
    m = pd.read_csv(src)
    m = m.sort_values("saved_at_utc").drop_duplicates("filename", keep="last")
    m["datetime"] = m["filename"].map(get_scan_datetime)
    m = m.dropna(subset=["datetime", "up_manual"])
    out = (
        m.assign(up_deg=m["up_manual"].astype(float))
        .loc[:, ["datetime", "up_deg", "filename", "note"]]
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    return len(out)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    n = write_perscan_lookup(MANUAL_CSV, UP_PERSCAN)
    logging.info("manual per-scan up lookup: %d scans -> %s", n, UP_PERSCAN.name)

    cfg = pl.load_config(pl.DEFAULT_CONFIG)
    cfg["transform"] = {
        "tilt": "rotation",
        "up_drift": True,
        "up_resolve": "scan",
        "up_lookup_csv": str(UP_PERSCAN.relative_to(REPO)),
    }
    cfg["paths"] = dict(cfg["paths"])
    cfg["paths"]["out_dir"] = "data/processed/proximal_rs/leaf/variants"
    cfg["paths"]["out_pattern"] = "leaf_{scan_type}_{year}_upmanual.parquet"

    pl.run_process(cfg, SCAN_TYPE, None, None)
    pl.run_temporal(cfg, SCAN_TYPE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
