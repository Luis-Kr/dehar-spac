#!/usr/bin/env python3
"""Build the offset + up-drift LEAF sensitivity variant (ADR 0005).

The paper headline is rotation + up-drift; this produces the **published-offset
tilt + up-drift** variant as a kept sensitivity (so the tilt offset-vs-rotation
choice can be revisited). Reuses scripts/process_leaf.py with the transform and
output paths overridden, writing to
``data/processed/proximal_rs/leaf/variants/leaf_{scan_type}_2025_offset_updrift.parquet``.

Run (after the canonical reprocess + the up lookup exist):
  /home/lk1167/miniconda3/envs/dehar-spac/bin/python \
      scripts/build_leaf_variant_offset_updrift.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import process_leaf as pl  # noqa: E402


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = pl.load_config(pl.DEFAULT_CONFIG)
    cfg["transform"] = {
        "tilt": "offset",
        "up_drift": True,
        "up_lookup_csv": cfg["transform"]["up_lookup_csv"],
    }
    cfg["paths"] = dict(cfg["paths"])
    cfg["paths"]["out_dir"] = "data/processed/proximal_rs/leaf/variants"
    cfg["paths"]["out_pattern"] = "leaf_{scan_type}_{year}_offset_updrift.parquet"

    for stype in cfg["scan_types"]:
        pl.run_process(cfg, stype, None, None)
        pl.run_temporal(cfg, stype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
