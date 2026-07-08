"""Build the audited per-scene Sentinel-2 ROI-mean product (issue #8 / ADR 0009).

Recomputes per-scene ROI means for every ring x (band + index) from the RAW
unmasked stacks, joins the hand-audit verdicts from the S2 inspector, and writes
the tables the analysis reads. This is a NEW, parallel product: it does NOT touch
the headline S2 path (``process_sentinel2.py`` -> ``*_indices.nc`` ->
``export_all_streams`` -> ``aggregate_daily_streams``), which keeps reading the
hardcoded ``S2_MANUAL_DATES_2025``. Promoting this audit to canonical is a
deliberate follow-up (ADR 0009, "not rewired yet").

All science lives in ``dehar.satellite.sentinel2_roi``; this script only
orchestrates (read raw -> call src -> write processed).

Run
---
    /home/lk1167/miniconda3/envs/dehar-spac/bin/python scripts/process_sentinel2_roi.py

Outputs (data/processed/satellite/sentinel2/)
---------------------------------------------
    s2_roi_means_by_scene.parquet   every scene, every ring x var, + verdict/flags
    s2_roi_means_clear.csv          verdict == "clear" subset (the audited product)
"""

import logging
from pathlib import Path

import pandas as pd
from dehar.satellite import sentinel2_roi as s2r

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/satellite/sentinel2")
AUDIT_CSV = Path("other/s2_inspector/s2_audit.csv")
OUT_DIR = Path("data/processed/satellite/sentinel2")
OUT_PARQUET = OUT_DIR / "s2_roi_means_by_scene.parquet"
OUT_CLEAR_CSV = OUT_DIR / "s2_roi_means_clear.csv"

AUDIT_COLS = ["verdict", "snow", "shadow", "haze", "note"]


def _load_audit(path: Path) -> pd.DataFrame:
    """Per-scene verdicts keyed by scene_id; empty frame if not yet audited."""
    if not path.exists():
        log.warning("no audit CSV at %s — verdicts left empty", path)
        return pd.DataFrame(columns=["scene_id", *AUDIT_COLS])
    a = pd.read_csv(path, dtype={"scene_id": str})
    for c in AUDIT_COLS:
        if c not in a.columns:
            a[c] = pd.NA
    return a[["scene_id", *AUDIT_COLS]]


def main() -> None:
    log.info("building ROI-mean table from raw stacks in %s", RAW_DIR)

    def _prog(d: int, total: int, _sid: str) -> None:
        if d % 100 == 0 or d == total:
            log.info("  %d / %d scenes", d, total)

    table = s2r.build_roi_table(RAW_DIR, progress=_prog)

    audit = _load_audit(AUDIT_CSV)
    merged = table.merge(audit, on="scene_id", how="left")
    n_clear = int((merged["verdict"] == "clear").sum())
    log.info(
        "%d scenes | %d audited | %d clear",
        len(merged), int(merged["verdict"].notna().sum()), n_clear,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT_PARQUET)
    log.info("wrote %s (%d rows, %d cols)", OUT_PARQUET, *merged.shape)

    clear = merged[merged["verdict"] == "clear"].copy()
    clear.to_csv(OUT_CLEAR_CSV, index=False)
    log.info("wrote %s (%d clear-sky scenes)", OUT_CLEAR_CSV, len(clear))

    if n_clear == 0:
        log.warning(
            "no clear scenes yet — run the inspector "
            "(other/s2_inspector/server.py) to audit, then re-run this script",
        )


if __name__ == "__main__":
    main()
