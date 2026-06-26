# Supplementary — LEAF hemispherical figures

Fisheye (hemispherical) renderings of the LEAF canopy scans for the paper
supplement. All figures read the canonical processed data
(`data/processed/proximal_rs/leaf/leaf_hemi_hi_2025.parquet`) plus the raw scans,
and share the helpers in `_leaf_hemi_common.py`. Every tunable (target dates,
scan hour, fonts, colour scales, bin sizes) sits in a `PARAMETERS` block at the
top of each script.

Run with the project Python (see repo `CLAUDE.md`):

```
/home/lk1167/miniconda3/envs/dehar-spac/bin/python \
    paper/40_supplementary/leaf_hemispherical/<script>.py
```

| script | figure | output |
|---|---|---|
| `fig_hemi_corrected_vs_uncorrected.py` | wide, transposed: 4 scenes as columns × 2 rows {uncorrected, corrected `seasonal_up`}, colour = height (0–20 m) | `outputs/hemi_uncorrected_vs_corrected.png` |
| `fig_hemi_seasonal_series.py` | wide: monthly night scenes as columns (Apr–Jun, Aug–Oct) × 3 rows {height, gap fraction binned (az/zen, log), gap fraction boolean}, `seasonal_up` | `outputs/hemi_seasonal_series.png` |

Prerequisite (seasonal "up")
- Both figures default to the `seasonal_up` mode, which re-folds each scan about
  the **smoothed daily "up"** from `data/processed/proximal_rs/leaf/leaf_up_daily_2025.csv`.
  Build it first (one-off, ~80 s):
  `…/python scripts/build_leaf_up_lookup.py`. This is robust to single-scan seam
  failures (e.g. the 2025-05-19 scan, whose own seam fit spuriously returns ~180°).

Notes
- Geometry modes: `none` (raw fold @180°), `rotation` (canonical tilt fix,
  issue #10 / ADR 0004), `self_calibrate` (per-scan up-drift fit, issue #11),
  `seasonal_up` (smoothed daily up lookup — the robust default here).
- The seasonal figure annotates each row with the **canonical** canopy-total
  `WeightedPAI` and prints a validation table (per-cell gap at 57.5° vs the
  canonical `Pgap_Z057.5`), so it doubles as a check on the canonical data.
- The fisheye gap / plant-area maps are a direct per-cell rendering of the
  returns (illustrative); the annotated PAI is the canonical Jupp (2009)
  FIRSTLAST / 5–70° inversion.
