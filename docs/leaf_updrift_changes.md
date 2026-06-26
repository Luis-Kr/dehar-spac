# LEAF "up"-drift correction — change notes (2026-06-26, issue #11)

Practical companion to **ADR 0005** (the decision record). What changed, where the
differences are, and how to reproduce. Read ADR 0005 for *why*.

## What changed (one paragraph)

The hemi scan folds each beam's zenith about a "straight-up" encoder point that
**drifts** as the scanner settles in sandy soil (~176° Apr → ~208° Sep–Oct). Left
uncorrected this biases leaf-off PAI **low** by ~1–1.4 m²m⁻². We now re-fold every
hemi scan about a **smoothed seasonal daily "up" lookup**, composed with the ADR 0004
tilt-rotation. The **dedicated hinge scan** turned out to be collateral (its fixed
setpoint, labelled 57.5°, actually drifts to ~30° by autumn and is unrecoverable), so
it is **demoted to a comparison** and the canonical 57.5° comes from the corrected hemi.

## Geometry model (the new 2 knobs)

`config/leaf_processing.yaml` → `transform:` is now `tilt` × `up_drift`:

| | up_drift OFF | up_drift ON |
|---|---|---|
| **tilt = rotation** (ADR 0004) | uncorrected baseline (`leaf_pre_updrift_fix_20260626/`, daily `…_uncorr` cols) | **HEADLINE** `leaf/leaf_{type}_2025.parquet` |
| **tilt = offset** (published TLS) | `leaf_pre_tilt_fix_20260625/` | sensitivity `leaf/variants/…_offset_updrift.parquet` |

Composition = **re-fold (hemi) then tilt**. The **hinge scan is never re-folded** (single
ring) → tilt-only in every run, so `leaf_hinge_2025.parquet` == its rotation baseline
(verified max|Δ| = 0). Manifest: `data/processed/proximal_rs/leaf/PARQUET_VARIANTS.md`.

## Where the differences are (numbers)

- **hemi_hi `WeightedPAI` total** (corrected − uncorrected): ≈0 May–Jun (up ≈ 180°), growing to
  **+0.50 (Oct), +1.37 (Nov), +1.03 (Dec)**; overall +0.47.
- **Tilt method is secondary**: Nov hemi PAI = up-drift+rotation **4.45**, up-drift+offset 4.06,
  **up-drift only (no tilt) 4.46** — rotation ≈ no tilt (azimuth-mean-zero); offset biases −0.4.
  Up-drift alone does essentially the whole correction (uncorrected 2.99).
- **hemi Hinge-angle total ≡ Weighted total** (1940/1940; max|Δ|=2.7e-15) — only PAVD shape differs.
  `LinearPAI` is the only inversion with a distinct total.
- **Dedicated hinge scan** (`pai_hinge_hinge`) diverges low in autumn (Nov ~2.35) — the drifting angle.
- **App C** (water-VOD ↔ predawn SWP, full season): **r 0.14 (ns, uncorrected) → 0.27 (***, corrected)**
  — the correction *improves* the VOD water proxy (regressor switched to `pai_hemi_hi_weighted`).
- **Part A**: corrected proximal-hemi PAI onset **2025-08-22** (bulk, near the water sensors); the
  early-warning/bulk PAI split is **retired** (its arm was the now-compromised dedicated hinge scan).
- **Validation**: hinge re-folded to its true angle θ(t) vs the corrected hemi at θ(t) —
  **r = 0.96, MAE = 0.010** (185 pairs, θ 29–61°) → independent proof it is angle-drift, not canopy.

## Code changed

- `src/proximal_rs/leaf.py`: `resolve_transform()` (2-knob + legacy aliases), `_apply_tilt_offset`,
  `load_up_lookup`/`up_on_date`, composition in `add_leaf_scan_to_profile` (+ `up_deg`), `up_for_scan`,
  `daily_up_lookup`. `config/leaf_processing.yaml`: 2-knob transform.
- `scripts/build_leaf_up_lookup.py` (NEW): per-scan seam fit → daily median → LOWESS → lookup csv/png.
  `scripts/build_leaf_variant_offset_updrift.py` (NEW). `scripts/process_leaf.py`: load lookup, pass
  `up_deg`, fail-loud if lookup missing.
- `scripts/export_all_streams_2025.py` + `aggregate_daily_streams_2025.py`: `pai_hemi_hi_{hinge,weighted}_uncorr_*`.
- `paper/config/analysis_config.yaml`: single bulk PAI; App C regressors → weighted (corr + uncorr).
  `paper/10_partA_sensor_response/onset_{bootstrap,satellite}.py`: PAI stream relabel.
  `paper/30_appC_vod_correction/vod_pai_correction.py`: iterate `regressors`.
  `paper/90_sensitivity/leaf_hinge_refold_validation.py` (NEW).
  `paper/40_supplementary/leaf_hemispherical/` (NEW figure scripts).
- Docs: `docs/adr/0005-…`, ADR 0003/0004 pointers, `CONTEXT.md` (4 terms), `data/README.md` §10
  (local; `data/` is gitignored), `…/leaf/PARQUET_VARIANTS.md`.
- Tests: `tests/test_leaf_processing.py` (+`resolve_transform`, hinge-invariance, hemi-refold) — 29 pass.

## Reproduce

```bash
PY=/home/lk1167/miniconda3/envs/dehar-spac/bin/python
$PY scripts/build_leaf_up_lookup.py                       # leaf_up_daily_2025.csv (+ QC png)
# archive current rotation parquets -> leaf_pre_updrift_fix_<date>/ first
$PY scripts/process_leaf.py                               # canonical (config = rotation + up_drift)
$PY scripts/build_leaf_variant_offset_updrift.py          # offset+updrift sensitivity
$PY scripts/export_all_streams_2025.py && $PY scripts/aggregate_daily_streams_2025.py
$PY paper/10_partA_sensor_response/onset_bootstrap.py     # + onset_satellite.py
$PY paper/30_appC_vod_correction/vod_pai_correction.py
$PY paper/90_sensitivity/leaf_hinge_refold_validation.py  # r=0.96 check
```

## Open / not done

- Late-season validation vs PhenoCam / leaf-angle.
- The **tilt offset-vs-rotation** final call: both produced; headline stays rotation. Flipping to the
  published offset would need an ADR superseding 0004.
- `{tilt: none, up_drift: on}` parquet (`leaf/variants/…_none_updrift.parquet`) is a one-off comparison,
  not part of the canonical pipeline.
