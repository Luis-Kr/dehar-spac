# 0008 — App C structural regressor: understory PAI (1.5–9 m), not whole-dome

- Status: accepted
- Date: 2026-07-01
- Builds on: [0006](0006-leaf-updrift-manual-up-canonical.md) (manual up → trustworthy vertical
  profiles), [0007](0007-appc-event-blind-biomass-fit.md) (event-blind fit).
- Supersedes (in part): [0005](0005-leaf-updrift-seasonal-correction.md) decision #6 — App C's
  structural regressor was the **whole-dome** hemi `WeightedPAI`; it is now the **understory band**.

## Context

App C recovers a water signal from GNSS-T VOD by subtracting a biomass term `a + b·PAI`. ADR 0005 #6
used the whole-dome hemi `WeightedPAI` (`pai_hemi_hi_weighted_mean_m2m2`). The height-resolved
analysis (`paper/40_supplementary/leaf_hemispherical/fig_height_resolved_dynamics.py`) showed DE-Har
is **two layers**:

- a **deciduous understory** (PAVD peak ~4–6 m) that carries the entire seasonal + August-drought PAI
  signal and is well sampled (near the below-canopy scanner);
- an **evergreen overstory** (~12 m) that is physically ~flat, but whose *apparent* PAI **rises in
  autumn as a from-below occlusion artifact** (once the understory drops leaves, more overstory
  returns get through) — not real growth.

Including the overstory in the whole-dome regressor injects that occlusion-artifact variation, which
**damps total-PAI's autumn decline** (understory falls while overstory spuriously rises), so total-PAI
matches VOD's autumn decline worse and leaves a dirtier water residual.

## Decision

1. **Canonical App C structural regressor = understory PAI (1.5–9 m)**,
   `pai_hemi_hi_weighted_understory_mean_m2m2`. Added to the aggregate pipeline: `leaf_totals`
   (`scripts/export_all_streams_2025.py`) emits per-scan understory/overstory `WeightedPAI` as the
   cumulative-PAI difference across `LEAF_LAYER_EDGES = (1.5, 9.0, 18.0)`, and `build_pai`
   (`scripts/aggregate_daily_streams_2025.py`) aggregates them daily.
2. `appC.biomass_col` and the first `corrections.regressors` entry point at the understory column.
   **Whole-dome `total-PAI` (ADR 0005) and `total-PAI (uncorrected)` are kept as comparisons.**
3. The split at 9 m is the PAVD density gap between the two layers (see the height figure).

## Measured effect (full 2025, event-blind fit)

| regressor | biomass fit R² | seasonal r(SWP) | event r(SWP) | Aug dip retention |
|---|---|---|---|---|
| **understory PAI (1.5–9 m)** | **0.79** | **0.45 \*\*\*** | **0.70** | 0.69 |
| total PAI (whole dome, ADR 0005) | 0.68 | 0.27 \*\*\* | 0.65 | 0.78 |
| total PAI (uncorrected) | 0.67 | 0.14 ns | — | 0.61 |
| LOESS ±30 d (PAI-free) | — | 0.31 \*\*\* | — | 0.21 |
| raw VOD | — | −0.05 ns | 0.73 | 1.00 |

The decisive point: with whole-dome total-PAI, the seasonal correlation only **tied** the cheap LOESS
detrend (0.27 vs 0.31) — PAI earned its keep only on the event dip. **Understory-PAI now beats LOESS
on _both_ axes** (season 0.45 > 0.31; dip 0.69 ≫ 0.21). "Is concurrent high-res PAI worth it?" → yes,
seasonally and at the event.

## Considered Options

- **Keep whole-dome total-PAI (ADR 0005).** Rejected: contaminated by the overstory occlusion
  artifact; ties LOESS on the seasonal metric. Retained as a comparison.
- **Overstory only (9–18 m).** Rejected: seasonal r ≈ 0.04 — it is occlusion noise with no
  water-relevant structural signal.
- **Understory (1.5–9 m).** Chosen: the varying, well-sampled, water-relevant layer.

## Consequences

- **+** A cleaner biomass regressor (fit R² 0.68 → 0.79); PAI beats the PAI-free detrend on both the
  seasonal correlation and the event dip.
- **+** Layer-PAI columns are now first-class in the daily table (understory/overstory mean+std).
- **−** **Mild event-circularity:** the understory PAI itself dips ~0.4 during the drought (a real
  structural response — leaf angle/area), so it is not purely static "biomass" in the event; dip
  retention falls to 0.69 (vs 0.78 for total-PAI) but stays far above LOESS (0.21). Flagged, accepted.
- **−** The overstory magnitude is **occlusion-limited from below** (the scanner looks up); the
  understory is the trustworthy layer, which is *why* understory-only is the right regressor here.
- **−** App C outputs regenerated. **Part A is unaffected** — it consumes the whole-canopy total PAI
  as "bulk biomass" for the onset cascade, a different use that is left on total-PAI.
