# App C water-VOD — change notes (2026-07-01)

Practical companion to **ADR 0006 / 0007 / 0008**. What changed in the App C
VOD→water correction this round, how the headline numbers moved, and what is now
canonical. Read the ADRs for the *why*; this is the "what / how much / what now".

## Summary

App C recovers a water signal from GNSS-T VOD by subtracting an independently
measured lidar **biomass** term (`water-VOD = VOD − (a + b·PAI)`). Three changes,
plus a figure/method clean-up and robustness sweeps:

1. the leaf **"up"** underpinning the PAI is now the hand-verified **manual
   reference** (ADR 0006), not the smoothed auto-lookup;
2. the biomass fit is **event-blind** (ADR 0007) — it no longer excludes the
   drought window (fair vs the LOESS baseline; standalone-proxy valid);
3. the structural regressor is the **understory PAI (1.5–9 m)** (ADR 0008), not
   the whole-dome total — the occlusion-limited overstory was contaminating it.

**Net:** the corrected water-VOD now tracks predawn SWP at **r = 0.45** (raw VOD
−0.05; a cheap PAI-free LOESS detrend 0.31) **and** preserves the August water
dip (retention **0.69** vs LOESS **0.21**). So concurrent lidar PAI beats the
cheap detrend on **both** the seasonal correlation *and* the event — previously
(whole-dome total-PAI) it only won the event.

## What changed

- **Leaf "up" → manual reference (ADR 0006).** Canonical hemi profiles are now
  re-folded about the hand-verified per-scan `up_manual` (`up_resolve: scan`);
  the smoothed lookup is a fallback. All PAI below is on this geometry.
- **Figures + method clean-up.** Dropped the retired drifting **hinge-PAI** from
  the App C headline (its seasonal-r "win" was angle-drift trend-matching; it
  inflated dip retention > 1). Added the **window-resolved** correlation figure
  (the signal lives in the stress window; raw VOD decouples in autumn, the
  correction stays coupled). Made `decomposition.png` legible (7 d smooth over
  faint raw). Flagged **sap flow as structure-contaminated** (raw VOD ≈ 0.80 with
  it — not a water validator; SWP + TWD are). Removed a stale `per_receiver.png`.
- **Event-blind fit (ADR 0007).** `appC.fit_exclude_event: false`. The exclude-Aug
  fit is kept as a sensitivity.
- **Understory-PAI regressor (ADR 0008).** New canonical daily columns
  `pai_hemi_hi_weighted_{understory,overstory}_{mean,std}_m2m2` (layer split at
  9 m, `LEAF_LAYER_EDGES`), added in `export_all_streams_2025.py` (`leaf_totals`)
  → `aggregate_daily_streams_2025.py` (`build_pai`). `appC.biomass_col` + first
  regressor point at the understory band.
- **Robustness sweeps** (off-pipeline, `other/`): VOD daily-metric sweep
  (max/percentiles/mean/median/min), LOESS half-width sweep, seasonal-fit PAI,
  sap-flow seasonal decomposition, drought-response-by-layer.

## How the results changed (VOD ↔ predawn SWP, full season unless noted)

| stage | regressor / setup | seasonal r | event r | dip retention | fit R² |
|---|---|---|---|---|---|
| raw VOD | none | −0.05 ns | — | 1.00 | — |
| ADR 0005 (prior) | whole-dome total, smoothed up, exclude-Aug | ~0.27 \*\*\* | 0.65 | 0.77 | 0.68 |
| **now (canonical)** | **understory (1.5–9 m), manual up, event-blind** | **0.45 \*\*\*** | **0.70** | **0.69** | **0.79** |
| reference | LOESS ±30 d (PAI-free, Humphrey 2023) | 0.31 \*\*\* | — | 0.21 | — |

Notes:
- The often-quoted **~0.40** seasonal r from earlier was a **stale figure** (a
  prototype 90-day rolling-median VOD baseline); today's like-for-like prior
  number is ~0.27, and the understory change lifts it to 0.45.
- **Whole-dome total-PAI tied the cheap LOESS detrend** (0.27 vs 0.31); the
  understory band clears it (0.45), because it drops the overstory's from-below
  occlusion artifact that damped total-PAI's autumn decline.
- **Robust to the VOD daily reducer:** seasonal r = 0.40–0.47 across
  max/p95/p90/p75/mean/median (min lowest, 0.40). Daily-max (predawn proxy,
  Yao 2024) also gives the best fit R² and dip retention, so it stays canonical.
- **Caveat (ADR 0008):** the understory PAI itself dips ~0.4 in the drought (a
  real structural response), so it is not purely static "biomass" in the event —
  hence retention 0.69 < 0.77 for total-PAI, but still ≫ LOESS 0.21.

## What is now canonical (App C)

- **VOD:** daily-max of the QC'd (RH ≤ 95 %, no rain 12 h) hourly-mean `nvod`
  ensemble (gps1/3/5) — predawn proxy (Yao 2024).
- **Biomass regressor:** **understory PAI (1.5–9 m)**,
  `pai_hemi_hi_weighted_understory_mean_m2m2`, from the manual-up hemi profiles.
- **Fit:** linear VOD ~ PAI, **event-blind** (sees all days).
- **Water targets:** predawn SWP (primary) + TWD (both raw-uncorrelated → the
  correction recovers them). Sap flow = structure-contaminated context only.
- **Comparisons kept:** total-PAI (whole dome, ADR 0005), total-PAI
  (uncorrected), LOESS ±30 d. **Dropped from the headline:** the drifting
  hinge-PAI (retired, ADR 0005) — sensitivity only.
- **Part A is unaffected** — it uses the whole-canopy **total** PAI as bulk
  biomass for the onset cascade, a different use.

## Where it lives / reproduce

- Contract: `paper/config/analysis_config.yaml` → `appC:`.
- Leaf up: `scripts/build_leaf_up_manual_lookup.py` → `scripts/process_leaf.py`
  (see `docs/runbooks/update-canonical-leaf-up.md`).
- Layer PAI: `scripts/export_all_streams_2025.py` → `scripts/aggregate_daily_streams_2025.py`.
- Stage: `paper/30_appC_vod_correction/vod_pai_correction.py` → `outputs/`.
- Explorations: `other/appc_*` , `other/leaf_height_dynamics.py`,
  `other/leaf_layer_drought_zoom.py`.
- Decisions: ADR 0006 (manual up), 0007 (event-blind), 0008 (understory regressor).
