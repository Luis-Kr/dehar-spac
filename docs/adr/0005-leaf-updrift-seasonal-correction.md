# 0005 — LEAF "up"-drift seasonal correction: canonical 57.5° from the corrected hemi

- Status: accepted; decision #1 (the **smoothed daily lookup** as the canonical "up") superseded by
  [0006](0006-leaf-updrift-manual-up-canonical.md) (2026-07-01) — the hand-verified **manual up
  reference** is now canonical and the smoothed lookup is a reproducible fallback. Everything else
  (the up-drift re-fold ∘ tilt composition, hinge-never-refolded, the 57.5° source) stands.
- Date: 2026-06-26
- Extends: [0004](0004-leaf-tilt-rotation-correction.md) — the up-drift re-fold is composed **on top of**
  the rotation tilt; the published offset is re-kept as a sensitivity.
- Supersedes (in part): [0003](0003-pai-hinge-hemi-not-sampling-artifact.md) — the "dedicated hinge
  scan = dense 57.5° early-warning" framing and the "keep both never-averaged series (hinge, hemi_hi)"
  decision (the hinge scan is demoted to a comparison series).

## Context

The hemi reconstruction recovers each beam's zenith by folding the mirror sweep about the encoder
"straight-up" point. At DE-Har that point **drifts** as the scanner settles in sandy soil: the
calibrated daily "up" runs ~176° (Apr) → ~208° (Sep–Oct) (see CONTEXT.md, *Hemi "up"-drift*).
Uncorrected, it mis-registers the two half-sweeps, pushes returns below ground (concentric rings near
the scanner), and biases the leaf-off **autumn/winter PAI low**. Issue #11.

Two findings forced a wider decision than a data refresh:

1. **Per-scan self-calibration is fragile.** Calibrating "up" from each scan's own seam can fail on a
   weak-seam scan (e.g. 2025-05-19 returns a spurious ~179° and leaves a visible seam) even though the
   seasonal drift is smooth.

2. **The dedicated hinge scan is collateral.** It fires at a fixed encoder setpoint (≈237.4°, labelled
   57.5° = `|237.4−180|`). Its *true* zenith is `|237.4 − up|`, which drifts **61° (Apr) → ≈57.5°
   (early May) → ~29–33° (Sep–Dec)**. The hinge scan is on its magic angle for only ~1–2 weeks, and
   being single-angle it **cannot be re-folded back** to 57.5°. So the headline "hinge-scan
   early-warning PAI" is partly an angle-drift artifact, and ADR 0003/0004's "dense 57.5° hinge scan"
   framing breaks after early May.

The up-corrected **hemi** scan, by contrast, images the whole dome and so samples a true,
season-stable 57.5° ring once re-folded.

## Decision

1. **Correct the hemi "up"-drift with a smoothed seasonal lookup, not per scan.**
   `scripts/build_leaf_up_lookup.py` calibrates "up" for every good hemi scan (from the **raw**
   geometry, so it is PAI-independent), takes a robust per-day median, and LOWESS-smooths over the
   season → `data/processed/proximal_rs/leaf/leaf_up_daily_2025.csv`. The trend is built from
   `hemi_hi` only (`hemi_low` seam fits are biased/unreliable at 16× fewer shots).

2. **Corrected geometry = up-drift re-fold ∘ tilt**, composed in that order. The transform config is
   refactored into two orthogonal knobs, `tilt ∈ {offset, rotation}` and `up_drift ∈ {off, on}`;
   the old `transform.method` names remain reproducible aliases.

3. **Paper headline canonical = `rotation` + `up_drift: on`** → the three
   `leaf_{hemi_hi,hemi_low,hinge}_2025.parquet`. The prior `rotation` / no-up-drift set is archived to
   `leaf_pre_updrift_fix_<date>/` and surfaced as `…_uncorr` daily columns. The **`offset` + up-drift**
   variant is produced and **kept as a sensitivity** (`leaf/variants/…_offset_updrift.parquet`, out of
   the daily table), preserving the published-library tilt for a future decision (flipping the headline
   tilt to offset would need an ADR superseding 0004).

4. **The canonical 57.5° "hinge-angle PAI" is taken from the up-corrected `hemi_hi` 57.5° ring**
   (`pai_hemi_hi_hinge`). The **dedicated hinge scan is demoted to a comparison series** (`leaf_hinge`,
   kept on tilt-rotation, labelled 57.5°, **not** re-folded — re-folding would re-label its single ring
   to ~30° and leave `HingePAI` with no 57.5° data). `leaf_seasonal_up` is one global method that
   re-folds hemi scans but only tilt-rotates the hinge scan.

5. **Part A consumes a single proximal "bulk biomass" PAI** — the canopy total of the up-corrected
   `hemi_hi` (`pai_hemi_hi_hinge`). On the hemi the Hinge-angle and Weighted inversions give the
   **same total** (verified, 1940/1940 scans; they differ only in PAVD shape), so there is no
   distinct "weighted bulk" to contrast against. The earlier **early-warning vs bulk PAI split is
   retired**: its early-warning arm was the dedicated hinge scan's dense 57.5° sampling, now shown to
   be a drifting-angle artifact. The dedicated hinge scan (`pai_hinge_hinge`) is retained as a
   flagged comparison and a possible future early-warning arm; the cascade's fast signal is carried by
   the water sensors (predawn SWP, VOD). The **full pylidar output** (Hinge / Linear / Weighted PAI
   + PAVD + `Pgap_Z*`) is still exported for every scan type.

6. **App C's structural regressor switches to `pai_hemi_hi_weighted`** (whole-dome bulk — the physical
   VOD structure proxy) and is run on **both** the uncorrected and corrected weighted PAI; headline =
   corrected.

7. **Re-folded-hinge validation.** To independently check the otherwise-borrowed assumption that the
   hinge shares the hemi "up", the hinge scan is *also* processed at its true angle θ(t) and its `Pgap`
   compared to the corrected hemi ring at θ(t); agreement confirms the drift (not canopy change) is the
   cause. Sensitivity artifact in `paper/90_sensitivity/`.

## Measured effect (full 2025)

- **hemi_hi `WeightedPAI` total**: overall **+0.47 m²m⁻²**; ~0 in May–June (up ≈ 180°), growing through
  the leaf-off season to **+0.50 (Oct), +1.37 (Nov), +1.03 (Dec)** — the suppressed-low autumn bias
  removed, tracking the drift.
- **hinge parquet unchanged** by the canonical run (max|Δ| = 0.000000): the hinge scan is tilt-rotated
  only, confirming the composition.
- **Re-folded-hinge validation**: the hinge at its true angle θ(t) vs the corrected hemi at θ(t) —
  **r = 0.96, MAE = 0.010** across 185 pairs (θ 29–61°), independent proof the hinge trajectory is
  angle-drift, not canopy (and that the hinge shares the hemi "up").
- **App C** (water-VOD ↔ predawn SWP, full season): the weighted structural correction **improves** —
  **r = 0.14 (ns, uncorrected) → 0.27 (p < 0.001, corrected)**.
- **Part A**: the corrected proximal-hemi PAI onsets **2025-08-22** (bulk, near the water sensors); the
  dedicated hinge-scan comparison onsets 08-15 (its apparent "lead", now flagged).
- **Tilt sensitivity** (Nov hemi WeightedPAI): offset+up-drift 4.18 vs rotation+up-drift 4.58 — the
  up-drift dominates (~+1.6), the tilt offset-vs-rotation choice is a secondary ~0.4.

## Considered Options

- **Per-scan self-calibration as canonical** (issue #11 first cut). Rejected: fragile on individual
  weak-seam scans; the smoothed lookup is robust and PAI-independent.
- **Re-invert the dedicated hinge scan at its true drifting angle** using the anglecam leaf-angle
  distribution. Rejected for the headline: reintroduces the leaf-angle dependence the 57.5° magic angle
  exists to avoid, and is very noisy near 30°. Kept only as the validation product.
- **Keep the dedicated hinge scan as the headline early-warning, flagged.** Rejected: knowingly
  headlines a drifting-angle series.
- **Revert the tilt to the published offset.** Not chosen as headline (ADR 0004 showed it is
  geometrically wrong), but the offset + up-drift variant is **kept** as a sensitivity so the choice can
  be revisited.

## Consequences

- **+** A true, season-stable 57.5° metric and an unbiased leaf-off hemi PAI; downstream PAI is no
  longer confounded by an instrument artifact.
- **+** The up-drift is independently validated (re-folded hinge vs hemi ring at θ).
- **−** **Headline numbers change and downstream artifacts go stale** and must be rebuilt: the
  daily/streams aggregates (`scripts/export_all_streams_2025.py` → `aggregate_daily_streams_2025.py`)
  and every PAI-consuming stage (Part A onset cascade, App C VOD correction). Expected direction:
  leaf-off **autumn/winter hemi PAI rises** (the uncorrected low bias removed). Exact deltas are
  recorded on regeneration.
- **−** **Part A's headline contrast is re-framed** (single-angle vs whole-dome on the corrected hemi);
  the early-warning *lead* may attenuate once the dense-hinge-scan + angle-drift mechanism is removed —
  itself a reportable finding.
- **−** More variants to track (the tilt × up_drift 2×2); a manifest documents where each lives, and
  only the headline feeds the daily table.
