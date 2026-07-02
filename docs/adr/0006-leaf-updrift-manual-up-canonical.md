# 0006 — LEAF "up"-drift: the hand-verified MANUAL up reference is canonical

- Status: accepted
- Date: 2026-07-01
- Extends: [0005](0005-leaf-updrift-seasonal-correction.md) — the up-drift re-fold ∘ tilt
  composition, the "correct the hemi, never re-fold the hinge" rule, and the 57.5° source are all
  unchanged. Only the **source of the "up"** changes.
- Supersedes (in part): [0005](0005-leaf-updrift-seasonal-correction.md) decision #1 — the
  *smoothed seasonal daily lookup* as the canonical "up". It is demoted to a reproducible fallback.

## Context

ADR 0005 fixed the seasonal "up"-drift (issue #11) by re-folding each hemi scan about an **automatic
smoothed daily lookup**: per-scan seam fits → robust per-day median → LOWESS over the season
(`scripts/build_leaf_up_lookup.py` → `leaf_up_daily_2025.csv`). That is robust to a single bad seam
fit, but the smoothing **blurs abrupt re-settling steps** and can lag the true "up" in the leaf-off
season, where the fold matters most.

The observer then **hand-verified the "up" for 300+ hemi scans** in the up-inspector app
(`other/up_inspector/`, built for issue #11): for a representative scan per day the folded-fisheye
seam is aligned by eye and the "up" that makes the canopy coherent is recorded
(`other/up_inspector/up_manual_corrections.csv`, one row per scan, near-daily 2025-04-16 → 12-15,
243/244 scan-days covered). This is a **direct physical measurement of the fold** — the gold-standard
"up".

**Validation against DHP.** Independent digital-hemispherical-photo LAI at 5 VOD positions
(`data/raw/proximal_rs/dhp/…`, 12 in-range scenes Apr–Nov) was compared against the hemi_hi total
WeightedPAI for every up variant (`other/dhp_lai_vs_pai_upmanual_2025.py`). Findings:

- On **full-window Pearson r the variants are indistinguishable** — manual 0.70, smoothed 0.74,
  uncorrected 0.74, all within n=12 sampling noise. The *drifting hinge scan* tops Pearson at 0.81, a
  **degenerate result**: r over 12 scenes rewards coincidental seasonal shape, not the leaf-off level
  the up-drift correction exists to fix. So DHP correlation **cannot discriminate** manual vs smoothed.
- On the metric the correction **is for — the leaf-off level — the manual best-matches DHP**: at the
  Nov scenes manual = 4.22 / 4.13 vs DHP 4.20 / 4.02, where smoothed overshoots (4.33 / 4.42) and
  uncorrected collapses (3.19 / 2.91). Manual also has the **highest Spearman (0.67)** of the
  geometry-correct variants.

So DHP gives the manual a **clean bill of health** (at least as good as the smoothed proxy, best on
leaf-off level) but does not prove a correlation advantage — none exists at this n.

## Decision

1. **Canonical "up" = the manual up reference, resolved per scan.** `transform.up_resolve: scan` with
   `up_lookup_csv: leaf_up_manual_perscan_2025.csv`, built from the up-inspector CSV by
   `scripts/build_leaf_up_manual_lookup.py`. **Basis of the decision: it is the hand-verified true
   "up"** (direct seam alignment over 300+ scans), with DHP confirming it is ≥ the smoothed proxy.
   It is **not** adopted for a better DHP correlation (that is a statistical wash).

2. **Per-scan, interpolated in time.** Each inspected scan folds about its exact `up_manual`; every
   other scan folds about a **linear-in-time interpolation** of its two nearest manual points;
   endpoints are held (no extrapolation). `dehar.proximal_rs.leaf.up_on_scan` (unit-tested). This is a
   change from ADR 0005's per-*day* resolution — a day's eight 3-hourly scans now each fold about
   their own "up". **No re-smoothing**, so the hand-verified re-settling steps are preserved.

3. **Everything else from ADR 0005 is unchanged.** Composition = up-drift re-fold ∘ tilt-rotation; the
   **hinge scan is never re-folded** (single ring) and its parquet is byte-identical; the canonical
   57.5° metric still comes from the up-corrected `hemi_hi` ring.

4. **The smoothed seasonal lookup is demoted to a reproducible fallback** (`up_resolve: date`,
   `leaf_up_daily_2025.csv` retained; `scripts/build_leaf_up_lookup.py` unchanged). The pre-switch
   smoothed-canonical parquets are snapshot to `data/processed/proximal_rs/leaf_pre_upmanual_20260701/`.

5. **The manual CSV is the canonical basis and is version-controlled.** Because it is a hand-made
   pipeline input (not derivable from code), `other/up_inspector/up_manual_corrections.csv` is
   committed; it *is* the provenance. Updating it (more inspection) → rebuild → reprocess → propagate
   is a documented, repeatable path: **`docs/runbooks/update-canonical-leaf-up.md`**.

## Measured effect (full 2025)

- **Manual vs smoothed is a small refinement.** hemi_hi total WeightedPAI vs the smoothed-canonical
  snapshot, over 227 shared days: **mean Δ −0.07, mean|Δ| 0.13, max|Δ| 0.82**. Nov 1 – Dec 15 mean:
  **manual 4.13** vs smoothed 4.24 vs uncorrected 2.90 (leaf-off Δ −0.12, toward DHP). The large
  leaf-off fix (uncorrected → any up-drift, ≈ +1.3) already landed in ADR 0005; the manual moves the
  level a further ≈ 0.1 toward DHP and pins the abrupt re-settling steps (e.g. the ≈ −4° "up" drop the
  observer resolved at 2025-09-24) that LOWESS blurred.
- **hinge parquet unchanged** (never re-folded — verified byte-identical to the snapshot).
- **Aggregates rebuilt** (`dehar_daily_2025.parquet`, `dehar_all_streams_2025.parquet`); the
  `pai_hemi_hi_*` columns now carry the manual "up" (mean|Δ| 0.13 vs before). **Part A / App C** deltas
  recorded when those stages are re-run — expected small.

## Considered Options

- **Keep the smoothed lookup canonical (ADR 0005).** Rejected: a hand-verified direct measurement of
  the fold now exists and is best on the leaf-off level; keeping an automatic proxy as the headline
  when the ground truth is available is second-best. Retained as the fallback.
- **Promote on best DHP Pearson r (the pre-committed gate).** Rejected: degenerate at n=12 — it
  crowns the drifting hinge scan and cannot separate manual from smoothed.
- **Re-define the acceptance metric to leaf-off level agreement.** Used only as *supporting* evidence
  (manual is best there), not as the basis — changing the gate after seeing results would be post-hoc.
  The basis is provenance (hand-verified true "up"), which DHP does not contradict.
- **Move the manual CSV out of `other/`.** Not done: the up-inspector app writes there and the
  observer keeps editing it; committing it in place keeps the source of truth and the app workflow
  aligned (the runbook makes the commit step explicit).

## Consequences

- **+** The canonical fold is the hand-verified true "up"; the leaf-off PAI best-matches independent
  DHP LAI.
- **+** A clear, repeatable update path for future inspection rounds (runbook), with a version-
  controlled basis.
- **−** The canonical pipeline now has a **hand-made input** (not reproducible from code alone).
  Mitigated: the CSV is version-controlled and is itself the provenance; the automatic smoothed lookup
  remains a byte-reproducible fallback (`up_resolve: date`).
- **−** **Downstream artifacts are stale and must be rebuilt to propagate**: canonical leaf parquets
  (done), the daily/streams aggregates (`export_all_streams_2025.py` → `aggregate_daily_streams_2025.py`),
  and every PAI-consuming stage (Part A onset cascade, App C VOD correction). The change is small, so
  headline numbers should move little; exact deltas recorded on regeneration.
- **−** More provenance to track (snapshot dir, two "up" lookups, the `up_resolve` knob). The config
  comments, this ADR, and the runbook document where each lives.
