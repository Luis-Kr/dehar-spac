# 0012 — Sentinel-2: the hand audit is promoted to the canonical S2 source

- Status: accepted
- Date: 2026-07-08
- Relates: the promised follow-up to [0009](0009-sentinel2-unmasked-hand-audit.md) ("not rewired
  yet"). Same manual-over-automated stance as [0006](0006-leaf-updrift-manual-up-canonical.md).
  Parallel in spirit to [0011](0011-sentinel1-grdfloat-canonical.md) (retire the opaque satellite
  product for the transparent in-repo one).

## Context

ADR 0009 re-exported the S2 archive unmasked and built a per-scene hand audit
(`other/s2_inspector/s2_audit.csv`), but deliberately left the headline pipeline on the old path:
`export_all_streams_2025.py` re-reduced the ROI mean from the SCL-masked `*_indices.nc` stacks and
`aggregate_daily_streams_2025.py` filtered to a hardcoded 33-date list (`S2_MANUAL_DATES_2025`,
2025-only). Promotion was gated on the audit "reproducing or deliberately superseding those 33
dates."

The audit is now complete: **693 scenes, 2017-04-10 → 2026-06-27**, each verdicted
(**229 clear / 24 partial / 440 cloud**). The gate is met — for 2025 the audit agrees with 30 of the
33 dates, reclassifies 3 as no-longer-clear (2025-02-27, 2025-10-30, 2025-11-19) and adds one
(2025-07-29): a small, defensible correction, not a reproduction.

Three things about the old headline path were wrong at once, and only promotion fixes all three
together:

1. **Wrong ROI centre.** The old `*_indices.nc` ROI was extracted on the pre-fix site coordinate,
   ~226 m off the true flux tower — the canonical 100 m ROI did not even contain the tower. The
   audited product is computed on the corrected tower grid (ADR 0009 tower fix).
2. **Masked values.** The old ROI means came from SCL/s2cloudless-masked stacks; the audited product
   recomputes from the raw unmasked stacks, so the clear-sky decision and the pixel values are
   consistent (both the audit's).
3. **2025-only, date-keyed filter.** `S2_MANUAL_DATES_2025` could not express the multi-year record
   and mis-keys the two-orbit overlap by date rather than scene.

## Decision

1. **The audited per-scene product is the single canonical S2 ROI source.**
   `data/processed/satellite/sentinel2/s2_roi_means_by_scene.parquet` (built by
   `scripts/process_sentinel2_roi.py` from raw unmasked stacks on the corrected tower grid, joined to
   the audit verdicts) is what the canonical daily table reads. `aggregate_daily_streams_2025.py`
   takes `verdict == clear`, the **100 m** ring, and drops `S2_MANUAL_DATES_2025`.

2. **Headline scope stays 2025-only for now.** App B / App C do within-2025 event-change, so the
   canonical daily S2 stream stays 2025 this pass. The full 2017–2026 audit lives in the parquet and
   is available for multi-year baselines/sensitivity; widening the headline window is a separate
   science decision, not part of this promotion.

3. **The old S2 method is retired, not left to coexist.** `scripts/process_sentinel2.py`, the S2
   re-reduce branch of `export_all_streams_2025.py`, and `S2_MANUAL_DATES_2025` are deleted (git
   history preserves them, as with `process_sentinel1.py`). The obsolete processed outputs
   (`*_indices.nc`, `sentinel2_filtered_2025.csv`) move to
   `data/processed/satellite/expired_old_method/`. The raw GEE download
   (`download_sentinel2.py`, `COPERNICUS/S2_SR_HARMONIZED`, unmasked) is unchanged — only the ROI
   product and clear-sky filter change.

4. **The analysis contract is updated.** `analysis_config.yaml` replaces `s2_clear_sky_dates` /
   `s2_source: gee_per_pass` with the audited-product + `verdict == clear` + 100 m ring convention,
   bound to this ADR.

5. **The audited product carries the full science-index menu.** The inspector's `sentinel2_roi.py`
   only computed 8 hand-written indices; the headline consumes ~12 (red-edge, water, senescence,
   kNDVI). The same `spyndex` menu the retired export used is ported into `sentinel2_roi.py`
   (`science_index_arrays`, index-then-ROI-mean), so the parquet is a drop-in superset. Two
   convention changes are made deliberately, both improvements over the retired path:
   - **kNDVI uses the canonical per-pixel sigma = 0.5·(N + R)** (Camps-Valls et al. 2021), so
     `kNDVI = tanh(NDVI**2)` and is monotonic in NDVI. The retired export used a single nonstandard
     global `median|N - R|` sigma over the whole datacube, which flattened kNDVI's seasonality.
   - **Spatial std is dropped.** No consumer reads `s2_*_std`; the daily table carries `s2_*_mean`
     only.

## Consequences

- **App B and App C headline numbers move** — new ROI centre, unmasked values, and the corrected
  clear set all feed `dehar_daily_2025.parquet`. This is accepted: the old numbers were computed on a
  wrong coordinate over masked pixels, so they were never right. Both stages are re-run after the
  rewire and their outputs reflect the corrected data.
- There is exactly one S2 clear-sky method and one S2 ROI product in the repo; the two-method
  coexistence ADR 0009 tolerated is ended.
- The clear-sky provenance of every scene is a committed artifact (`s2_audit.csv`); regenerating the
  canonical daily S2 is `process_sentinel2_roi.py` → aggregate, no hidden date list.
- Re-download of the raw archive remains a manual, GEE-authenticated step (not reproducible
  headlessly); the audited product and daily table are reproducible from the committed raw + audit.
