# 0009 — Sentinel-2: raw archive is re-exported unmasked; a per-scene hand audit is the cloud filter

- Status: accepted
- Date: 2026-07-02
- Relates: issue #8 (satellite data audit + inspection tool); ADR
  [0001](0001-sentinel1-sentle-weekly.md) (S2 stays GEE per-pass, S1 = sentle). Shares the
  **manual-over-automated** stance of [0005](0005-leaf-updrift-seasonal-correction.md) /
  [0006](0006-leaf-updrift-manual-up-canonical.md) (the hand-verified up reference is canonical).

## Context

The Sentinel-2 raw archive at DE-Har was in **two incompatible regimes**. Years **2017–2024** were
exported from GEE with SCL + s2cloudless masking on: cloudy pixels are `NaN`, so each scene is only
45–71 % filled. Years **2025–2026** were re-exported with masking **off**
(`CLOUD_PROB_THRESH=100`, `_mask_clouds` disabled in `src/satellite/download_sentinel2.py`): every
scene is 100 % filled, clouds and all. A per-scene ROI mean therefore means a different thing in each
half of the record, which contaminates any multi-year baseline.

Automated masking is also the wrong tool for this site. SCL/s2cloudless **over-mask** pixels adjacent
to thin cloud and misclassify bright leaf-off / snow, silently dropping otherwise-usable clear pixels
over the tower ROI, while still **leaking thin cirrus** that a human eye catches immediately. The
only usability decision that matters is *"is the tower ROI clear on this scene?"* — a judgement the
project already makes by hand elsewhere (the manual up reference, ADR 0006; the hardcoded
`S2_MANUAL_DATES_2025` list of 33 clear-sky dates in `scripts/aggregate_daily_streams_2025.py`).

That hardcoded list is 2025-only, date-keyed, and buried in an aggregation script sourced from a
notebook. Hartheim sits in the overlap of two S2 relative orbits, so a calendar *date* can carry two
scenes with different geometry and cloud — a date is the wrong unit.

## Decision

1. **Re-export the whole S2 archive unmasked.** All years (2017 → present) are downloaded from
   `COPERNICUS/S2_SR_HARMONIZED` with cloud masking disabled, so every scene carries full raw
   reflectance and the record is uniform. (2025–2026 are already in this state; 2017–2024 need a
   re-run with `START_DATE="2017-01-01"`.)

2. **A per-scene hand audit is the single cloud filter.** Every scene is reviewed by eye over the
   tower ROI in the S2 inspector (`other/s2_inspector/`, built for issue #8) and assigned a verdict —
   **clear / partial / cloud** (+ snow/shadow/haze flags + note) — written **per scene, keyed by
   acquisition datetime** to `other/s2_inspector/s2_audit.csv`. This CSV supersedes
   `S2_MANUAL_DATES_2025` (which becomes the 2025 seed). SCL/s2cloudless is **not** trusted for
   usability. Headline consumers take `verdict == clear` only; `partial` is available for sensitivity.

3. **The audit judges the ROI, not the tile.** A scene clear over the tower but cloudy at the tile
   edge is `clear`. True-color and a SWIR false-color rendering plus a computed ROI cloud-score assist
   the eye; the score pre-sorts scenes but never sets the verdict.

4. **The headline pipeline is not rewired yet.** `process_sentinel2.py` emits per-scene ROI means
   (all rings × indices, clear-only and all) to `data/processed/satellite/sentinel2/` for inspection,
   but `paper/config/analysis_config.yaml` (`s2_clear_sky_dates`) and
   `scripts/aggregate_daily_streams_2025.py` keep reading the existing 33 dates until the multi-year
   audit is trusted. Promoting `s2_audit.csv` to canonical — gated on the 2025 audit reproducing or
   deliberately superseding those 33 dates — is a **follow-up** and will get its own ADR + runbook.

## Consequences

- ~480 scenes across 2017–2026 must be reviewed by hand once; thereafter the CSV is append/update and
  the provenance of every clear-sky decision is a committed artifact (like the up-inspector CSV).
- The raw archive grows (unmasked scenes are ~2× the masked size) and `data/raw` re-download is a
  manual, GEE-authenticated step — not reproducible headlessly on the analysis server.
- Subjectivity enters the clear-sky call; mitigated by the graded verdict, the recorded score/flags,
  and a single reviewer, exactly as for the manual up reference.
- Until the pipeline is rewired, `s2_audit.csv` and the existing 33-date list coexist; the audit does
  not silently move any headline number.
