# 0002 — PhenoCam GCC is the canonical greenness (anglecam GCC demoted to alternative)

- Status: accepted
- Date: 2026-06-22

## Context

Until now the only canopy greenness in `dehar_daily_2025.parquet` was an **in-house GCC**
computed from the upward leaf-angle cameras' time-lapse frames
(`scripts/process_gcc_anglecam.py` → `export_all_streams_2025.py::load_phenology` →
`aggregate_daily_streams_2025.py::build_gcc`, columns `gcc_cam60..70_p90`, ensemble
`gcc_mean/std/p90`). It was a **compensatory product** built before network PhenoCam data was
available. The four headline consumers (`paper/10_partA_sensor_response/onset_bootstrap.py`,
`paper/20_appB_satellite_mlr/satellite_mlr.py`, `fig_state_vs_area.py`,
`analysis/cascade/changepoint_detection.py`) each rebuilt the anglecam ensemble **inline** from
the per-camera columns — there was no canonical greenness column and no config selector. (Two further GCC consumers,
`analysis/cascade/changepoint_detection.py` and `analysis/satellite_comparison/index_sensitivity.py`,
read a *separate legacy* CSV `dehar_daily_season_2025_filtered.csv` with an old column schema —
they are not headline `paper/` stages and are out of scope for this change.)

Standard **PhenoCam-network products** now exist for the site
(`data/raw/proximal_rs/phenocam/`, ROI `hartheim2_UN_1000`): 1-day and 3-day **GCC** and
camera **NDVI** summary files, full record 2018-12 → 2026-06 (2025 coverage: 352/365 days for
1-day `gcc_90`). These are the gold-standard, directly citable greenness products
(Sonnentag 2012; Richardson 2018).

A scientific subtlety: the PhenoCam ROI is veg-type **UN (understory)**, a top-down network
camera at 47.9338 N, 7.5981 E, whereas the anglecam GCC is the **upward** view of individually
tagged hornbeams. They sample **different canopy elements** — yet PhenoCam is now the default.

## Decision

Make **PhenoCam GCC the canonical greenness**; keep anglecam GCC in the table as an alternative.

- New `streams.greenness` config block: `gcc_source: phenocam | anglecam` (default `phenocam`)
  and a shared `phenocam_aggregation: 1day | 3day` knob (default `1day`) driving both PhenoCam
  GCC and PhenoCam NDVI. Resolved by `paper_common.greenness_col(cfg)`.
- **Canonical statistic = raw `gcc_90`** (the 90th-percentile, analog of the anglecam `_p90`),
  **not** the smoothed `smooth_gcc_90`: smoothing would bias the changepoint *timing* that
  Part A measures. `smooth_gcc_90` is carried but non-canonical.
- New `src/proximal_rs/phenocam.py` loaders + `scripts/process_phenocam.py` write the **full
  multi-year** record to `data/processed/proximal_rs/phenocam/phenocam_daily.parquet`;
  `aggregate_daily_streams_2025.py` joins the **2025 slice** into the daily table.
- Columns: `gcc_phenocam_{1day,3day}_p90` (+ `_mean`, `_std`, `_smooth_p90`, `image_count`)
  and `ndvi_phenocam_{1day,3day}_p90` (+ `_mean`, `_std`, `image_count`). Anglecam ensemble
  renamed `gcc_mean/std/p90` → `gcc_anglecam_*`; per-camera `gcc_cam*` untouched.
- 3-day products land at their **native cadence with NaN between** — never forward-filled at
  the storage boundary (matches the repo's "missing = NaN" rule).
- The three headline `paper/` GCC consumers — `paper/10_partA_sensor_response/onset_bootstrap.py`,
  `paper/20_appB_satellite_mlr/satellite_mlr.py`, `paper/20_appB_satellite_mlr/fig_state_vs_area.py`
  — are rewired to consume `greenness_col(cfg)` (Part A / App B GCC numbers move). **PhenoCam NDVI
  is made available only** (columns + `streams.ndvi_phenocam` + `paper_common.ndvi_phenocam_col(cfg)`);
  no stage consumes it yet ("future computations").

## Considered Options

- **Keep anglecam canonical, add PhenoCam as data-only.** Rejected: PhenoCam is the citable
  gold standard; the anglecam product was always a stopgap.
- **Magic `gcc_canonical_*` column whose contents depend on the selector.** Rejected: the
  stored parquet would silently change meaning with config and stop self-describing. Provenance
  stays in the column *name* (cf. ADR 0001's S1 source split).
- **Use `smooth_gcc_90` as canonical.** Rejected: pre-smoothing biases changepoint timing.

## Consequences

- **+** Greenness is now a citable, standard product; the daily table self-describes which GCC
  is which; switching products (or 1-day↔3-day) is a one-line config edit, no rebuild.
- **−** Headline Part A / App B GCC numbers change; the rewire of the four stages is part of
  this decision, not an incidental edit.
- **−** Default greenness now samples the **understory** ROI, a different canopy element than
  the anglecam hornbeam view — recorded here so it isn't a silent confound.
- Reversible: set `gcc_source: anglecam` to fall back to `gcc_anglecam_*` (still in the table).
