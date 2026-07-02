# Runbook — update the canonical LEAF "up" (manual reference)

**When to run this:** you inspected more scans in the up-inspector app (so
`other/up_inspector/up_manual_corrections.csv` changed) and want the change to
flow into the canonical LEAF PAI and everything downstream.

**Why it works this way:** the canonical up-drift re-fold folds each hemi scan
about the **hand-verified "up"** you recorded per scan (ADR 0006). That CSV is
the *source of truth*; the pipeline reads a derived per-scan lookup. So an update
is always: **edit the CSV → rebuild the lookup → reprocess → propagate.** See
[ADR 0006](../adr/0006-leaf-updrift-manual-up-canonical.md) for the decision and
[ADR 0005](../adr/0005-leaf-updrift-seasonal-correction.md) for the drift itself.

All commands use the env Python by absolute path (`conda run` is broken here):

```
PY=/home/lk1167/miniconda3/envs/dehar-spac/bin/python
```

## The one-shot (copy-paste)

```bash
cd /mnt/data/lk1167/projects/dehar-spac
PY=/home/lk1167/miniconda3/envs/dehar-spac/bin/python

# 0. (do this first, interactively) inspect more scans:
#    $PY other/up_inspector/server.py   → http://127.0.0.1:8000 → Submit scans
#    This appends/overwrites rows in other/up_inspector/up_manual_corrections.csv.

# 1. rebuild the per-scan manual "up" lookup from the CSV
$PY scripts/build_leaf_up_manual_lookup.py

# 2. rebuild the canonical LEAF parquets (hemi_hi, hemi_low, hinge) with the new up
$PY scripts/process_leaf.py --scan-type all --steps all

# 3. rebuild the daily / streams aggregate tables (the entry point for analysis)
$PY scripts/export_all_streams_2025.py
$PY scripts/aggregate_daily_streams_2025.py

# 4. re-run the PAI-consuming paper stages (headline numbers)
$PY paper/10_partA_sensor_response/onset_bootstrap.py
$PY paper/10_partA_sensor_response/onset_satellite.py
$PY paper/30_appC_vod_correction/vod_pai_correction.py

# 5. commit the basis + the regenerated committed artifacts (see "What to commit")
```

## Step by step (what each does)

1. **`scripts/build_leaf_up_manual_lookup.py`** — reads
   `other/up_inspector/up_manual_corrections.csv` (latest submit wins per scan),
   keys each correction by its filename's UTC scan datetime, and writes the
   canonical lookup `data/processed/proximal_rs/leaf/leaf_up_manual_perscan_2025.csv`
   (`datetime, up_deg, …`). No smoothing.

2. **`scripts/process_leaf.py`** — with `transform.up_resolve: scan` (already set
   in `config/leaf_processing.yaml`) folds each hemi scan about its exact
   `up_manual` where inspected, and about a **linear-in-time interpolation** of
   the two neighbouring manual points elsewhere (endpoints held). Overwrites the
   three canonical parquets `leaf_{hemi_hi,hemi_low,hinge}_2025.parquet`. The
   **hinge is never re-folded**, so `leaf_hinge_2025.parquet` stays byte-identical.

3. **`export_all_streams_2025.py` → `aggregate_daily_streams_2025.py`** — rebuild
   `dehar_all_streams_2025.parquet` and `dehar_daily_2025.parquet`, which carry the
   `pai_hemi_hi_*` columns headline stages read. (The `_uncorr` comparison columns
   come from the pre-up-drift snapshot `leaf_pre_updrift_fix_20260626/` and do
   **not** change.)

4. **Paper stages** — only the PAI consumers need re-running:
   `onset_bootstrap.py` + `onset_satellite.py` (Part A cascade) and
   `vod_pai_correction.py` (App C). Each writes into its own stage `outputs/`.

## What to commit

`data/` is gitignored, so the rebuilt parquets/tables are **not** committed —
they are reproducible from the CSV by the steps above. Commit:

- `other/up_inspector/up_manual_corrections.csv` — the canonical basis (provenance).
- any regenerated **committed** paper outputs your workflow tracks (figures/tables
  under `paper/*/outputs/` that are not gitignored).

## Notes / safety

- **First-time snapshot already taken.** The pre-switch smoothed-canonical parquets
  are at `data/processed/proximal_rs/leaf_pre_upmanual_20260701/`. Re-running the
  steps above overwrites the live canonical in place; it does not touch that
  snapshot.
- **Revert to the automatic smoothed "up"** (ADR 0005 fallback): set
  `transform.up_resolve: date` and `up_lookup_csv:
  data/processed/proximal_rs/leaf/leaf_up_daily_2025.csv` in
  `config/leaf_processing.yaml`, then re-run from step 2. (`leaf_up_daily_2025.csv`
  is rebuilt by `scripts/build_leaf_up_lookup.py`.)
- **Sanity check after step 2:** the leaf-off (Nov–Dec) hemi_hi total WeightedPAI
  should sit near the DHP LAI (~4), not collapse toward ~2.9 (that would mean the
  up-drift correction is off). `other/dhp_lai_vs_pai_upmanual_2025.py` redraws the
  DHP-vs-PAI check.
- **Coverage:** the lookup need not cover every scan — scans between inspected
  ones are interpolated, and scans before the first / after the last inspected
  scan hold the nearest endpoint. More inspected scans = less interpolation.
