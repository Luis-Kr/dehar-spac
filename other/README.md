# `other/` — off-pipeline checks

Scratch space for analyses that are **not** part of the reproducible paper
pipeline (`paper/`, `scripts/`, `analysis/`). Nothing here feeds a headline
number; treat it as a lab notebook.

## LEAF 2026 cross-year double-check (Har_01 + Har_02)

`leaf_2026_doublecheck.py` — process a **2026** LEAF station (hemi-high + hinge)
and plot the PAI time series, to confirm the 2025 result reproduces.

Run:

```bash
# Har_01 — standard pylidar tilt (the originally-requested correction):
/home/lk1167/miniconda3/envs/dehar-spac/bin/python other/leaf_2026_doublecheck.py
# Har_02 — primary uses rotation (standard offset saturates its hinge, see below):
.../python other/leaf_2026_doublecheck.py --station Har_02 --tilt rotation
# Har_02 — standard-correction record, kept for comparison:
.../python other/leaf_2026_doublecheck.py --station Har_02 --tilt offset --tag stdtilt
```

How it differs from the canonical `scripts/process_leaf.py` (deliberate, for a
clean cross-year sanity check):

1. **Raw tree** — reads `/mnt/gsdata/projects/icos_har/strucnet/data/raw/<station>/2026`
   (not the committed 2025 `data/raw/` tree). `--station` = `Har_01` | `Har_02` | `all`.
2. **Geometry** — `--tilt` picks the tilt correction, with **no** up-drift re-fold:
   `offset` = the **standard pylidar correction** (upstream scalar offset,
   `LeafScanFile(transform=True)`); `rotation` = the DE-Har per-beam re-levelling
   (ADR 0004).
3. **No met context** — 2026 meteorology is not in the processed pipeline yet,
   so per-scan quality flags are skipped; the plot uses a robust daily median.

Everything method-bound (inversion grid, instrument geometry, scan-type globs,
per-type total-PAI metric) is loaded from `config/leaf_processing.yaml` so it
cannot silently diverge from the 2025 contract. The hinge series uses the
canonical 5-ring scan (`_0005_8500`); the few single-ring `_0001_8500` files
(both years, both stations) are a sparse mode that saturates the inversion and
are excluded.

### Tilt correction per station (important)

The standard scalar **offset** adds the instrument's full lean to *every* beam,
so it shifts the dedicated hinge ring off the 57.5° magic angle by the lean
magnitude:

| station | sensor | instrument lean | hinge under `offset` | primary tilt used |
|---|---|---|---|---|
| Har_01 | ESS00353 | ~1.1° | ring → 58.6°, fine | **`offset` (standard)** |
| Har_02 | ESS00351 | ~5.5° | ring → ~63°, **88% saturated (degenerate)** | **`rotation` (ADR 0004)** |

So Har_02's hinge is unusable under the standard correction; `rotation` re-levels
per beam (the shift averages to ~0 over the azimuth sweep, keeping the ring on
57.5°) and restores a clean hinge (median ~6.3, 0% saturated). Har_02 hemi-high is
robust to the lean under either correction. The standard-correction Har_02 files
are kept with a `_stdtilt` suffix for the record.

### Outputs (all in `other/`; `<station>` = Har_01 | Har_02)

| file | what |
|---|---|
| `leaf_hemi_hi_<station>_2026.parquet` | hemi-high long-format profiles (row per scan × height) |
| `leaf_hinge_<station>_2026.parquet` | hinge long-format profiles |
| `leaf_pai_daily_<station>_2026.csv` | tidy daily total-PAI medians + scan counts (the plotted data) |
| `pai_timeseries_<station>_2026.png` | the PAI time-series figure |
| `*_Har_02_2026_stdtilt.*` | Har_02 under the standard offset (degenerate hinge), kept for the record |

Primary corrections: **Har_01 = `offset` (standard)**, **Har_02 = `rotation`**.

### Result (2026-03-09 → 06-24, 213 hemi-high + ~640 hinge scans per station)

The 2025 pattern reproduces at **both** stations — a clear **spring leaf-out
rise** and **hinge total PAI ≤ hemi total PAI** by the June plateau:

| station (tilt) | mid-March (hemi / hinge) | June plateau (hemi / hinge) | June hinge<hemi gap |
|---|---|---|---|
| Har_01 (`offset`) | 4.28 / 4.04 | 5.54 / 5.23 | 0.32 |
| Har_02 (`rotation`) | 5.14 / 4.34 | 6.78 / 6.31 | 0.47 |

Har_02 carries a denser canopy (plateau ~6.8 vs ~5.5) and its hinge tracks faster
during the April–May leaf-out before settling below the hemi by June.

## LEAF 2026 hemispherical seasonal series

`fig_hemi_seasonal_2026.py` — a 2026 version of the paper figure
`paper/40_supplementary/leaf_hemispherical/fig_hemi_seasonal_series.py`, with the
**rotation** tilt fix (ADR 0004) for both stations.

Run:

```bash
/home/lk1167/miniconda3/envs/dehar-spac/bin/python \
    other/fig_hemi_seasonal_2026.py --station all
```

Differences from the paper figure (by request):

* **Two rows only** — canopy height (row 1) and binned gap fraction (row 2); the
  paper figure's third boolean-gap row is dropped.
* **4 representative clean scenes** per station, evenly spread across the season.
  2026 has no processed meteorology, so rain/fog/wind-disturbed scans are
  **inferred as outliers**: a robust LOWESS trend is fit to the hemi-high night
  (hour-20) total-PAI series, and scans deviating by more than `--k-mad` (default
  3) robust SDs are flagged and excluded; the 4 scenes are then picked from the
  clean nights. The night-scan PAI series is recomputed here with the rotation
  geometry so selection and the header PAI both match the rendering.

Picked scenes (both stations land on the same clean nights): **Mar 15, Apr 15,
May 17, Jun 18**. Outliers caught: 9 (Har_01) / 8 (Har_02) — the obvious PAI
crashes (z down to −7…−9, e.g. Mar 25, Apr 13, May 11). The header PAI shows the
leaf-out rise (Har_01 4.26→5.25; Har_02 5.06→6.72) and the gap panels show the
canopy closing from March to June.

Outputs (per station):

| file | what |
|---|---|
| `hemi_seasonal_2026_<station>.png` | the 2-row fisheye seasonal series |
| `hemi_seasonal_2026_<station>_scenes.csv` | the 4 picked scenes + outlier diagnostics |
