# LEAF up-vector inspector

A small local web app to hand-check and correct the seasonal **up-drift** of the
LEAF hemispherical scans (ADR 0005, issue #11). For one representative non-noisy
scan per day it draws the canopy-height fisheye and lets you drag an **up** slider;
the scan is re-folded live in the browser and the fisheye redraws instantly. When
the fold is right the disk fills coherently and the canopy structure is sharp;
when the up is wrong the structure smears and the horizon ring mis-aligns. Submit
stores your chosen up to a CSV.

## Run

```
/home/lk1167/miniconda3/envs/dehar-spac/bin/python other/up_inspector/server.py
# then open http://127.0.0.1:8000
```

Options: `--host --port --max-points --night-hour (default 23)`.
Plotly is vendored in `static/` so it works offline.

## Two modes (toggle at the top of the Day card)

- **New scenes** — click through *fresh* uncorrected scans (the default flow below).
- **Redo done** — step through the scans you have already corrected (sorted in time) to
  fix mistakes. Each loads its stored `up_manual` preselected; adjust and Submit to
  **overwrite** that row (the count stays the same). `[` `]` / Submit / Skip move through
  the corrected list.

## Rounds (it gets denser each pass)

The CSV accumulates **one row per scan** (keyed by filename), and each *New scenes* pass
picks a *fresh* scan per day, so coverage grows the more you run it:

- **Default up** — once the CSV has any corrections, the default for a scan is the
  **nearest-in-time `up_manual`** (your manual curve), not the smoothed seasonal value.
  The label reads `default (manual)`; with an empty CSV it falls back to
  `default (smooth)`. So round 2 onward you are validating round 1's curve.
- **Which scan** — the representative is a clean scan **not yet corrected**, chosen as
  far as possible in time-of-day from scans already checked that day. Round 1 lands on
  the night scan; round 2 lands on a different one (often midday), etc.
- **Resume / progress** — startup jumps to the first day whose representative is not yet
  corrected; the dropdown marks corrected scans with `✓`. Accept a scan by skipping
  (no save); only **corrections you submit** are written, so re-folds that already fit
  cost nothing and the picture sharpens where it disagrees.

## How it works

- **Index** — `data/processed/proximal_rs/leaf/leaf_hemi_hi_2025.parquet` is grouped
  per scan (datetime, filename, `quality_all`, PAI, timestamp). Days are listed with all
  8 three-hourly scans; the representative is the fresh-scan / round logic above. The
  default up is the nearest-in-time correction, falling back to the smoothed seasonal
  value (`leaf_up_daily_2025.csv` `up_smooth_deg`) when the CSV is empty.
- **Re-fold (client-side, exact)** — the server sends each beam's raw encoder angle
  `s`, raw azimuth, and `range1/range2`. The browser computes, per beam,
  `zen = |s − up|`, flips azimuth by π where `s < up`, and `h = range·cos(zen) + 1.5` —
  identical to `dehar.proximal_rs.leaf.recenter_leaf_data` (verified to 0 difference).
  Only the slider re-fold is needed per tick, so updates are instant (no round-trip).
  Switching scans fetches once as a compact **binary** payload (`/api/scan.bin`:
  `uint32 n` then `float32 s,az,r1,r2`) — `--max-points` returns (default 120 k,
  ~2 MB) so the fisheye is as dense as the published reference figure.
- **Fisheye** — zenith as radius (N up, clockwise), height as colour (turbo). Two
  representations via the **Points / Gridded** toggle:
  - *Points* — dense WebGL scatter of every return (fine markers, like the reference).
  - *Gridded* — mean canopy height in fine (~1.3°) fisheye cells with bilinear
    smoothing, empty cells transparent. A smooth continuous field that sharpens the
    **seam** (the N–S meridian, vertical line) where the two folded half-sweeps meet: at
    the right up they join smoothly; a wrong up shows a step.
  Default colour limits are the 2nd/98th height percentiles; both are adjustable. Toggle
  "show below-horizon beams" to extend past the 90° horizon (the downward half, where a
  bad fold punches through the ground).

## Controls

- **Day nav** `◀ ▶` or `[` `]`; tags show clean-scan count / all-noisy / corrected.
- **Scan this day** dropdown — pick a different scan (`✓` = already corrected, `rep` = this
  round's pick).
- **Up** slider, ±1 / ±0.1 buttons, number box; arrow keys nudge (Shift = ±1);
  `reset` returns to the default (nearest manual). The pill shows Δ from that default —
  i.e. how far this scan deviates from your manual curve's prediction.
- **Colour scale** low/high percentiles + `rescale` (recompute limits at current up).
- **Submit & next** (`Enter`) saves this scan and jumps to the next uncorrected
  representative; **Skip** (`S`) accepts/moves on without saving.

## Output

`other/up_inspector/up_manual_corrections.csv` — one row per **scan** (keyed by filename,
latest submit wins), so corrections accumulate across rounds:

```
date, filename, scan_hour, up_default_smooth, up_manual, delta,
color_p_lo, color_p_hi, note, saved_at_utc
```

`up_default_smooth`/`delta` stay relative to the smoothed seasonal curve (round-comparable);
the UI's live Δ is relative to your manual curve. Feed `up_manual` (with `filename`'s
timestamp) back as a per-scan up lookup to replace the smoothed `up_smooth_deg`.

## Make it canonical (after inspecting)

This CSV is the **canonical** LEAF "up" basis (ADR 0006). After inspecting more scans,
propagate the change with the runbook
[`docs/runbooks/update-canonical-leaf-up.md`](../../docs/runbooks/update-canonical-leaf-up.md).
In short: rebuild the lookup (`scripts/build_leaf_up_manual_lookup.py`), reprocess
(`scripts/process_leaf.py`), rebuild the aggregates, re-run the PAI stages, and **commit
this CSV** (it is the provenance of the canonical data).
