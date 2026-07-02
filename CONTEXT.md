# DE-Har SPAC

Domain glossary for the DE-Har (Hartheim) drought-stress SPAC analysis. Defines the
project-specific terms whose meaning is not obvious from the code, so the same word means the
same thing across sensors, scripts, config, and the paper.

## Language

### Greenness

**PhenoCam GCC**:
Green Chromatic Coordinate from the network PhenoCam (RGB), delivered as standard 1-day and
3-day summary products. The **gold-standard, citable** canopy greenness for this project;
canonical value is the per-period 90th percentile (`gcc_90`).
_Avoid_: phenology cam, greenness index (ambiguous)

**Anglecam GCC**:
GCC computed in-house from the upward leaf-angle cameras' time-lapse frames. A
**compensatory legacy product** built before PhenoCam data was available; retained for
comparison/sensitivity, no longer the default greenness.
_Avoid_: phenology GCC

**PhenoCam NDVI**:
Camera-derived NDVI from the PhenoCam's paired RGB+IR images (1-day and 3-day products). A
proximal NDVI, always labelled `phenocam` to distinguish it from the satellite `s2_ndvi`.
_Avoid_: camera NDVI, proximal NDVI (unqualified)

**Canonical greenness**:
The single greenness series the headline stages consume, selected in
`paper/config/analysis_config.yaml` and resolved by `paper_common.greenness_col()`. Default =
PhenoCam GCC, 1-day aggregation.

### Onset & breakpoints (Part A)

**Changepoint**:
Any raw shift in level detected in a stream by the changepoint search. Mechanism-level and
unfiltered — most are not interesting on their own.
_Avoid_: break, shift

**Breakpoint**:
A *qualifying* changepoint: one whose jump is in the stress direction and large enough to clear
the stream's own noise floor. The unit whose full-season distribution Part A studies (where do
breakpoints cluster over June–September?).
_Avoid_: changepoint (raw, unfiltered), onset (the single elected one)

**Onset**:
The single breakpoint elected to represent a stream's response — either the earliest
(first-departure, the headline cascade) or the largest (acute-event). The headline unit; a
breakpoint collapsed to one date.
_Avoid_: breakpoint, changepoint

### Proximal PAI (LEAF / TLS)

The word "hinge" rides on **two independent axes** — a *scan mode* and an *inversion metric*.
Keep them separate; the headline column name `pai_hinge_hinge_mean_m2m2` collides them, and a
swap would invert the early-warning story.

**Hinge scan**:
A LEAF acquisition *mode* that stares at a fixed encoder setpoint (nominally the 57.5° zenith
ring) and sweeps all azimuths, sampling that ring **densely**. The dedicated single-angle scan.
Because it shares the scan head's drifting "up" (see **Hinge "up"-drift**) its *true* zenith
walks off 57.5° after early May and cannot be recovered (single parked angle), so it is **kept
only as a comparison/sensitivity series**, not the canonical 57.5° source.
_Avoid_: hinge (unqualified — could mean the metric), hinge angle, treating the dedicated hinge
scan as a fixed 57.5° measurement after early May

**Hemi scan**:
A LEAF acquisition *mode* that images the whole sky dome (fisheye). Two resolutions:
`hemi_hi` (dense) and `hemi_low` (sparse). At the 57.5° ring it samples **far fewer** beams
than the hinge scan.
_Avoid_: hemispherical (unqualified), hemi (unqualified)

**Hinge-angle PAI**:
The *inversion metric* `-1.1 · log(Pgap)` evaluated at the single 57.5° ring (Jupp et al.
2009). A method, not a scan. The **canonical** 57.5° measurement is taken from the
**up-corrected `hemi_hi` scan's 57.5° ring** (which samples a true, season-stable 57.5° once
the "up"-drift is corrected); the dedicated **hinge scan**'s same-named inversion is retained
only for comparison. Contrast metrics: **Linear PAI** (multi-angle regression over all zeniths)
and **Weighted PAI** (solid-angle weighted). On the hemi the Hinge-angle and Weighted inversions give
the **same canopy total** (Weighted rescales to `max(HingePAI)`); they differ only in PAVD shape, so
**Linear PAI is the only inversion with a distinct total**.
_Avoid_: hinge PAI (ambiguous with the scan), HingePAI as a synonym for the hinge scan,
sourcing the canonical 57.5° metric from the dedicated hinge scan, treating hemi Weighted and
Hinge-angle totals as different series

**Canonical proximal PAI (bulk)**:
Part A consumes a **single** proximal PAI series — the canopy total of the **up-corrected
`hemi_hi`** scan (`pai_hemi_hi_hinge`, which **equals** `pai_hemi_hi_weighted` by construction: on
the hemi the Hinge-angle and Weighted inversions give the *same total*, differing only in PAVD
shape). The earlier **early-warning vs bulk PAI split is retired**: its early-warning arm was the
dedicated hinge scan's dense 57.5° sampling, now shown to be a drifting-angle artifact (see **Hinge
"up"-drift**). The dedicated hinge scan (`pai_hinge_hinge`) is retained only as a flagged comparison
and a possible future early-warning arm; the cascade's fast signal is carried by the water sensors
(predawn SWP, VOD), not a second PAI.
_Avoid_: any "hinge-vs-hemi" or "hinge-vs-weighted" early-warning-vs-bulk **PAI** contrast (the hemi
Hinge and Weighted totals are identical; the dedicated hinge scan is compromised)

**Hemi "up"-point drift**:
The hemi reconstruction recovers each beam's zenith by folding the mirror sweep about its "straight
up" point. At DE-Har that point **drifts over the season** as the scanner settles in the sandy soil
(about 0° offset in April, growing to about +24° by December). Left uncorrected it mis-registers the
two half-sweeps, pushes returns below the ground (concentric rings near the scanner), and biases the
leaf-off **autumn/winter PAI low**. Corrected by re-folding each scan about an **"up" reference**
(see *Up reference* below). The **same drift shifts the dedicated hinge scan** off 57.5° (it shares
the scan head's encoder home), but the hinge scan cannot be re-folded back (single parked angle) —
hence the canonical 57.5° comes from the corrected hemi.
_Avoid_: trusting raw hemi 3-D geometry off the 57.5° ring for scans after early May.

**Up reference**:
The per-beam "straight up" a hemi scan is re-folded about to undo the *Hemi "up"-point drift*. Two
distinct sources, kept separate because they can disagree by several degrees in the leaf-off season:

- **Manual up reference** — the hand-verified estimate and the **canonical "up" (ADR 0006)**: a human
  aligns the folded-fisheye seam per scan in the up-inspector app and records the "up" that makes the
  canopy coherent (`up_manual`, 300+ scans). Resolved per scan and interpolated in time for scans no
  one inspected (`up_resolve: scan`), so it keeps abrupt re-settling steps the smoothing loses.
  Independent DHP LAI confirms it is at least as good as the smoothed proxy and best-matches the
  leaf-off level.
- **Smoothed up lookup** — the automatic estimate: per-scan seam fits across all hemi scans, robust
  per-day median, LOWESS-smoothed over the season. Robust to single-scan seam failures; the ADR 0005
  method, now a **reproducible fallback** (`up_resolve: date`). Can lag real re-settling steps because
  the smoothing blurs them.

_Avoid_: "the up curve" (unqualified — say *smoothed up lookup* or *manual up reference*); calling the
manual curve "self-calibration" (that was the rejected fragile per-scan seam-fit, not the hand check).
