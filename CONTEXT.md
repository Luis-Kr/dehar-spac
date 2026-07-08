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

### Satellite clear-sky (Sentinel-2)

**S2 scene**:
A single Sentinel-2 overpass of the DE-Har tile, keyed by its exact acquisition *datetime*. The
**unit of the clear-sky audit** — because Hartheim sits in the overlap of two relative orbits, a
calendar *date* can carry **two scenes** with different view geometry and cloud cover, so a scene,
not a date, is judged.
_Avoid_: "S2 date" as the audit unit (a date may bundle two scenes); "pass"/"image" (unqualified).

**Clear-sky audit**:
The **hand review of every S2 scene** for cloud/quality over the tower ROI, recorded as an
*audit verdict* in `s2_audit.csv`. It is the project's **single cloud filter** — the raw archive is
re-exported with SCL/s2cloudless masking **off** so the eye, not an automated mask, decides which
scenes are usable (same manual-over-automated stance as the *Manual up reference*). It is the
**canonical S2 source** (ADR 0012): the daily table reads the audited per-scene ROI product and takes
`verdict == clear`; the old `S2_MANUAL_DATES_2025` date list and the masked `*_indices.nc` re-reduce
are removed.
_Avoid_: "cloud mask" (that is the rejected automated SCL path); treating SCL/s2cloudless as the
usability decision.

**Audit verdict**:
The per-scene ROI judgement, a three-level ordinal: **clear** (ROI unobscured — usable in headline),
**partial** (thin haze / cloud edge in ROI — sensitivity only), **cloud** (ROI obscured — reject).
Headline stages consume `clear` only. Carries optional flags (snow, shadow, haze) and a note.
_Avoid_: binary pass/fail (loses the *partial* middle ground); a *partial* scene silently entering a
headline number.

**S2 ROI ring**:
A circular buffer around the tower over which S2 pixels are averaged. Canonical science radius is
**100 m** (`analysis_config.yaml`); **30 m** is the GNSS-T VOD-footprint match, **50 m** a sensitivity
radius. The **500 m** ring is **QA / spatial-context only** — at ~78 ha it mixes the dying-pine stand,
broadleaf edges, roads and floodplain, so its mean is not a tower-stand canopy value and never feeds a
headline number.
_Avoid_: treating the 500 m ROI mean as a canopy signal; changing the canonical radius away from 100 m
without updating `analysis_config.yaml` and the bound citation.

### Radar backscatter (Sentinel-1)

**S1 backscatter (VV, VH)**:
Dual-pol σ⁰ from Sentinel-1 IW GRD. All averaging and speckle filtering happen in **linear power**
(the playground starts from `S1_GRD_FLOAT`); **dB** (10·log₁₀) is only a display/interpretation
transform applied *last*. The `angle` band is the per-pixel incidence angle.
_Avoid_: averaging or speckle-filtering in dB (the current GEE download's 15 m focal median is done
in dB — a bug the playground exists to expose); calling the GEE per-pass export "raw" (it is already
angle-normalised + speckle-filtered).

**Cross-ratio (CR)**:
`VH/VV` in linear (equivalently `VH_dB − VV_dB`). The vegetation-sensitive backscatter ratio; being a
ratio it is **offset-invariant**, so a constant angle-normalisation shift cancels out of it.
_Avoid_: "ratio" unqualified; the inverted `VV/VH`.

**SPAN / RVI**:
**SPAN** = total power `VV + VH` (linear). **RVI** = Radar Vegetation Index `4·VH/(VV + VH)` (linear).
_Avoid_: computing either in dB.

**Orbit handling**:
Ascending and descending are **never blended**, and distinct **relative orbits** are kept separate —
their geometry and incidence angle differ. DE-Har is covered by **two relative orbits per direction**,
at different incidence angles over the ROI: ascending 15 = 32.2°, 88 = 41.8°; descending 66 = 44.0°,
139 = 34.8°. Within a single relative orbit the incidence angle barely varies over the ROI, so **angle
normalisation is a near-constant offset there**; it only changes the signal when combining orbits/swaths
(a ~10° step here). The **descending** pass is the ~05:41 UTC **morning** acquisition (overnight-
rehydrated / near-predawn state — pre-sunrise only in the winter half-year); **ascending** is the ~17:22
UTC evening/depleted pass. **Canonical S1 = the transparent GRD_FLOAT product** (ADR 0011): a single
relative orbit per direction — **descending 139** (morning, ~35°) + **ascending 15** (evening, ~32° —
the asc orbit closest in incidence to 139, so the two form a comparable diurnal pair). Linear-domain
processing, ROI mean over 50/100/200 m, temporal moving-mean windows raw + w3…w13; the **headline
series is 100 m, w5** (~30 d). The **sentle** weekly composite (ADR 0001) and the old angle-normalised
GEE per-pass export are **retired** — their pipelines were opaque; this one is fully in-repo.
_Avoid_: mixing asc+desc or multiple relative orbits into one series; copying another site's "orbit 66 =
36°" (relative-orbit incidence is site-specific); assuming angle-norm matters for a single fixed-orbit
ROI time series (there it is ~a DC shift the z-scores remove); "sentle" or the old `*_indices.nc` as an
S1 source (both removed).
