# 0003 — PAI hinge vs hemi: same inversion, different angle and footprint (not a sampling-count artifact)

- Status: accepted; finding #4 superseded by [0004](0004-leaf-tilt-rotation-correction.md) (2026-06-25);
  the "dedicated hinge scan = dense 57.5° early-warning / keep both never-averaged series" framing
  superseded by [0005](0005-leaf-updrift-seasonal-correction.md) (2026-06-26)
- Date: 2026-06-23

> **Superseded in part.** Finding #4 below measured the hinge scan at "~59 deg" using the upstream
> *offset* tilt transform. ADR 0004 shows that ~59 deg was largely an artifact of that crude offset
> (it adds the full ~2 deg tilt to every beam); a proper per-beam rotation puts the hinge scan back
> at ~57.5 deg. The shot-count and azimuth findings (#2, #3) and the footprint component still hold.
> The committed PAI parquets were regenerated with the rotation; the numbers quoted here reflect the
> pre-fix offset data, retained at `data/processed/proximal_rs/leaf_pre_tilt_fix_20260625/`.

## Context

Part A rests on keeping PAI as **two never-averaged series** — `pai_hinge_hinge_mean_m2m2`
(fast/early-warning) and `pai_hemi_hi_hinge_mean_m2m2` (slow/bulk), config
`streams.pai.series: [hinge, hemi_hi]`. Both are produced by the **same** Jupp (2009)
hinge-angle inversion (`-1.1 * ln(Pgap)` at the 57.5 deg ring,
`lib/pylidar-tls-canopy/.../plant_profile.py:calcHingePlantProfiles`); they differ only in which
**scan** supplied the ring. GitHub issue #6 asked whether a *processing* artifact — chiefly the
much higher shot density of the dedicated hinge scan (~34k beams in the 55-60 deg ring vs ~8.8k
for hemi_hi) — could manufacture the early hinge drop and thereby the headline conclusion.

The audit (`paper/90_sensitivity/pai_*.py`, figures in `outputs/appendix/`) found:

1. **The inversion is correct and matches upstream.** `leaf_io.py` is byte-identical to
   `armstonj/pylidar-tls-canopy`; `plant_profile.py` differed only by one commit
   (`c1c4e86`, "nhbins -> len"). For the DE-Har binning (`MAX_H/HRES = 25/0.5`, zenith `65/5`,
   azimuth `360/45` — all exact integers) that fix is **numerically inert** (it only bites on
   non-integer ratios), so our numbers already equal upstream's.
2. **Not shot count.** Subsampling the hinge ring down to hemi_hi density closes **-0.1 %** of
   the hinge-hemi gap (`pai_subsample_hinge_to_hemi.py`, validated to 4 decimals against the
   parquet). Reducing beams 3.85x does essentially nothing.
3. **Not azimuth-sampling bias.** At matched zenith *and* matched azimuth the ratio is unchanged
   (0.51 = 0.51; `pai_mechanism_azimuth.py`).
4. **It is effective angle + footprint.** The dedicated hinge scan does **not** sit at 57.5 deg:
   instrument leveling (~1.9 deg tilt from the `Tilt` header) plus the mechanical setpoint put
   its 5 rings at ~58.4-60.2 deg (mean ~59.2 deg), and the 5 deg-wide bin lumps them under the
   "57.5 deg" label (one ring even spills past 60 deg and is dropped). hemi_hi genuinely spans
   55-60 deg, mean ~57.4 deg. Even at an identical narrow zenith band the hemi scan reads ~2x
   less gap (a real footprint/resolution effect). Matched zenith+azimuth reproduces the **full**
   ~0.75 PAI gap (`pai_pgap_vs_zenith.py`, `pai_ring_sensitivity.py`).
5. **Two incidental facts.** `pai_*_weighted_mean` ≡ `pai_*_hinge_mean` for the canopy **total**
   (the solid-angle profile is rescaled to `max(HingePAI)`, `plant_profile.py:270`); only the
   PAVD/vertical shape differs. `pai_hinge_linear_mean` is **degenerate** (~10, single-angle
   regression is ill-posed) and must never be consumed.

The "is it a sampling artifact" question therefore resolves to **no**. What remains is a naming
hazard (both columns say `hinge`) and a measurement caveat (the two series are different-angle,
different-footprint measurements, and the hinge number is internally fragile at peak canopy where
`Pgap` nears the `1e-5` log floor). Biological interpretation of the early signal is deliberately
out of scope here.

## Decision

- **Keep both series, unchanged**, and keep the hinge-angle inversion for both. The early/strong
  hinge response is **not** a shot-count or azimuth-sampling artifact.
- **Pin the vendored library** at `c1c4e86` (adopts the `nhbins -> len` fix as insurance against
  a future non-integer `MAX_H/HRES`; numerically inert today). The parent repo records this
  submodule commit.
- **Frame the contrast as sampling geometry, not method**: hinge-scan vs hemi-scan PAI under the
  *same* hinge-angle inversion differ by **effective angle (~59 deg vs ~57.4 deg) + footprint**,
  not by inversion choice. See `CONTEXT.md` ("Proximal PAI").
- **Treat `pai_*_weighted_mean` as redundant with `pai_*_hinge_mean`** at the total level (use
  Weighted only for the PAVD); **never consume `pai_hinge_linear_mean`**.
- **Lock the maths with tests** (`tests/test_plant_profile.py`): hinge formula + log floor,
  linear model recovery, the negative-slope fallback, the Weighted≡Hinge-total invariant, and the
  LEAF zenith-fold / hemi azimuth-flip transform.

## Considered Options

- **Collapse to one PAI series or average hinge+hemi.** Rejected: they measure different angles
  and footprints; averaging would destroy the very contrast Part A studies and hide the caveat.
- **Re-derive the hinge scan with a constant matched to its true ~59 deg.** Rejected for now: the
  `1.1` hinge constant is the documented Jupp (2009) value at ~57.3 deg; changing it would diverge
  from the cited method. The ~59 deg placement is recorded as a caveat instead.
- **Edit the vendored `plant_profile.py` in place.** Rejected: it is a git submodule; bumping the
  pinned commit keeps provenance clean instead of creating a permanent local fork.

## Consequences

- **+** The early-warning claim is defended against the specific artifact issue #6 raised: it is
  not sampling count or azimuth coverage.
- **+** The maths is now regression-tested; the library is pinned to a known upstream commit.
- **−** The two `_hinge_` columns are different-angle, different-footprint measurements wearing one
  label, and the hinge series is least reliable at peak canopy (Pgap near the log floor). Any paper
  text must state this; it is a measurement caveat, not a code fix.
- Evidence and figures live under `paper/90_sensitivity/outputs/appendix/`; this ADR and the
  `data/README.md` LEAF section are the durable home of the caveats (previously notebook prose).
