# 0004 — LEAF tilt correction: adopt a true per-beam rotation (re-frames 0003's effective angle)

- Status: accepted; extended by [0005](0005-leaf-updrift-seasonal-correction.md) (2026-06-26), which
  composes a seasonal "up"-drift re-fold on top of this rotation and re-keeps the offset as a sensitivity
- Date: 2026-06-25
- Supersedes (in part): [0003](0003-pai-hinge-hemi-not-sampling-artifact.md) finding #4 and its
  "keep the offset transform" decision.

## Context

GitHub issue #10 asked to (1) move the LEAF processing out of the notebook into a single
config-driven, multiprocessing CLI, and (2) **solve the tilt** in the scans. Part 1 produced
`scripts/process_leaf.py` + `src/proximal_rs/leaf.py` + `config/leaf_processing.yaml`, reproducing
the committed parquet bit-for-bit. Part 2 is this decision.

The LEAF instrument is not level: its `Tilt` header (an accelerometer reading of the world-vertical
up-direction in the instrument frame) shows a steady **~2.0 deg lean at azimuth ~130 deg** at
DE-Har. The vendored library's transform (`leaf_io.py:131-135`, applied whenever `transform=True`)
"corrects" this by a **scalar offset**: it adds the tilt's zenith and azimuth to **every** beam
equally. That is geometrically wrong — a tilt changes a beam's zenith by `~tilt * cos(az - tilt_az)`,
which is `+tilt` only for beams pointing toward the lean, `-tilt` for the opposite side, and ~0
perpendicular.

The consequence is specific and measurable on the **hinge scan** (5 rings × 8500 azimuth, fired at
the instrument's 57.5 deg mechanical setpoint):

| Hinge ring, mean effective zenith | value |
|---|---|
| no correction (true setpoint) | **57.45 deg** |
| offset (upstream, what 0003 measured) | **59.25 deg** |
| rotation (this ADR) | **57.51 deg** |

So ADR 0003's finding #4 — that the dedicated hinge scan "sits at ~59 deg (leveling tilt +
setpoint)" — was reading an **artifact of the crude offset**, not the true geometry. The offset
adds the full ~2 deg to a scan that is already at 57.5 deg, manufacturing the ~59 deg. The
footprint/resolution part of 0003's finding stands; the **angle** part does not.

## Decision

- **Adopt `transform.method: leaf_tilt_rotation` as the default** in `config/leaf_processing.yaml`.
  It applies the minimal 3-D rotation that brings the measured up-vector to vertical
  (`dehar.proximal_rs.leaf.leveling_rotation` / `level_leaf_data`), re-levelling each beam by
  `~tilt * cos(az - tilt_az)` (mean zero over an azimuth sweep) instead of a uniform `+tilt`.
- **Regenerate** the three committed parquets (`leaf_{hinge,hemi_hi,hemi_low}_2025.parquet`) with
  rotation. The pre-fix (offset) files are retained at
  `data/processed/proximal_rs/leaf_pre_tilt_fix_20260625/`.
- **Keep the `1.1` hinge constant and the 57.5 deg ring**: rotation puts the hinge scan back **on**
  the Jupp (2009) magic angle, so the documented constant now applies correctly (it did not under
  the offset). The earlier rejection in 0003 of "re-deriving the hinge with a matched ~59 deg
  constant" is moot — there is no longer a ~59 deg to match.
- **Keep both never-averaged series** (`hinge`, `hemi_hi`); that decision from 0003 is unchanged.

## Measured effect (full 2025)

- Hinge effective zenith **59.25 -> 57.51 deg** (back on the magic angle).
- Hinge total `HingePAI` **essentially unchanged** (Δ mean -0.04 m²m⁻²): the early-warning series is
  robust to the fix — the correction's azimuthal symmetry means the ring's mean `Pgap` barely moves.
- `hemi_hi` total `WeightedPAI` **+0.20 m²m⁻² (~+4%)**.
- Offset mode remains available (`method: leaf_tilt_offset`) and **byte-identical** to the pre-fix
  parquet, so 0003's audit numbers stay reproducible.

## Considered Options

- **Keep the offset (status quo of 0003).** Rejected: it is geometrically incorrect and biases the
  hinge scan off the magic angle on which the whole hinge-angle inversion rests.
- **New ADR vs. editing 0003 in place.** Chose a new ADR that supersedes 0003's finding #4, per the
  immutable-record convention; 0003 keeps a status pointer here.
- **Edit the vendored `leaf_io.py` transform.** Rejected: it is a pinned submodule (0003). The fix
  lives in our wrapper instead, leaving provenance clean.

## Consequences

- **+** The hinge scan is measured at its intended 57.5 deg; the "~59 deg effective angle" caveat is
  removed, not just documented.
- **+** The fix is regression-tested (`tests/test_leaf_processing.py`: `R·up = vertical`,
  orthonormality, identity-when-level, hinge re-centring on a real scan), and the offset path is
  proven byte-identical to the pre-fix data.
- **−** Headline PAI changes (hemi_hi ~+4%; hinge total ~flat but per-scan shifts). **Downstream
  artifacts are now stale and must be rebuilt to propagate**: the daily/streams aggregates
  (`scripts/export_all_streams_2025.py` → `aggregate_daily_streams_2025.py`) and every paper stage
  that consumes PAI (Part A onset cascade, App C VOD correction). The `paper/90_sensitivity/pai_*.py`
  audit scripts and ADR 0003 figures were computed on the offset data and should be re-read against
  `leaf_pre_tilt_fix_20260625/` or re-run.
- **−** ADR 0003's prose and `data/README.md` §10 caveat are updated to point here; any paper text
  asserting the hinge scan "sits at ~59 deg" must be revised.
