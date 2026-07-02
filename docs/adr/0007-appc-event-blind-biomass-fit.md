# 0007 — App C biomass fit is event-blind (do not exclude the drought from the fit)

- Status: accepted
- Date: 2026-07-01
- Scope: `paper/30_appC_vod_correction/` (Application C), config `appC.fit_exclude_event`.

## Context

App C recovers a water signal from GNSS-T VOD by subtracting an independently-measured biomass term:
`water-VOD = VOD − (a + b·PAI)`, where `a,b` come from regressing daily VOD on hemi-PAI over the
season. The prototype **excluded the Aug 7–22 drought window from that regression** ("so the water
drop cannot bias the slope").

Two problems with excluding it:

1. **It makes the headline comparison unfair.** App C's question is "is concurrent high-res PAI worth
   it vs a cheap PAI-free LOESS ±30 d detrend?" But only the PAI regression got to *peek* at the event
   and drop it; the LOESS filter sees everything. That asymmetry advantages PAI for free.
2. **It smuggles in the answer.** To exclude Aug 7–22 you must already know the drought is Aug 7–22.
   The whole pitch is that PAI-corrected VOD is a **standalone** water proxy — if you need external
   stress data to place the exclusion, it is not standalone ("if we don't have the SWP data we don't
   know where the stress happens"). The event window is *forcing*-defined (soil moisture + VPD), so it
   is not strictly circular with the SWP validation **target**, but it still needs external stress
   knowledge VOD alone would not have.

## Decision

1. **The canonical biomass fit is EVENT-BLIND** (`appC.fit_exclude_event: false`): the regression sees
   **all** days, including the drought. `corrections()` uses a no-op exclusion by default.
2. **The exclude-Aug fit is retained as a sensitivity** (`fit_exclude_event: true`;
   `paper/90_sensitivity/appC_sensitivity.py` sweep 1, now labelled include=canonical / exclude=sensitivity).
3. The **event window** (`appC.event_exclude`, key name kept for continuity) is still used for the
   August dip-retention metric and figure shading — it is the *reporting* window, no longer a fit input.

## Measured effect (hemi-PAI, full 2025)

Including vs excluding the event is **nearly identical**, because biomass is ~static in the 2-week
event (hemi-PAI 6.32 baseline → 6.02 event), so those days have negligible leverage on the slope:

| fit | slope b | event dip retention | seasonal r(SWP) |
|---|---|---|---|
| **include Aug (canonical)** | 0.0779 | **0.778** | 0.269 |
| exclude Aug (old, sensitivity) | 0.0798 | 0.773 | 0.277 |

The headline survives event-blind: **hemi-PAI still preserves the water dip (0.78) that the LOESS
detrend smears (0.21)** — now with no "but you peeked at the event" objection. It is stronger: *with no
knowledge of when the drought occurred*, PAI keeps the dip a cheap detrend destroys.

**Why PAI still beats LOESS event-blind:** PAI removes biomass using an **independent structural
measurement**, blind to VOD's own time series, so it cannot touch the water dip. The LOESS ±30 d is a
**self-referential temporal filter** — its baseline follows VOD's August dip down, so subtracting it
smears the dip. The advantage is inherent to the method, not to the exclusion.

## Considered Options

- **Keep the exclusion (prototype).** Rejected: unfair to the event-blind LOESS, and non-standalone,
  for ≈ zero benefit (biomass static in August).
- **Make LOESS exclude the event too (symmetric exclusion).** Rejected: a ±30 d temporal filter cannot
  cleanly remove a mid-series 2-week hole, and it still requires a-priori drought knowledge.
- **Event-blind for all methods.** Chosen.

## Consequences

- **+** Fair PAI-vs-LOESS comparison; a genuine standalone-water-proxy claim (no a-priori drought
  knowledge); stronger paper wording.
- **−** Headline numbers move negligibly (recorded above); regenerated
  `paper/30_appC_vod_correction/outputs/`.
- **−** The config key `event_exclude` is now a slight misnomer (it is the reporting/event window);
  kept for continuity and documented inline. A separate up-drift-variant sensitivity
  (`paper/90_sensitivity/vod_pai_method_sweep.py`) retains its own event handling and is not a
  headline product.
