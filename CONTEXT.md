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
