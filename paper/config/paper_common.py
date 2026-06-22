"""Shared helpers for the paper/ analysis stages — config + canonical data loaders."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "paper" / "config" / "analysis_config.yaml"


def load_config() -> dict:
    with open(CONFIG) as fh:
        return yaml.safe_load(fh)


def load_daily(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    df = pd.read_parquet(REPO / cfg["meta"]["daily_parquet"])
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    return df


def load_sensors_meta(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    return pd.read_csv(REPO / cfg["meta"]["sensors_meta"])


def load_event_windows() -> pd.DataFrame:
    """Stage-00 output; run paper/00_event_definition/define_event.py first."""
    p = REPO / "paper" / "00_event_definition" / "outputs" / "tables" / "event_windows.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — run stage 00 first.")
    ev = pd.read_csv(p, parse_dates=["start", "end"])
    return ev


def hornbeam_qc_cams(df: pd.DataFrame, cfg: dict | None = None, window=None):
    """Inter-camera coherence QC for the hornbeam AngleCams (non-circular: viewpoint quality, not
    event response). Each cam's daylight leaf-angle is correlated with the leave-one-out canopy
    consensus over the leaf-on season; cams with r >= coherence_min_r are kept.
    Returns (kept_cams: list[int], coherence: dict[int, float])."""
    cfg = cfg or load_config()
    la = cfg["streams"]["leaf_angle"]
    cams = la["all_hornbeam_cams"]
    thr = la.get("coherence_min_r", 0.6)
    w = window or cfg["onset"]["search_window"]
    cols = {c: f"leaf_angle_cam{c}_daylight_mean_deg" for c in cams
            if f"leaf_angle_cam{c}_daylight_mean_deg" in df.columns}
    sub = df.loc[pd.Timestamp(str(w[0])):pd.Timestamp(str(w[1]))]
    coh = {}
    for c in cols:
        others = [cols[k] for k in cols if k != c]
        cons = sub[others].mean(axis=1)
        d = pd.concat([sub[cols[c]], cons], axis=1).dropna()
        coh[c] = float(np.corrcoef(d.iloc[:, 0], d.iloc[:, 1])[0, 1]) if len(d) > 3 else np.nan
    kept = [c for c in cols if np.isfinite(coh[c]) and coh[c] >= thr]
    return kept, coh


def greenness_col(cfg: dict | None = None) -> str:
    """Canonical greenness column, resolved from ``streams.greenness`` (ADR 0002).

    ``gcc_source: phenocam`` -> ``gcc_phenocam_{phenocam_aggregation}_p90`` (default);
    ``gcc_source: anglecam`` -> ``gcc_anglecam_p90`` (legacy/compensatory contrast).
    """
    cfg = cfg or load_config()
    g = cfg["streams"]["greenness"]
    source = g.get("gcc_source", "phenocam")
    if source == "anglecam":
        return "gcc_anglecam_p90"
    if source != "phenocam":
        raise ValueError(
            f"streams.greenness.gcc_source must be phenocam|anglecam, got {source!r}")
    agg = g.get("phenocam_aggregation", "1day")
    stat = g.get("canonical_stat", "p90")
    return f"gcc_phenocam_{agg}_{stat}"


def ndvi_phenocam_col(cfg: dict | None = None) -> str:
    """Canonical proximal (PhenoCam) NDVI column, resolved from
    ``streams.ndvi_phenocam`` and the shared ``phenocam_aggregation`` knob."""
    cfg = cfg or load_config()
    agg = cfg["streams"]["greenness"].get("phenocam_aggregation", "1day")
    stat = cfg["streams"]["greenness"].get("canonical_stat", "p90")
    return f"ndvi_phenocam_{agg}_{stat}"


def stars(p: float) -> str:
    """R-style significance code from a p-value."""
    if p is None or p != p:   # NaN
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def paper_style() -> None:
    """Shared figure style for all paper/ stages: THIN grid lines, THICK data lines."""
    import matplotlib as mpl
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.7)
    mpl.rcParams.update({
        "grid.linewidth": 0.4, "grid.color": "0.85", "grid.alpha": 0.7,
        "axes.linewidth": 0.8, "axes.edgecolor": "0.45",
        "lines.linewidth": 2.2, "lines.markersize": 5,
        "patch.linewidth": 0.4,
        "savefig.dpi": 150, "savefig.bbox": "tight",
    })
