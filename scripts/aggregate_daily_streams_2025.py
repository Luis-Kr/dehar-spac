"""
Filter + daily-aggregate the DE-Har 2025 all-streams parquet into one analysis file.

Reads ``data/processed/dehar_all_streams_2025.parquet`` (native / 30-min, unfiltered) and
produces ``data/processed/dehar_daily_2025.parquet`` — one row per day, every stream filtered
and aggregated with the time-of-day window and statistic that fit its convention. Column headers
carry units (``var_stat_unit``; dimensionless variables carry no unit token).

Per-stream handling (see the project plan for the full table)
-------------------------------------------------------------
* Air temp   — 24 h mean / min / max / std                          [degC]
* RH         — 24 h mean / std                                      [pct]
* VPD        — 24 h mean / max / std                                [hPa]
* Rg/PAR/u*  — 24 h mean                                            [Wm2 / umol_m2_s / ms]
* Precip     — 24 h SUM                                             [mm]
* Fluxes     — 24 h mean (GPP<0 masked+interp); ET = Σ·0.5 SUM      [umol_m2_s / Wm2 / mm]
* Soil moist — 24 h mean / std per probe + plot ensemble            [pct]
* Sap flow   — 24 h SUM (min_count) per tree + ensemble             [sum]
* SWP        — predawn (2 h pre-sunrise) per tree + ensemble±ci95   [MPa]
* TWD        — predawn per tree + ensemble                          [um]
* VOD        — RH≤95 % & no-rain-12 h + noisy-hour clean; predawn / daylight /
               nighttime / 24 h, per receiver + ensemble            [dimensionless]
* Leaf angle — daylight / nighttime / 24 h daily mean of per-frame mean/median/std,
               all cams + ensemble                                  [deg]
* GCC        — anglecam: rain+wind mask, daytime 09-15 UTC, mean / p90 ensemble  [dimensionless]
* PhenoCam   — daily-native GCC + camera-NDVI (1-day/3-day, p90 canonical), 2025 slice joined
* PAI        — quality_all scans, 24 h mean / std (hemi_hi + hinge)  [m2m2]
* Sentinel-2 — only manual clear-sky dates; all indices+bands mean/std   [dimensionless / refl]
* Sentinel-1 — both orbits, dB indices + RVI, on acquisition date   [dB / dimensionless]

Run from the repo root (needs the ``dehar-spac`` env):
    python scripts/aggregate_daily_streams_2025.py
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("aggregate_daily")

# ── Config ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data/processed/dehar_all_streams_2025.parquet"
OUT_PATH = ROOT / "data/processed/dehar_daily_2025.parquet"

RG_DAY_THR_WM2 = 5.0          # Rg above this = "day"; sunrise = first such step
PREDAWN_HOURS = 2.0           # predawn window length before sunrise

PHYS_TREES = ["h10545", "h10546", "h10560"]   # SWP+TWD+sapflow matched trees (SWP ensemble)

# VOD cleaning
VOD_RH_THRESH_PCT = 95.0
VOD_RAIN_THRESH_MM = 0.001
VOD_RAIN_LOOKBACK_H = 12.0
VOD_NOISY_K = 5.0             # drop hours with hourly σ > μ + K·σ of the hourly-σ distribution
VOD_DEVICES = ["gps1", "gps3", "gps5"]

# GCC masking (anglecam)
GCC_PRECIP_THR_MM = 0.0001
GCC_PRECIP_LOCK_H = 6.0
GCC_USTAR_MAX_MS = 0.6
GCC_DAYTIME_UTC = (9, 15)
GCC_INTERP_LIMIT_D = 5

# PhenoCam (daily-native; joined, not resampled — ADR 0002)
TARGET_YEAR = 2025
PHENOCAM_PATH = ROOT / "data/processed/proximal_rs/phenocam/phenocam_daily.parquet"

SAPFLOW_MIN_COUNT = 40        # min 30-min steps required for a daily sap-flow sum
GPP_INTERP_LIMIT_D = 5

# Sentinel-2 canonical clear-sky source (ADR 0012): the hand-audited per-scene
# ROI product. Headline stages take verdict == "clear" at the 100 m ring; the
# retired S2_MANUAL_DATES_2025 date list is gone.
S2_ROI_PARQUET = (
    ROOT / "data/processed/satellite/sentinel2/s2_roi_means_by_scene.parquet"
)
S2_RING_M = 100          # canonical science ring (analysis_config.yaml)
S2_HEADLINE_YEAR = 2025  # ADR 0012: canonical daily S2 stays 2025-only this pass


# ── Generic daily reducers ────────────────────────────────────────────────────
def _daily(s: pd.Series, how: str, **kw) -> pd.Series:
    return getattr(s.resample("1D"), how)(**kw)


def _sunrise_hour(rg: pd.Series) -> pd.Series:
    """Fractional UTC hour of the first daytime step (Rg > thr) per day, indexed by date."""
    up = rg[rg > RG_DAY_THR_WM2]
    first = up.groupby(up.index.normalize()).apply(lambda g: g.index.min())
    return first.dt.hour + first.dt.minute / 60.0


def _predawn_daily(series: pd.Series, sunrise_h: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Mean / std / n over [sunrise-PREDAWN_HOURS, sunrise] for each day."""
    s = series.dropna()
    means, stds, ns = {}, {}, {}
    for date, sh in sunrise_h.items():
        end = date + pd.Timedelta(hours=float(sh))
        win = s.loc[end - pd.Timedelta(hours=PREDAWN_HOURS):end]
        if len(win):
            means[date], stds[date], ns[date] = win.mean(), win.std(ddof=1), len(win)
    return pd.Series(means), pd.Series(stds), pd.Series(ns)


def _masked_daily_mean_std(series: pd.Series, mask: pd.Series, suffix: str) -> pd.DataFrame:
    """Daily mean+std of `series` where `mask` (aligned bool) is True."""
    s = series.where(mask)
    return pd.DataFrame({f"{suffix}_mean": _daily(s, "mean"), f"{suffix}_std": _daily(s, "std")})


# ── Per-stream builders ───────────────────────────────────────────────────────
def build_meteo(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=_daily(df["meteo_tair_c"], "mean").index)
    t = df["meteo_tair_c"]
    out["tair_mean_degC"] = _daily(t, "mean")
    out["tair_min_degC"] = _daily(t, "min")
    out["tair_max_degC"] = _daily(t, "max")
    out["tair_std_degC"] = _daily(t, "std")
    out["rh_mean_pct"] = _daily(df["meteo_rh_pct"], "mean")
    out["rh_std_pct"] = _daily(df["meteo_rh_pct"], "std")
    v = df["meteo_vpd_hpa"]
    out["vpd_mean_hPa"] = _daily(v, "mean")
    out["vpd_max_hPa"] = _daily(v, "max")
    out["vpd_std_hPa"] = _daily(v, "std")
    out["rg_mean_Wm2"] = _daily(df["meteo_rg_wm2"], "mean")
    out["par_mean_umol_m2_s"] = _daily(df["meteo_par_umol_m2s"], "mean")
    out["ustar_mean_ms"] = _daily(df["meteo_ustar_ms"], "mean")
    out["precip_sum_mm"] = _daily(df["precip_mm"], "sum")
    return out


def build_fluxes(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=_daily(df["flux_gpp_f_umol_m2s"], "mean").index)
    # GPP: mask physically impossible negatives, then gap-fill short runs.
    gpp = _daily(df["flux_gpp_f_umol_m2s"], "mean")
    gpp_sd = _daily(df["flux_gpp_f_sd_umol_m2s"], "mean")
    neg = gpp < 0
    out["gpp_mean_umol_m2_s"] = gpp.mask(neg).interpolate("time", limit=GPP_INTERP_LIMIT_D)
    out["gpp_sd_umol_m2_s"] = gpp_sd.mask(neg).interpolate("time", limit=GPP_INTERP_LIMIT_D)
    out["nee_mean_umol_m2_s"] = _daily(df["flux_nee_f_umol_m2s"], "mean")
    out["nee_sd_umol_m2_s"] = _daily(df["flux_nee_f_sd_umol_m2s"], "mean")
    out["reco_mean_umol_m2_s"] = _daily(df["flux_reco_f_umol_m2s"], "mean")
    out["reco_sd_umol_m2_s"] = _daily(df["flux_reco_f_sd_umol_m2s"], "mean")
    out["le_mean_Wm2"] = _daily(df["flux_le_wm2"], "mean")
    out["h_mean_Wm2"] = _daily(df["flux_h_wm2"], "mean")
    # ET: 30-min mm h-1 → mm per step (×0.5) → daily sum.
    out["et_sum_mm"] = _daily(df["flux_et_f_mm_h"], "sum", min_count=1) * 0.5
    return out


def build_soil_moisture(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith("sm_vwc_pct_")]
    out = pd.DataFrame(index=_daily(df[cols[0]], "mean").index)
    for c in cols:                                   # sm_vwc_pct_a_5cm -> sm_a_5cm
        probe = c.replace("sm_vwc_pct_", "sm_")
        out[f"{probe}_mean_pct"] = _daily(df[c], "mean")
        out[f"{probe}_std_pct"] = _daily(df[c], "std")
    # plot ensemble: depth-average within each profile, then mean/std across profiles
    profiles = sorted({c.split("_")[3] for c in cols})
    prof_daily = {p: _daily(df[[c for c in cols if c.split("_")[3] == p]].mean(axis=1), "mean")
                  for p in profiles}
    prof_df = pd.DataFrame(prof_daily)
    out["sm_mean_pct"] = prof_df.mean(axis=1)
    out["sm_std_pct"] = prof_df.std(axis=1)
    return out


def build_sapflow(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith("js_h")]
    daily = df[cols].resample("1D").sum(min_count=SAPFLOW_MIN_COUNT)
    out = pd.DataFrame(index=daily.index)
    for c in cols:                                   # js_h10545 -> sapflow_h10545_sum
        out[f"sapflow_{c[3:]}_sum"] = daily[c]
    out["sapflow_sum_mean"] = daily.mean(axis=1)
    out["sapflow_sum_std"] = daily.std(axis=1)
    return out


def build_predawn(df: pd.DataFrame, sunrise_h: pd.Series, prefix: str, src_prefix: str,
                  trees: list[str], unit: str, ensemble_trees: list[str],
                  with_ci95: bool = False) -> pd.DataFrame:
    """Per-tree predawn daily mean + inter-tree ensemble (mean/std[/ci95])."""
    out = None
    per_tree = {}
    for c in [c for c in df.columns if c.startswith(src_prefix)]:
        tree = c.split("_")[-1]
        m, _, _ = _predawn_daily(df[c], sunrise_h)
        per_tree[tree] = m
        col = f"{prefix}_{tree}_pd_{unit}"
        out = m.to_frame(col) if out is None else out.join(m.rename(col), how="outer")
    ens = pd.DataFrame({t: per_tree[t] for t in ensemble_trees if t in per_tree})
    out[f"{prefix}_pd_mean_{unit}"] = ens.mean(axis=1)
    out[f"{prefix}_pd_std_{unit}"] = ens.std(axis=1, ddof=1)
    if with_ci95:
        n = ens.notna().sum(axis=1)
        out[f"{prefix}_pd_ci95_{unit}"] = 1.96 * out[f"{prefix}_pd_std_{unit}"] / np.sqrt(n)
    return out


def build_vod(df: pd.DataFrame, sunrise_h: pd.Series) -> pd.DataFrame:
    day = df["meteo_rg_wm2"] > RG_DAY_THR_WM2
    night = df["meteo_rg_wm2"] <= RG_DAY_THR_WM2
    precip_12h = df["precip_mm"].fillna(0).rolling(f"{int(VOD_RAIN_LOOKBACK_H)}h").sum()

    out = None
    ens = {w: [] for w in ("predawn", "daylight", "nighttime", "24h")}
    for dev in VOD_DEVICES:
        col = f"nvod_{dev}"
        if col not in df.columns:
            continue
        v = df[col]
        # RH + rain filter, then drop noisy hours
        keep = (df["meteo_rh_pct"] <= VOD_RH_THRESH_PCT) & (precip_12h <= VOD_RAIN_THRESH_MM)
        v = v.where(keep)
        hourly_std = v.resample("1h").std()
        thr = hourly_std.mean() + VOD_NOISY_K * hourly_std.std()
        noisy = list(hourly_std[hourly_std > thr].index.floor("h"))
        in_noisy = pd.Index(v.index.floor("h")).isin(noisy)
        v = v.where(~in_noisy)

        frames = {}
        pm, ps, _ = _predawn_daily(v, sunrise_h)
        frames["predawn"] = pd.DataFrame({f"vod_{dev}_predawn_mean": pm,
                                          f"vod_{dev}_predawn_std": ps})
        frames["daylight"] = _masked_daily_mean_std(v, day, f"vod_{dev}_daylight")
        frames["nighttime"] = _masked_daily_mean_std(v, night, f"vod_{dev}_nighttime")
        frames["24h"] = pd.DataFrame({f"vod_{dev}_24h_mean": _daily(v, "mean"),
                                      f"vod_{dev}_24h_std": _daily(v, "std")})
        for w, fr in frames.items():
            ens[w].append(fr.iloc[:, 0].rename(dev))            # the mean column for ensemble
            out = fr if out is None else out.join(fr, how="outer")
    # cross-device ensemble per window
    for w in ens:
        e = pd.concat(ens[w], axis=1)
        out[f"vod_{w}_mean"] = e.mean(axis=1)
        out[f"vod_{w}_std"] = e.std(axis=1, ddof=1)
    return out


def build_leaf_angle(df: pd.DataFrame) -> pd.DataFrame:
    day = df["meteo_rg_wm2"] > RG_DAY_THR_WM2
    night = df["meteo_rg_wm2"] <= RG_DAY_THR_WM2
    windows = {"daylight": day, "nighttime": night, "24h": pd.Series(True, index=df.index)}
    cams = sorted({c.split("_")[2] for c in df.columns if c.startswith("leaf_angle_cam")})
    out = None
    ens = {(w, stat): [] for w in windows for stat in ("mean", "median", "std")}
    for cam in cams:
        for stat in ("mean", "median", "std"):
            src = f"leaf_angle_{cam}_{stat}"
            if src not in df.columns:
                continue
            for w, mask in windows.items():
                daily = _daily(df[src].where(mask), "mean")
                col = f"leaf_angle_{cam}_{w}_{stat}_deg"
                out = daily.to_frame(col) if out is None else out.join(daily.rename(col), how="outer")
                ens[(w, stat)].append(daily.rename(cam))
    for (w, stat), series_list in ens.items():
        e = pd.concat(series_list, axis=1)
        out[f"leaf_angle_{w}_{stat}_deg"] = e.mean(axis=1)
        if stat == "mean":
            out[f"leaf_angle_{w}_camstd_deg"] = e.std(axis=1, ddof=1)
    return out


def build_gcc(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith("gcc_cam")]
    # quality mask: rain lockout + low turbulence + daytime
    wet = (df["precip_mm"].fillna(0) >= GCC_PRECIP_THR_MM).astype(float)
    n_lock = max(1, int(np.ceil(GCC_PRECIP_LOCK_H * 60 / 30)))
    rain_ok = wet.rolling(n_lock, min_periods=1).max() == 0
    wind_ok = df["meteo_ustar_ms"].fillna(0) <= GCC_USTAR_MAX_MS
    h0, h1 = GCC_DAYTIME_UTC
    daytime = (df.index.hour >= h0) & (df.index.hour < h1)
    mask = rain_ok & wind_ok & pd.Series(daytime, index=df.index)

    g = df[cols].where(mask, axis=0)
    dmean = g.resample("1D").mean()
    dp90 = g.resample("1D").quantile(0.90)
    out = pd.DataFrame(index=dmean.index)
    for c in cols:
        out[f"{c}_mean"] = dmean[c].interpolate("time", limit=GCC_INTERP_LIMIT_D)
        out[f"{c}_p90"] = dp90[c].interpolate("time", limit=GCC_INTERP_LIMIT_D)
    # Anglecam ensemble — demoted to an alternative greenness (ADR 0002); kept in the
    # table under an explicit ``gcc_anglecam_*`` name so it never collides with the
    # canonical PhenoCam GCC. Per-camera ``gcc_cam*`` columns above are unchanged.
    out["gcc_anglecam_mean"] = dmean.mean(axis=1).interpolate("time", limit=GCC_INTERP_LIMIT_D)
    out["gcc_anglecam_std"] = dmean.std(axis=1).interpolate("time", limit=GCC_INTERP_LIMIT_D)
    out["gcc_anglecam_p90"] = dp90.mean(axis=1).interpolate("time", limit=GCC_INTERP_LIMIT_D)
    return out


def build_phenocam() -> pd.DataFrame:
    """Join the daily-native PhenoCam GCC/NDVI products (the TARGET_YEAR slice).

    PhenoCam summary products are already daily, so they bypass the sub-daily
    export/resample path: ``scripts/process_phenocam.py`` writes the full record,
    and here we slice to TARGET_YEAR and align to the UTC daily index. The 3-day
    products keep their native cadence (NaN between window dates — no fill).
    """
    if not PHENOCAM_PATH.exists():
        raise FileNotFoundError(
            f"{PHENOCAM_PATH} missing — run scripts/process_phenocam.py first.")
    pc = pd.read_parquet(PHENOCAM_PATH)
    if pc.index.tz is None:
        pc.index = pc.index.tz_localize("UTC")
    pc = pc[pc.index.year == TARGET_YEAR].sort_index()
    pc.index = pc.index.normalize()              # UTC midnight, matches the daily grid
    return pc


def build_pai(df: pd.DataFrame) -> pd.DataFrame:
    out = None
    metric_map = {"HingePAI": "hinge", "LinearPAI": "linear", "WeightedPAI": "weighted"}
    for scan in ("hemi_hi", "hinge"):
        qcol = f"leaf_{scan}_quality_all"
        if qcol not in df.columns:
            continue
        good = df[qcol] == True                                   # noqa: E712
        for raw, short in metric_map.items():
            src = f"leaf_{scan}_{raw}_total"
            s = df[src].where(good)
            m = _daily(s, "mean").rename(f"pai_{scan}_{short}_mean_m2m2")
            sd = _daily(s, "std").rename(f"pai_{scan}_{short}_std_m2m2")
            fr = pd.concat([m, sd], axis=1)
            out = fr if out is None else out.join(fr, how="outer")

    # canopy-layer PAI (understory 1.5-9 m / overstory 9-18 m) from the hemi profile;
    # App C's canonical structural regressor is the understory band (ADR 0008):
    # pai_hemi_hi_weighted_{understory,overstory}_{mean,std}_m2m2
    hqcol = "leaf_hemi_hi_quality_all"
    if hqcol in df.columns:
        hgood = df[hqcol] == True                                 # noqa: E712
        for band in ("understory", "overstory"):
            src = f"leaf_hemi_hi_WeightedPAI_{band}"
            if src not in df.columns:
                continue
            s = df[src].where(hgood)
            m = _daily(s, "mean").rename(f"pai_hemi_hi_weighted_{band}_mean_m2m2")
            sd = _daily(s, "std").rename(f"pai_hemi_hi_weighted_{band}_std_m2m2")
            out = out.join(pd.concat([m, sd], axis=1), how="outer")

    # uncorrected (rotation, no up-drift) headline comparison columns (ADR 0005):
    # pai_hemi_hi_{hinge,weighted}_uncorr_{mean,std}
    uqcol = "leaf_hemi_hi_uncorr_quality_all"
    if uqcol in df.columns:
        ugood = df[uqcol] == True                                 # noqa: E712
        for raw, short in {"HingePAI": "hinge", "WeightedPAI": "weighted"}.items():
            src = f"leaf_hemi_hi_uncorr_{raw}_total"
            if src not in df.columns:
                continue
            s = df[src].where(ugood)
            m = _daily(s, "mean").rename(f"pai_hemi_hi_{short}_uncorr_mean_m2m2")
            sd = _daily(s, "std").rename(f"pai_hemi_hi_{short}_uncorr_std_m2m2")
            out = out.join(pd.concat([m, sd], axis=1), how="outer")
    return out


def build_sentinel2() -> pd.DataFrame:
    """Daily clear-sky S2 ROI means from the audited per-scene product (ADR 0012).

    Reads the hand-audited ROI table (`s2_roi_means_by_scene.parquet`), keeps
    ``verdict == "clear"`` scenes at the canonical 100 m ring for the headline
    year, and resamples to daily means. Bands/indices are renamed
    ``<VAR>_r100 -> s2_<var>_mean``. Replaces the retired 33-date filter.
    """
    tab = pd.read_parquet(S2_ROI_PARQUET)
    tab = tab[tab["verdict"] == "clear"].copy()
    tab["datetime"] = pd.to_datetime(tab["datetime"])
    tab = tab[tab["datetime"].dt.year == S2_HEADLINE_YEAR]

    suffix = f"_r{S2_RING_M}"
    val_cols = [
        c for c in tab.columns if c.endswith(suffix) and c != f"n{suffix}"
    ]
    s2 = tab.set_index("datetime")[val_cols].sort_index()
    s2.columns = [f"s2_{c[: -len(suffix)].lower()}_mean" for c in val_cols]
    if s2.index.tz is None:
        s2.index = s2.index.tz_localize("UTC")
    return s2.resample("1D").mean().dropna(how="all")


def build_sentinel1(df: pd.DataFrame) -> pd.DataFrame:
    keep, rename = [], {}
    for orb in ("asc", "desc"):
        for var in ("VV", "VH", "SPAN", "CR"):
            for stat in ("mean", "std"):
                src = f"s1_{orb}_{var}_dB_{stat}"
                if src in df.columns:
                    keep.append(src)
                    rename[src] = f"s1_{orb}_{var.lower()}_{stat}_dB"
        for stat in ("mean", "std"):
            src = f"s1_{orb}_RVI_{stat}"
            if src in df.columns:
                keep.append(src)
                rename[src] = f"s1_{orb}_rvi_{stat}"
    s1 = df[keep].dropna(how="all")
    daily = s1.resample("1D").mean().dropna(how="all").rename(columns=rename)
    return daily


# ── Orchestration ─────────────────────────────────────────────────────────────
def main() -> None:
    log.info("Reading %s", IN_PATH)
    df = pd.read_parquet(IN_PATH).sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    sunrise_h = _sunrise_hour(df["meteo_rg_wm2"])

    builders = {
        "meteo/precip": lambda: build_meteo(df),
        "fluxes":       lambda: build_fluxes(df),
        "soil_moist":   lambda: build_soil_moisture(df),
        "sapflow":      lambda: build_sapflow(df),
        "swp":          lambda: build_predawn(df, sunrise_h, "swp", "swp_mpa_h",
                                              PHYS_TREES, "MPa", PHYS_TREES, with_ci95=True),
        "twd":          lambda: build_predawn(df, sunrise_h, "twd", "twd_um_h",
                                              PHYS_TREES, "um", PHYS_TREES),
        "vod":          lambda: build_vod(df, sunrise_h),
        "leaf_angle":   lambda: build_leaf_angle(df),
        "gcc":          lambda: build_gcc(df),
        "phenocam":     lambda: build_phenocam(),
        "pai":          lambda: build_pai(df),
        "sentinel2":    lambda: build_sentinel2(),
        "sentinel1":    lambda: build_sentinel1(df),
    }

    parts = {}
    for name, fn in builders.items():
        try:
            part = fn()
            parts[name] = part
            log.info("  %-12s %3d cols", name, part.shape[1])
        except Exception as exc:                                     # noqa: BLE001
            log.warning("  %-12s FAILED: %s", name, exc)

    daily = pd.concat(parts.values(), axis=1, join="outer").sort_index()
    daily.index.name = "date"

    dupes = daily.columns[daily.columns.duplicated()].tolist()
    if dupes:
        raise ValueError(f"Duplicate columns: {dupes}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(OUT_PATH, index=True)

    filled = 100 * daily.notna().to_numpy().sum() / daily.size
    log.info("─" * 60)
    log.info("Daily file : %d days × %d cols  (%.1f%% filled)", *daily.shape, filled)
    log.info("Range      : %s → %s", daily.index.min().date(), daily.index.max().date())
    log.info("Saved      : %s (%.2f MB)", OUT_PATH, OUT_PATH.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
