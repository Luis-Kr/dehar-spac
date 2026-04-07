"""Extract and average eddy covariance fluxes across u* thresholds."""

import pandas as pd


def extract_fluxes(df: pd.DataFrame) -> pd.DataFrame:
    """Extract gap-filled fluxes averaged across u* thresholds.

    For NEE, GPP, and Reco the U05/U50/U95 gap-filled columns are
    averaged to give a single best estimate, following the standard
    REddyProc convention.  The standard deviation across thresholds
    is kept as an uncertainty estimate.

    Also extracts the raw (unfilled) NEE for high-resolution analyses,
    gap-filled ET, and sensible/latent heat fluxes.

    Parameters
    ----------
    df : pd.DataFrame
        Full DataFrame from :func:`dehar.atmosphere.meteo.read_atmosphere_soil_raw`.

    Returns
    -------
    pd.DataFrame
        Flux time series with columns:

        - ``nee_umol_m2s`` — raw u*-filtered NEE (with gaps)
        - ``nee_f_umol_m2s`` — gap-filled NEE (mean of U05/U50/U95)
        - ``nee_f_sd_umol_m2s`` — std across u* thresholds
        - ``gpp_f_umol_m2s`` — gap-filled GPP (mean of U05/U50/U95)
        - ``gpp_f_sd_umol_m2s`` — std across u* thresholds
        - ``reco_f_umol_m2s`` — gap-filled Reco (mean of U05/U50/U95)
        - ``reco_f_sd_umol_m2s`` — std across u* thresholds
        - ``le_wm2`` — raw latent heat flux
        - ``h_wm2`` — raw sensible heat flux
        - ``et_f_mm_h`` — gap-filled evapotranspiration
    """
    out = pd.DataFrame(index=df.index)

    # Raw u*-filtered NEE
    if "NEE_uStar_orig" in df.columns:
        out["nee_umol_m2s"] = df["NEE_uStar_orig"]

    # Gap-filled carbon fluxes: mean across u* thresholds
    for var, unit in [("NEE", "umol_m2s"), ("GPP", "umol_m2s"), ("Reco", "umol_m2s")]:
        cols = [f"{var}_U05_f", f"{var}_U50_f", f"{var}_U95_f"]
        # Reco columns have no _f suffix
        if var == "Reco":
            cols = ["Reco_U05", "Reco_U50", "Reco_U95"]
        present = [c for c in cols if c in df.columns]
        if present:
            subset = df[present]
            out[f"{var.lower()}_f_{unit}"] = subset.mean(axis=1)
            out[f"{var.lower()}_f_sd_{unit}"] = subset.std(axis=1)

    # Raw energy fluxes
    if "LE" in df.columns:
        out["le_wm2"] = df["LE"]
    if "H" in df.columns:
        out["h_wm2"] = df["H"]

    # Gap-filled ET
    if "ET_f" in df.columns:
        out["et_f_mm_h"] = df["ET_f"]
    elif "ET" in df.columns:
        out["et_f_mm_h"] = df["ET"]

    if out.empty:
        raise ValueError("No flux columns found in input DataFrame.")

    return out
