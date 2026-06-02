"""Mechanistic test of Application A: does a leaf-angle change *alone* reproduce
the observed August change of each Sentinel-2 index?

PROSAIL (PROSPECT-5 + 4SAIL) is run with the **baseline** vs **event** mean leaf
angle, holding LAI (= baseline PAI), chlorophyll, water and everything else
fixed. If a simulated index change has the same sign and a comparable magnitude
to the observed change, leaf angle *causes* that part of the signal (the
confound). Model-based, so it sidesteps the ~7-day satellite cadence.

Parametrization (literature-aligned)
------------------------------------
PROSPECT-5 leaf: N=1.5, Cab=45, Car=10, Cw=0.012 cm, Cm=0.008 g/cm^2 -- typical
  temperate broadleaf (Jacquemoud et al. 2009; Berger et al. 2018 ranges). Held
  fixed so only leaf angle varies.
4SAIL canopy: LAI = baseline PAI (held), hotspot=0.05, mean leaf angle =
  AngleCam mean tilt (typelidf=2, ellipsoidal LIDF; Goel & Strebel 1984;
  Kattenborn et al. 2024).
Geometry: SZA=39 deg (Sentinel-2 ~10:30 mean local solar time at Hartheim
  47.9 N, mid-Aug), VZA=3 deg, rel.azimuth=0. Soil: rsoil=1, psoil=1 (dry).

Sentinel-2 index definitions (standard band combinations):
  NDVI  (Rouse 1974)        = (B8-B4)/(B8+B4)
  kNDVI (Camps-Valls 2021)  = tanh(NDVI^2)
  EVI   (Huete et al. 2002) = 2.5(B8-B4)/(B8+6 B4-7.5 B2+1)
  NIRv  (Badgley et al.2017)= NDVI * B8
  SAVI  (Huete 1988, L=0.5) = 1.5(B8-B4)/(B8+B4+0.5)
  CIre  (Gitelson 2003)     = B7/B5 - 1
  MTCI  (Dash & Curran 2004)= (B6-B5)/(B5-B4)
  NDII  (Hardisky 1983)     = (B8A-B11)/(B8A+B11)
Band reflectance is averaged over nominal S2A band windows (not full spectral
response functions -- a refinement). For red-edge indices (CIre, MTCI) check the
band choices match the processing pipeline before quoting the explained fraction.

Run
---
    python analysis/satellite_comparison/prosail_leaf_angle.py --csv <file>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import prosail

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "data" / "processed" / "dehar_daily_season_2025_filtered.csv"
BASELINE = ("2025-07-28", "2025-08-06")   # recovered, non-stressed
EVENT = ("2025-08-10", "2025-08-22")      # around the stress peak

# Sentinel-2A nominal band windows (nm).
BANDS = {"B2": (459, 525), "B4": (650, 680), "B5": (697, 712), "B6": (733, 748),
         "B7": (773, 793), "B8": (780, 886), "B8A": (855, 875), "B11": (1565, 1655)}
# Fixed leaf/canopy/geometry (everything except leaf angle is held constant).
FIXED = dict(n=1.5, cab=45.0, car=10.0, cbrown=0.0, cw=0.012, cm=0.008, ant=0.0,
             hspot=0.05, tts=39.0, tto=3.0, psi=0.0)
# index name (matches s2_<key>_savgol) -> function of a band dict.
INDEX_FUNCS = {
    "ndvi": lambda b: (b["B8"] - b["B4"]) / (b["B8"] + b["B4"]),
    "kndvi": lambda b: np.tanh(((b["B8"] - b["B4"]) / (b["B8"] + b["B4"])) ** 2),
    "evi": lambda b: 2.5 * (b["B8"] - b["B4"]) / (b["B8"] + 6 * b["B4"] - 7.5 * b["B2"] + 1),
    "nirv": lambda b: (b["B8"] - b["B4"]) / (b["B8"] + b["B4"]) * b["B8"],
    "savi": lambda b: 1.5 * (b["B8"] - b["B4"]) / (b["B8"] + b["B4"] + 0.5),
    "cire": lambda b: b["B7"] / b["B5"] - 1,
    "mtci": lambda b: (b["B6"] - b["B5"]) / (b["B5"] - b["B4"]),
    "ndii": lambda b: (b["B8A"] - b["B11"]) / (b["B8A"] + b["B11"]),
}
NIR_SENSITIVE = {"ndvi", "kndvi", "evi", "nirv", "savi", "ndii"}  # vs red-edge only


def _bands(rho: np.ndarray) -> dict[str, float]:
    return {k: float(rho[lo - 400:hi - 400 + 1].mean()) for k, (lo, hi) in BANDS.items()}


def simulate(leaf_angle: float, lai: float) -> dict[str, float]:
    rho = prosail.run_prosail(
        FIXED["n"], FIXED["cab"], FIXED["car"], FIXED["cbrown"], FIXED["cw"],
        FIXED["cm"], lai, leaf_angle, FIXED["hspot"], FIXED["tts"], FIXED["tto"],
        FIXED["psi"], ant=FIXED["ant"], typelidf=2, rsoil=1.0, psoil=1.0,
    )
    b = _bands(np.asarray(rho))
    return {name: float(f(b)) for name, f in INDEX_FUNCS.items()}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="PROSAIL leaf-angle confound, all S2 indices")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    a = p.parse_args(argv)

    df = pd.read_csv(a.csv, parse_dates=["date"]).set_index("date").sort_index()
    df["leaf_angle"] = df[["leaf_angle_cam65_mean", "leaf_angle_cam67_mean"]].mean(axis=1)

    def wm(col, w):
        return float(df[col].loc[w[0]:w[1]].mean())

    ang_b, ang_e = wm("leaf_angle", BASELINE), wm("leaf_angle", EVENT)
    lai = wm("pai_total_sg", BASELINE)
    sim_b, sim_e = simulate(ang_b, lai), simulate(ang_e, lai)

    print(f"Leaf angle: {ang_b:.1f}deg -> {ang_e:.1f}deg ({ang_e - ang_b:+.1f})  | "
          f"LAI held = {lai:.2f} | SZA {FIXED['tts']:.0f}deg, Cab {FIXED['cab']:.0f}\n")
    print(f"{'index':6s} {'NIR?':4s} {'sim_d':>8s} {'obs_d':>8s} {'explained':>10s} {'sign':>9s}")
    rows = []
    for key in INDEX_FUNCS:
        ocol = f"s2_{key}_savgol"
        if ocol not in df:
            continue
        d_sim = sim_e[key] - sim_b[key]
        d_obs = wm(ocol, EVENT) - wm(ocol, BASELINE)
        frac = d_sim / d_obs if abs(d_obs) > 1e-6 else np.nan
        sign = "same" if d_sim * d_obs > 0 else "opp"
        fr = f"{frac:8.0%}" if np.isfinite(frac) else "     n/a"
        nir = "NIR" if key in NIR_SENSITIVE else "RE"
        print(f"{key:6s} {nir:4s} {d_sim:+8.4f} {d_obs:+8.4f} {fr:>10s} {sign:>9s}")
        rows.append({"index": key, "nir_sensitive": key in NIR_SENSITIVE,
                     "sim_baseline": sim_b[key], "sim_event": sim_e[key],
                     "d_sim": d_sim, "d_obs": d_obs, "frac_explained": frac, "sign": sign})
    tab = pd.DataFrame(rows)

    # LAI sensitivity (PAI_total includes wood -> true green LAI lower).
    print("\nExplained fraction vs assumed green LAI:")
    print("  LAI " + " ".join(f"{k:>6s}" for k in INDEX_FUNCS))
    lai_rows = []
    for lai_t in (2.0, 3.0, 4.0, 5.0, 6.0):
        sb, se = simulate(ang_b, lai_t), simulate(ang_e, lai_t)
        cells, rec = [], {"lai": lai_t}
        for key in INDEX_FUNCS:
            d_obs = wm(f"s2_{key}_savgol", EVENT) - wm(f"s2_{key}_savgol", BASELINE)
            fr = (se[key] - sb[key]) / d_obs if abs(d_obs) > 1e-6 else np.nan
            rec[key] = fr
            cells.append(f"{fr:6.0%}" if np.isfinite(fr) else "   n/a")
        lai_rows.append(rec)
        print(f"  {lai_t:>3.0f} " + " ".join(cells))

    out = REPO_ROOT / "data" / "processed" / "prosail_all_indices.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out, index=False)
    pd.DataFrame(lai_rows).to_csv(
        REPO_ROOT / "data" / "processed" / "prosail_lai_sensitivity.csv", index=False)

    # Sensitivity-curve figure: each index vs leaf angle (LAI fixed).
    angles = np.arange(15, 70.1, 2.5)
    sims = [simulate(float(x), lai) for x in angles]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, key in zip(axes.ravel(), INDEX_FUNCS):
        ax.plot(angles, [s[key] for s in sims], "-", color="#2166ac")
        ax.plot(ang_b, sim_b[key], "o", color="#b2182b", ms=8, label="baseline")
        ax.plot(ang_e, sim_e[key], "s", color="#b2182b", ms=8, label="event")
        ax.set(xlabel="mean leaf angle (deg)", ylabel=key.upper(),
               title=f"{key.upper()} ({'NIR' if key in NIR_SENSITIVE else 'red-edge'})")
        ax.legend(fontsize=7)
    fig.suptitle(f"PROSAIL leaf-angle-only sensitivity, all S2 indices "
                 f"(LAI={lai:.1f}, Cab={FIXED['cab']:.0f} fixed)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    figp = REPO_ROOT / "figures" / "prosail_all_indices_sensitivity.png"
    figp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figp, dpi=140)
    plt.close(fig)

    print("\nNDII/CIre/MTCI also respond to canopy water/chlorophyll (held fixed),")
    print("so their explained fraction is the structural (leaf-angle) part only.")
    print(f"\nWrote {out}\nWrote {figp}")


if __name__ == "__main__":
    main()
