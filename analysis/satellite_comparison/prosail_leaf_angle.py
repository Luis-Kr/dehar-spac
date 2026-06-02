"""Mechanistic test of Application A: does a leaf-angle change *alone* reproduce
the observed August satellite change?

We run PROSAIL (PROSPECT + 4SAIL) twice: once with the **baseline** mean leaf
angle and once with the **event** mean leaf angle, holding **LAI (= baseline
PAI), chlorophyll, water and everything else fixed**. If the simulated change in
NDVI / NIRv / NDII has the same sign and a comparable magnitude to the observed
change, then leaf angle *causes* the satellite signal — the confound is
mechanistic, not just correlational. This sidesteps the ~7-day satellite cadence
because it is model-based.

Leaf angle (AngleCam mean tilt) enters PROSAIL as the ellipsoidal LIDF mean
(typelidf=2). NDVI saturates at high LAI, so the leaf-angle effect is expected to
be largest on NIR reflectance / NIRv (consistent with the observed signal moving
more in NIRv/NDII than in NDVI).

Refs: Jacquemoud et al. 2009 (PROSAIL), Verhoef 1984 (SAIL), Kattenborn 2024.
Caveats printed at runtime: band reflectance is sampled over nominal Sentinel-2
band windows (not full spectral response functions); NDII also responds to canopy
water (held fixed here), so leaf angle explains only the *structural* part of it.

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

# Sentinel-2 nominal band windows (nm) -> indices into the 400-2500 nm spectrum.
BANDS = {"red": (650, 680), "nir": (780, 880), "nir_a": (855, 875), "swir": (1565, 1655)}
# Fixed leaf/canopy/geometry parameters (everything except leaf angle is held).
FIXED = dict(n=1.5, cab=45.0, car=10.0, cbrown=0.0, cw=0.012, cm=0.008, ant=0.0,
             hspot=0.05, tts=38.0, tto=5.0, psi=0.0)  # Hartheim ~48N, S2 ~10:30


def band(rho: np.ndarray, name: str) -> float:
    lo, hi = BANDS[name]
    return float(rho[lo - 400:hi - 400 + 1].mean())


def indices(rho: np.ndarray) -> dict[str, float]:
    red, nir, nir_a, swir = (band(rho, b) for b in ("red", "nir", "nir_a", "swir"))
    ndvi = (nir - red) / (nir + red)
    return {"NDVI": ndvi, "NIRv": ndvi * nir, "NDII": (nir_a - swir) / (nir_a + swir)}


def simulate(leaf_angle: float, lai: float) -> dict[str, float]:
    rho = prosail.run_prosail(
        FIXED["n"], FIXED["cab"], FIXED["car"], FIXED["cbrown"], FIXED["cw"],
        FIXED["cm"], lai, leaf_angle, FIXED["hspot"], FIXED["tts"], FIXED["tto"],
        FIXED["psi"], ant=FIXED["ant"], typelidf=2, rsoil=1.0, psoil=1.0,
    )
    return indices(np.asarray(rho))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="PROSAIL leaf-angle confound test")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    a = p.parse_args(argv)

    df = pd.read_csv(a.csv, parse_dates=["date"]).set_index("date").sort_index()
    df["leaf_angle"] = df[["leaf_angle_cam65_mean", "leaf_angle_cam67_mean"]].mean(axis=1)

    def win_mean(col, w):
        return float(df[col].loc[w[0]:w[1]].mean())

    ang_base, ang_evt = win_mean("leaf_angle", BASELINE), win_mean("leaf_angle", EVENT)
    lai = win_mean("pai_total_sg", BASELINE)
    pai_evt = win_mean("pai_total_sg", EVENT)

    sim_base, sim_evt = simulate(ang_base, lai), simulate(ang_evt, lai)
    obs = {"NDVI": "s2_ndvi_savgol", "NIRv": "s2_nirv_savgol", "NDII": "s2_ndii_savgol"}

    print(f"Leaf angle:  baseline {ang_base:.1f}deg  ->  event {ang_evt:.1f}deg "
          f"(Delta {ang_evt - ang_base:+.1f}deg)")
    print(f"PAI (held as LAI={lai:.2f}):  baseline {lai:.2f} -> event {pai_evt:.2f} "
          f"(Delta {pai_evt - lai:+.2f}, ~flat)\n")
    print(f"{'index':6s} {'sim_base':>9s} {'sim_event':>10s} {'d_sim':>8s} "
          f"{'d_obs':>8s} {'explained':>10s}")
    rows = []
    for idx in ("NDVI", "NIRv", "NDII"):
        d_sim = sim_evt[idx] - sim_base[idx]
        d_obs = win_mean(obs[idx], EVENT) - win_mean(obs[idx], BASELINE)
        frac = d_sim / d_obs if d_obs != 0 else np.nan
        same = "same sign" if (d_sim * d_obs > 0) else "OPP sign"
        print(f"{idx:6s} {sim_base[idx]:9.3f} {sim_evt[idx]:10.3f} {d_sim:+8.3f} "
              f"{d_obs:+8.3f} {frac:9.0%}  {same}")
        rows.append({"index": idx, "sim_baseline": sim_base[idx], "sim_event": sim_evt[idx],
                     "d_sim": d_sim, "d_obs": d_obs, "frac_explained": frac})

    # Robustness: PAI_total includes wood, so true green LAI is lower and NDVI
    # is less saturated. Report the explained fraction across plausible green LAI.
    print("\nSensitivity to assumed green LAI (NIRv is robust; NDVI is LAI-dependent):")
    print(f"{'LAI':>4} {'NDVI%':>7} {'NIRv%':>7} {'NDII%':>7}")
    lai_rows = []
    for lai_t in (2.0, 3.0, 4.0, 5.0, 6.0):
        sb, se = simulate(ang_base, lai_t), simulate(ang_evt, lai_t)
        fr = {}
        for idx in ("NDVI", "NIRv", "NDII"):
            d_obs = win_mean(obs[idx], EVENT) - win_mean(obs[idx], BASELINE)
            fr[idx] = (se[idx] - sb[idx]) / d_obs if d_obs else float("nan")
        print(f"{lai_t:>4.0f} {fr['NDVI']:>6.0%} {fr['NIRv']:>6.0%} {fr['NDII']:>6.0%}")
        lai_rows.append({"lai": lai_t, **{f"frac_{k}": v for k, v in fr.items()}})
    pd.DataFrame(lai_rows).to_csv(
        REPO_ROOT / "data" / "processed" / "prosail_lai_sensitivity.csv", index=False)

    # Sensitivity curves: indices vs leaf angle, LAI fixed.
    angles = np.arange(15, 75.1, 2.5)
    curves = {k: [] for k in ("NDVI", "NIRv", "NDII")}
    for ang in angles:
        s = simulate(float(ang), lai)
        for k in curves:
            curves[k].append(s[k])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, idx in zip(axes, ("NDVI", "NIRv", "NDII")):
        ax.plot(angles, curves[idx], "-", color="#2166ac")
        for ang, mk, lab in [(ang_base, "o", "baseline"), (ang_evt, "s", "event")]:
            ax.plot(ang, simulate(ang, lai)[idx], mk, color="#b2182b", ms=8, label=lab)
        ax.set(xlabel="mean leaf angle (deg)", ylabel=idx,
               title=f"{idx}  (LAI={lai:.1f}, Cab={FIXED['cab']:.0f} fixed)")
        ax.legend(fontsize=8)
    fig.suptitle("PROSAIL: leaf-angle-only sensitivity vs the observed event change",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    figp = REPO_ROOT / "figures" / "prosail_leaf_angle_sensitivity.png"
    figp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figp, dpi=140)
    plt.close(fig)

    outp = REPO_ROOT / "data" / "processed" / "prosail_leaf_angle.csv"
    outp.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outp, index=False)
    print("\nNote: NDII also responds to canopy water (fixed here), so its 'explained'")
    print("fraction is a lower bound on the structural (leaf-angle) contribution.")
    print(f"\nWrote {figp}\nWrote {outp}")


if __name__ == "__main__":
    main()
