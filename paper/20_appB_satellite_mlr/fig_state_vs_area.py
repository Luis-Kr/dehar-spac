"""
Stage 20 / Application B — APPENDIX figure: "canopy STATE change, not leaf AREA change".

The August VI zigzag (greenness/chlorophyll indices dip on 08-18 then rebound by 08-26) is driven by
the STATE of the existing leaves changing, not by a change in leaf AREA:
  - leaf colour (senescence pigments)  -> PSRI peaks exactly at the NDVI/CIg trough
  - leaf orientation (turgor/wilting)  -> hornbeam + bird-cherry leaf angle peak at the same date,
                                           then RECOVER (reversible) tracking the VI rebound
  - bulk leaf area (hemi-PAI)          -> FLAT through the acute window; the irreversible leaf-area
                                           loss only begins in September.
So leaf angle and leaf colour are COLLINEAR over the event (same-day peaks) and both sit on the
non-area side of the ledger; the genuine area control is the flat hemi-PAI, and the area loss is
TIME-SEPARATED (September). This figure makes that argument on one shared August->September axis.

Run:
  /home/lk1167/miniconda3/envs/dehar-spac/bin/python paper/20_appB_satellite_mlr/fig_state_vs_area.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "config"))
from paper_common import load_config, load_daily, paper_style, hornbeam_qc_cams, greenness_col  # noqa: E402

paper_style()

X0, X1 = pd.Timestamp("2025-07-25"), pd.Timestamp("2025-09-20")
TROUGH = pd.Timestamp("2025-08-18")          # NDVI/CIg trough = PSRI peak = leaf-angle peak
ACUTE = (pd.Timestamp("2025-08-06"), pd.Timestamp("2025-08-22"))
C_GREEN, C_CIG = "#1b7837", "#5aae61"        # satellite greenness / chlorophyll
C_GCC = "#35978f"                            # proximal phenocam greenness (GCC)
C_PSRI = "#b35806"                           # senescence pigment
C_HB, C_BC = "#2166ac", "#9970ab"            # leaf angle: hornbeam / bird cherry
C_HEMI, C_HINGE = "#8c510a", "#bf812d"       # PAI bulk / early-warning


def _scene(df, col):
    """S2 index on its actual clear-sky scene dates within the window (sparse markers)."""
    return df[col].dropna().loc[X0:X1]


def _angle(df, cams):
    s = df[[f"leaf_angle_cam{c}_daylight_mean_deg" for c in cams]].mean(axis=1).loc[X0:X1]
    return s, s.rolling(3, center=True, min_periods=1).mean()


def main() -> int:
    cfg = load_config()
    df = load_daily(cfg)
    kept, _ = hornbeam_qc_cams(df, cfg)
    bc = cfg["streams"]["leaf_angle"]["birdcherry_cams"]

    ndvi, cig, psri = _scene(df, "s2_ndvi_mean"), _scene(df, "s2_cig_mean"), _scene(df, "s2_psri_mean")
    gcc_raw = df[greenness_col(cfg)].loc[X0:X1]   # canonical greenness = PhenoCam GCC (ADR 0002)
    gcc = gcc_raw.rolling(3, center=True, min_periods=1).mean()
    hb_raw, hb = _angle(df, kept)
    bc_raw, bcs = _angle(df, bc)
    hemi = df["pai_hemi_hi_hinge_mean_m2m2"].loc[X0:X1]
    hinge = df["pai_hinge_hinge_mean_m2m2"].loc[X0:X1]

    fig, axes = plt.subplots(5, 1, figsize=(9, 13.5), sharex=True)
    axA, axB, axC, axD, axE = axes

    def decorate(ax):
        ax.axvspan(*ACUTE, color="#fdded2", alpha=0.6, zorder=0)
        ax.axvline(TROUGH, color="#b2182b", lw=1.2, ls="--", zorder=1)

    # --- A: greenness / chlorophyll (the zigzag) ---
    decorate(axA)
    axA.plot(ndvi.index, ndvi.values, "-o", color=C_GREEN, label="NDVI (greenness)")
    axA.set_ylabel("NDVI", color=C_GREEN)
    axA.tick_params(axis="y", labelcolor=C_GREEN)
    axAr = axA.twinx()
    axAr.plot(cig.index, cig.values, "-s", color=C_CIG, label="CIg (chlorophyll)")
    axAr.set_ylabel("CIg", color=C_CIG)
    axAr.tick_params(axis="y", labelcolor=C_CIG)
    axAr.grid(False)
    axA.set_title("A  Greenness / chlorophyll indices dip, then rebound", loc="left", fontsize=11)
    axA.annotate("trough\n08-18", (TROUGH, ndvi.loc[:TROUGH].iloc[-1]), xytext=(6, -28),
                 textcoords="offset points", fontsize=8, color="#b2182b")

    # --- B: proximal understory greenness (GCC) declines and plateaus -- NO rebound ---
    decorate(axB)
    axB.plot(gcc_raw.index, gcc_raw.values, color=C_GCC, lw=0.7, alpha=0.35)
    axB.plot(gcc.index, gcc.values, color=C_GCC, lw=2.4)
    axB.set_ylabel("GCC (phenocam)")
    axB.set_title("B  Proximal understory greenness (GCC) declines and plateaus  ->  NO rebound",
                  loc="left", fontsize=11)
    g_post = gcc.loc[ACUTE[1]:].iloc[0]
    axB.annotate("understory does NOT re-green\n(satellite rebound = geometry + evergreen, not this layer)",
                 (gcc.loc[ACUTE[1]:].index[3], g_post), xytext=(6, 22), textcoords="offset points",
                 fontsize=8, color=C_GCC,
                 arrowprops=dict(arrowstyle="->", color=C_GCC, lw=0.9))

    # --- C: senescence pigment (PSRI) climbs through the event ---
    decorate(axC)
    axC.plot(psri.index, psri.values, "-^", color=C_PSRI)
    axC.set_ylabel("PSRI")
    axC.set_title("C  Senescence pigment (PSRI) climbs through the event  ->  leaves turning red",
                  loc="left", fontsize=11)

    # --- D: leaf angle (hornbeam + bird cherry) peaks then RECOVERS (reversible) ---
    decorate(axD)
    axD.plot(hb_raw.index, hb_raw.values, color=C_HB, lw=0.7, alpha=0.35)
    axD.plot(hb.index, hb.values, color=C_HB, lw=2.4, label="hornbeam (QC)")
    axD.plot(bc_raw.index, bc_raw.values, color=C_BC, lw=0.7, alpha=0.35)
    axD.plot(bcs.index, bcs.values, color=C_BC, lw=2.4, label="bird cherry")
    axD.set_ylabel("leaf angle (deg)")
    axD.legend(fontsize=8, loc="upper right", framealpha=0.9)
    axD.set_title("D  Leaf angle peaks at the trough, then RECOVERS  ->  reversible turgor / wilting",
                  loc="left", fontsize=11)

    # --- E: bulk PAI flat through the event; leaf area falls only in September ---
    decorate(axE)
    axE.plot(hemi.index, hemi.values, color=C_HEMI, lw=2.4, label="hemi-PAI (bulk leaf area)")
    axE.plot(hinge.index, hinge.values, color=C_HINGE, lw=1.8, ls="--", label="hinge-PAI (early-warning)")
    axE.set_ylabel("PAI (m$^2$ m$^{-2}$)")
    axE.legend(fontsize=8, loc="lower left", framealpha=0.9)
    axE.set_title("E  Bulk leaf area (hemi-PAI) is FLAT in the event  ->  area loss only in September",
                  loc="left", fontsize=11)
    hemi_aug = hemi.loc[ACUTE[0]:ACUTE[1]]
    axE.annotate("area conserved\n(state changes, not amount)",
                 (hemi_aug.index[len(hemi_aug) // 2], hemi_aug.mean()), xytext=(0, 26),
                 textcoords="offset points", fontsize=8, color=C_HEMI, ha="center",
                 arrowprops=dict(arrowstyle="-", color=C_HEMI, lw=0.8))
    sep = hemi.loc["2025-09-05":X1]
    axE.annotate("leaf area lost\n(senescence)", (sep.index[len(sep) // 2], sep.mean()),
                 xytext=(0, -34), textcoords="offset points", fontsize=8, color=C_HEMI, ha="center",
                 arrowprops=dict(arrowstyle="->", color=C_HEMI, lw=0.9))

    axE.set_xlim(X0, X1)
    axE.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
    axE.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    for lab in axE.get_xticklabels():
        lab.set_rotation(45)
        lab.set_ha("right")

    # fig.suptitle("App B — the August VI change is canopy STATE (colour + angle + water), not leaf AREA\n"
    #              "leaf angle (peak 08-18) and colour co-vary over the event; hemi-PAI is the area control "
    #              "and stays flat until September", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = HERE / "outputs" / "appendix" / "state_vs_area.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)

    # quick console summary for the caption numbers
    print(f"kept hornbeam cams: {kept}")
    print(f"NDVI trough {ndvi.idxmin().date()} = {ndvi.min():.3f}; NDVI 08-26 = {ndvi.asof('2025-08-26'):.3f} (rebound)")
    print(f"GCC pre-event {gcc.loc[:ACUTE[0]].iloc[-1]:.4f} -> trough {gcc.min():.4f} ({gcc.idxmin().date()}) "
          f"-> 08-26 {gcc.asof('2025-08-26'):.4f} (NO rebound)")
    print(f"PSRI peak {psri.idxmax().date()} = {psri.max():.3f}")
    print(f"hornbeam angle peak {hb.idxmax().date()} = {hb.max():.1f} deg; bird cherry peak "
          f"{bcs.idxmax().date()} = {bcs.max():.1f} deg")
    print(f"hemi-PAI acute mean {hemi.loc[ACUTE[0]:ACUTE[1]].mean():.2f}; "
          f"hemi-PAI 2025-09-15 {hemi.asof('2025-09-15'):.2f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
