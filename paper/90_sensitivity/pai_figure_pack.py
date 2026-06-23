"""
Stage 90 — Figure pack for the paper/appendix (issue #6). All HingePAI method.

Generates a spread of candidate figures (hinge vs hemi scan), to be cherry-picked:
  1  season canopy-total HingePAI (hinge vs hemi)
  2  PAI(zenith) x time heatmap, per scan        (HingePAI per zenith ring)
  3  PAVD(height) x time heatmap, per scan        (HingePAVD vertical layers)
  4  season gap fraction at the hinge ring (raw signal, hinge vs hemi)
  5  gap fraction(zenith) x time heatmap, per scan (raw Pgap)
  6  gap fraction(height) x time heatmap @57.5 deg, per scan (vertical gap build-up)

Per-zenith canopy-total Pgap = min over height of Pgap_Z* (= fully accumulated gap);
HingePAI = -1.1*ln(Pgap). quality_all scans, daily aggregation.

Run:
  /home/lk1167/miniconda3/envs/dehar-spac/bin/python paper/90_sensitivity/pai_figure_pack.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "config"))
from paper_common import paper_style  # noqa: E402

paper_style()

ROOT = HERE.parents[1]
LEAF_DIR = ROOT / "data" / "processed" / "proximal_rs" / "leaf"
OUT = HERE / "outputs" / "appendix"
OUT.mkdir(parents=True, exist_ok=True)
SCANS = {"hinge": "hinge scan", "hemi_hi": "hemi_hi scan"}
C = {"hinge": "#1b7837", "hemi_hi": "#762a83"}
MAXH = 25


def _load(scan: str) -> pd.DataFrame:
    d = pd.read_parquet(LEAF_DIR / f"leaf_{scan}_2025.parquet")
    d["datetime"] = pd.to_datetime(d["datetime"], utc=True)
    return d


def per_zenith_pgap(scan: str) -> pd.DataFrame:
    """Daily canopy-total Pgap per zenith bin (quality_all, daily mean)."""
    d = _load(scan)
    pcols = sorted([c for c in d.columns if c.startswith("Pgap_Z")])
    qual = d.groupby("datetime")["quality_all"].first()
    pg = d.groupby("datetime")[pcols].min()
    pg = pg[qual.reindex(pg.index).fillna(False)]
    pg.columns = [float(c.replace("Pgap_Z", "")) for c in pg.columns]
    return pg.resample("1D").mean()


def height_matrix(scan: str, col: str) -> pd.DataFrame:
    """height x date matrix (daily median, quality_all, interior gaps filled)."""
    d = _load(scan)
    sel = d[d["quality_all"]]
    piv = sel.pivot_table(values=col, index="datetime", columns="height", aggfunc="median")
    daily = piv.resample("1D").median().interpolate(method="time", limit_area="inside")
    return daily.T.sort_index()


def hingepai(pg):
    return -1.1 * np.log(np.clip(pg, 1e-5, None))


def _heat(ax, mat, cmap, vmax=None, vmin=None, label=""):
    x = mdates.date2num(pd.to_datetime(mat.columns))
    y = mat.index.astype(float).to_numpy()
    mesh = ax.pcolormesh(x, y, mat.values, cmap=cmap, shading="nearest",
                         vmin=vmin, vmax=vmax)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    cb = ax.figure.colorbar(mesh, ax=ax, pad=0.015)
    cb.set_label(label, fontsize=9)
    return mesh


# ── 1. season canopy-total HingePAI ───────────────────────────────────────────
def fig_season_pai():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for scan in SCANS:
        s = hingepai(per_zenith_pgap(scan)[57.5]).dropna()
        ax.plot(s.index, s.values, color=C[scan], lw=2.4, label=SCANS[scan])
    ax.set(ylabel=r"HingePAI ($m^2\,m^{-2}$)", xlabel="2025",
           title="Canopy-total HingePAI over the season (57.5 deg ring)")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "pack1_hingepai_season.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── 2. PAI(zenith) x time heatmap ─────────────────────────────────────────────
def fig_pai_zenith_heat():
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    vmax = 7
    for ax, scan in zip(axes, SCANS):
        m = hingepai(per_zenith_pgap(scan)).T          # zenith x date
        m = m.dropna(how="all")
        _heat(ax, m, "viridis", vmax=vmax, vmin=0, label=r"HingePAI ($m^2 m^{-2}$)")
        ax.set(ylabel="zenith (deg)", title=f"{SCANS[scan]} — PAI per zenith ring")
        ax.axhline(57.5, color="w", ls="--", lw=1)
    axes[-1].set_xlabel("2025")
    fig.tight_layout()
    fig.savefig(OUT / "pack2_pai_zenith_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── 3. PAVD(height) x time heatmap ────────────────────────────────────────────
def fig_pavd_heat():
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    mats = {s: height_matrix(s, "HingePAVD") for s in SCANS}
    vmax = np.nanpercentile(np.concatenate([m.values.ravel() for m in mats.values()]), 98)
    for ax, scan in zip(axes, SCANS):
        _heat(ax, mats[scan], "Greens", vmax=vmax, vmin=0, label=r"HingePAVD ($m^2 m^{-3}$)")
        ax.set(ylabel="height (m)", ylim=(0, MAXH),
               title=f"{SCANS[scan]} — vertical foliage layers (HingePAVD)")
    axes[-1].set_xlabel("2025")
    fig.tight_layout()
    fig.savefig(OUT / "pack3_pavd_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── 4. raw gap fraction at the hinge ring ─────────────────────────────────────
def fig_gap_season():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for scan in SCANS:
        s = per_zenith_pgap(scan)[57.5].dropna()
        ax.plot(s.index, s.values, color=C[scan], lw=2.4, label=SCANS[scan])
    ax.set(ylabel="gap fraction Pgap @ 57.5 deg", yscale="log", xlabel="2025",
           title="Raw signal: canopy gap fraction at the hinge ring "
                 "(low gap = dense canopy)")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "pack4_gapfraction_season.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── 5. gap fraction(zenith) x time heatmap ────────────────────────────────────
def fig_gap_zenith_heat():
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for ax, scan in zip(axes, SCANS):
        m = per_zenith_pgap(scan).T.dropna(how="all")
        _heat(ax, np.log10(m), "magma_r", label=r"log$_{10}$ gap fraction")
        ax.set(ylabel="zenith (deg)", title=f"{SCANS[scan]} — gap fraction per zenith ring")
        ax.axhline(57.5, color="w", ls="--", lw=1)
    axes[-1].set_xlabel("2025")
    fig.tight_layout()
    fig.savefig(OUT / "pack5_gapfraction_zenith_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── 6. gap fraction(height) x time @ 57.5 deg ─────────────────────────────────
def fig_gap_height_heat():
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for ax, scan in zip(axes, SCANS):
        m = height_matrix(scan, "Pgap_Z057.5")
        _heat(ax, m, "magma_r", vmin=0, vmax=1, label="gap fraction @ 57.5 deg")
        ax.set(ylabel="height (m)", ylim=(0, MAXH),
               title=f"{SCANS[scan]} — gap fraction vs height (gaps close upward)")
    axes[-1].set_xlabel("2025")
    fig.tight_layout()
    fig.savefig(OUT / "pack6_gapfraction_height_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for fn in (fig_season_pai, fig_pai_zenith_heat, fig_pavd_heat,
               fig_gap_season, fig_gap_zenith_heat, fig_gap_height_heat):
        fn()
        print(f"  done: {fn.__name__}")
    print(f"\n6 candidate figures in {OUT}/pack*.png")


if __name__ == "__main__":
    main()
