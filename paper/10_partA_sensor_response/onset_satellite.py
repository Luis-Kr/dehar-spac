"""
Stage 10 / Part A — satellite-focused onset set.

A SECOND onset/cascade caterpillar (separate from the headline), restricted to the key in-situ
anchors + ALL satellite indices: when does each satellite index DEPART relative to SWP / VOD /
leaf angle / PAI? Same engine + same plot style as onset_bootstrap.py; distinct output filenames
so the headline onset_* outputs are untouched.

Stream set (23 rows): Predawn SWP, GNSS-T VOD, leaf angle (hornbeam QC + bird cherry), PAI hinge,
PAI hemi, all S1 (VH/VV/SPAN/CR/RVI), curated App-B S2 (12 indices).

Run:
  /home/lk1167/miniconda3/envs/dehar-spac/bin/python paper/10_partA_sensor_response/onset_satellite.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "config"))
import onset_bootstrap as ob  # noqa: E402


def curated_streams() -> list[ob.Stream]:
    s2 = ["ndvi", "nirv", "kndvi", "evi", "cire", "mtci", "ndrei", "ndii", "ndmi", "nmdi", "psri", "cig"]
    streams = [
        ob.Stream("Predawn SWP", "swp_pd_mean_MPa", "decrease", "physiology"),
        ob.Stream("GNSS-T VOD", "vod_predawn_mean", "decrease", "proximal"),
        ob.Stream("Leaf angle (hornbeam)", "leaf_angle_hornbeam_daylight_mean", "auto", "proximal"),
        ob.Stream("Leaf angle (bird cherry)", "leaf_angle_birdcherry_daylight_mean", "auto", "proximal"),
        ob.Stream("PAI (proximal hemi)", "pai_hemi_hi_hinge_mean_m2m2", "decrease", "proximal"),
        ob.Stream("PAI hinge-scan (comparison)", "pai_hinge_hinge_mean_m2m2", "decrease", "proximal"),
        ob.Stream("S1 VH", "s1_asc_vh_mean_dB", "auto", "satellite"),
        ob.Stream("S1 VV", "s1_asc_vv_mean_dB", "auto", "satellite"),
        ob.Stream("S1 SPAN", "s1_asc_span_mean_dB", "auto", "satellite"),
        ob.Stream("S1 CR", "s1_asc_cr_mean_dB", "auto", "satellite"),
        ob.Stream("S1 RVI", "s1_asc_rvi_mean", "auto", "satellite"),
    ]
    streams += [ob.Stream(f"S2 {i.upper()}", f"s2_{i}_mean", "auto", "satellite") for i in s2]
    return streams


def main() -> int:
    cfg = ob.load_config()
    ev = ob.load_event_windows()
    ref_date = ev.loc[ev.stage == "soil_limit", "start"].iloc[0]
    acute_start = ev.loc[ev.stage == "acute", "start"].iloc[0]
    peak = pd.Timestamp("2025-08-20")

    _unused, df = ob.build_streams_and_frame(cfg)   # df carries the derived leaf-angle ensembles
    streams = curated_streams()
    win = [pd.Timestamp(d) for d in cfg["onset"]["search_window"]]
    df = df.loc[win[0]:win[1]]
    n_boot = int(os.environ.get("ONSET_NBOOT", cfg["onset"].get("n_boot", 1000)))

    tdir = HERE / "outputs" / "tables"; tdir.mkdir(parents=True, exist_ok=True)
    fdir = HERE / "outputs" / "figures"; fdir.mkdir(parents=True, exist_ok=True)
    marks = [("soil-limit", ref_date, "#8c8c8c"), ("acute start", acute_start, "#b2182b"),
             ("Psi min", peak, "#1b7837")]
    rule_titles = {
        "first-departure": "Part A — satellite-index onset vs in-situ anchors (FIRST DEPARTURE)",
        "acute-event": "Part A — satellite-index onset vs in-situ anchors (ACUTE EVENT)",
    }
    fname = {"first-departure": "first_departure", "acute-event": "acute_event"}

    for rule in ["first-departure", "acute-event"]:
        res = ob.run_rule(df, streams, rule, ref_date, model="l2", pen_scale=2.0,
                          n_boot=n_boot, block=5, seed=ob.SEED)
        res.to_csv(tdir / f"onset_satellite_{fname[rule]}.csv", index=False)
        ob.plot_caterpillar(res, fdir / f"onset_caterpillar_satellite_{fname[rule]}.png",
                            ref_date, marks, rule_titles[rule])

        grid = ob.grid_search(df, streams, rule, ref_date)
        summ = ob.stability_summary(grid)
        summ.to_csv(tdir / f"onset_grid_stability_satellite_{fname[rule]}.csv", index=False)
        ob.plot_grid_stability(grid, summ, fdir / f"onset_grid_robustness_satellite_{fname[rule]}.png",
                               rule_titles[rule])

        show = res.copy()
        for c in ("onset_date", "ci_lo", "ci_hi"):
            show[c] = show[c].dt.strftime("%Y-%m-%d")
        print(f"\n=== {rule} ===")
        print(show[["stream", "group", "n_obs", "onset_date", "ci_lo", "ci_hi",
                    "ci_days", "onset_rel_days"]].to_string(index=False, na_rep="—"))
        print(f"ordering robustness: mean Kendall-tau = {summ.attrs['mean_tau']:.2f}, "
              f"min = {summ.attrs['min_tau']:.2f}")

    print(f"\nwrote tables -> {tdir}\nwrote figures -> {fdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
