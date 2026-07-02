"""Interactive "up"-vector inspector for LEAF hemispherical scans (off-pipeline).

A small Flask app to hand-check the seasonal up-drift correction (ADR 0005,
issue #11). For one representative non-noisy scan per day it serves the raw
per-beam encoder/range arrays; the browser re-folds the scan live as you drag an
"up" slider (zen = |s - up|, azimuth flips by pi where s < up -- exactly
``dehar.proximal_rs.leaf.recenter_leaf_data``) and redraws the canopy-height
fisheye in WebGL, so it updates instantly with no server round-trip. The
smoothed seasonal "up" for the day is the preloaded default; submit stores your
chosen value to a CSV.

Run
---
    /home/lk1167/miniconda3/envs/dehar-spac/bin/python other/up_inspector/server.py
    # then open http://127.0.0.1:8000

Options: --host --port --year --max-points --night-hour
Output CSV: other/up_inspector/up_manual_corrections.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
from dehar.proximal_rs.leaf import read_leaf_scan  # noqa: E402

RAW_DIR = REPO / "data/raw/proximal_rs/leaf"
SENSOR_HEIGHT = 1.5
OUT_CSV = HERE / "up_manual_corrections.csv"
CSV_FIELDS = [
    "date", "filename", "scan_hour", "up_default_smooth",
    "up_manual", "delta", "color_p_lo", "color_p_hi", "note", "saved_at_utc",
]

app = Flask(__name__, static_folder=str(HERE / "static"), static_url_path="/static")


# ── day / scan index ──────────────────────────────────────────────────────────
def _circular_night_dist(hour: int, night: int) -> int:
    d = abs(int(hour) - night)
    return min(d, 24 - d)


def build_days(parquet: Path, up_daily: Path, night_hour: int) -> tuple[list[dict], dict]:
    """One entry per date: all scans (hour/quality/pai/ts), smoothed fallback up, rep.

    The representative is the *least-checked* clean scan: a clean scan not yet in the
    corrections CSV, chosen as far as possible in time-of-day from any already-corrected
    scan that day. So each pass (round) lands on a fresh scan and coverage grows.
    Returns ``(days, filename->epoch_seconds)``.
    """
    h = pd.read_parquet(parquet, columns=[
        "datetime", "filename", "scan_hour", "WeightedPAI", "quality_all"])
    g = (h.groupby("datetime")
         .agg(filename=("filename", "first"),
              hour=("scan_hour", "first"),
              quality=("quality_all", "first"),
              pai=("WeightedPAI", "max"))
         .reset_index())
    g["date"] = g["datetime"].dt.strftime("%Y-%m-%d")
    g["ts"] = g["datetime"].apply(lambda t: int(t.timestamp()))   # epoch seconds (UTC)
    fn_ts = dict(zip(g["filename"], g["ts"].astype(int)))

    up = pd.read_csv(up_daily, parse_dates=["date"])
    up["date"] = pd.to_datetime(up["date"], utc=True)
    up_s = up.set_index("date")["up_smooth_deg"].sort_index()
    dates = sorted(g["date"].unique())
    sd_ts = pd.to_datetime(dates).tz_localize("UTC")
    pos = up_s.index.get_indexer(sd_ts, method="nearest")
    smooth_up = {d: round(float(up_s.iloc[p]), 2) for d, p in zip(dates, pos)}

    corrected_by_date: dict[str, set] = {}
    corrected_fns: set = set()
    for r in _read_csv_rows():
        corrected_fns.add(r["filename"])
        corrected_by_date.setdefault(r["date"], set()).add(r["filename"])

    days = []
    for d, sub in g.groupby("date"):
        scans = [
            {"filename": r.filename, "hour": int(r.hour), "quality": bool(r.quality),
             "pai": round(float(r.pai), 2), "ts": int(r.ts)}
            for r in sub.sort_values("hour").itertuples()
        ]
        done_here = corrected_by_date.get(d, set())
        done_hours = [s["hour"] for s in scans if s["filename"] in done_here]
        fresh = [s for s in scans if s["quality"] and s["filename"] not in corrected_fns]
        if fresh and done_hours:                    # farthest in time-of-day from a checked scan
            rep = max(fresh, key=lambda s: min(_circular_night_dist(s["hour"], h) for h in done_hours))
        elif fresh:                                 # nothing checked here yet -> avoid the night default
            rep = max(fresh, key=lambda s: _circular_night_dist(s["hour"], night_hour))
        else:                                       # all clean scans checked -> fall back to a clean one
            pool = [s for s in scans if s["quality"]] or scans
            rep = min(pool, key=lambda s: _circular_night_dist(s["hour"], night_hour))
        days.append({
            "date": d,
            "default_up": smooth_up[d],
            "n_clean": sum(s["quality"] for s in scans),
            "n_done_here": len(done_here),
            "all_noisy": not any(s["quality"] for s in scans),
            "rep": rep["filename"],
            "scans": scans,
        })
    return days, fn_ts


# ── per-scan refold arrays (cached) ───────────────────────────────────────────
@lru_cache(maxsize=24)
def _scan_arrays(filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(s, az_raw [rad], range1, range2) for every beam of a raw scan."""
    leaf = read_leaf_scan(str(RAW_DIR / filename), sensor_height=SENSOR_HEIGHT,
                          transform_cfg={"enabled": False})
    d = leaf.data
    nsteps = 2.56e4 if leaf.header["Firmware ver."] >= 4.11 else 1e4
    s = d["scan_encoder"].to_numpy() / nsteps * 2 * np.pi + leaf.zenith_offset
    az = d["rotary_encoder"].to_numpy() / 2e4 * 2 * np.pi
    return s, az, d["range1"].to_numpy(float), d["range2"].to_numpy(float)


def _round_or_none(a: np.ndarray, nd: int) -> list:
    return [None if not np.isfinite(v) else round(float(v), nd) for v in a]


# ── routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/days")
def api_days():
    return jsonify({"days": app.config["DAYS"], "saved": _load_saved(),
                    "corrections": _corrections(), "sensor_height": SENSOR_HEIGHT})


def _corrections() -> list[dict]:
    """Sorted [{ts, up}] from the CSV — the manual up curve, for nearest-in-time lookup."""
    fn_ts = app.config.get("FN_TS", {})
    out = []
    for r in _read_csv_rows():
        ts = fn_ts.get(r["filename"])
        if ts is None:                              # filename not in this season's parquet
            try:
                ts = int(pd.Timestamp(r["date"], tz="UTC").timestamp()) \
                    + int(float(r.get("scan_hour") or 0)) * 3600
            except (ValueError, TypeError):
                continue
        out.append({"ts": int(ts), "up": float(r["up_manual"])})
    out.sort(key=lambda c: c["ts"])
    return out


def _scan_indices(fn: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(s, az, r1, r2, idx) for a scan; idx = finite-return rows, capped subsample."""
    s, az, r1, r2 = _scan_arrays(fn)
    idx = np.flatnonzero(np.isfinite(r1) | np.isfinite(r2))
    cap = app.config["MAX_POINTS"]
    if idx.size > cap:
        idx = np.sort(np.random.default_rng(0).choice(idx, cap, replace=False))
    return s, az, r1, r2, idx


@app.route("/api/scan.bin")
def api_scan_bin():
    """Binary payload: uint32 n, then float32 s[n], az[n], r1[n], r2[n] (NaN = no return).

    ~4x smaller and far faster to parse than JSON, so we can ship enough returns to
    match the dense reference fisheye without a sluggish scan switch.
    """
    fn = request.args.get("file", "")
    if not (RAW_DIR / fn).exists():
        return ("scan not found", 404)
    s, az, r1, r2, idx = _scan_indices(fn)
    n = int(idx.size)
    out = np.empty(4 * n, dtype="<f4")
    out[0:n], out[n:2 * n], out[2 * n:3 * n], out[3 * n:4 * n] = \
        s[idx], az[idx], r1[idx], r2[idx]
    body = np.array([n], dtype="<u4").tobytes() + out.tobytes()
    return app.response_class(body, mimetype="application/octet-stream")


@app.route("/api/scan")
def api_scan():
    fn = request.args.get("file", "")
    if not (RAW_DIR / fn).exists():
        return jsonify({"error": f"scan not found: {fn}"}), 404
    s, az, r1, r2 = _scan_arrays(fn)
    fin = np.isfinite(r1) | np.isfinite(r2)
    idx = np.flatnonzero(fin)
    cap = app.config["MAX_POINTS"]
    if idx.size > cap:
        idx = np.sort(np.random.default_rng(0).choice(idx, cap, replace=False))
    return jsonify({
        "filename": fn,
        "sensor_height": SENSOR_HEIGHT,
        "n": int(idx.size),
        "s": np.round(s[idx], 5).tolist(),
        "az": np.round(az[idx], 5).tolist(),
        "r1": _round_or_none(r1[idx], 3),
        "r2": _round_or_none(r2[idx], 3),
    })


@app.route("/api/save", methods=["POST"])
def api_save():
    p = request.get_json(force=True)
    row = {
        "date": p["date"], "filename": p["filename"],
        "scan_hour": p.get("scan_hour", ""),
        "up_default_smooth": p.get("up_default_smooth", ""),
        "up_manual": round(float(p["up_manual"]), 3),
        "delta": round(float(p["up_manual"]) - float(p.get("up_default_smooth") or 0), 3),
        "color_p_lo": p.get("color_p_lo", ""),
        "color_p_hi": p.get("color_p_hi", ""),
        "note": p.get("note", ""),
        "saved_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    rows = _read_csv_rows()
    rows = [r for r in rows if r.get("filename") != row["filename"]]   # one row per scan (latest wins)
    rows.append(row)
    rows.sort(key=lambda r: (r["date"], r["filename"]))
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return jsonify({"ok": True, "saved": _load_saved(),
                    "corrections": _corrections(), "n_saved": len(rows)})


def _read_csv_rows() -> list[dict]:
    if not OUT_CSV.exists():
        return []
    with OUT_CSV.open(newline="") as f:
        return list(csv.DictReader(f))


def _load_saved() -> dict:
    return {r["filename"]: {"up_manual": float(r["up_manual"]), "date": r["date"]}
            for r in _read_csv_rows()}


def main() -> int:
    global OUT_CSV
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--year", default="2025")
    ap.add_argument("--max-points", type=int, default=400000)
    ap.add_argument("--night-hour", type=int, default=23)
    ap.add_argument("--csv", default=str(OUT_CSV),
                    help="corrections CSV to read/write (point tests at a throwaway copy)")
    a = ap.parse_args()
    OUT_CSV = Path(a.csv)

    parquet = REPO / "data/processed/proximal_rs/leaf/leaf_hemi_hi_2025.parquet"
    up_daily = REPO / "data/processed/proximal_rs/leaf/leaf_up_daily_2025.csv"
    days, fn_ts = build_days(parquet, up_daily, a.night_hour)
    app.config["DAYS"] = days
    app.config["FN_TS"] = fn_ts
    app.config["MAX_POINTS"] = a.max_points
    n_done = len(_load_saved())
    print(f"[up-inspector] {len(days)} days indexed (year {a.year}); "
          f"{n_done} scans already corrected; CSV -> {OUT_CSV}")
    print(f"[up-inspector] open http://{a.host}:{a.port}")
    app.run(host=a.host, port=a.port, threaded=True, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
