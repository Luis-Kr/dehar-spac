"""
Compute Green Chromatic Coordinate (GCC) from angle-camera time-lapse videos
at Hartheim (Dehar SPAC), 01 May – 30 November 2025.

GCC  =  mean(G) / ( mean(R) + mean(G) + mean(B) )

computed per RGB image over the full frame.

Raw-data locations (merged transparently)
-----------------------------------------
LOCAL_ROOT  2025-05-01 – 2025-09-15
    /mnt/data/lk1167/projects/other/data/icos-har/raw

NAS_ROOT    2025-09-16 – 2025-11-30
    /mnt/gsdata/projects/icos_har/anglecam/data/raw/videos_sept_oct_nov

Directory layout (same for both roots):
    <root>/<YYYY-MM-DD>/<camera>/<camera>_frame_data.csv
    <root>/<YYYY-MM-DD>/<camera>/<camera>_output_video.mp4

Output
------
One parquet per camera (fast, compressed, resumable):
    data/processed/proximal_rs/phenology/gcc_hartheim_2025_<camera>.parquet

Columns: timestamp (str "YYYY-MM-DD HH:MM:00"), gcc (float32, 3 dp)

Intermediate scratch (written immediately after each cam-day completes):
    data/processed/proximal_rs/phenology/scratch/<camera>_<YYYY-MM-DD>.parquet

On re-run, existing scratch files are skipped — full resume capability.
"""

import logging
import os
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
LOCAL_ROOT  = Path("/mnt/data/lk1167/projects/other/data/icos-har/raw")
NAS_ROOT    = Path("/mnt/gsdata/projects/icos_har/anglecam/data/raw/videos_sept_oct_nov")
OUT_DIR     = Path("data/processed/proximal_rs/phenology")
SCRATCH_DIR = OUT_DIR / "scratch"

# ── Date range ─────────────────────────────────────────────────────────────────
DATE_START = date(2025, 5,  1)
DATE_END   = date(2025, 11, 30)
DATE_SPLIT = date(2025, 9,  16)   # first date on NAS_ROOT

CAMERAS = [f"G5Bullet_{i}" for i in range(60, 71)]   # _60 … _70

# ── Parallelism ────────────────────────────────────────────────────────────────
_NCPU         = os.cpu_count() or 8
LOCAL_WORKERS = max(1, _NCPU - 30)
NAS_WORKERS   = 4


# ── Worker (runs in subprocess) ────────────────────────────────────────────────

def _gcc_image(frame: np.ndarray) -> float:
    """GCC = mean(G) / (mean(R) + mean(G) + mean(B)).  BGR uint8 input."""
    b = float(frame[:, :, 0].mean())
    g = float(frame[:, :, 1].mean())
    r = float(frame[:, :, 2].mean())
    denom = r + g + b
    return round(g / denom, 3) if denom > 0.0 else float("nan")


def process_cam_day(
    cam: str, day: date, root: Path
) -> list[tuple[str, float]]:
    """Process one camera × day.

    Returns list of (timestamp_str, gcc) where timestamp_str is
    "YYYY-MM-DD HH:MM:00" (floored to minute, seconds always :00).
    """
    cam_dir  = root / day.isoformat() / cam
    csv_path = cam_dir / f"{cam}_frame_data.csv"
    vid_path = cam_dir / f"{cam}_output_video.mp4"

    if not csv_path.exists() or not vid_path.exists():
        return []

    try:
        meta = pd.read_csv(
            csv_path,
            usecols=lambda c: c in {"frame_number", "frame_date", "mode"},
            dtype={"frame_number": np.int32, "mode": str},
        )
        meta["frame_date"] = pd.to_datetime(meta["frame_date"], errors="coerce")
        meta = meta.dropna(subset=["frame_date"])
        rgb_rows = meta[meta["mode"] == "RGB"].set_index("frame_number")
    except Exception as exc:
        log.warning("CSV parse error  %s  %s: %s", day, cam, exc)
        return []

    if rgb_rows.empty:
        return []

    rgb_sorted = sorted(rgb_rows.index.tolist())
    rgb_set    = set(rgb_sorted)
    # Floor to minute, encode seconds as literal :00
    rgb_ts = rgb_rows["frame_date"].dt.floor("1min").dt.strftime("%Y-%m-%d %H:%M:00")

    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        log.warning("Cannot open video  %s  %s", day, cam)
        return []

    cap.set(cv2.CAP_PROP_POS_FRAMES, rgb_sorted[0])
    fn      = rgb_sorted[0]
    fn_last = rgb_sorted[-1]

    records: list[tuple[str, float]] = []
    while fn <= fn_last:
        ret, frame = cap.read()
        if not ret:
            break
        if fn in rgb_set:
            records.append((rgb_ts.loc[fn], _gcc_image(frame)))
        fn += 1
    cap.release()
    return records


# ── Scratch helpers ────────────────────────────────────────────────────────────

_EMPTY_SCHEMA = {"timestamp": pd.Series(dtype="str"), "gcc": pd.Series(dtype="float32")}


def _scratch_path(cam: str, day: date) -> Path:
    return SCRATCH_DIR / f"{cam}_{day.isoformat()}.parquet"


def _save_scratch(records: list[tuple], cam: str, day: date) -> None:
    path = _scratch_path(cam, day)
    if not records:
        pd.DataFrame(_EMPTY_SCHEMA).to_parquet(path, index=False)
        return
    df = pd.DataFrame(records, columns=["timestamp", "gcc"])
    df["gcc"] = df["gcc"].astype("float32")
    df.to_parquet(path, index=False)


# ── Task building (skips already-finished scratch files) ──────────────────────

def _build_tasks() -> tuple[list[tuple], list[tuple]]:
    local_tasks: list[tuple] = []
    nas_tasks:   list[tuple] = []
    d = DATE_START
    while d <= DATE_END:
        root = LOCAL_ROOT if d < DATE_SPLIT else NAS_ROOT
        for cam in CAMERAS:
            if _scratch_path(cam, d).exists():
                continue   # already processed; will be picked up in merge
            if d < DATE_SPLIT:
                local_tasks.append((cam, d, root))
            else:
                nas_tasks.append((cam, d, root))
        d += timedelta(days=1)
    return local_tasks, nas_tasks


# ── Merge scratch → per-camera parquet ────────────────────────────────────────

def _merge_cameras() -> None:
    for cam in CAMERAS:
        scratch_files = sorted(SCRATCH_DIR.glob(f"{cam}_*.parquet"))
        if not scratch_files:
            log.warning("No scratch files for %s — skipping", cam)
            continue

        dfs: list[pd.DataFrame] = []
        for f in scratch_files:
            try:
                chunk = pd.read_parquet(f)
                if not chunk.empty:
                    dfs.append(chunk)
            except Exception as exc:
                log.warning("Unreadable scratch file %s: %s", f.name, exc)

        if not dfs:
            log.warning("No data for %s", cam)
            continue

        df = pd.concat(dfs, ignore_index=True)
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        out_path = OUT_DIR / f"gcc_hartheim_2025_{cam}.parquet"
        df.to_parquet(out_path, index=False)
        log.info("%-20s  %7d rows  →  %s", cam, len(df), out_path.name)


# ── Orchestration ──────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    existing = len(list(SCRATCH_DIR.glob("*.parquet")))
    if existing:
        log.info(
            "Resuming: %d scratch files already present (those cam-days will be skipped).",
            existing,
        )

    local_tasks, nas_tasks = _build_tasks()
    total = len(local_tasks) + len(nas_tasks)

    if total == 0:
        log.info("All cam-days already processed. Proceeding straight to merge.")
    else:
        log.info(
            "Tasks: %d local (workers=%d)  +  %d NAS (workers=%d)  =  %d total",
            len(local_tasks), LOCAL_WORKERS,
            len(nas_tasks),   NAS_WORKERS,
            total,
        )

        done = 0
        lock = threading.Lock()

        def _run_pool(tasks: list[tuple], max_workers: int) -> None:
            nonlocal done
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futs = {pool.submit(process_cam_day, *t): t for t in tasks}
                for fut in as_completed(futs):
                    cam, day, _ = futs[fut]
                    try:
                        recs = fut.result()
                    except Exception as exc:
                        log.error("Worker error  %s  %s: %s", day, cam, exc)
                        recs = []
                    # Save immediately — crash-safe, no data accumulation in RAM
                    try:
                        _save_scratch(recs, cam, day)
                    except Exception as exc:
                        log.error("Scratch save failed  %s  %s: %s", day, cam, exc)
                    with lock:
                        done += 1
                        n = done
                    if n % 200 == 0 or n == total:
                        log.info("Progress: %d / %d cam-days  (%.0f%%)", n, total, 100 * n / total)

        t_local = threading.Thread(target=_run_pool, args=(local_tasks, LOCAL_WORKERS))
        t_nas   = threading.Thread(target=_run_pool, args=(nas_tasks,   NAS_WORKERS))
        t_local.start()
        t_nas.start()
        t_local.join()
        t_nas.join()

        log.info("All workers done. Merging scratch files into per-camera parquet ...")

    _merge_cameras()
    log.info("Done. Output: %s", OUT_DIR)


if __name__ == "__main__":
    main()
