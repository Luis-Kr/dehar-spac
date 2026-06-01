"""Teaching demo: bootstrap confidence intervals for a breakpoint (onset) date.

This is a *self-contained, synthetic-data* illustration of how we put an
uncertainty band on a detected onset, so that "sensor X responded before
sensor Y" becomes a defensible statement instead of an eyeballed one.

The idea in one sentence
------------------------
Your data = a clean step (the onset) + random day-to-day wiggle (noise).
You only observed *one* realisation of that wiggle. To find out how much the
detected onset would move if the wiggle had come out differently, you reuse
the wiggle you actually measured (the residuals), reshuffle it in short
blocks, glue it back onto the same step, and re-detect the onset — a thousand
times. The spread of those thousand onsets is the uncertainty.

Run it
------
    python analysis/cascade/breakpoint_bootstrap_demo.py

It prints onset dates with 95% CIs and writes two figures to ``figures/``:
    - breakpoint_bootstrap_single.png   (one series + its onset CI + histogram)
    - breakpoint_bootstrap_caterpillar.png  (5 sensors -> fast vs slow clusters)

Notes for the real pipeline
---------------------------
- The single-breakpoint search here is exhaustive least-squares (transparent
  and exact for one change). For multiple breakpoints, swap in
  ``ruptures.Pelt`` (already a project dependency) — the bootstrap logic is
  identical: re-fit on each synthetic series and record the onset.
- Block resampling (not single-day) preserves the autocorrelation of daily
  data; resampling single days would make the CI falsely narrow.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: save figures, no display
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"
SEED = 20250801


# --------------------------------------------------------------------------- #
# 1. Toy data                                                                 #
# --------------------------------------------------------------------------- #
def make_toy_series(
    rng: np.random.Generator,
    n_before: int,
    n_after: int,
    mean_before: float,
    mean_after: float,
    noise_sd: float,
    ar1: float = 0.5,
) -> np.ndarray:
    """Return a daily series: a step from ``mean_before`` to ``mean_after``
    plus sticky (autocorrelated, AR(1)) noise.

    ar1 controls "stickiness": 0 = white noise, ->1 = strongly persistent.
    Real sensor data is sticky, which is exactly why we block-bootstrap below.
    """
    n = n_before + n_after
    step = np.concatenate([np.full(n_before, mean_before),
                           np.full(n_after, mean_after)])
    noise = np.zeros(n)
    noise[0] = rng.normal(0, noise_sd)
    for t in range(1, n):  # AR(1): today's noise partly echoes yesterday's
        noise[t] = ar1 * noise[t - 1] + rng.normal(0, noise_sd * (1 - ar1**2) ** 0.5)
    return step + noise


# --------------------------------------------------------------------------- #
# 2. Single-breakpoint detector (transparent stand-in for PELT with 1 change) #
# --------------------------------------------------------------------------- #
def fit_single_breakpoint(y: np.ndarray, min_seg: int = 3) -> int:
    """Return the index ``cp`` (first day of the 'after' segment) that splits
    ``y`` into two pieces with the smallest total within-segment error.

    cost(cp) = SSE(before) + SSE(after), where SSE = sum of squared deviations
    from the segment mean. This is the one-change-point version of what PELT
    optimises.
    """
    n = len(y)
    candidates = range(min_seg, n - min_seg + 1)
    costs = []
    for cp in candidates:
        before, after = y[:cp], y[cp:]
        sse = ((before - before.mean()) ** 2).sum() + ((after - after.mean()) ** 2).sum()
        costs.append(sse)
    return list(candidates)[int(np.argmin(costs))]


def skeleton_and_residuals(y: np.ndarray, cp: int) -> tuple[np.ndarray, np.ndarray]:
    """Split ``y`` at ``cp`` and return (fitted step, residuals).

    fitted step = the two segment means (the 'clean signal');
    residual    = actual - its segment mean (the 'wiggle' the step misses).
    """
    fitted = np.empty_like(y, dtype=float)
    fitted[:cp] = y[:cp].mean()
    fitted[cp:] = y[cp:].mean()
    residuals = y - fitted
    return fitted, residuals


# --------------------------------------------------------------------------- #
# 3. The bootstrap                                                            #
# --------------------------------------------------------------------------- #
def block_resample(residuals: np.ndarray, rng: np.random.Generator,
                   block_len: int) -> np.ndarray:
    """Moving-block bootstrap: rebuild a residual series of the same length by
    pasting together randomly chosen blocks of ``block_len`` consecutive
    residuals. Drawing blocks (not single points) keeps the day-to-day
    stickiness, so the CI stays honest.
    """
    n = len(residuals)
    n_blocks = int(np.ceil(n / block_len))
    max_start = n - block_len
    out = []
    for _ in range(n_blocks):
        start = rng.integers(0, max_start + 1)  # random block start
        out.append(residuals[start:start + block_len])
    return np.concatenate(out)[:n]


def bootstrap_breakpoint(
    y: np.ndarray,
    n_boot: int = 1000,
    block_len: int = 5,
    min_seg: int = 3,
    seed: int = SEED,
) -> tuple[int, tuple[int, int], np.ndarray]:
    """Bootstrap CI for the onset of ``y``.

    Returns (onset, (ci_lo, ci_hi), all_bootstrap_onsets) as 0-based indices.

    Procedure (this *is* the 5-step recipe):
      1. fit onset on the real data
      2. residual = actual - segment mean
      3. synthetic = same step + block-resampled residuals; re-detect onset
      4. repeat n_boot times
      5. 2.5th-97.5th percentile of the onsets = 95 % CI
    """
    rng = np.random.default_rng(seed)
    cp = fit_single_breakpoint(y, min_seg)                      # step 1
    fitted, residuals = skeleton_and_residuals(y, cp)          # step 2
    boot = np.empty(n_boot, dtype=int)
    for b in range(n_boot):                                     # step 4
        synthetic = fitted + block_resample(residuals, rng, block_len)  # step 3
        boot[b] = fit_single_breakpoint(synthetic, min_seg)
    lo, hi = np.percentile(boot, [2.5, 97.5])                  # step 5
    return cp, (int(round(lo)), int(round(hi))), boot


# --------------------------------------------------------------------------- #
# 4. Figures                                                                  #
# --------------------------------------------------------------------------- #
def plot_single(y: np.ndarray, cp: int, ci: tuple[int, int],
                boot: np.ndarray, path: Path) -> None:
    days = np.arange(1, len(y) + 1)  # 1-based days for display
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(days, y, "o-", color="#333", label="daily value")
    ax1.axvspan(ci[0] + 1, ci[1] + 1, color="tab:orange", alpha=0.25,
                label="95% CI of onset")
    ax1.axvline(cp + 1, color="tab:red", lw=2, label=f"onset = day {cp + 1}")
    ax1.set(xlabel="day", ylabel="sensor value", title="Series + detected onset")
    ax1.legend(fontsize=8)

    ax2.hist(boot + 1, bins=np.arange(0.5, len(y) + 1.5),
             color="tab:blue", alpha=0.8)
    ax2.axvspan(ci[0] + 1, ci[1] + 1, color="tab:orange", alpha=0.25)
    ax2.set(xlabel="onset day", ylabel="# of bootstrap runs",
            title=f"1000 bootstrap onsets\n95% CI: day {ci[0] + 1}-{ci[1] + 1}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_caterpillar(results: list[dict], path: Path) -> None:
    """Caterpillar plot: one row per sensor (onset point + CI bar), ordered by
    onset. Overlapping bars => onsets not separable (report as co-temporal).
    """
    results = sorted(results, key=lambda r: r["onset"])
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, r in enumerate(results):
        lo, hi = r["ci"]
        ax.plot([lo + 1, hi + 1], [i, i], color="tab:gray", lw=6, alpha=0.5)
        ax.plot(r["onset"] + 1, i, "o", color="tab:red", zorder=3)
        ax.text(hi + 1.5, i, f"day {r['onset'] + 1}  (CI {lo + 1}-{hi + 1})",
                va="center", fontsize=8)
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels([r["name"] for r in results])
    ax.set(xlabel="onset day", title="Onset ordering with 95% CIs (caterpillar)")
    ax.margins(x=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 5. Run the two demos                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    # --- Demo 1: one sensor, strong step -> tight CI -------------------------
    y = make_toy_series(rng, n_before=20, n_after=20,
                        mean_before=20, mean_after=30, noise_sd=2.0)
    cp, ci, boot = bootstrap_breakpoint(y)
    print("Demo 1 - single sensor")
    print(f"  detected onset: day {cp + 1}")
    print(f"  95% CI:         day {ci[0] + 1} to day {ci[1] + 1}")
    print(f"  (strong step vs small noise -> tight CI)\n")
    plot_single(y, cp, ci, boot, FIG_DIR / "breakpoint_bootstrap_single.png")

    # --- Demo 2: five sensors with different true onsets ---------------------
    # Mimics the real story: a FAST cluster (physiology/leaf-angle/VOD) and a
    # SLOW cluster (greenness/PAI). Noise differs per sensor -> CIs differ.
    specs = [
        # name,            onset, mean0, mean1, noise
        ("Stem Psi",          6,   -0.5,  -1.8, 0.18),
        ("Leaf angle",       11,   35.0,  48.0, 4.0),
        ("GNSS-T VOD",       13,    0.55,  0.42, 0.05),
        ("GCC",              22,    0.40,  0.36, 0.012),
        ("PAI",              25,    3.2,   2.7,  0.18),
    ]
    results = []
    for name, onset, m0, m1, sd in specs:
        s = make_toy_series(rng, n_before=onset, n_after=40 - onset,
                            mean_before=m0, mean_after=m1, noise_sd=sd)
        cp_i, ci_i, _ = bootstrap_breakpoint(s, seed=SEED + onset)
        results.append({"name": name, "onset": cp_i, "ci": ci_i})

    print("Demo 2 - onset ordering (caterpillar)")
    for r in sorted(results, key=lambda r: r["onset"]):
        print(f"  {r['name']:<12} onset day {r['onset'] + 1:>2}  "
              f"(95% CI {r['ci'][0] + 1}-{r['ci'][1] + 1})")
    print("\n  Reading rule: if two sensors' CIs do NOT overlap, their order is")
    print("  defensible; if they overlap, report them as co-temporal.")
    plot_caterpillar(results, FIG_DIR / "breakpoint_bootstrap_caterpillar.png")

    print(f"\nFigures written to: {FIG_DIR}")


if __name__ == "__main__":
    main()
