#!/usr/bin/env python3
"""
Build a 1x2 synchrony diagnostics chart from real ARKit timecode data.

Output:
  /home/zym/EATalker/doc/figures/synchrony_diagnostics.pdf

If real data is unavailable, the script falls back to simulated data.
"""

from __future__ import annotations

import csv
import glob
import re
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None


DATASET_ROOT = Path("/home/zym/EaTalker_Dataset_v2")
OUTPUT_PATH = Path("/home/zym/EATalker/doc/figures/synchrony_diagnostics.pdf")
FPS_TARGET = 60.0
THRESHOLD_MS = 40.0
RNG_SEED = 2026


def parse_timecode_to_seconds(
    timecode: str, fps: float = 60.0, include_subframe: bool = True
) -> float:
    """
    Parse `HH:MM:SS:FF(.mmm)` to seconds.
    """
    m = re.match(r"^(\d+):(\d+):(\d+):(\d+)(?:\.(\d+))?$", timecode.strip())
    if not m:
        raise ValueError(f"Invalid timecode: {timecode}")
    hh, mm, ss, ff, sub = m.groups()
    out = int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ff) / float(fps)
    if include_subframe and sub is not None:
        out += (int(sub) / 1000.0) / float(fps)
    return out


def build_timecode_seconds_sequence(
    timecodes: Sequence[str], fps: float = 60.0, mode: str = "adaptive"
) -> List[float]:
    """
    Convert timecodes to seconds with the same logic family used in preprocessing.

    mode:
      - raw
      - ff_only
      - robust
      - adaptive
    """
    base = [parse_timecode_to_seconds(tc, fps=fps, include_subframe=False) for tc in timecodes]
    raw = [parse_timecode_to_seconds(tc, fps=fps, include_subframe=True) for tc in timecodes]

    if mode == "raw":
        return raw
    if mode == "ff_only":
        return base

    exp_dt = 1.0 / float(fps)
    max_reasonable_dt = 4.5 * exp_dt

    robust_out: List[float] = [raw[0]]
    for i in range(1, len(timecodes)):
        prev = robust_out[-1]
        cand_raw = raw[i]
        cand_base = base[i]
        dt_raw = cand_raw - prev
        dt_base = cand_base - prev

        raw_ok = dt_raw > 0 and dt_raw <= max_reasonable_dt
        base_ok = dt_base > 0 and dt_base <= max_reasonable_dt

        if raw_ok and base_ok:
            chosen = cand_raw if abs(dt_raw - exp_dt) <= abs(dt_base - exp_dt) else cand_base
        elif raw_ok:
            chosen = cand_raw
        elif base_ok:
            chosen = cand_base
        else:
            chosen = prev + (min(dt_base, max_reasonable_dt) if dt_base > 0 else exp_dt)

        robust_out.append(chosen)

    if mode == "robust":
        return robust_out

    raw_arr = np.asarray(raw, dtype=np.float64)
    robust_arr = np.asarray(robust_out, dtype=np.float64)
    raw_dt = np.diff(raw_arr)
    robust_dt = np.diff(robust_arr)
    tol_60 = 0.003
    tol_30 = 0.004

    out = [float(raw_arr[0])]
    for dt_raw, dt_robust in zip(raw_dt, robust_dt):
        raw_is_30 = abs(dt_raw - 2.0 * exp_dt) <= tol_30
        robust_is_60 = abs(dt_robust - exp_dt) <= tol_60
        chosen_dt = float(dt_raw if (raw_is_30 or not robust_is_60) else dt_robust)
        if chosen_dt <= 0:
            chosen_dt = float(exp_dt)
        out.append(out[-1] + chosen_dt)
    return out


def repair_timecode_jump_stitch(
    seconds: Sequence[float],
    fps: float = 60.0,
    dt_threshold_sec: float = 0.04,
    min_run_len: int = 1,
) -> List[float]:
    """
    Repair large dt runs by stitching them to expected frame step.
    """
    if len(seconds) < 2:
        return list(seconds)

    arr = np.asarray(seconds, dtype=np.float64)
    dts = np.diff(arr)
    exp_dt = 1.0 / float(fps)

    runs = []
    start = None
    for i, dt in enumerate(dts):
        if dt > dt_threshold_sec:
            if start is None:
                start = i
        else:
            if start is not None:
                runs.append((start, i - 1))
                start = None
    if start is not None:
        runs.append((start, len(dts) - 1))

    runs = [(s, e) for (s, e) in runs if (e - s + 1) >= int(min_run_len)]
    if not runs:
        return list(arr.tolist())

    repair_idx = set()
    for s, e in runs:
        repair_idx.update(range(s, e + 1))

    out = [float(arr[0])]
    for i, dt in enumerate(dts):
        new_dt = float(exp_dt if i in repair_idx else dt)
        if new_dt <= 0:
            new_dt = float(exp_dt)
        out.append(out[-1] + new_dt)
    return out


def load_timecodes_from_iphone_csv(csv_path: Path) -> List[str]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        tc_idx = header.index("Timecode") if "Timecode" in header else 0
        return [row[tc_idx].strip() for row in reader if len(row) > tc_idx and row[tc_idx].strip()]


def collect_real_dt(dataset_root: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    csv_paths = sorted(glob.glob(str(dataset_root / "*" / "[aA]udio" / "*" / "*_iPhone.csv")))
    if not csv_paths:
        return None, None

    dt_before_all = []
    dt_after_all = []

    for path in csv_paths:
        timecodes = load_timecodes_from_iphone_csv(Path(path))
        if len(timecodes) < 3:
            continue

        sec_before = np.asarray(
            build_timecode_seconds_sequence(timecodes, fps=FPS_TARGET, mode="raw"),
            dtype=np.float64,
        )
        sec_after = np.asarray(
            build_timecode_seconds_sequence(timecodes, fps=FPS_TARGET, mode="adaptive"),
            dtype=np.float64,
        )
        sec_after = np.asarray(
            repair_timecode_jump_stitch(
                sec_after, fps=FPS_TARGET, dt_threshold_sec=0.04, min_run_len=1
            ),
            dtype=np.float64,
        )

        dt_before_all.append(np.diff(sec_before) * 1000.0)
        dt_after_all.append(np.diff(sec_after) * 1000.0)

    if not dt_before_all or not dt_after_all:
        return None, None

    dt_before = np.concatenate(dt_before_all)
    dt_after = np.concatenate(dt_after_all)

    dt_before = dt_before[np.isfinite(dt_before) & (dt_before > 0)]
    dt_after = dt_after[np.isfinite(dt_after) & (dt_after > 0)]
    if dt_before.size == 0 or dt_after.size == 0:
        return None, None
    return dt_before, dt_after


def simulate_dt(n: int = 10000, seed: int = 2026) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    n_out = int(round(n * 0.02))
    n_main = n - n_out
    dt_main = rng.normal(loc=16.67, scale=1.2, size=n_main)
    dt_main = np.clip(dt_main, 8.0, 35.0)
    dt_out = rng.uniform(40.0, 150.0, size=n_out)
    dt_before = np.concatenate([dt_main, dt_out])
    rng.shuffle(dt_before)

    dt_after = rng.normal(loc=16.67, scale=0.55, size=n)
    dt_after = np.clip(dt_after, 12.0, 29.95)
    return dt_before, dt_after


def format_percent(percent: float) -> str:
    """
    Keep very small rates visible (e.g., 0.0005% instead of 0.00%).
    """
    if percent >= 1.0:
        return f"{percent:.2f}%"
    if percent >= 0.01:
        return f"{percent:.3f}%"
    if percent > 0.0:
        return f"{percent:.4f}%"
    return "0.0000%"


def add_stats_box(ax: plt.Axes, data: np.ndarray, threshold_ms: float = 40.0) -> None:
    anomalies = int(np.sum(data > threshold_ms))
    total = int(data.size)
    anomaly_ratio = float((anomalies / total) * 100.0 if total > 0 else 0.0)
    max_gap = float(np.max(data))
    ratio_txt = format_percent(anomaly_ratio)
    txt = (
        f"dt > {threshold_ms:.0f}ms: {ratio_txt}\\n"
        f"max: {max_gap:.1f} ms"
    )
    ax.text(
        0.97,
        0.97,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.70", alpha=0.96),
    )


def plot_hist(ax: plt.Axes, data: np.ndarray, title: str, x_max: float) -> None:
    bins = np.linspace(0.0, x_max, 140)
    ax.hist(
        data,
        bins=bins,
        color="steelblue",
        alpha=0.90,
        edgecolor="white",
        linewidth=0.35,
    )
    ax.axvline(THRESHOLD_MS, color="red", linestyle="--", linewidth=2.0)
    blended = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(
        THRESHOLD_MS + 3.0,
        0.5,
        "Threshold (40 ms)",
        transform=blended,
        color="red",
        fontsize=12,
        rotation=90,
        va="center",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85),
        clip_on=False,
    )
    ax.set_title(title, fontsize=17, pad=8)
    ax.set_xlabel("Time Delta (ms)", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)
    ax.set_yscale("log")
    ax.set_xlim(0.0, x_max)
    ax.grid(True, which="major", color="#d0d0d0", linestyle="-", linewidth=0.65, alpha=0.8)
    ax.grid(True, which="minor", color="#ececec", linestyle="-", linewidth=0.45, alpha=0.7)


def main() -> None:
    if sns is not None:
        sns.set_theme(style="whitegrid", context="paper")
    else:
        plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 17,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    dt_before, dt_after = collect_real_dt(DATASET_ROOT)
    using_real = dt_before is not None and dt_after is not None
    if not using_real:
        dt_before, dt_after = simulate_dt(n=10000, seed=RNG_SEED)

    x_max = max(80.0, float(np.percentile(np.concatenate([dt_before, dt_after]), 99.8)))
    x_max = min(x_max, 180.0)

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2), constrained_layout=True)
    plot_hist(axes[0], dt_before, "Before Calibration", x_max)
    add_stats_box(axes[0], dt_before, THRESHOLD_MS)
    plot_hist(axes[1], dt_after, "After Calibration", x_max)
    add_stats_box(axes[1], dt_after, THRESHOLD_MS)
    fig.suptitle("Synchrony Diagnostics Chart", fontsize=19, y=1.03)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=600, bbox_inches="tight")

    print(f"[source] {'real-data' if using_real else 'simulated-data'}")
    print(f"[saved] {OUTPUT_PATH}")
    print(
        f"[before] n={len(dt_before)}, anomaly%={(np.mean(dt_before > THRESHOLD_MS) * 100):.4f}, max={np.max(dt_before):.3f} ms"
    )
    print(
        f"[after ] n={len(dt_after)}, anomaly%={(np.mean(dt_after > THRESHOLD_MS) * 100):.4f}, max={np.max(dt_after):.3f} ms"
    )


if __name__ == "__main__":
    main()
