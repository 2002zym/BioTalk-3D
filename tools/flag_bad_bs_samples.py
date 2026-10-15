#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flag suspicious blendshape samples using only BS statistics (no audio).

Heuristics:
  - Extremely low motion (nearly static).
  - Extremely high motion spikes (velocity/acceleration outliers).
  - Abnormal BS magnitude distribution (very small or very large).
  - (Optional) mouth opening outliers from bs->vertices.

Output:
  - JSON summary with per-sample metrics and a final anomaly score.
  - Console print of top-K suspicious samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_bs_array(path: Path) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 1:
        arr = arr[None]
    if arr.ndim == 2:
        if arr.shape[1] == 51:
            return arr
        if arr.shape[0] == 51:
            return arr.T
    raise ValueError(f"Unsupported bs shape {arr.shape} in {path}")


def robust_z(x: np.ndarray, med: float, mad: float) -> np.ndarray:
    # 0.6745 rescales MAD to std for normal distributions.
    return 0.6745 * (x - med) / (mad + 1e-8)


def compute_metrics(bs: np.ndarray) -> Dict[str, float]:
    # bs: (T, 51)
    if bs.shape[0] < 2:
        return {
            "frames": int(bs.shape[0]),
            "bs_mag_mean": float(np.linalg.norm(bs, axis=1).mean())
            if bs.shape[0] > 0 else 0.0,
            "bs_mag_std": 0.0,
            "vel_mean": 0.0,
            "vel_std": 0.0,
            "vel_max": 0.0,
            "acc_std": 0.0,
            "acc_max": 0.0,
            "flat_ratio": 1.0,
        }

    mag = np.linalg.norm(bs, axis=1)
    vel = np.linalg.norm(bs[1:] - bs[:-1], axis=1)
    acc = np.linalg.norm(vel[1:] - vel[:-1], axis=0) if len(vel) > 1 else np.array([0.0], dtype=np.float32)

    # Ratio of near-zero velocity frames (detect near-static segments).
    vel_eps = 1e-4
    flat_ratio = float((vel < vel_eps).mean())

    return {
        "frames": int(bs.shape[0]),
        "bs_mag_mean": float(mag.mean()),
        "bs_mag_std": float(mag.std()),
        "vel_mean": float(vel.mean()),
        "vel_std": float(vel.std()),
        "vel_max": float(vel.max()),
        "acc_std": float(acc.std()),
        "acc_max": float(acc.max()),
        "flat_ratio": flat_ratio,
    }


def bs_to_verts(bs: np.ndarray, blendshape_flat: np.ndarray, mean_flat: np.ndarray, v_count: int) -> np.ndarray:
    verts_flat = bs @ blendshape_flat + mean_flat
    return verts_flat.reshape(bs.shape[0], v_count, 3)


def mouth_opening_metric(verts: np.ndarray, mouth_idx: np.ndarray) -> Dict[str, float]:
    mouth = verts[:, mouth_idx, :]
    y = mouth[..., 1]
    opening = y.max(axis=1) - y.min(axis=1)
    return {
        "mouth_open_mean": float(opening.mean()),
        "mouth_open_std": float(opening.std()),
        "mouth_open_max": float(opening.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flag suspicious BS samples using only BS statistics.")
    parser.add_argument("--data-dir",
                        required=True,
                        type=Path,
                        help="Directory containing *_bs.npy files (e.g., train).")
    parser.add_argument("--out",
                        type=Path,
                        default=Path("bs_quality_report.json"),
                        help="Output JSON report path.")
    parser.add_argument("--top-k",
                        type=int,
                        default=20,
                        help="Print top-K suspicious samples.")
    parser.add_argument("--decimate",
                        type=int,
                        default=1,
                        help="Temporal decimation factor (e.g., 2 for 60->30fps).")
    parser.add_argument("--use-mouth",
                        action="store_true",
                        help="Include mouth opening metrics (requires bs->vertices).")
    parser.add_argument("--static-mouth-only",
                        action="store_true",
                        help="Only report samples flagged as 'static but mouth open'.")
    parser.add_argument("--basis-npz",
                        type=Path,
                        default=Path("/home/zym/EATalker/resources/binary_resources/inhouse_arkit.npz"),
                        help="Blendshape basis npz (meanshape/blendshape).")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise NotADirectoryError(data_dir)

    bs_files = sorted(data_dir.glob("*_bs.npy"))
    if not bs_files:
        raise RuntimeError(f"No *_bs.npy found in {data_dir}")

    use_mouth = bool(args.use_mouth)
    if use_mouth:
        basis = np.load(args.basis_npz)
        meanshape = basis["meanshape"].astype(np.float32)
        blendshape = basis["blendshape"].astype(np.float32)
        inner_idx = basis.get("inner_mouth_idx", None)
        outer_idx = basis.get("outer_mouth_idx", None)
        if inner_idx is None or outer_idx is None:
            raise RuntimeError("basis npz missing inner_mouth_idx/outer_mouth_idx")
        mouth_idx = np.unique(np.concatenate([inner_idx, outer_idx])).astype(np.int64)
        blendshape_flat = blendshape.reshape(51, -1)
        mean_flat = meanshape.reshape(-1)
        v_count = meanshape.shape[0]
    else:
        mouth_idx = None
        blendshape_flat = mean_flat = None
        v_count = 0

    records: List[Dict[str, float]] = []
    for bs_path in bs_files:
        bs = load_bs_array(bs_path)
        if args.decimate and args.decimate > 1:
            bs = bs[::args.decimate]
        metrics = compute_metrics(bs)
        if use_mouth:
            verts = bs_to_verts(bs, blendshape_flat, mean_flat, v_count)
            metrics.update(mouth_opening_metric(verts, mouth_idx))
        metrics["file"] = bs_path.name
        records.append(metrics)

    # Build robust z-scores across dataset for each metric.
    metric_keys = [
        "bs_mag_mean",
        "bs_mag_std",
        "vel_mean",
        "vel_std",
        "vel_max",
        "acc_std",
        "acc_max",
    ]
    if use_mouth:
        metric_keys += ["mouth_open_mean", "mouth_open_std", "mouth_open_max"]

    stats = {}
    for k in metric_keys:
        vals = np.array([r[k] for r in records], dtype=np.float32)
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        stats[k] = {"median": med, "mad": mad}

    # Score each sample by maximum absolute z-score across metrics,
    # with extra penalty for very high flat_ratio.
    for r in records:
        z_scores = []
        for k in metric_keys:
            z = robust_z(r[k], stats[k]["median"], stats[k]["mad"])
            z_scores.append(abs(float(z)))
        score = max(z_scores) if z_scores else 0.0
        # Extra flag for nearly static sequences
        if r["flat_ratio"] > 0.98:
            score = max(score, 5.0)
        r["score"] = float(score)

        # Flag: long static but mouth stays open/large.
        # Heuristic: high flat_ratio + unusually large mouth_open_max
        # and unusually low velocity std/mean.
        r["static_mouth_flag"] = False
        if use_mouth:
            z_mouth_max = robust_z(r["mouth_open_max"],
                                   stats["mouth_open_max"]["median"],
                                   stats["mouth_open_max"]["mad"])
            z_vel_std = robust_z(r["vel_std"],
                                 stats["vel_std"]["median"],
                                 stats["vel_std"]["mad"])
            z_vel_mean = robust_z(r["vel_mean"],
                                  stats["vel_mean"]["median"],
                                  stats["vel_mean"]["mad"])
            if (r["flat_ratio"] > 0.90 and z_mouth_max > 2.0 and
                    (z_vel_std < -2.0 or z_vel_mean < -2.0)):
                r["static_mouth_flag"] = True
                r["score"] = max(r["score"], 6.0)

    # Sort by score descending
    records_sorted = sorted(records, key=lambda x: x["score"], reverse=True)
    static_mouth = [r for r in records_sorted if r.get("static_mouth_flag")]

    # Save report
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "data_dir": str(data_dir),
                "decimate": int(args.decimate),
                "use_mouth": use_mouth,
                "stats": stats,
                "records": records_sorted,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved report to {out_path}")

    if args.static_mouth_only:
        print(f"Static-mouth flagged samples: {len(static_mouth)}")
        for r in static_mouth:
            print(
                f"  {r['file']} | score={r['score']:.2f} | "
                f"mouth_max={r.get('mouth_open_max', 0.0):.4f} | "
                f"vel_std={r['vel_std']:.4f} | flat={r['flat_ratio']:.2f}"
            )
        return

    # Print top-K
    top_k = max(1, int(args.top_k))
    print(f"Top {top_k} suspicious samples:")
    for r in records_sorted[:top_k]:
        print(
            f"  {r['file']} | score={r['score']:.2f} | "
            f"vel_max={r['vel_max']:.4f} | acc_max={r['acc_max']:.4f} | "
            f"flat={r['flat_ratio']:.2f}"
        )

    if use_mouth:
        print(f"Static-mouth flagged samples: {len(static_mouth)}")


if __name__ == "__main__":
    main()
