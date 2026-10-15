#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare GT vs prediction mouth motion amplitude distributions.

This script:
- Loads predicted vertices from eat_demo-generated npz.
- Loads GT blendshape npy from a directory and converts to vertices.
- Computes per-frame mouth motion amplitude (relative to per-sequence mean).
- Summarizes distributions and outputs a JSON report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from EATalker.loss.loss import BlendShapeLoss


def load_pred_vertices(npz_path: Path) -> Dict[str, np.ndarray]:
    out = {}
    npz = np.load(npz_path, allow_pickle=True)
    for key in npz.files:
        item = npz[key]
        if isinstance(item, np.ndarray) and item.dtype == object:
            item = item.item()
        if isinstance(item, dict):
            # pick the first dataset if multiple exist
            if len(item) == 1:
                verts = next(iter(item.values()))
            else:
                # prefer EATalker if present, else first
                verts = item.get('EATalker', next(iter(item.values())))
        else:
            verts = item
        out[key] = np.asarray(verts)
    return out


def mouth_amplitude(verts: np.ndarray, mouth_idx: np.ndarray) -> np.ndarray:
    """Compute per-frame mouth motion amplitude (relative to per-seq mean)."""
    # verts: [T, V, 3]
    mouth = verts[:, mouth_idx, :]
    mean = mouth.mean(axis=0, keepdims=True)
    diff = mouth - mean
    per_frame = np.linalg.norm(diff, axis=2).mean(axis=1)
    return per_frame


def summarize(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {}
    return {
        'mean': float(arr.mean()),
        'p50': float(np.percentile(arr, 50)),
        'p90': float(np.percentile(arr, 90)),
        'p99': float(np.percentile(arr, 99)),
        'max': float(arr.max()),
    }


def load_gt_vertices(
    gt_bs_path: Path,
    bs_loss: BlendShapeLoss,
) -> np.ndarray:
    bs = np.load(gt_bs_path).astype(np.float32)
    if bs.ndim != 2:
        raise ValueError(f'Unexpected GT bs shape {bs.shape} for {gt_bs_path}')
    bs_t = torch.from_numpy(bs)
    with torch.no_grad():
        verts = bs_loss.get_vertices(bs_t).cpu().numpy()
    return verts


def find_gt_bs(gt_root: Path, stem: str) -> Path | None:
    """Find *_bs.npy in gt_root or common subdirs."""
    candidates = [
        gt_root / f'{stem}_bs.npy',
        gt_root / 'train' / f'{stem}_bs.npy',
        gt_root / 'test' / f'{stem}_bs.npy',
        gt_root / 'blendshapes' / f'{stem}_bs.npy',
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Compare GT vs prediction mouth amplitude distributions')
    parser.add_argument('--pred-npz', type=str, required=True,
                        help='Prediction npz from eat_demo')
    parser.add_argument('--gt-dir', type=str, required=True,
                        help='GT root dir; searches in root/train/test/blendshapes')
    parser.add_argument('--mouth-idx', type=str,
                        default='resources/binary_resources/05_inhouse_arkit_mouth_idx.npy',
                        help='Mouth vertex indices npy')
    parser.add_argument('--blendshape-path', type=str,
                        default='resources/binary_resources/inhouse_arkit.npz',
                        help='Blendshape basis npz for bs->verts')
    parser.add_argument('--out', type=str,
                        default='mouth_amp_report.json',
                        help='Output JSON path')
    parser.add_argument('--top-k', type=int, default=20,
                        help='Print top-k lowest amplitude ratio samples')
    args = parser.parse_args()

    pred_npz = Path(args.pred_npz)
    gt_dir = Path(args.gt_dir)
    out_path = Path(args.out)

    if not pred_npz.exists():
        raise FileNotFoundError(f'Pred npz not found: {pred_npz}')
    if not gt_dir.exists():
        raise FileNotFoundError(f'GT dir not found: {gt_dir}')

    pred_map = load_pred_vertices(pred_npz)
    mouth_idx = np.load(args.mouth_idx).astype(np.int64)
    bs_loss = BlendShapeLoss(
        mouth_indices_path=args.mouth_idx,
        blendshape_path=args.blendshape_path,
        bs_beta=0.0,
    )

    per_sample: Dict[str, Dict[str, float]] = {}
    ratios: List[Tuple[str, float]] = []
    agg_pred: List[float] = []
    agg_gt: List[float] = []
    mismatched: List[str] = []
    missing: List[str] = []

    for wav_rel, pred_verts in pred_map.items():
        stem = Path(wav_rel).stem
        gt_bs = find_gt_bs(gt_dir, stem)
        if gt_bs is None:
            missing.append(wav_rel)
            continue

        try:
            gt_verts = load_gt_vertices(gt_bs, bs_loss)
        except Exception as exc:
            mismatched.append(f'{wav_rel}: {exc}')
            continue

        if pred_verts.ndim != 3 or gt_verts.ndim != 3:
            mismatched.append(f'{wav_rel}: pred {pred_verts.shape}, gt {gt_verts.shape}')
            continue

        t = min(pred_verts.shape[0], gt_verts.shape[0])
        if t <= 1:
            mismatched.append(f'{wav_rel}: too short (pred {pred_verts.shape[0]}, gt {gt_verts.shape[0]})')
            continue

        pred_amp = mouth_amplitude(pred_verts[:t], mouth_idx)
        gt_amp = mouth_amplitude(gt_verts[:t], mouth_idx)

        pred_stats = summarize(pred_amp)
        gt_stats = summarize(gt_amp)

        ratio = None
        if gt_stats and gt_stats['mean'] > 1e-8:
            ratio = pred_stats['mean'] / gt_stats['mean']
            ratios.append((wav_rel, ratio))

        per_sample[wav_rel] = {
            'pred_mean': pred_stats.get('mean', 0.0),
            'gt_mean': gt_stats.get('mean', 0.0),
            'ratio_mean': ratio if ratio is not None else 0.0,
            'pred_p90': pred_stats.get('p90', 0.0),
            'gt_p90': gt_stats.get('p90', 0.0),
        }

        agg_pred.extend(pred_amp.tolist())
        agg_gt.extend(gt_amp.tolist())

    report = {
        'pred_npz': str(pred_npz),
        'gt_dir': str(gt_dir),
        'num_pred': len(pred_map),
        'num_missing_gt': len(missing),
        'num_mismatched': len(mismatched),
        'overall_pred': summarize(np.asarray(agg_pred)),
        'overall_gt': summarize(np.asarray(agg_gt)),
        'per_sample': per_sample,
        'missing_gt': missing,
        'mismatched': mismatched,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    if ratios:
        ratios_sorted = sorted(ratios, key=lambda x: x[1])
        print(f'Top {args.top_k} lowest pred/gt mouth amplitude ratios:')
        for name, ratio in ratios_sorted[:args.top_k]:
            print(f'  {name}: {ratio:.3f}')
    print(f'Report saved to {out_path}')


if __name__ == '__main__':
    main()
