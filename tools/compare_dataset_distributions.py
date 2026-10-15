#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare full (no sampling) BS and audio distributions across datasets.

Example:
  PYTHONPATH=/home/zym /home/zym/micromamba/bin/micromamba run -n eatalker \
  python -m EATalker.tools.compare_dataset_distributions \
    --eatalker /home/zym/EATalker/EATalker_data_template/EATalker \
    --d6 /home/zym/D6_unitalker_Chinese_speech \
    --d7 /home/zym/D7_unitalker_song \
    --out /home/zym/experiments/test_results/dataset_dist_report.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf


def _d6_order() -> List[str]:
    return [
        'EyeBlinkLeft', 'EyeBlinkRight', 'EyeSquintLeft', 'EyeSquintRight',
        'EyeLookDownLeft', 'EyeLookDownRight', 'EyeLookInLeft', 'EyeLookInRight',
        'EyeWideLeft', 'EyeWideRight', 'EyeLookOutLeft', 'EyeLookOutRight',
        'EyeLookUpLeft', 'EyeLookUpRight', 'BrowDownLeft', 'BrowDownRight',
        'BrowInnerUp', 'BrowOuterUpLeft', 'BrowOuterUpRight', 'JawOpen',
        'MouthClose', 'JawLeft', 'JawRight', 'JawForward', 'MouthUpperUpLeft',
        'MouthUpperUpRight', 'MouthLowerDownLeft', 'MouthLowerDownRight',
        'MouthRollUpper', 'MouthRollLower', 'MouthSmileLeft', 'MouthSmileRight',
        'MouthDimpleLeft', 'MouthDimpleRight', 'MouthStretchLeft', 'MouthStretchRight',
        'MouthFrownLeft', 'MouthFrownRight', 'MouthPressLeft', 'MouthPressRight',
        'MouthPucker', 'MouthFunnel', 'MouthLeft', 'MouthRight',
        'MouthShrugLower', 'MouthShrugUpper', 'NoseSneerLeft', 'NoseSneerRight',
        'CheekPuff', 'CheekSquintLeft', 'CheekSquintRight'
    ]


def _mouth_bs_idx() -> List[int]:
    order = _d6_order()
    mouth_names = {
        'JawOpen', 'JawLeft', 'JawRight', 'JawForward', 'MouthClose',
        'MouthFunnel', 'MouthPucker', 'MouthLeft', 'MouthRight',
        'MouthSmileLeft', 'MouthSmileRight', 'MouthFrownLeft', 'MouthFrownRight',
        'MouthDimpleLeft', 'MouthDimpleRight', 'MouthStretchLeft', 'MouthStretchRight',
        'MouthRollUpper', 'MouthRollLower', 'MouthShrugLower', 'MouthShrugUpper',
        'MouthPressLeft', 'MouthPressRight', 'MouthUpperUpLeft', 'MouthUpperUpRight',
        'MouthLowerDownLeft', 'MouthLowerDownRight'
    }
    return [i for i, name in enumerate(order) if name in mouth_names]


JAW_OPEN_IDX = _d6_order().index('JawOpen')  # 0-based, expected 19
MOUTH_IDX = _mouth_bs_idx()


@dataclass
class RunningMoments:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_v: float = float('inf')
    max_v: float = float('-inf')
    near_zero: int = 0

    def update(self, arr: np.ndarray, zero_eps: float = 1e-6) -> None:
        if arr.size == 0:
            return
        flat = arr.reshape(-1).astype(np.float64, copy=False)
        n = int(flat.size)
        self.min_v = float(min(self.min_v, float(np.min(flat))))
        self.max_v = float(max(self.max_v, float(np.max(flat))))
        self.near_zero += int(np.sum(np.abs(flat) <= zero_eps))
        # Welford update in batch
        batch_mean = float(np.mean(flat))
        batch_m2 = float(np.sum((flat - batch_mean) ** 2))
        if self.count == 0:
            self.count = n
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        delta = batch_mean - self.mean
        new_count = self.count + n
        self.mean = self.mean + delta * n / new_count
        self.m2 = self.m2 + batch_m2 + delta * delta * self.count * n / new_count
        self.count = new_count

    def as_dict(self) -> Dict[str, float]:
        var = self.m2 / max(self.count, 1)
        std = float(math.sqrt(var)) if var >= 0 else float('nan')
        zero_frac = float(self.near_zero / max(self.count, 1))
        return {
            'count': int(self.count),
            'mean': float(self.mean),
            'std': std,
            'min': float(self.min_v) if self.count else float('nan'),
            'max': float(self.max_v) if self.count else float('nan'),
            'near_zero_frac': zero_frac,
        }


class RunningHistogram:
    def __init__(self, lo: float, hi: float, bins: int):
        self.lo = float(lo)
        self.hi = float(hi)
        self.bins = int(bins)
        self.edges = np.linspace(self.lo, self.hi, self.bins + 1, dtype=np.float64)
        self.counts = np.zeros(self.bins, dtype=np.int64)
        self.under = 0
        self.over = 0

    def update(self, arr: np.ndarray) -> None:
        if arr.size == 0:
            return
        flat = arr.reshape(-1).astype(np.float64, copy=False)
        self.under += int(np.sum(flat < self.lo))
        self.over += int(np.sum(flat > self.hi))
        clipped = np.clip(flat, self.lo, self.hi)
        h, _ = np.histogram(clipped, bins=self.edges)
        self.counts += h.astype(np.int64)

    def quantile(self, q: float) -> float:
        q = float(q)
        if not (0.0 <= q <= 1.0):
            raise ValueError('q must be in [0,1]')
        total = int(self.counts.sum())
        if total == 0:
            return float('nan')
        target = int(math.ceil(q * total))
        cumsum = np.cumsum(self.counts)
        idx = int(np.searchsorted(cumsum, target, side='left'))
        idx = min(max(idx, 0), self.bins - 1)
        return float((self.edges[idx] + self.edges[idx + 1]) * 0.5)

    def as_dict(self) -> Dict[str, float]:
        return {
            'lo': self.lo,
            'hi': self.hi,
            'bins': self.bins,
            'under': int(self.under),
            'over': int(self.over),
            'p50': self.quantile(0.50),
            'p90': self.quantile(0.90),
            'p95': self.quantile(0.95),
            'p99': self.quantile(0.99),
        }


def _find_split_dirs(root: Path) -> List[Path]:
    candidates = []
    for name in ['train', 'test', 'val']:
        d = root / name
        if d.exists() and d.is_dir():
            candidates.append(d)
    if candidates:
        return candidates
    return [root]


def _find_pair_bs_for_wav(wav_path: Path) -> Optional[Path]:
    stem = wav_path.stem
    parent = wav_path.parent
    candidates = [
        parent / f'{stem}.npy',
        parent / f'{stem}_bs.npy',
    ]
    # EATalker often stores wav without "_bs", and bs as "<stem>_bs.npy"
    if stem.endswith('_bs'):
        candidates.append(parent / f'{stem[:-3]}.npy')
    for c in candidates:
        if c.exists():
            return c
    return None


def _safe_load_bs(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f'Unexpected bs shape {arr.shape} in {path}')
    return arr.astype(np.float32, copy=False)


def _audio_basic_stats(wav_path: Path, frame_ms: float = 20.0) -> Dict[str, float]:
    info = sf.info(str(wav_path))
    sr = int(info.samplerate)
    frames = int(info.frames)
    duration = float(frames / sr) if sr > 0 else 0.0
    # Read audio (mono) for RMS/silence/clipping
    audio, _sr = sf.read(str(wav_path), dtype='float32', always_2d=True)
    if _sr != sr:
        sr = int(_sr)
    y = audio.mean(axis=1)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    rms = float(np.sqrt(np.mean(y * y))) if y.size else 0.0
    # normalize by peak for silence ratio estimate
    if peak > 0:
        yn = y / peak
    else:
        yn = y
    hop = max(1, int(sr * frame_ms / 1000.0))
    # frame RMS
    if yn.size == 0:
        silent_ratio = 1.0
    else:
        n = int(math.ceil(yn.size / hop))
        # pad for reshape
        pad = n * hop - yn.size
        if pad > 0:
            yn_pad = np.pad(yn, (0, pad), mode='constant')
        else:
            yn_pad = yn
        frames_y = yn_pad.reshape(n, hop)
        frms = np.sqrt(np.mean(frames_y * frames_y, axis=1))
        silent_ratio = float(np.mean(frms < 0.01))  # -40 dBFS relative to peak
    clip_ratio = float(np.mean(np.abs(y) >= 0.99)) if y.size else 0.0
    return {
        'sr': sr,
        'duration_sec': duration,
        'peak': peak,
        'rms': rms,
        'silent_ratio': silent_ratio,
        'clip_ratio': clip_ratio,
    }


def _summarize_list(values: List[float]) -> Dict[str, float]:
    if not values:
        return {'count': 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        'count': int(arr.size),
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'min': float(arr.min()),
        'p10': float(np.percentile(arr, 10)),
        'p50': float(np.percentile(arr, 50)),
        'p90': float(np.percentile(arr, 90)),
        'max': float(arr.max()),
    }


def analyze_dataset(root: Path,
                    name: str,
                    bs_range: Tuple[float, float] = (-0.2, 1.2),
                    bins: int = 400) -> Dict[str, object]:
    split_dirs = _find_split_dirs(root)
    wav_paths: List[Path] = []
    bs_paths: List[Path] = []
    missing_bs: List[str] = []
    for split in split_dirs:
        wavs = sorted(split.glob('*.wav'))
        wav_paths.extend(wavs)
        for w in wavs:
            bs = _find_pair_bs_for_wav(w)
            if bs is None:
                missing_bs.append(str(w))
            else:
                bs_paths.append(bs)

    # Audio aggregated
    sr_counts: Dict[str, int] = {}
    durations: List[float] = []
    rms_list: List[float] = []
    peak_list: List[float] = []
    silent_list: List[float] = []
    clip_list: List[float] = []
    for w in wav_paths:
        st = _audio_basic_stats(w)
        sr_counts[str(st['sr'])] = sr_counts.get(str(st['sr']), 0) + 1
        durations.append(float(st['duration_sec']))
        rms_list.append(float(st['rms']))
        peak_list.append(float(st['peak']))
        silent_list.append(float(st['silent_ratio']))
        clip_list.append(float(st['clip_ratio']))

    # BS aggregated
    all_mom = RunningMoments()
    mouth_mom = RunningMoments()
    jaw_mom = RunningMoments()
    vel_mom = RunningMoments()
    jaw_vel_mom = RunningMoments()
    all_hist = RunningHistogram(bs_range[0], bs_range[1], bins=bins)
    mouth_hist = RunningHistogram(bs_range[0], bs_range[1], bins=bins)
    jaw_hist = RunningHistogram(bs_range[0], bs_range[1], bins=bins)

    dims_seen = None
    bs_files_processed = 0
    for bs_path in bs_paths:
        bs = _safe_load_bs(bs_path)
        if dims_seen is None:
            dims_seen = int(bs.shape[1])
        else:
            if int(bs.shape[1]) != int(dims_seen):
                raise ValueError(f'Dim mismatch: expected {dims_seen}, got {bs.shape[1]} in {bs_path}')

        all_mom.update(bs)
        all_hist.update(bs)

        if bs.shape[1] > JAW_OPEN_IDX:
            jaw = bs[:, JAW_OPEN_IDX]
            jaw_mom.update(jaw)
            jaw_hist.update(jaw)
            if jaw.shape[0] > 1:
                jaw_vel = np.diff(jaw, axis=0)
                jaw_vel_mom.update(jaw_vel)

        if bs.shape[1] >= max(MOUTH_IDX) + 1:
            mouth = bs[:, MOUTH_IDX]
            mouth_mom.update(mouth)
            mouth_hist.update(mouth)

        if bs.shape[0] > 1:
            vel = np.diff(bs, axis=0)
            vel_l2 = np.linalg.norm(vel, axis=1)
            vel_mom.update(vel_l2)

        bs_files_processed += 1

    return {
        'name': name,
        'root': str(root),
        'splits': [d.name for d in split_dirs],
        'pairs': {
            'num_wavs': len(wav_paths),
            'num_bs': len(bs_paths),
            'missing_bs': missing_bs[:50],  # avoid huge output
            'missing_bs_count': len(missing_bs),
        },
        'audio': {
            'sr_counts': sr_counts,
            'duration_sec': _summarize_list(durations),
            'rms': _summarize_list(rms_list),
            'peak': _summarize_list(peak_list),
            'silent_ratio': _summarize_list(silent_list),
            'clip_ratio': _summarize_list(clip_list),
        },
        'bs': {
            'dims': int(dims_seen or 0),
            'all': {**all_mom.as_dict(), **all_hist.as_dict()},
            'mouth': {**mouth_mom.as_dict(), **mouth_hist.as_dict()},
            'jawOpen': {**jaw_mom.as_dict(), **jaw_hist.as_dict()},
            'vel_l2': vel_mom.as_dict(),
            'jawOpen_vel': jaw_vel_mom.as_dict(),
        },
    }


def ratio(a: float, b: float) -> Optional[float]:
    if b == 0 or math.isnan(b) or math.isnan(a):
        return None
    return float(a / b)


def compare_three(eat: Dict[str, object], d6: Dict[str, object],
                  d7: Dict[str, object]) -> Dict[str, object]:
    def get(d, *keys):
        cur = d
        for k in keys:
            cur = cur[k]
        return cur

    out = {'summary': {}}
    for scope in ['all', 'mouth', 'jawOpen']:
        e_p99 = get(eat, 'bs', scope, 'p99')
        d6_p99 = get(d6, 'bs', scope, 'p99')
        d7_p99 = get(d7, 'bs', scope, 'p99')
        out['summary'][f'{scope}_p99_ratio'] = {
            'EATalker_vs_D6': ratio(e_p99, d6_p99),
            'EATalker_vs_D7': ratio(e_p99, d7_p99),
            'EATalker_p99': e_p99,
            'D6_p99': d6_p99,
            'D7_p99': d7_p99,
        }
        e_mean = get(eat, 'bs', scope, 'mean')
        d6_mean = get(d6, 'bs', scope, 'mean')
        d7_mean = get(d7, 'bs', scope, 'mean')
        out['summary'][f'{scope}_mean_ratio'] = {
            'EATalker_vs_D6': ratio(e_mean, d6_mean),
            'EATalker_vs_D7': ratio(e_mean, d7_mean),
            'EATalker_mean': e_mean,
            'D6_mean': d6_mean,
            'D7_mean': d7_mean,
        }
    return out


def main():
    ap = argparse.ArgumentParser(
        description='Compare full BS+audio distributions across EATalker vs D6/D7 (no sampling).'
    )
    ap.add_argument('--eatalker',
                    type=Path,
                    required=True,
                    help='EATalker dataset root (contains train/test dirs)')
    ap.add_argument('--d6',
                    type=Path,
                    required=True,
                    help='D6 dataset root (contains train/test/val)')
    ap.add_argument('--d7',
                    type=Path,
                    required=True,
                    help='D7 dataset root (contains train/test/val)')
    ap.add_argument('--out',
                    type=Path,
                    required=True,
                    help='Output JSON path')
    ap.add_argument('--bs-range',
                    type=str,
                    default='-0.2,1.2',
                    help='Histogram range for BS values, e.g. "-0.2,1.2"')
    ap.add_argument('--bins',
                    type=int,
                    default=400,
                    help='Histogram bins')
    args = ap.parse_args()

    lo_s, hi_s = args.bs_range.split(',')
    bs_range = (float(lo_s), float(hi_s))

    eat = analyze_dataset(args.eatalker, 'EATalker', bs_range=bs_range, bins=args.bins)
    d6 = analyze_dataset(args.d6, 'D6', bs_range=bs_range, bins=args.bins)
    d7 = analyze_dataset(args.d7, 'D7', bs_range=bs_range, bins=args.bins)
    cmp = compare_three(eat, d6, d7)

    report = {
        'datasets': {
            'EATalker': eat,
            'D6': d6,
            'D7': d7,
        },
        'compare': cmp,
        'meta': {
            'jawOpen_idx': int(JAW_OPEN_IDX),
            'mouth_idx': MOUTH_IDX,
            'bs_hist_range': list(bs_range),
            'bins': int(args.bins),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'Saved report to {args.out}')
    for k, v in cmp['summary'].items():
        print(k, v)


if __name__ == '__main__':
    main()

