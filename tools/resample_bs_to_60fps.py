#!/usr/bin/env python3
"""
Resample existing *_bs.npy sequences to strict 60 fps using corresponding WAV duration.

Writes fixed files alongside originals (suffix _60fps.npy) or to an output dir.

Usage examples:
  # Dry-run first 10 files
  PYTHONPATH=/home/zym python3 EATalker/tools/resample_bs_to_60fps.py \
    --root /home/zym/EATalker/EATalker_data_template/EATalker --max 10 --dry

  # Apply to all train/test and write next to originals with _60fps suffix
  PYTHONPATH=/home/zym python3 EATalker/tools/resample_bs_to_60fps.py \
    --root /home/zym/EATalker/EATalker_data_template/EATalker

  # Write to a separate output directory (mirrors train/test structure)
  PYTHONPATH=/home/zym python3 EATalker/tools/resample_bs_to_60fps.py \
    --root /home/zym/EATalker/EATalker_data_template/EATalker \
    --out /home/zym/experiments/bs_60fps_fixed
"""

import argparse
import os
import wave
import contextlib
import numpy as np


def wav_duration_seconds(path: str) -> float:
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        return wf.getnframes() / float(wf.getframerate())


def linear_resample(arr: np.ndarray, target_len: int) -> np.ndarray:
    if arr.shape[0] == target_len:
        return arr.astype(np.float32, copy=True)
    if arr.shape[0] <= 1:
        return np.tile(arr.astype(np.float32, copy=True), (target_len, 1))
    x = np.linspace(0.0, 1.0, arr.shape[0], dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, target_len, dtype=np.float64)
    out = np.empty((target_len, arr.shape[1]), dtype=np.float32)
    for d in range(arr.shape[1]):
        out[:, d] = np.interp(x_new, x, arr[:, d].astype(np.float64))
    return out


def process_split(root: str, split: str, out_dir: str = None, max_n: int = 0, dry: bool = False):
    split_dir = os.path.join(root, split)
    if not os.path.isdir(split_dir):
        return
    names = sorted([f[:-7] for f in os.listdir(split_dir) if f.endswith('_bs.npy')])
    cnt = 0
    for name in names:
        bs_path = os.path.join(split_dir, name + '_bs.npy')
        wav_path = os.path.join(split_dir, name + '.wav')
        if not os.path.exists(wav_path):
            continue
        try:
            dur = wav_duration_seconds(wav_path)
        except Exception:
            continue
        target = int(round(dur * 60.0))
        try:
            bs = np.load(bs_path)
        except Exception:
            continue
        src = bs.shape[0]
        if target <= 0 or src <= 0:
            continue
        if abs(src - target) <= 1:
            # already ok-ish
            out_bs = bs.astype(np.float32, copy=True)
        else:
            out_bs = linear_resample(bs, target)
        if dry:
            print(f'[{split}] {name}: {src} -> {target} (no write)')
        else:
            if out_dir:
                os.makedirs(os.path.join(out_dir, split), exist_ok=True)
                out_path = os.path.join(out_dir, split, name + '_bs.npy')
            else:
                out_path = os.path.join(split_dir, name + '_60fps.npy')
            np.save(out_path, out_bs)
            print(f'[{split}] {name}: {src}->{target} written to {out_path}')
        cnt += 1
        if max_n and cnt >= max_n:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='Path to .../EATalker_data_template/EATalker')
    ap.add_argument('--out', default='', help='Optional output dir; if empty, writes _60fps.npy next to originals')
    ap.add_argument('--max', type=int, default=0, help='Max items per split (0 = all)')
    ap.add_argument('--dry', action='store_true', help='Dry-run without writing files')
    args = ap.parse_args()

    for split in ('train', 'test'):
        process_split(args.root, split, args.out or None, args.max, args.dry)


if __name__ == '__main__':
    main()

