#!/usr/bin/env python3
"""
Overwrite existing *_bs.npy with strict-60fps versions (keep filename unchanged).

Scopes:
- train, test: if a sibling *_60fps.npy exists (from resample_bs_to_60fps.py),
  load it and overwrite the original *_bs.npy; otherwise compute from WAV.
- blendshapes: compute target length from audio/<name>.wav and resample inline.

Usage:
  PYTHONPATH=/home/zym python3 EATalker/tools/overwrite_bs_with_60fps.py \
    --root /home/zym/EATalker/EATalker_data_template/EATalker
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


def overwrite_from_sidecar(split_dir: str) -> int:
    """For train/test: if *_60fps.npy exists, use it to overwrite *_bs.npy.
    Fallback: resample from WAV.
    """
    if not os.path.isdir(split_dir):
        return 0
    count = 0
    for fname in os.listdir(split_dir):
        if not fname.endswith('_bs.npy'):
            continue
        base = fname[:-7]  # strip _bs.npy
        bs_path = os.path.join(split_dir, fname)
        sidecar = os.path.join(split_dir, base + '_60fps.npy')
        if os.path.exists(sidecar):
            data = np.load(sidecar)
            np.save(bs_path, data.astype(np.float32))
            count += 1
        else:
            # fallback: resample from WAV
            wav_path = os.path.join(split_dir, base + '.wav')
            if not os.path.exists(wav_path):
                continue
            dur = wav_duration_seconds(wav_path)
            target = int(round(dur * 60.0))
            src = np.load(bs_path)
            fixed = linear_resample(src, target)
            np.save(bs_path, fixed)
            count += 1
    return count


def overwrite_blendshapes(root: str) -> int:
    """For blendshapes dir: resample each *_bs.npy using audio/<name>.wav."""
    bdir = os.path.join(root, 'blendshapes')
    adir = os.path.join(root, 'audio')
    if not os.path.isdir(bdir) or not os.path.isdir(adir):
        return 0
    count = 0
    for fname in os.listdir(bdir):
        if not fname.endswith('_bs.npy'):
            continue
        base = fname[:-7]
        bs_path = os.path.join(bdir, fname)
        wav_path = os.path.join(adir, base + '.wav')
        if not os.path.exists(wav_path):
            continue
        try:
            dur = wav_duration_seconds(wav_path)
            target = int(round(dur * 60.0))
            src = np.load(bs_path)
            fixed = linear_resample(src, target)
            np.save(bs_path, fixed)
            count += 1
        except Exception:
            continue
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='Path to .../EATalker_data_template/EATalker')
    args = ap.parse_args()

    total = 0
    for split in ('train', 'test'):
        d = os.path.join(args.root, split)
        c = overwrite_from_sidecar(d)
        print(f'[overwrite] {split}: {c} files')
        total += c
    c = overwrite_blendshapes(args.root)
    print(f'[overwrite] blendshapes: {c} files')
    total += c
    print(f'[done] total overwritten: {total}')


if __name__ == '__main__':
    main()

