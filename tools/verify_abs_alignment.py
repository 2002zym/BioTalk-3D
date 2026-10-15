#!/usr/bin/env python3
"""
Verify absolute start/end alignment across Audio / CSV / EEG for a single sample.

Limits:
- Audio absolute time is inferred from per-session start time (from known mapping)
  plus the per-utterance relative duration (from the WAV itself). If you don't
  have the per-utterance relative offsets file (标签 1.txt), we only verify
  duration consistency and CSV timecode step (fps) correctness.
- EEG absolute time requires access to the original raw file (SET/CNT) to read
  meas_date / markers. If only per-utterance EEG *.npy exists, absolute time
  cannot be reconstructed here.

Usage example:
  PYTHONPATH=/home/zym python3 EATalker/tools/verify_abs_alignment.py \
    --sample /home/zym/EATalker/EATalker_data_template/EATalker/test/sentence4_liu2_happy_bs.npy
"""

import argparse
import os
import re
import wave
import contextlib
import numpy as np
import csv


PERSON_CSV_MAP = {
    'liu1': 'MySlate_10_iPhone_liu1.csv',
    'liu2': 'MySlate_11_iPhone_liu2.csv',
    'pj1':  'MySlate_9_iPhone.csv',
    'pj2':  'MySlate_8_iPhone_pj2.csv',
}

PERSON_START_LOCAL = {
    # Local start time strings taken from generic_data_processor
    'liu1': '10:38:35:16.995',
    'liu2': '10:53:38:51.926',
    'pj1':  '22:13:04:12.344',
    'pj2':  '21:51:00:17.838',
}


def wav_duration_seconds(path: str):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        return wf.getnframes() / float(wf.getframerate())


def parse_timecode_seconds(timecode: str, fps: float = 60.0, mode: str = 'frames_only') -> float:
    m = re.match(r'^(\d+):(\d+):(\d+):(\d+)\.(\d+)$', timecode)
    if not m:
        return 0.0
    h, mi, s, frames, ms = map(int, m.groups())
    base = h * 3600 + mi * 60 + s
    if mode == 'frames_only':
        sub = frames / fps
    elif mode == 'millis_only':
        sub = ms / 1000.0
    else:
        # previous buggy behavior would add both; we keep for comparison only
        sub = frames / fps + ms / 1000.0
    return base + sub


def estimate_csv_fps(csv_path: str, sample_n: int = 100) -> float:
    times = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= sample_n:
                break
            tc = row.get('Timecode') or row.get('timecode') or ''
            if not tc:
                continue
            # The files appear to be UTC; local vs UTC is irrelevant for fps.
            t_frames = parse_timecode_seconds(tc, mode='frames_only')
            t_ms = parse_timecode_seconds(tc, mode='millis_only')
            times.append((t_frames, t_ms))
    if len(times) < 2:
        return 0.0
    # Use frames_only interpretation to compute step; ideal step ~ 1/60
    steps = [times[i+1][0] - times[i][0] for i in range(len(times)-1)]
    mean_step = sum(steps) / len(steps)
    if mean_step <= 0:
        return 0.0
    return 1.0 / mean_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', required=True, help='Path to *_bs.npy under .../EATalker/{train,test}')
    ap.add_argument('--root', default='/home/zym/EATalker/EATalker_data_template', help='Dataset template root')
    args = ap.parse_args()

    bs_path = args.sample
    if not os.path.exists(bs_path):
        print(f'[ERR] bs not found: {bs_path}')
        return
    base = os.path.basename(bs_path).replace('_bs.npy', '')
    m = re.match(r'^sentence(\d+)_([a-z0-9]+)_(\w+)$', base)
    if not m:
        print(f'[ERR] cannot parse sample name from {base}')
        return
    sid, person, emotion = m.groups()
    split = os.path.basename(os.path.dirname(bs_path))  # train or test
    ds_root = os.path.join(args.root, 'EATalker')
    wav_path = os.path.join(ds_root, split, f'{base}.wav')
    eeg_path = os.path.join(ds_root, split, f'{base}_eeg.npy')

    if not os.path.exists(wav_path):
        print(f'[ERR] wav not found: {wav_path}')
        return

    dur = wav_duration_seconds(wav_path)
    bs_frames = int(np.load(bs_path).shape[0])
    eff_bs_fps = bs_frames / dur if dur > 0 else 0.0

    csv_name = PERSON_CSV_MAP.get(person)
    csv_path = os.path.join(args.root, csv_name) if csv_name else None
    csv_fps_est = estimate_csv_fps(csv_path) if csv_path and os.path.exists(csv_path) else 0.0

    print('--- Sample ---')
    print(f'Name      : {base} ({split})')
    print(f'Audio sec : {dur:.3f}s')
    print(f'BS frames : {bs_frames}  (effective ~{eff_bs_fps:.2f} fps)')
    if csv_path:
        print(f'CSV       : {csv_path}  (estimated fps ~{csv_fps_est:.2f})')
    else:
        print('CSV       : [not mapped for this person]')

    # Absolute time check requires session relative start offsets; not available here.
    # We can still conclude on alignment consistency:
    aligned = abs(eff_bs_fps - 60.0) < 0.5  # near-60 fps
    if not aligned:
        print('[ALIGN] Blendshape timebase deviates from 60 fps → likely mis-parsed timecode (frames + milliseconds).')
    else:
        print('[ALIGN] Blendshape timebase ≈60 fps.')

    if os.path.exists(eeg_path):
        eeg_T = int(np.load(eeg_path).shape[0])  # assume [T, C]
        eeg_eff = eeg_T / dur if dur > 0 else 0.0
        print(f'EEG T     : {eeg_T} (effective ~{eeg_eff:.1f} Hz)')
    else:
        print('EEG       : [per-utterance *.npy not found or not used for abs time]')

    print('\nConclusion:')
    if not aligned:
        print('- BS timebase inconsistent with 60 fps. Fix CSV timecode parsing before retraining.')
    else:
        print('- BS timebase looks fine; if FDD still high, inspect EEG start/markers and fusion regularization.')


if __name__ == '__main__':
    main()

