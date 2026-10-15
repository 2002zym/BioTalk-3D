#!/usr/bin/env python3
"""Sanity check one BioTalk sample for shape/time consistency."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf


def effective_fps(fps: int) -> int:
    if fps == 60:
        return 30
    if fps == 100:
        return 25
    return int(fps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity check one split sample.")
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-diff-sec", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.split_json.read_text(encoding="utf-8"))
    data = payload.get("data", [])
    if not data:
        raise RuntimeError(f"No samples in {args.split_json}")

    rng = random.Random(args.seed)
    sample = data[rng.randrange(len(data))]

    annot = np.load(sample["annot_path"]).astype(np.float32)
    eeg = np.load(sample["eeg_path"]).astype(np.float32)
    audio, sr = sf.read(sample["audio_path"], always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    src_fps = int(sample.get("fps", 60))
    tgt_fps = effective_fps(src_fps)
    effective_annot = annot[::2] if src_fps == 60 else annot
    effective_annot = effective_annot[::4] if src_fps == 100 else effective_annot

    bs_duration = len(effective_annot) / float(tgt_fps)
    eeg_duration = eeg.shape[0] / 1000.0
    audio_duration = len(audio) / float(sr)

    report = {
        "sample_id": sample.get("sample_id", ""),
        "subject": sample.get("subject", ""),
        "session": sample.get("session", ""),
        "language": sample.get("language", ""),
        "audio_shape": list(audio.shape),
        "audio_sr": int(sr),
        "annot_shape_raw": list(annot.shape),
        "annot_shape_effective": list(effective_annot.shape),
        "eeg_shape": list(eeg.shape),
        "source_fps": src_fps,
        "target_fps": tgt_fps,
        "duration_sec": {
            "audio": round(audio_duration, 4),
            "blendshape_effective": round(bs_duration, 4),
            "eeg": round(eeg_duration, 4),
        },
        "abs_diff_sec": {
            "audio_vs_bs": round(abs(audio_duration - bs_duration), 4),
            "audio_vs_eeg": round(abs(audio_duration - eeg_duration), 4),
            "bs_vs_eeg": round(abs(bs_duration - eeg_duration), 4),
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if abs(audio_duration - bs_duration) > args.max_diff_sec:
        raise AssertionError(
            f"audio-vs-bs duration diff too large: {abs(audio_duration - bs_duration):.4f}s"
        )
    if abs(audio_duration - eeg_duration) > args.max_diff_sec:
        raise AssertionError(
            f"audio-vs-eeg duration diff too large: {abs(audio_duration - eeg_duration):.4f}s"
        )


if __name__ == "__main__":
    main()
