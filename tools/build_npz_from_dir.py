#!/usr/bin/env python3
"""
Build a vertices npz from a directory of (wav, bs.npy) pairs without dataset JSON.

Usage example:

PYTHONPATH=/home/zym python3 -m EATalker.tools.build_npz_from_dir --in-dir /home/zym/EATalker/zym --out /home/zym/experiments/test_results/zym_vertices.npz --decimate 2

Then render:

PYTHONPATH=/home/zym python3 -m EATalker.main.eat_render /home/zym/experiments/test_results/zym_vertices.npz /home/zym/EATalker/zym /home/zym/experiments/test_results/zym_video

This script converts each 51-dim blendshape npy (T,51) into vertices (T,N,3)
using the configured blendshape basis (default: inhouse_arkit.npz), and packs
them into an npz whose keys are the matching audio filenames in the same folder.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from EATalker.loss.loss import BlendShapeLoss


def find_audio_for_stem(folder: Path, stem: str) -> Optional[Path]:
    """Return an audio path matching the given stem.

    Many datasets save blendshape npy as "<name>_bs.npy" while the audio is
    "<name>.wav". Try both the original stem and the stem with a trailing
    "_bs" removed. Match common audio extensions case-insensitively.
    """
    candidate_stems = []
    # Prefer trimmed stem first if it endswith _bs
    if stem.endswith('_bs'):
        candidate_stems.append(stem[:-3])
    candidate_stems.append(stem)

    exts = ('.wav', '.mp3', '.m4a', '.WAV', '.MP3', '.M4A')
    for s in candidate_stems:
        for ext in exts:
            p = folder / f"{s}{ext}"
            if p.exists():
                return p
    return None


def load_bs_array(path: Path) -> np.ndarray:
    arr = np.load(str(path))
    arr = arr.astype(np.float32)
    if arr.ndim == 1:
        arr = arr[None]
    # Heuristics: accept (T,51) or (51,T)
    if arr.ndim == 2:
        if arr.shape[1] == 51:
            return arr
        if arr.shape[0] == 51:
            return arr.T
    raise ValueError(f"Unsupported bs shape {arr.shape} in {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build vertices npz from a folder of wav/npy pairs")
    ap.add_argument("--in-dir", required=True, type=Path, help="Directory containing *.npy and matching audio files")
    ap.add_argument("--out", required=True, type=Path, help="Output npz path")
    ap.add_argument("--mouth-idx", type=Path,
                    default=Path("/home/zym/EATalker/resources/binary_resources/05_inhouse_arkit_mouth_idx.npy"),
                    help="Mouth index npy (only required to init BlendShapeLoss)")
    ap.add_argument("--basis-npz", type=Path,
                    default=Path("/home/zym/EATalker/resources/binary_resources/inhouse_arkit.npz"),
                    help="Blendshape basis npz (meanshape+blendshape+faces)")
    ap.add_argument("--decimate", type=int, default=1, help="Optional temporal decimation factor (e.g., 2 for 60->30fps)")
    args = ap.parse_args()

    folder: Path = args.in_dir
    if not folder.is_dir():
        raise NotADirectoryError(folder)

    loss = BlendShapeLoss(
        mouth_indices_path=str(args.mouth_idx),
        blendshape_path=str(args.basis_npz),
        bs_beta=0.0,
    )

    out_dict: Dict[str, Dict[str, np.ndarray]] = {}
    npy_list = sorted([p for p in folder.glob("*.npy")])
    kept = 0
    for npy in npy_list:
        stem = npy.stem
        audio = find_audio_for_stem(folder, stem)
        if audio is None:
            # No matching audio -> skip
            continue
        bs = load_bs_array(npy)
        if args.decimate and args.decimate > 1:
            bs = bs[::args.decimate]
        # 通过 BlendShapeLoss.get_vertices 间接调用 bs2vertices：
        # 将 (T,51) 展平成 (T,51) → 顶点 (T,N,3)，再写入 npz
        bs_t = torch.tensor(bs, dtype=torch.float32)            # (T,51)
        verts_t = loss.get_vertices(bs_t)                        # (T,N,3)
        verts = verts_t.numpy()
        # Key must match audio filename
        key = audio.name
        out_dict[key] = { 'EATalker': verts }
        kept += 1

    if not out_dict:
        raise RuntimeError("No (wav,npy) pairs found in the folder (try removing '_bs' suffix or check audio filenames).")

    os.makedirs(args.out.parent, exist_ok=True)
    np.savez(str(args.out), **out_dict)
    print(f"Saved {kept} items -> {args.out}")


if __name__ == "__main__":
    main()
