#!/usr/bin/env python3
"""
Render ground-truth videos for the EATalker test split.

This script:
1) Builds a vertices npz from (wav, *_bs.npy) pairs in the test folder (GT).
2) Renders mp4 videos into the output directory.
3) Verifies that rendered mp4s match the test wav list.

Example:
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/home/zym \\
    python3 -m EATalker.tools.render_testset_gt \\
      --test-dir /home/zym/EATalker/EATalker_data_template/EATalker/test \\
      --out-npz /home/zym/experiments/test_results/test_gt_vertices.npz \\
      --out-videos /home/zym/EATalker/render_audio/test_videos
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from EATalker.loss.loss import BlendShapeLoss
from EATalker.main.eat_render import array_to_video, get_obj_faces, render_meshes


def load_bs_array(path: Path) -> np.ndarray:
    arr = np.load(str(path)).astype(np.float32)
    if arr.ndim == 1:
        arr = arr[None]
    if arr.ndim == 2:
        if arr.shape[1] == 51:
            return arr
        if arr.shape[0] == 51:
            return arr.T
    raise ValueError(f"Unsupported bs shape {arr.shape} in {path}")


def build_vertices_npz(
    test_dir: Path,
    out_npz: Path,
    decimate: int = 1,
    basis_npz: Path = Path(
        "/home/zym/EATalker/resources/binary_resources/inhouse_arkit.npz"),
    mouth_idx: Path = Path(
        "/home/zym/EATalker/resources/binary_resources/05_inhouse_arkit_mouth_idx.npy"
    ),
) -> List[str]:
    test_dir = Path(test_dir)
    bs_files = sorted(test_dir.glob("*_bs.npy"))
    if not bs_files:
        raise RuntimeError(f"No *_bs.npy found in {test_dir}")

    loss = BlendShapeLoss(
        mouth_indices_path=str(mouth_idx),
        blendshape_path=str(basis_npz),
        bs_beta=0.0,
    )

    out_dict: Dict[str, Dict[str, np.ndarray]] = {}
    wav_names: List[str] = []
    for bs_path in bs_files:
        stem = bs_path.stem
        wav_stem = stem[:-3] if stem.endswith("_bs") else stem
        wav_path = test_dir / f"{wav_stem}.wav"
        if not wav_path.exists():
            continue
        bs = load_bs_array(bs_path)
        if decimate and decimate > 1:
            bs = bs[::decimate]
        bs_t = torch.tensor(bs, dtype=torch.float32)
        verts_t = loss.get_vertices(bs_t)  # (T,N,3)
        out_dict[wav_path.name] = {"EATalker": verts_t.numpy()}
        wav_names.append(wav_path.name)

    if not out_dict:
        raise RuntimeError(f"No (wav, *_bs.npy) pairs found in {test_dir}")

    os.makedirs(out_npz.parent, exist_ok=True)
    np.savez(str(out_npz), **out_dict)
    return sorted(wav_names)


def render_npz_to_videos(
    npz_path: Path,
    audio_dir: Path,
    out_videos: Path,
    device: str,
    img_size: int = 256,
) -> List[str]:
    audio_dir = Path(audio_dir)
    out_videos = Path(out_videos)
    out_videos.mkdir(parents=True, exist_ok=True)
    data = np.load(str(npz_path), allow_pickle=True)

    rendered: List[str] = []
    for wav_name in data.keys():
        wav_path = audio_dir / wav_name
        cur = data[wav_name]
        if isinstance(cur, np.ndarray):
            cur = cur.item()
        for datasetname, verts in cur.items():
            faces = get_obj_faces("inhouse_blendshape_weight")
            verts_t = torch.tensor(verts, dtype=torch.float32, device=device)
            faces_t = torch.tensor(faces[None], dtype=torch.long, device=device)
            with torch.no_grad():
                img_array = render_meshes(
                    verts_t,
                    faces_t,
                    img_size=(img_size, img_size),
                    aa_factor=1,
                )
                img_array = (img_array[..., :3].cpu().numpy() * 255).astype(
                    np.uint8)

            out_key = wav_name[:-4]
            out_path = out_videos / f"{out_key}_{datasetname}.mp4"
            tmp_path = out_videos / "tmp.mp4"
            if tmp_path.exists():
                tmp_path.unlink()
            array_to_video(img_array, output_path=str(tmp_path), fps=30)

            cmd = [
                "ffmpeg",
                "-loglevel",
                "error",
                "-nostdin",
                "-i",
                str(tmp_path),
                "-i",
                str(wav_path),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                "-strict",
                "-2",
                str(out_path),
                "-y",
            ]
            subprocess.run(cmd, check=True)
            if tmp_path.exists():
                tmp_path.unlink()
            rendered.append(out_path.name)
    return sorted(rendered)


def verify_alignment(test_wavs: List[str],
                     out_videos: Path) -> Tuple[List[str], List[str]]:
    out_videos = Path(out_videos)
    expected = {f"{Path(w).stem}_EATalker.mp4" for w in test_wavs}
    actual = {p.name for p in out_videos.glob("*.mp4")}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    return missing, extra


def main() -> None:
    ap = argparse.ArgumentParser(description="Render GT videos for the test split")
    ap.add_argument("--test-dir",
                    type=Path,
                    required=True,
                    help="EATalker/test folder containing wav + *_bs.npy")
    ap.add_argument("--out-npz",
                    type=Path,
                    required=True,
                    help="Output vertices npz path")
    ap.add_argument("--out-videos",
                    type=Path,
                    required=True,
                    help="Output mp4 directory")
    ap.add_argument("--decimate",
                    type=int,
                    default=1,
                    help="Temporal decimation (e.g., 2 for 60->30fps)")
    ap.add_argument("--device",
                    type=str,
                    default=None,
                    help="cuda / cpu (default: auto)")
    ap.add_argument("--skip-render",
                    action="store_true",
                    help="Only build npz and verify (no rendering)")
    ap.add_argument("--img-size",
                    type=int,
                    default=256,
                    help="Rendered frame size, e.g. 256 or 512")
    args = ap.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    test_wavs = build_vertices_npz(args.test_dir, args.out_npz, decimate=args.decimate)
    print(f"Built npz: {args.out_npz} ({len(test_wavs)} wavs)")

    if not args.skip_render:
        rendered = render_npz_to_videos(args.out_npz,
                                        args.test_dir,
                                        args.out_videos,
                                        device=device,
                                        img_size=args.img_size)
        print(f"Rendered videos: {len(rendered)} -> {args.out_videos}")

    missing, extra = verify_alignment(test_wavs, args.out_videos)
    if missing:
        print("Missing mp4:")
        for m in missing:
            print("  ", m)
    if extra:
        print("Extra mp4:")
        for e in extra:
            print("  ", e)
    if not missing and not extra:
        print("OK: test wavs and rendered mp4s match 1:1")


if __name__ == "__main__":
    main()
