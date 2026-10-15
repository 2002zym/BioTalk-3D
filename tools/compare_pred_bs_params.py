#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Predict blendshape params and compare to GT blendshape params."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import soxr
import torch
from transformers import Wav2Vec2FeatureExtractor

from EATalker.dataset.dataset_config import dataset_config
from EATalker.models.eatalker import EATalker
from EATalker.utils.config import load_cfg_from_cfg_file, merge_cfg_from_list
from EATalker.utils.utils import get_audio_encoder_dim, get_template_verts


def get_all_audios(audio_root: str) -> List[str]:
    wav_names = []
    for root, _, files in os.walk(audio_root):
        for fname in files:
            if fname.endswith((".wav", ".mp3")):
                rel_path = os.path.relpath(os.path.join(root, fname), audio_root)
                wav_names.append(rel_path)
    wav_names.sort()
    return wav_names


def split_long_audio(audio: np.ndarray,
                     processor: Wav2Vec2FeatureExtractor) -> List[np.ndarray]:
    a, b = 25, 5
    sr = 16000
    total_length = len(audio) / sr
    reps = max(0, int(np.ceil((total_length - a) / (a - b)))) + 1
    splits = []
    start, end = 0, int(a * sr)
    step = int((a - b) * sr)
    for _ in range(reps):
        chunk = audio[start:end]
        chunk = np.squeeze(
            processor(chunk, sampling_rate=sr).input_values).astype(np.float32)
        splits.append(chunk)
        start += step
        end += step
    return splits


def merge_out_list(out_list: List[np.ndarray], fps: int) -> np.ndarray:
    if len(out_list) == 1:
        return out_list[0]
    a, b = 25, 5
    left_weight = np.linspace(1, 0, b * fps)[:, np.newaxis]
    right_weight = 1 - left_weight
    a_frames = a * fps
    b_frames = b * fps
    offset = a_frames - b_frames

    out_length = len(out_list[-1]) + offset * (len(out_list) - 1)
    merged = np.empty((out_length, out_list[-1].shape[-1]),
                      dtype=out_list[-1].dtype)
    merged[:a_frames] = out_list[0]
    cursor = a_frames
    for out_piece in out_list[1:]:
        merged[cursor - b_frames:cursor] = (
            left_weight * merged[cursor - b_frames:cursor] +
            right_weight * out_piece[:b_frames])
        merged[cursor:cursor + offset] = out_piece[b_frames:]
        cursor += offset
    return merged


def trim_audio_duration(audio: np.ndarray,
                        sample_rate: int,
                        max_seconds: float = 20.0) -> np.ndarray:
    max_len = int(max_seconds * sample_rate)
    if len(audio) <= max_len:
        return audio
    return audio[:max_len]


def find_gt_bs(gt_root: Path, stem: str) -> Optional[Path]:
    for sub in ["", "train", "test", "blendshapes"]:
        p = gt_root / sub / f"{stem}_bs.npy"
        if p.exists():
            return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare predicted bs to GT bs.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weight_path", type=str, required=True)
    parser.add_argument("--wav-dir", type=str, required=True)
    parser.add_argument("--gt-root", type=str, required=True,
                        help="Dataset root containing train/test/blendshapes")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--use-eeg", action="store_true")
    parser.add_argument("--pure-eeg", action="store_true")
    parser.add_argument("--opts", nargs=argparse.REMAINDER)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limit number of wavs (0 = all)")
    args_cli = parser.parse_args()

    cfg = load_cfg_from_cfg_file(args_cli.config)
    if args_cli.opts:
        cfg = merge_cfg_from_list(cfg, args_cli.opts)
    if args_cli.use_eeg:
        cfg.use_eeg = True
    cfg.pure_eeg = args_cli.pure_eeg

    cfg.dataset = cfg.dataset.split(",") if isinstance(cfg.dataset, str) else cfg.dataset
    cfg.demo_dataset = cfg.demo_dataset.split(",") if isinstance(cfg.demo_dataset, str) else cfg.demo_dataset
    if isinstance(cfg.duplicate_list, str):
        cfg.duplicate_list = [int(i) for i in cfg.duplicate_list.split(",") if i.strip()]

    weight_path = args_cli.weight_path
    checkpoint = torch.load(weight_path, map_location="cpu")
    cfg.identity_num = len(checkpoint["decoder.learnable_style_emb.weight"])

    renamed = {}
    for k, v in checkpoint.items():
        if k.startswith("decoder.audio_feature_map."):
            renamed[k.replace("audio_feature_map", "multimodal_feature_map")] = v
    checkpoint.update(renamed)

    cfg.audio_encoder_feature_dim = get_audio_encoder_dim(cfg.audio_encoder_repo)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EATalker(cfg)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    model.to(device)

    processor = Wav2Vec2FeatureExtractor.from_pretrained(cfg.audio_encoder_repo)
    wav_names = get_all_audios(args_cli.wav_dir)
    if args_cli.max_samples > 0:
        wav_names = wav_names[:args_cli.max_samples]

    gt_root = Path(args_cli.gt_root)
    out_path = Path(args_cli.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    missing_gt = []

    with torch.no_grad():
        for wav_rel in wav_names:
            name_lower = wav_rel.lower()
            inferred_id = 1 if "pj" in name_lower else 0
            wav_path = os.path.join(args_cli.wav_dir, wav_rel)
            audio_wave, sr = sf.read(wav_path, always_2d=False)
            if audio_wave.ndim > 1:
                audio_wave = audio_wave.mean(axis=1)
            audio_wave = audio_wave.astype(np.float32)
            if sr != 16000:
                audio_wave = soxr.resample(audio_wave, sr, 16000)
                sr = 16000
            audio_wave = trim_audio_duration(audio_wave, sr)

            if cfg.pure_eeg:
                audio_splits = [None]
            else:
                audio_splits = split_long_audio(audio_wave, processor)

            for dataset_name in cfg.demo_dataset:
                annot_type = dataset_config[dataset_name]["annot_type"]
                template_idx = inferred_id
                template_np = get_template_verts(cfg.data_root, dataset_name, template_idx)
                template = torch.tensor(template_np.reshape(1, -1),
                                        dtype=torch.float32, device=device)
                template = dataset_config[dataset_name]["scale"] * template
                condition_idx = inferred_id % cfg.identity_num
                condition_id = torch.tensor([condition_idx], dtype=torch.long, device=device)

                fps = 30
                chunk_outputs: List[np.ndarray] = []
                for chunk in audio_splits:
                    if chunk is not None:
                        audio_tensor = torch.tensor(chunk[None], dtype=torch.float32, device=device)
                    else:
                        audio_tensor = None
                    out, _, _ = model(
                        audio=audio_tensor,
                        eeg=None,
                        template=template,
                        face_motion=None,
                        style_idx=condition_id,
                        annot_type=annot_type,
                        fps=fps,
                    )
                    chunk_outputs.append(out.cpu().numpy().squeeze(0))
                pred_bs = merge_out_list(chunk_outputs, fps)

                stem = Path(wav_rel).stem
                gt_bs_path = find_gt_bs(gt_root, stem)
                if gt_bs_path is None:
                    missing_gt.append(wav_rel)
                    continue
                gt_bs = np.load(gt_bs_path).astype(np.float32)

                t = min(len(pred_bs), len(gt_bs))
                if t == 0:
                    continue
                pred_bs_t = pred_bs[:t]
                gt_bs_t = gt_bs[:t]
                diff = pred_bs_t - gt_bs_t
                l1 = float(np.mean(np.abs(diff)))
                l2 = float(np.sqrt(np.mean(diff ** 2)))
                per_dim_l1 = np.mean(np.abs(diff), axis=0).tolist()

                pred_max = pred_bs_t.max(axis=0).tolist()
                gt_max = gt_bs_t.max(axis=0).tolist()
                results[wav_rel] = {
                    "frames": int(t),
                    "l1_mean": l1,
                    "l2_mean": l2,
                    "per_dim_l1": per_dim_l1,
                    "pred_max": pred_max,
                    "gt_max": gt_max,
                }

    # Aggregate max ratio per dimension
    per_dim_ratio = None
    per_dim_pred_max = None
    per_dim_gt_max = None
    dim_count = 0
    for v in results.values():
        if "pred_max" not in v or "gt_max" not in v:
            continue
        pred_max = np.asarray(v["pred_max"], dtype=np.float64)
        gt_max = np.asarray(v["gt_max"], dtype=np.float64)
        if pred_max.shape != gt_max.shape:
            continue
        ratio = pred_max / (gt_max + 1e-8)
        if per_dim_ratio is None:
            per_dim_ratio = ratio.copy()
            per_dim_pred_max = pred_max.copy()
            per_dim_gt_max = gt_max.copy()
        else:
            per_dim_ratio += ratio
            per_dim_pred_max += pred_max
            per_dim_gt_max += gt_max
        dim_count += 1
    if dim_count > 0:
        per_dim_ratio = (per_dim_ratio / dim_count).tolist()
        per_dim_pred_max = (per_dim_pred_max / dim_count).tolist()
        per_dim_gt_max = (per_dim_gt_max / dim_count).tolist()

    report = {
        "pred_npz": None,
        "wav_dir": args_cli.wav_dir,
        "gt_root": args_cli.gt_root,
        "num_wavs": len(wav_names),
        "num_compared": len(results),
        "missing_gt": missing_gt,
        "samples": results,
        "per_dim_ratio_mean": per_dim_ratio,
        "per_dim_pred_max_mean": per_dim_pred_max,
        "per_dim_gt_max_mean": per_dim_gt_max,
    }
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved report to {out_path}")
    if missing_gt:
        print(f"Missing GT for {len(missing_gt)} files (e.g., {missing_gt[:3]})")


if __name__ == "__main__":
    main()
