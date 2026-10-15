#!/usr/bin/env python3
"""Precompute BioTalk audio/EEG features for faster training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor

from EATalker.models.eeg_encoder import EEGEncoder
from EATalker.models.wav2vec2 import Wav2Vec2Model
from EATalker.models.wavlm import WavLMModel
from EATalker.utils import config as cfgmod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache audio/EEG features")
    parser.add_argument("--config",
                        type=str,
                        default="/home/zym/EATalker/config/eatalker.yaml")
    parser.add_argument("--split-json",
                        type=str,
                        nargs="+",
                        required=True,
                        help="One or more split json files")
    parser.add_argument("--out-dir",
                        type=str,
                        required=True,
                        help="Directory for cached .npy features")
    parser.add_argument("--manifest-out",
                        type=str,
                        default="",
                        help="Output manifest path; default <out-dir>/cache_manifest.json")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--use-audio", action="store_true", default=True)
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--use-eeg", action="store_true", default=True)
    parser.add_argument("--no-eeg", action="store_true")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    args = parser.parse_args()
    if args.no_audio:
        args.use_audio = False
    if args.no_eeg:
        args.use_eeg = False
    return args


def _load_cfg(args: argparse.Namespace):
    cfg = cfgmod.load_cfg_from_cfg_file(args.config)
    if args.opts:
        cfg = cfgmod.merge_cfg_from_list(cfg, args.opts)
    return cfg


def _safe_sample_id(item: dict) -> str:
    sample_id = item.get("sample_id", "")
    if isinstance(sample_id, str) and sample_id:
        return sample_id
    annot_path = Path(item.get("annot_path", ""))
    stem = annot_path.stem
    if stem.endswith("_bs"):
        stem = stem[:-3]
    return stem or "unknown"


def _safe_key(subject: str, sample_id: str) -> str:
    if subject:
        return f"{subject}::{sample_id}"
    return sample_id


def _safe_filename(name: str) -> str:
    return (name.replace("/", "_").replace("\\", "_").replace(":",
                                                              "__").replace(
                                                                  " ", "_"))


def _resolve_fps(annot_array: np.ndarray, fps: int) -> Tuple[np.ndarray, int]:
    if int(fps) == 60:
        return annot_array[::2], 30
    if int(fps) == 100:
        return annot_array[::4], 25
    return annot_array, int(fps)


def _load_audio(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32)
    wav, _ = librosa.load(path, sr=16000)
    return wav.astype(np.float32)


def _build_models(cfg, device: torch.device, use_audio: bool, use_eeg: bool):
    audio_encoder = None
    audio_processor = None
    eeg_encoder = None

    if use_audio:
        audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            cfg.audio_encoder_repo)
        if "wav2vec2" in cfg.audio_encoder_repo:
            audio_encoder = Wav2Vec2Model.from_pretrained(cfg.audio_encoder_repo)
        elif "wavlm" in cfg.audio_encoder_repo:
            audio_encoder = WavLMModel.from_pretrained(cfg.audio_encoder_repo)
        else:
            raise ValueError(
                f"Unsupported audio_encoder_repo: {cfg.audio_encoder_repo}")
        audio_encoder = audio_encoder.to(device).eval()

    if use_eeg:
        eeg_encoder = EEGEncoder(
            eeg_channels=int(getattr(cfg, "eeg_channels", 60)),
            eeg_length=int(getattr(cfg, "eeg_length", 256)),
            output_dim=int(getattr(cfg, "eeg_output_dim", 128)),
            dropout_rate=float(getattr(cfg, "eeg_dropout", 0.5)),
            kernel_length=int(getattr(cfg, "eeg_kernel_length", 64)),
            kernel_length2=int(getattr(cfg, "eeg_kernel_length2", 16)),
            F1=int(getattr(cfg, "eeg_F1", 8)),
            D=int(getattr(cfg, "eeg_D", 2)),
            F2=int(getattr(cfg, "eeg_F2", 16)),
            use_lstm=bool(getattr(cfg, "eeg_use_lstm", False)),
            lstm_hidden=int(getattr(cfg, "eeg_lstm_hidden", 64)),
            freeze_eegnet=False,
        ).to(device).eval()

    return audio_processor, audio_encoder, eeg_encoder


def main() -> None:
    args = parse_args()
    cfg = _load_cfg(args)
    out_dir = Path(args.out_dir).resolve()
    out_audio = out_dir / "audio"
    out_eeg = out_dir / "eeg"
    out_audio.mkdir(parents=True, exist_ok=True)
    out_eeg.mkdir(parents=True, exist_ok=True)
    manifest_out = Path(args.manifest_out).resolve() if args.manifest_out else (
        out_dir / "cache_manifest.json")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cache] device={device}")

    audio_processor, audio_encoder, eeg_encoder = _build_models(
        cfg, device, use_audio=bool(args.use_audio), use_eeg=bool(args.use_eeg))
    interpolate_pos = int(getattr(cfg, "interpolate_pos", 1))

    entries: Dict[str, Dict[str, object]] = {}
    total_seen = 0
    total_done = 0
    max_samples = int(args.max_samples or 0)

    for split_json in args.split_json:
        payload = json.loads(Path(split_json).read_text(encoding="utf-8"))
        data = payload.get("data", [])
        for item in tqdm(data, desc=f"cache:{Path(split_json).name}"):
            if max_samples > 0 and total_done >= max_samples:
                break
            total_seen += 1
            subject = str(item.get("subject", "") or "")
            sample_id = _safe_sample_id(item)
            key = _safe_key(subject, sample_id)
            file_tag = _safe_filename(key)
            row: Dict[str, object] = {
                "sample_id": sample_id,
                "subject": subject,
            }

            annot = np.load(item["annot_path"]).astype(np.float32)
            annot, fps = _resolve_fps(annot, int(item.get("fps", 30)))
            duration = min(float(len(annot)) / max(float(fps), 1.0), 20.0)

            if args.use_audio:
                audio_path = item.get("audio_path", "")
                if audio_path and os.path.exists(audio_path):
                    audio = _load_audio(audio_path)
                    duration = min(duration, float(len(audio)) / 16000.0, 20.0)
                    frame_num = max(1, int(round(duration * fps)))
                    audio_len = max(1, int(round(duration * 16000)))
                    out_path = out_audio / f"{file_tag}.npy"
                    if args.overwrite or (not out_path.exists()):
                        audio = audio[:audio_len]
                        audio_input = np.squeeze(
                            audio_processor(audio,
                                            sampling_rate=16000).input_values
                        ).astype(np.float32)
                        with torch.no_grad():
                            audio_tensor = torch.from_numpy(audio_input).unsqueeze(
                                0).to(device)
                            feat = audio_encoder(
                                audio_tensor,
                                frame_num=frame_num,
                                interpolate_pos=interpolate_pos,
                            ).last_hidden_state.squeeze(0).detach().cpu().numpy(
                            )
                        np.save(out_path, feat.astype(np.float32))
                    row["audio_feature_path"] = str(out_path)

            if args.use_eeg:
                eeg_path = item.get("eeg_path", "")
                if eeg_path and os.path.exists(eeg_path):
                    frame_num = max(1, int(round(duration * fps)))
                    out_path = out_eeg / f"{file_tag}.npy"
                    if args.overwrite or (not out_path.exists()):
                        eeg = np.load(eeg_path).astype(np.float32)
                        with torch.no_grad():
                            eeg_tensor = torch.from_numpy(eeg).unsqueeze(0).to(
                                device)
                            feat = eeg_encoder(
                                eeg_tensor, frame_num=frame_num,
                                interpolate_pos=interpolate_pos
                            ).last_hidden_state.squeeze(0).detach().cpu().numpy()
                        np.save(out_path, feat.astype(np.float32))
                    row["eeg_feature_path"] = str(out_path)

            entries[key] = row
            # Backward-compat fallback key when unique.
            if sample_id not in entries:
                entries[sample_id] = row
            total_done += 1
        if max_samples > 0 and total_done >= max_samples:
            break

    manifest = {
        "version": 1,
        "entries": entries,
        "source_splits": [str(Path(p).resolve()) for p in args.split_json],
        "num_entries": len(entries),
        "num_samples_processed": total_done,
        "device": str(device),
    }
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[cache] done seen={total_seen} processed={total_done} manifest={manifest_out}")


if __name__ == "__main__":
    main()

