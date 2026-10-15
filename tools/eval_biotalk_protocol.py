#!/usr/bin/env python3
"""Evaluate BioTalk protocol split with optional EEG temporal shift and visuals."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm


def _ensure_sys_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    project_parent = project_root.parent
    for path in (project_parent, project_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_ensure_sys_path()

from EATalker.dataset.dataset import get_single_dataset  # noqa: E402
from EATalker.loss.loss import EATalkerLoss  # noqa: E402
from EATalker.main.eat_render import (  # noqa: E402
    array_to_video,
    get_obj_faces,
    render_meshes,
)
from EATalker.models.eatalker import EATalker  # noqa: E402
from EATalker.utils import config as cfgmod  # noqa: E402
from EATalker.utils.utils import get_audio_encoder_dim, load_ckpt  # noqa: E402


ARKIT51_ORDER = [
    "EyeBlinkLeft",
    "EyeBlinkRight",
    "EyeSquintLeft",
    "EyeSquintRight",
    "EyeLookDownLeft",
    "EyeLookDownRight",
    "EyeLookInLeft",
    "EyeLookInRight",
    "EyeWideLeft",
    "EyeWideRight",
    "EyeLookOutLeft",
    "EyeLookOutRight",
    "EyeLookUpLeft",
    "EyeLookUpRight",
    "BrowDownLeft",
    "BrowDownRight",
    "BrowInnerUp",
    "BrowOuterUpLeft",
    "BrowOuterUpRight",
    "JawOpen",
    "MouthClose",
    "JawLeft",
    "JawRight",
    "JawForward",
    "MouthUpperUpLeft",
    "MouthUpperUpRight",
    "MouthLowerDownLeft",
    "MouthLowerDownRight",
    "MouthRollUpper",
    "MouthRollLower",
    "MouthSmileLeft",
    "MouthSmileRight",
    "MouthDimpleLeft",
    "MouthDimpleRight",
    "MouthStretchLeft",
    "MouthStretchRight",
    "MouthFrownLeft",
    "MouthFrownRight",
    "MouthPressLeft",
    "MouthPressRight",
    "MouthPucker",
    "MouthFunnel",
    "MouthLeft",
    "MouthRight",
    "MouthShrugLower",
    "MouthShrugUpper",
    "NoseSneerLeft",
    "NoseSneerRight",
    "CheekPuff",
    "CheekSquintLeft",
    "CheekSquintRight",
]

MOUTH_NAMES = {
    "JawOpen",
    "JawLeft",
    "JawRight",
    "JawForward",
    "MouthClose",
    "MouthFunnel",
    "MouthPucker",
    "MouthLeft",
    "MouthRight",
    "MouthSmileLeft",
    "MouthSmileRight",
    "MouthFrownLeft",
    "MouthFrownRight",
    "MouthDimpleLeft",
    "MouthDimpleRight",
    "MouthStretchLeft",
    "MouthStretchRight",
    "MouthRollUpper",
    "MouthRollLower",
    "MouthShrugLower",
    "MouthShrugUpper",
    "MouthPressLeft",
    "MouthPressRight",
    "MouthUpperUpLeft",
    "MouthUpperUpRight",
    "MouthLowerDownLeft",
    "MouthLowerDownRight",
}
UPPER_NAMES = {
    "EyeBlinkLeft",
    "EyeBlinkRight",
    "EyeSquintLeft",
    "EyeSquintRight",
    "EyeLookDownLeft",
    "EyeLookDownRight",
    "EyeLookInLeft",
    "EyeLookInRight",
    "EyeWideLeft",
    "EyeWideRight",
    "EyeLookOutLeft",
    "EyeLookOutRight",
    "EyeLookUpLeft",
    "EyeLookUpRight",
    "BrowDownLeft",
    "BrowDownRight",
    "BrowInnerUp",
    "BrowOuterUpLeft",
    "BrowOuterUpRight",
}

MOUTH_IDX = [i for i, name in enumerate(ARKIT51_ORDER) if name in MOUTH_NAMES]
UPPER_IDX = [i for i, name in enumerate(ARKIT51_ORDER) if name in UPPER_NAMES]


@dataclass
class SampleRecord:
    sample_id: str
    subject: str
    session: str
    language: str
    audio_path: str
    annot_path: str
    annot_type: str
    num_frames: int
    bs_mae_51d: float
    bs_mse_51d: float
    upper_face_mae_51d: float
    mouth_mae_51d: float
    velocity_mae: float
    velocity_mse: float
    acceleration_mae: float
    acceleration_mse: float
    MVE: float
    LVE: float
    LVD: float
    UFVE: float
    FDD: float


def shift_eeg_array(eeg: np.ndarray, shift_steps: int, channels: int) -> np.ndarray:
    """Shift EEG in time with zero padding and keep original shape."""
    if shift_steps == 0:
        return eeg.copy()
    arr = np.asarray(eeg, dtype=np.float32)
    if arr.ndim == 2:
        transposed = False
        if arr.shape[-1] == channels:  # [T, C]
            work = arr
        elif arr.shape[0] == channels:  # [C, T]
            work = arr.T
            transposed = True
        else:
            raise ValueError(
                f"2D EEG shape {arr.shape} incompatible with channels={channels}"
            )
        out = np.zeros_like(work)
        if shift_steps > 0:
            out[shift_steps:] = work[:-shift_steps]
        else:
            steps = abs(shift_steps)
            out[:-steps] = work[steps:]
        return out.T if transposed else out

    if arr.ndim == 3:
        transposed = False
        if arr.shape[-1] == channels:  # [B, T, C]
            work = arr
        elif arr.shape[1] == channels:  # [B, C, T]
            work = np.transpose(arr, (0, 2, 1))
            transposed = True
        else:
            raise ValueError(
                f"3D EEG shape {arr.shape} incompatible with channels={channels}"
            )
        out = np.zeros_like(work)
        if shift_steps > 0:
            out[:, shift_steps:] = work[:, :-shift_steps]
        else:
            steps = abs(shift_steps)
            out[:, :-steps] = work[:, steps:]
        return np.transpose(out, (0, 2, 1)) if transposed else out
    raise ValueError(f"Unsupported EEG ndim={arr.ndim}")


def shift_eeg_tensor(eeg: torch.Tensor, shift_steps: int, channels: int) -> torch.Tensor:
    shifted = shift_eeg_array(eeg.detach().cpu().numpy(), shift_steps, channels)
    return torch.tensor(shifted, dtype=eeg.dtype, device=eeg.device)


def _as_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return list(value)


def _to_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _extract_scalar(batch: Dict[str, Any], key: str, default: str = "") -> str:
    val = batch.get(key, default)
    if isinstance(val, (list, tuple)):
        if not val:
            return default
        return str(val[0])
    return str(val)


def _extract_fps(batch: Dict[str, Any]) -> Optional[float]:
    fps_tensor = batch.get("fps")
    if fps_tensor is None:
        return None
    if isinstance(fps_tensor, torch.Tensor):
        return float(fps_tensor[0].item())
    if isinstance(fps_tensor, (list, tuple)):
        return float(fps_tensor[0])
    return float(fps_tensor)


def _extract_num(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return _extract_num(value[0])
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_audio_duration_sec(batch: Dict[str, Any]) -> Optional[float]:
    original_audio = batch.get("original_audio")
    if isinstance(original_audio, torch.Tensor) and original_audio.numel() > 0:
        length = int(original_audio.shape[-1]) if original_audio.dim() >= 1 else int(original_audio.numel())
        return length / 16000.0
    audio = batch.get("audio")
    if isinstance(audio, torch.Tensor) and audio.numel() > 0:
        length = int(audio.shape[-1]) if audio.dim() >= 1 else int(audio.numel())
        return length / 16000.0
    return None


def _extract_timestamp_field(batch: Dict[str, Any], key: str) -> Optional[float]:
    timestamps = batch.get("timestamps")
    if isinstance(timestamps, dict):
        return _extract_num(timestamps.get(key))
    return None


def _build_valid_mask(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    fps: Optional[float],
    batch: Dict[str, Any],
    sample_id: str,
    sanity_checks: bool,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if pred.shape[:2] != target.shape[:2]:
        raise AssertionError(
            f"[sanity] T mismatch for {sample_id}: T_pred={pred.shape[1]}, T_gt={target.shape[1]}"
        )
    bsz, t_gt = target.shape[:2]
    length_from_audio: Optional[int] = None
    audio_duration = _extract_audio_duration_sec(batch)
    if fps is not None and audio_duration is not None:
        length_from_audio = int(round(audio_duration * fps))
    elif fps is not None:
        ts_duration = _extract_timestamp_field(batch, "duration_sec")
        if ts_duration is not None:
            length_from_audio = int(round(ts_duration * fps))

    if length_from_audio is None:
        valid_lengths = torch.full((bsz,), t_gt, dtype=torch.long, device=target.device)
    else:
        valid_lengths = torch.full(
            (bsz,),
            max(0, min(t_gt, int(length_from_audio))),
            dtype=torch.long,
            device=target.device,
        )
    steps = torch.arange(t_gt, device=target.device).unsqueeze(0)
    valid_mask = steps < valid_lengths.unsqueeze(1)

    ts_target_fps = _extract_timestamp_field(batch, "target_fps")
    ts_source_fps = _extract_timestamp_field(batch, "source_fps")
    if sanity_checks:
        if length_from_audio is None:
            raise AssertionError(
                f"[sanity] Unable to compute expected length from audio/timestamps for {sample_id}"
            )
        if int(length_from_audio) != int(t_gt):
            raise AssertionError(
                f"[sanity] Length mismatch for {sample_id}: "
                f"T_pred={pred.shape[1]}, T_gt={t_gt}, expected={length_from_audio}, "
                f"audio_dur={audio_duration}, fps_used={fps}"
            )
        if fps is None:
            raise AssertionError(f"[sanity] Missing fps in batch for {sample_id}")
        if ts_target_fps is not None and int(round(ts_target_fps)) != int(round(fps)):
            raise AssertionError(
                f"[sanity] FPS mismatch for {sample_id}: batch_fps={fps}, target_fps={ts_target_fps}"
            )
        print(
            "[sanity][fps_len] "
            f"sample={sample_id} T_pred={pred.shape[1]} T_gt={t_gt} "
            f"expected={length_from_audio} audio_sec={audio_duration:.6f} fps_used={fps} "
            f"source_fps={ts_source_fps} target_fps={ts_target_fps}"
        )

    meta = {
        "t_pred": int(pred.shape[1]),
        "t_gt": int(t_gt),
        "expected_t": int(length_from_audio) if length_from_audio is not None else None,
        "audio_duration_sec": float(audio_duration) if audio_duration is not None else None,
        "source_fps": ts_source_fps,
        "target_fps": ts_target_fps,
    }
    return valid_mask, meta


def _masked_mean(values: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
    if values.shape[:2] != time_mask.shape:
        raise ValueError(
            f"mask shape mismatch: values={values.shape}, mask={time_mask.shape}"
        )
    mask = time_mask
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(values).to(dtype=values.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def _compute_dyn_errors(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    device = pred.device
    zero = torch.tensor(0.0, device=device)
    if pred.shape[1] > 1:
        pred_v = pred[:, 1:] - pred[:, :-1]
        target_v = target[:, 1:] - target[:, :-1]
        vel_diff = pred_v - target_v
        vel_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
        vel_mae = _masked_mean(vel_diff.abs(), vel_mask)
        vel_mse = _masked_mean(vel_diff**2, vel_mask)
    else:
        vel_mae = zero
        vel_mse = zero

    if pred.shape[1] > 2:
        pred_a = pred_v[:, 1:] - pred_v[:, :-1]
        target_a = target_v[:, 1:] - target_v[:, :-1]
        acc_diff = pred_a - target_a
        vel_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
        acc_mask = vel_mask[:, 1:] & vel_mask[:, :-1]
        acc_mae = _masked_mean(acc_diff.abs(), acc_mask)
        acc_mse = _masked_mean(acc_diff**2, acc_mask)
    else:
        acc_mae = zero
        acc_mse = zero
    return vel_mae, vel_mse, acc_mae, acc_mse


def _compute_vertex_metrics(
    loss_module: EATalkerLoss,
    pred: torch.Tensor,
    target: torch.Tensor,
    template: torch.Tensor,
    annot_type: str,
    valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    device = pred.device
    zero = torch.tensor(0.0, device=device)
    mve = zero
    lve = zero
    lvd = zero
    ufve = zero
    fdd = zero

    if hasattr(loss_module, "get_vertices_sequence"):
        try:
            pred_vertices = loss_module.get_vertices_sequence(pred, annot_type)
            target_vertices = loss_module.get_vertices_sequence(target, annot_type)
            v_diff_norm = (pred_vertices - target_vertices).norm(dim=-1)  # [B,T,V]
            mve = _masked_mean(v_diff_norm, valid_mask)

            # Mouth metrics in vertex space with the same definition as loss.mouth_metric.
            mouth_idx = getattr(loss_module.loss_module_dict[annot_type], "mouth_idx", None)
            if isinstance(mouth_idx, torch.Tensor) and mouth_idx.numel() > 0:
                mouth_idx = mouth_idx.to(pred_vertices.device)
                mouth_diff_sq = (target_vertices - pred_vertices).pow(2).sum(dim=-1)
                mouth_diff_sq = mouth_diff_sq.index_select(2, mouth_idx)
                l2_per_frame = mouth_diff_sq.max(dim=-1).values  # [B,T]
                lve = _masked_mean(l2_per_frame, valid_mask)
                lvd = _masked_mean(l2_per_frame.sqrt(), valid_mask)

            upper_idx = loss_module.get_upper_face_indices(annot_type)
            if isinstance(upper_idx, torch.Tensor) and upper_idx.numel() > 0:
                upper_idx = upper_idx.to(pred_vertices.device)
                upper_dist = v_diff_norm.index_select(2, upper_idx)
                ufve = _masked_mean(upper_dist, valid_mask)

                if template is not None and pred_vertices.size(1) > 1:
                    if template.dim() == 3 and template.size(1) == 1:
                        template_flat = template[:, 0]
                    elif template.dim() == 2:
                        template_flat = template
                    else:
                        template_flat = template.reshape(template.size(0), -1)
                    template_vertices = loss_module.get_vertices(
                        template_flat.detach(), annot_type
                    )
                    if isinstance(template_vertices, torch.Tensor) and template_vertices.numel() > 0:
                        template_vertices = template_vertices.to(pred_vertices.device)
                        fdd_vals: List[torch.Tensor] = []
                        for b in range(pred_vertices.shape[0]):
                            frame_mask = valid_mask[b]
                            if frame_mask.sum().item() <= 1:
                                continue
                            pred_b = pred_vertices[b][frame_mask].index_select(1, upper_idx)
                            gt_b = target_vertices[b][frame_mask].index_select(1, upper_idx)
                            tmpl_b = template_vertices[b].index_select(0, upper_idx).unsqueeze(0)
                            motion_pred = pred_b - tmpl_b
                            motion_gt = gt_b - tmpl_b
                            energy_pred = motion_pred.pow(2).sum(dim=-1)
                            energy_gt = motion_gt.pow(2).sum(dim=-1)
                            pred_motion_std = energy_pred.std(dim=0, unbiased=False).mean()
                            gt_motion_std = energy_gt.std(dim=0, unbiased=False).mean()
                            fdd_vals.append((gt_motion_std - pred_motion_std).abs())
                        if fdd_vals:
                            fdd = torch.stack(fdd_vals).mean()
        except NotImplementedError:
            pass
    return mve, lve, lvd, ufve, fdd


def _render_comparison(
    *,
    loss_module: EATalkerLoss,
    pred_bs: torch.Tensor,
    gt_bs: torch.Tensor,
    annot_type: str,
    audio_path: str,
    output_path: Path,
    device: torch.device,
    img_size: int,
) -> None:
    faces = get_obj_faces(annot_type)
    faces_t = torch.tensor(faces[None], dtype=torch.long, device=device)
    with torch.no_grad():
        pred_v = loss_module.get_vertices_sequence(pred_bs, annot_type).squeeze(0)
        gt_v = loss_module.get_vertices_sequence(gt_bs, annot_type).squeeze(0)
        pred_img = render_meshes(pred_v.to(device), faces_t, img_size=(img_size, img_size))
        gt_img = render_meshes(gt_v.to(device), faces_t, img_size=(img_size, img_size))

    pred_img = (pred_img[..., :3].detach().cpu().numpy() * 255).astype(np.uint8)
    gt_img = (gt_img[..., :3].detach().cpu().numpy() * 255).astype(np.uint8)
    tmp_dir = output_path.parent / ".tmp_render"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    gt_tmp = tmp_dir / f"{output_path.stem}_gt.mp4"
    pred_tmp = tmp_dir / f"{output_path.stem}_pred.mp4"
    array_to_video(gt_img, str(gt_tmp), fps=30)
    array_to_video(pred_img, str(pred_tmp), fps=30)

    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(gt_tmp),
        "-i",
        str(pred_tmp),
        "-i",
        audio_path,
        "-filter_complex",
        "[0:v][1:v]hstack=inputs=2[v]",
        "-map",
        "[v]",
        "-map",
        "2:a",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    gt_tmp.unlink(missing_ok=True)
    pred_tmp.unlink(missing_ok=True)


def _append_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _write_rows_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = cfgmod.load_cfg_from_cfg_file(args.config)
    if args.opts:
        cfg = cfgmod.merge_cfg_from_list(cfg, args.opts)
    cfg.dataset = _as_list(getattr(cfg, "dataset", "EATalker"))
    cfg.demo_dataset = _as_list(getattr(cfg, "demo_dataset", "EATalker"))
    if isinstance(getattr(cfg, "duplicate_list", None), str):
        cfg.duplicate_list = [int(x) for x in cfg.duplicate_list.split(",") if x]

    cfg.use_audio = bool(args.use_audio)
    cfg.use_eeg = bool(args.use_eeg)
    cfg.audio_encoder_feature_dim = get_audio_encoder_dim(cfg.audio_encoder_repo)
    cfg.workers = int(getattr(cfg, "workers", 0))
    if args.sanity_checks:
        cfg.workers = 0

    loader = get_single_dataset(cfg, json_path=args.split_json)
    cfg.identity_num = loader.dataset.get_identity_num()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: Optional[EATalker]
    if args.sanity_pred_eq_gt:
        model = None
    else:
        model = EATalker(cfg).to(device)
        load_ckpt(
            model,
            args.weight_path,
            style_emb_init_indices=getattr(cfg, "style_emb_init_indices", None),
        )
        model.eval()
    loss_module = EATalkerLoss(cfg).to(device)
    loss_module.eval()

    channels = int(getattr(cfg, "eeg_channels", 60))
    shift_steps = int(round(args.eeg_shift_ms * args.eeg_sample_rate_hz / 1000.0))

    split_info: Dict[str, Any] = {}
    split_json_path = Path(args.split_json).resolve()
    try:
        split_data = json.loads(split_json_path.read_text(encoding="utf-8"))
        split_info = split_data.get("info", {})
    except Exception:  # noqa: BLE001
        split_info = {}

    if args.sanity_checks:
        id_template_path = split_json_path.parent / "id_template.npy"
        id_template_shape = None
        if id_template_path.exists():
            id_template_shape = np.load(id_template_path).shape
        data_items = getattr(loader.dataset, "data_list", [])
        subject_to_ids: Dict[str, set] = {}
        unique_ids: set = set()
        for item in data_items:
            subject = str(getattr(item, "subject", "") or "unknown")
            condition_id = int(getattr(item, "identity_idx", -1))
            unique_ids.add(condition_id)
            subject_to_ids.setdefault(subject, set()).add(condition_id)
        print("[sanity][identity] subject_id -> condition_id mapping:")
        for subject in sorted(subject_to_ids.keys()):
            ids_sorted = sorted(subject_to_ids[subject])
            print(f"[sanity][identity] {subject} -> {ids_sorted}")
            if len(ids_sorted) != 1:
                raise AssertionError(
                    f"[sanity] subject {subject} maps to multiple ids: {ids_sorted}"
                )
        if id_template_shape is not None:
            print(f"[sanity][template] id_template.npy shape={id_template_shape}")
            if unique_ids:
                max_id = max(unique_ids)
                if id_template_shape[0] <= max_id:
                    raise AssertionError(
                        f"[sanity] id_template rows {id_template_shape[0]} cannot cover max id {max_id}"
                    )
            info_ids = split_info.get("id_list", [])
            if info_ids and id_template_shape[0] < len(info_ids):
                raise AssertionError(
                    f"[sanity] id_template rows {id_template_shape[0]} < len(info.id_list)={len(info_ids)}"
                )

    records: List[SampleRecord] = []
    nomask_accum: Dict[str, List[float]] = {
        "bs_mae_51d": [],
        "bs_mse_51d": [],
        "upper_face_mae_51d": [],
        "mouth_mae_51d": [],
        "velocity_mae": [],
        "velocity_mse": [],
        "acceleration_mae": [],
        "acceleration_mse": [],
        "MVE": [],
        "LVE": [],
        "LVD": [],
        "UFVE": [],
        "FDD": [],
    }
    for batch in tqdm(loader, desc=f"eval:{args.protocol}:{args.baseline}"):
        if args.max_samples > 0 and len(records) >= args.max_samples:
            break
        gt = batch["data"].to(device, non_blocking=True)
        template = batch["template"].to(device, non_blocking=True)
        identity = batch["id"].to(device, non_blocking=True)
        annot_type = _extract_scalar(batch, "annot_type", "inhouse_blendshape_weight")
        sample_id = _extract_scalar(batch, "sample_id", Path(_extract_scalar(batch, "annot_path")).stem)
        if sample_id.endswith("_bs"):
            sample_id = sample_id[:-3]
        audio = None
        if cfg.use_audio:
            a = batch.get("audio")
            if isinstance(a, torch.Tensor):
                audio = a.to(device, non_blocking=True)
        eeg = None
        if cfg.use_eeg:
            e = batch.get("eeg")
            if isinstance(e, torch.Tensor):
                eeg = e.to(device, non_blocking=True)
                if shift_steps != 0 and eeg.numel() > 0:
                    eeg = shift_eeg_tensor(eeg, shift_steps=shift_steps, channels=channels)

        fps = _extract_fps(batch)
        if args.sanity_pred_eq_gt:
            pred = gt.clone()
        else:
            if model is None:
                raise RuntimeError("Model is not initialized.")
            with torch.no_grad():
                pred, _, _ = model(
                    audio=audio,
                    eeg=eeg,
                    template=template,
                    face_motion=gt,
                    style_idx=identity,
                    annot_type=annot_type,
                    fps=fps,
                )

        valid_mask, mask_meta = _build_valid_mask(
            pred=pred,
            target=gt,
            fps=fps,
            batch=batch,
            sample_id=sample_id,
            sanity_checks=args.sanity_checks,
        )
        full_mask = torch.ones_like(valid_mask, dtype=torch.bool)
        diff = pred - gt

        bs_mae = _masked_mean(diff.abs(), valid_mask)
        bs_mse = _masked_mean(diff**2, valid_mask)
        upper_mae = (
            _masked_mean(diff[:, :, UPPER_IDX].abs(), valid_mask)
            if UPPER_IDX
            else torch.tensor(0.0, device=device)
        )
        mouth_mae = (
            _masked_mean(diff[:, :, MOUTH_IDX].abs(), valid_mask)
            if MOUTH_IDX
            else torch.tensor(0.0, device=device)
        )
        vel_mae, vel_mse, acc_mae, acc_mse = _compute_dyn_errors(pred, gt, valid_mask)
        mve, lve, lvd, ufve, fdd = _compute_vertex_metrics(
            loss_module,
            pred,
            gt,
            template,
            annot_type,
            valid_mask,
        )

        bs_mae_nomask = _masked_mean(diff.abs(), full_mask)
        bs_mse_nomask = _masked_mean(diff**2, full_mask)
        upper_mae_nomask = (
            _masked_mean(diff[:, :, UPPER_IDX].abs(), full_mask)
            if UPPER_IDX
            else torch.tensor(0.0, device=device)
        )
        mouth_mae_nomask = (
            _masked_mean(diff[:, :, MOUTH_IDX].abs(), full_mask)
            if MOUTH_IDX
            else torch.tensor(0.0, device=device)
        )
        vel_mae_nomask, vel_mse_nomask, acc_mae_nomask, acc_mse_nomask = _compute_dyn_errors(
            pred, gt, full_mask
        )
        mve_nomask, lve_nomask, lvd_nomask, ufve_nomask, fdd_nomask = _compute_vertex_metrics(
            loss_module,
            pred,
            gt,
            template,
            annot_type,
            full_mask,
        )
        nomask_accum["bs_mae_51d"].append(_to_float(bs_mae_nomask))
        nomask_accum["bs_mse_51d"].append(_to_float(bs_mse_nomask))
        nomask_accum["upper_face_mae_51d"].append(_to_float(upper_mae_nomask))
        nomask_accum["mouth_mae_51d"].append(_to_float(mouth_mae_nomask))
        nomask_accum["velocity_mae"].append(_to_float(vel_mae_nomask))
        nomask_accum["velocity_mse"].append(_to_float(vel_mse_nomask))
        nomask_accum["acceleration_mae"].append(_to_float(acc_mae_nomask))
        nomask_accum["acceleration_mse"].append(_to_float(acc_mse_nomask))
        nomask_accum["MVE"].append(_to_float(mve_nomask))
        nomask_accum["LVE"].append(_to_float(lve_nomask))
        nomask_accum["LVD"].append(_to_float(lvd_nomask))
        nomask_accum["UFVE"].append(_to_float(ufve_nomask))
        nomask_accum["FDD"].append(_to_float(fdd_nomask))
        if args.sanity_checks:
            print(
                "[sanity][mask_compare] "
                f"sample={sample_id} "
                f"MVE_nomask={_to_float(mve_nomask):.8f} MVE_masked={_to_float(mve):.8f} "
                f"LVE_nomask={_to_float(lve_nomask):.8f} LVE_masked={_to_float(lve):.8f} "
                f"UFVE_nomask={_to_float(ufve_nomask):.8f} UFVE_masked={_to_float(ufve):.8f} "
                f"FDD_nomask={_to_float(fdd_nomask):.8f} FDD_masked={_to_float(fdd):.8f}"
            )
            print(
                "[sanity][len_meta] "
                f"sample={sample_id} t_pred={mask_meta['t_pred']} t_gt={mask_meta['t_gt']} "
                f"expected_t={mask_meta['expected_t']} audio_duration_sec={mask_meta['audio_duration_sec']}"
            )
        records.append(
            SampleRecord(
                sample_id=sample_id,
                subject=_extract_scalar(batch, "subject", ""),
                session=_extract_scalar(batch, "session", ""),
                language=_extract_scalar(batch, "language", ""),
                audio_path=_extract_scalar(batch, "audio_path", ""),
                annot_path=_extract_scalar(batch, "annot_path", ""),
                annot_type=annot_type,
                num_frames=int(pred.shape[1]),
                bs_mae_51d=_to_float(bs_mae),
                bs_mse_51d=_to_float(bs_mse),
                upper_face_mae_51d=_to_float(upper_mae),
                mouth_mae_51d=_to_float(mouth_mae),
                velocity_mae=_to_float(vel_mae),
                velocity_mse=_to_float(vel_mse),
                acceleration_mae=_to_float(acc_mae),
                acceleration_mse=_to_float(acc_mse),
                MVE=_to_float(mve),
                LVE=_to_float(lve),
                LVD=_to_float(lvd),
                UFVE=_to_float(ufve),
                FDD=_to_float(fdd),
            ))

    if not records:
        raise RuntimeError("No samples loaded from split json.")

    def mean_of(name: str) -> float:
        return float(np.mean([getattr(r, name) for r in records]))

    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "run_id": args.run_id,
        "protocol": args.protocol,
        "baseline": args.baseline,
        "phase": args.phase,
        "split_json": str(Path(args.split_json).resolve()),
        "weight_path": str(Path(args.weight_path).resolve()),
        "use_audio": bool(cfg.use_audio),
        "use_eeg": bool(cfg.use_eeg),
        "eeg_shift_ms": float(args.eeg_shift_ms),
        "num_samples": len(records),
        "bs_mae_51d": mean_of("bs_mae_51d"),
        "bs_mse_51d": mean_of("bs_mse_51d"),
        "upper_face_mae_51d": mean_of("upper_face_mae_51d"),
        "mouth_mae_51d": mean_of("mouth_mae_51d"),
        "velocity_mae": mean_of("velocity_mae"),
        "velocity_mse": mean_of("velocity_mse"),
        "acceleration_mae": mean_of("acceleration_mae"),
        "acceleration_mse": mean_of("acceleration_mse"),
        "MVE": mean_of("MVE"),
        "LVE": mean_of("LVE"),
        "LVD": mean_of("LVD"),
        "UFVE": mean_of("UFVE"),
        "FDD": mean_of("FDD"),
    }
    if args.sanity_checks:
        for key, values in nomask_accum.items():
            summary[f"{key}_nomask"] = float(np.mean(values)) if values else float("nan")
        if args.sanity_pred_eq_gt:
            for key in ("MVE", "LVE", "UFVE", "FDD"):
                value = float(summary[key])
                if not np.isfinite(value) or value > args.sanity_zero_tolerance:
                    raise AssertionError(
                        f"[sanity] {key} is not near zero in pred==gt mode: "
                        f"value={value}, tolerance={args.sanity_zero_tolerance}"
                    )

    if args.per_sample_csv:
        rows = [r.__dict__.copy() for r in records]
        _write_rows_csv(Path(args.per_sample_csv), rows)

    if args.visual_root and (args.render_topk > 0 or args.render_worstk > 0):
        visual_root = Path(args.visual_root) / args.protocol / args.baseline
        visual_root.mkdir(parents=True, exist_ok=True)
        render_device = torch.device(args.render_device)

        sorted_records = sorted(records, key=lambda r: r.bs_mae_51d)
        selected: List[Tuple[str, SampleRecord]] = []
        for r in sorted_records[: args.render_topk]:
            selected.append(("top", r))
        for r in sorted_records[-args.render_worstk:]:
            selected.append(("worst", r))

        selected_ids = {r.sample_id for _, r in selected}
        rows = []
        for batch in tqdm(loader, desc="render_visuals"):
            sample_id = _extract_scalar(batch, "sample_id", Path(_extract_scalar(batch, "annot_path")).stem)
            if sample_id.endswith("_bs"):
                sample_id = sample_id[:-3]
            if sample_id not in selected_ids:
                continue
            gt = batch["data"].to(device, non_blocking=True)
            template = batch["template"].to(device, non_blocking=True)
            identity = batch["id"].to(device, non_blocking=True)
            annot_type = _extract_scalar(batch, "annot_type", "inhouse_blendshape_weight")
            audio_path = _extract_scalar(batch, "audio_path", "")
            audio = None
            if cfg.use_audio:
                a = batch.get("audio")
                if isinstance(a, torch.Tensor):
                    audio = a.to(device, non_blocking=True)
            eeg = None
            if cfg.use_eeg:
                e = batch.get("eeg")
                if isinstance(e, torch.Tensor):
                    eeg = e.to(device, non_blocking=True)
                    if shift_steps != 0 and eeg.numel() > 0:
                        eeg = shift_eeg_tensor(eeg, shift_steps=shift_steps, channels=channels)
            fps = _extract_fps(batch)
            if args.sanity_pred_eq_gt:
                pred = gt.clone()
            else:
                if model is None:
                    raise RuntimeError("Model is not initialized.")
                with torch.no_grad():
                    pred, _, _ = model(
                        audio=audio,
                        eeg=eeg,
                        template=template,
                        face_motion=gt,
                        style_idx=identity,
                        annot_type=annot_type,
                        fps=fps,
                    )

            tag = "other"
            metric = 0.0
            for bucket, rec in selected:
                if rec.sample_id == sample_id:
                    tag = bucket
                    metric = rec.bs_mae_51d
                    break
            out_name = f"{tag}_{sample_id}_gt_vs_pred.mp4"
            out_path = visual_root / out_name
            try:
                _render_comparison(
                    loss_module=loss_module,
                    pred_bs=pred.detach(),
                    gt_bs=gt.detach(),
                    annot_type=annot_type,
                    audio_path=audio_path,
                    output_path=out_path,
                    device=render_device,
                    img_size=args.img_size,
                )
                rows.append(
                    {
                        "sample_id": sample_id,
                        "bucket": tag,
                        "bs_mae_51d": metric,
                        "video_path": str(out_path),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "sample_id": sample_id,
                        "bucket": tag,
                        "bs_mae_51d": metric,
                        "video_path": "",
                        "error": str(exc),
                    }
                )
        if rows:
            _write_rows_csv(visual_root / "visual_summary.csv", rows)

    if args.metrics_csv:
        _append_csv(Path(args.metrics_csv), summary)
    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BioTalk protocol split")
    parser.add_argument("--config", type=str, default="/home/zym/EATalker/config/eatalker.yaml")
    parser.add_argument("--weight-path", type=str, required=True)
    parser.add_argument("--split-json", type=str, required=True)
    parser.add_argument("--protocol", type=str, required=True)
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--phase", type=str, default="test")
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--use-audio", action="store_true", default=True)
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--use-eeg", action="store_true")
    parser.add_argument("--eeg-shift-ms", type=float, default=0.0)
    parser.add_argument("--eeg-sample-rate-hz", type=int, default=1000)
    parser.add_argument("--metrics-csv", type=str, default="")
    parser.add_argument("--per-sample-csv", type=str, default="")
    parser.add_argument("--out-json", type=str, default="")
    parser.add_argument("--visual-root", type=str, default="")
    parser.add_argument("--render-topk", type=int, default=0)
    parser.add_argument("--render-worstk", type=int, default=0)
    parser.add_argument("--render-device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--sanity-checks", action="store_true")
    parser.add_argument("--sanity-pred-eq-gt", action="store_true")
    parser.add_argument("--sanity-zero-tolerance", type=float, default=1e-6)
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    args = parser.parse_args()
    if args.baseline.lower() in ("audio+eeg", "b", "audio_eeg"):
        args.use_eeg = True
    if args.baseline.lower() in ("audio-only", "audio_only", "a"):
        args.use_eeg = False
    if args.no_audio:
        args.use_audio = False
    return args


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
