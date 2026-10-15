#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline evaluate a checkpoint on a dataset split with *current* metric definitions.

Why this exists:
  - Training logs/tensorboard may use older metric definitions (e.g. FDD).
  - This script recomputes metrics without retraining.

Outputs (mixed_dataset averages):
  - rec_L2_loss: total_loss (same composition as training/validation)
  - LVE, LVD, MVE, UFVE, FDD: same as current training code
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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

from EATalker.dataset.dataset import get_dataloaders  # noqa: E402
from EATalker.loss.loss import EATalkerLoss  # noqa: E402
from EATalker.models.eatalker import EATalker  # noqa: E402
from EATalker.utils import config as cfgmod  # noqa: E402
from EATalker.utils.utils import get_audio_encoder_dim, load_ckpt  # noqa: E402


def _cfg_postprocess(cfg: Any) -> Any:
    cfg.dataset = cfg.dataset.split(",") if isinstance(cfg.dataset, str) else list(cfg.dataset)
    cfg.demo_dataset = cfg.demo_dataset.split(",") if isinstance(cfg.demo_dataset, str) else list(cfg.demo_dataset)
    if isinstance(getattr(cfg, "duplicate_list", None), str):
        cfg.duplicate_list = [int(x) for x in cfg.duplicate_list.split(",") if x]
    return cfg


def _to_device(x: Any, device: torch.device) -> Any:
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=False)
    return x


def _extract_fps(batch: Dict[str, Any]) -> Optional[float]:
    fps_tensor = batch.get("fps", None)
    if fps_tensor is None:
        return None
    if isinstance(fps_tensor, torch.Tensor):
        return float(fps_tensor[0].item())
    return float(fps_tensor[0])


def _compute_total_loss_and_metrics(
    model: torch.nn.Module,
    loss_module: EATalkerLoss,
    batch: Dict[str, Any],
    device: torch.device,
    args: Any,
) -> Dict[str, float]:
    use_audio = bool(getattr(args, "use_audio", True))
    use_eeg = bool(getattr(args, "use_eeg", True))

    data = _to_device(batch["data"], device)
    template = _to_device(batch["template"], device)
    identity = _to_device(batch["id"], device)

    audio = None
    if use_audio:
        audio_tensor = batch.get("audio", None)
        if isinstance(audio_tensor, torch.Tensor):
            audio = _to_device(audio_tensor, device)

    eeg = None
    if use_eeg:
        eeg_tensor = batch.get("eeg", None)
        if isinstance(eeg_tensor, torch.Tensor):
            eeg = _to_device(eeg_tensor, device)

    annot_type_list = batch.get("annot_type")
    if not isinstance(annot_type_list, (list, tuple)):
        annot_type_list = [annot_type_list]
    annot_type = annot_type_list[0]

    fps_value = _extract_fps(batch)

    out_motion, out_pca, gt_pca = model(
        audio=audio,
        eeg=eeg,
        template=template,
        face_motion=data,
        style_idx=identity,
        annot_type=annot_type,
        fps=fps_value,
    )

    rec_loss = loss_module(out_motion, data, annot_type)
    mouth_metric_L2, mouth_metric_L2_norm = loss_module.mouth_metric(out_motion, data, annot_type)

    zero = torch.tensor(0.0, device=device)
    mve = zero
    ufve = zero
    fdd = zero

    if hasattr(loss_module, "get_vertices_sequence"):
        try:
            pred_vertices = loss_module.get_vertices_sequence(out_motion, annot_type)
            target_vertices = loss_module.get_vertices_sequence(data, annot_type)
            diff = pred_vertices - target_vertices
            mve = diff.norm(dim=-1).mean()

            upper_idx = None
            if hasattr(loss_module, "get_upper_face_indices"):
                upper_idx = loss_module.get_upper_face_indices(annot_type)
            if isinstance(upper_idx, torch.Tensor) and upper_idx.numel() > 0:
                upper_idx = upper_idx.to(pred_vertices.device)
                pred_upper = pred_vertices.index_select(2, upper_idx)
                target_upper = target_vertices.index_select(2, upper_idx)
                ufve = (pred_upper - target_upper).norm(dim=-1).mean()

                if template is not None and pred_upper.size(1) > 1 and hasattr(loss_module, "get_vertices"):
                    template_flat: torch.Tensor
                    if template.dim() == 3 and template.size(1) == 1:
                        template_flat = template[:, 0]
                    elif template.dim() == 2:
                        template_flat = template
                    else:
                        template_flat = template.reshape(template.size(0), -1)
                    template_vertices = loss_module.get_vertices(template_flat.detach(), annot_type)
                    if isinstance(template_vertices, torch.Tensor) and template_vertices.numel() > 0:
                        template_vertices = template_vertices.to(pred_vertices.device).unsqueeze(1)
                        motion_pred = pred_vertices - template_vertices
                        motion_gt = target_vertices - template_vertices
                        motion_pred = motion_pred.index_select(2, upper_idx)
                        motion_gt = motion_gt.index_select(2, upper_idx)
                        energy_pred = motion_pred.pow(2).sum(dim=-1)  # [B, T, V]
                        energy_gt = motion_gt.pow(2).sum(dim=-1)
                        pred_motion_std = energy_pred.std(dim=1, unbiased=False).mean(dim=-1)  # [B]
                        gt_motion_std = energy_gt.std(dim=1, unbiased=False).mean(dim=-1)  # [B]
                        fdd = (gt_motion_std - pred_motion_std).abs().mean()
        except NotImplementedError:
            pass

    speed_weight = float(getattr(args, "velocity_loss_weight", 0.0) or 0.0)
    acc_weight = float(getattr(args, "acceleration_loss_weight", 0.0) or 0.0)

    speed_loss = zero
    acc_loss = zero
    if out_motion.shape[1] > 1:
        pred_vel = out_motion[:, 1:] - out_motion[:, :-1]
        gt_vel = data[:, 1:] - data[:, :-1]
        speed_loss = torch.nn.functional.mse_loss(pred_vel, gt_vel)
        if out_motion.shape[1] > 2:
            pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
            gt_acc = gt_vel[:, 1:] - gt_vel[:, :-1]
            acc_loss = torch.nn.functional.mse_loss(pred_acc, gt_acc)

    mouth_speed_weight = float(getattr(args, "mouth_velocity_loss_weight", 0.0) or 0.0)
    mouth_acc_weight = float(getattr(args, "mouth_acceleration_loss_weight", 0.0) or 0.0)
    mouth_under_std_weight = float(getattr(args, "mouth_under_std_weight", 0.0) or 0.0)

    mouth_speed_loss = zero
    mouth_acc_loss = zero
    mouth_under_std_loss = zero
    mouth_idx = getattr(loss_module, "mouth_bs_idx", None)
    if isinstance(mouth_idx, torch.Tensor) and mouth_idx.numel() > 0:
        mouth_idx = mouth_idx.to(device)
        pred_mouth = out_motion.index_select(2, mouth_idx)
        gt_mouth = data.index_select(2, mouth_idx)
        if mouth_under_std_weight > 0.0:
            pred_std = pred_mouth.std(dim=1, unbiased=False)
            gt_std = gt_mouth.std(dim=1, unbiased=False)
            mouth_under_std_loss = torch.relu(gt_std - pred_std).mean()
        if out_motion.shape[1] > 1:
            pred_m_vel = pred_mouth[:, 1:] - pred_mouth[:, :-1]
            gt_m_vel = gt_mouth[:, 1:] - gt_mouth[:, :-1]
            if mouth_speed_weight > 0.0:
                mouth_speed_loss = torch.nn.functional.mse_loss(pred_m_vel, gt_m_vel)
            if out_motion.shape[1] > 2 and mouth_acc_weight > 0.0:
                pred_m_acc = pred_m_vel[:, 1:] - pred_m_vel[:, :-1]
                gt_m_acc = gt_m_vel[:, 1:] - gt_m_vel[:, :-1]
                mouth_acc_loss = torch.nn.functional.mse_loss(pred_m_acc, gt_m_acc)

    pca_weight = float(getattr(args, "pca_weight", 0.0) or 0.0)
    if out_pca is not None:
        pca_loss = loss_module.pca_loss(out_pca, gt_pca)
    else:
        pca_loss = zero

    total_loss = (
        rec_loss
        + pca_weight * pca_loss
        + speed_weight * speed_loss
        + acc_weight * acc_loss
        + mouth_speed_weight * mouth_speed_loss
        + mouth_acc_weight * mouth_acc_loss
        + mouth_under_std_weight * mouth_under_std_loss
    )

    return {
        "rec_L2_loss": float(total_loss.detach().cpu().item()),
        "LVE": float(mouth_metric_L2.detach().cpu().item()),
        "LVD": float(mouth_metric_L2_norm.detach().cpu().item()),
        "MVE": float(mve.detach().cpu().item()),
        "UFVE": float(ufve.detach().cpu().item()),
        "FDD": float(fdd.detach().cpu().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline evaluate a ckpt with current metric definitions.")
    parser.add_argument("--config", type=str, default="/home/zym/EATalker/config/eatalker.yaml", help="Config yaml path")
    parser.add_argument("--weight_path", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Which split json to evaluate")
    parser.add_argument("--out", type=str, default=None, help="Optional path to save json report")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None, help="Optional KEY VALUE pairs to override config")
    args_cli = parser.parse_args()

    cfg = cfgmod.load_cfg_from_cfg_file(args_cli.config)
    if args_cli.opts:
        cfg = cfgmod.merge_cfg_from_list(cfg, args_cli.opts)
    cfg = _cfg_postprocess(cfg)
    cfg.audio_encoder_feature_dim = get_audio_encoder_dim(cfg.audio_encoder_repo)

    cfg.distributed = False
    cfg.local_rank = 0
    cfg.world_size = 1
    cfg.rank = 0

    cfg.do_train = False
    cfg.do_validate = False
    cfg.do_test = True

    _, _, test_loader = get_dataloaders(cfg)
    if args_cli.split != "test":
        raise NotImplementedError("Currently only supports split=test (EATalker has no val by default).")

    cfg.identity_num = test_loader.dataset.get_identity_num()
    if getattr(cfg, "random_id_prob", 0.0) > 0.0:
        cfg.identity_num = cfg.identity_num + 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = EATalker(cfg).to(device)
    loss_module = EATalkerLoss(cfg).to(device)

    load_ckpt(
        model,
        args_cli.weight_path,
        re_init_decoder_and_head=bool(getattr(cfg, "re_init_decoder_and_head", False)),
        style_emb_init_indices=getattr(cfg, "style_emb_init_indices", None),
    )

    model.eval()
    loss_module.eval()

    sums = {k: 0.0 for k in ("rec_L2_loss", "LVE", "LVD", "MVE", "UFVE", "FDD")}
    n = 0
    for batch in tqdm(test_loader, desc=f"eval({Path(args_cli.weight_path).name})"):
        metrics = _compute_total_loss_and_metrics(model, loss_module, batch, device, cfg)
        for k in sums:
            sums[k] += float(metrics[k])
        n += 1

    report = {
        "rec_L2_loss": sums["rec_L2_loss"] / max(1, n),
        "LVE": sums["LVE"] / max(1, n),
        "LVD": sums["LVD"] / max(1, n),
        "MVE": sums["MVE"] / max(1, n),
        "UFVE": sums["UFVE"] / max(1, n),
        "FDD": sums["FDD"] / max(1, n),
        "num_samples": float(n),
        "weight_path": str(args_cli.weight_path),
        "split": args_cli.split,
        "use_eeg": bool(getattr(cfg, "use_eeg", True)),
        "use_audio": bool(getattr(cfg, "use_audio", True)),
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args_cli.out:
        out_path = Path(args_cli.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

