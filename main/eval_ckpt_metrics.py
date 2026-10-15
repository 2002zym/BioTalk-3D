#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线重算 ckpt 的客观指标（不训练）。

用途：
- 训练代码更新了指标定义（尤其是 FDD）后，不想重训，只想对旧 ckpt 重新计算“新口径”的指标；
- 支持纯 audio / 双模态（EEG+audio），由配置/opts 决定 use_eeg。

输出指标（与训练日志/TensorBoard一致的命名）：
- rec_L2_loss（这里是 total_loss：rec_loss + 各类速度/加速度/under-std 正则）
- LVE, LVD（mouth_metric 返回）
- MVE（全脸顶点欧氏距离均值）
- UFVE（上半脸顶点欧氏距离均值）
- FDD（模板对齐后，上半脸位移能量 std 差的 ABS；与 main/compute_objective_metrics.py 的 ABS FDD 对齐）
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch

from EATalker.dataset.dataset import get_single_dataset
from EATalker.dataset.dataset_config import dataset_config
from EATalker.loss.loss import EATalkerLoss
from EATalker.models.eatalker import EATalker
from EATalker.utils.config import load_cfg_from_cfg_file, merge_cfg_from_list
from EATalker.utils.utils import get_audio_encoder_dim, load_ckpt


@dataclass
class MetricSums:
    total_loss: float = 0.0
    lve: float = 0.0
    lvd: float = 0.0
    mve: float = 0.0
    ufve: float = 0.0
    fdd: float = 0.0
    count: int = 0

    def add(self, *, total_loss, lve, lvd, mve, ufve, fdd):
        self.total_loss += float(total_loss)
        self.lve += float(lve)
        self.lvd += float(lvd)
        self.mve += float(mve)
        self.ufve += float(ufve)
        self.fdd += float(fdd)
        self.count += 1

    def mean(self) -> Dict[str, float]:
        denom = max(self.count, 1)
        return {
            "rec_L2_loss": self.total_loss / denom,
            "LVE": self.lve / denom,
            "LVD": self.lvd / denom,
            "MVE": self.mve / denom,
            "UFVE": self.ufve / denom,
            "FDD": self.fdd / denom,
            "num_samples": float(self.count),
        }


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return list(value)


def _resolve_split_json(cfg, split: str) -> str:
    dataset_name = cfg.dataset[0]
    dataset_dir = dataset_config[dataset_name]["dirname"]
    if split == "train":
        json_name = getattr(cfg, "train_json_key", "train.json")
    else:
        json_name = f"{split}.json"
    return os.path.join(cfg.data_root, dataset_dir, json_name)


def _compute_total_loss(
    *,
    cfg,
    loss_module: EATalkerLoss,
    out_motion: torch.Tensor,
    gt: torch.Tensor,
    annot_type: str,
) -> torch.Tensor:
    zero = torch.tensor(0.0, device=out_motion.device)
    rec_loss = loss_module(out_motion, gt, annot_type)

    speed_weight = float(getattr(cfg, "velocity_loss_weight", 0.0) or 0.0)
    acc_weight = float(getattr(cfg, "acceleration_loss_weight", 0.0) or 0.0)
    speed_loss = zero
    acc_loss = zero
    if out_motion.shape[1] > 1 and speed_weight > 0.0:
        pred_vel = out_motion[:, 1:] - out_motion[:, :-1]
        gt_vel = gt[:, 1:] - gt[:, :-1]
        speed_loss = torch.nn.functional.mse_loss(pred_vel, gt_vel)
        if out_motion.shape[1] > 2 and acc_weight > 0.0:
            pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
            gt_acc = gt_vel[:, 1:] - gt_vel[:, :-1]
            acc_loss = torch.nn.functional.mse_loss(pred_acc, gt_acc)

    mouth_speed_weight = float(
        getattr(cfg, "mouth_velocity_loss_weight", 0.0) or 0.0
    )
    mouth_acc_weight = float(
        getattr(cfg, "mouth_acceleration_loss_weight", 0.0) or 0.0
    )
    mouth_under_std_weight = float(
        getattr(cfg, "mouth_under_std_weight", 0.0) or 0.0
    )
    mouth_speed_loss = zero
    mouth_acc_loss = zero
    mouth_under_std_loss = zero

    mouth_idx = getattr(loss_module, "mouth_bs_idx", None)
    if isinstance(mouth_idx, torch.Tensor) and mouth_idx.numel() > 0:
        mouth_idx = mouth_idx.to(out_motion.device)
        pred_mouth = out_motion.index_select(2, mouth_idx)
        gt_mouth = gt.index_select(2, mouth_idx)

        if mouth_under_std_weight > 0.0:
            pred_std = pred_mouth.std(dim=1, unbiased=False)
            gt_std = gt_mouth.std(dim=1, unbiased=False)
            mouth_under_std_loss = torch.relu(gt_std - pred_std).mean()

        if out_motion.shape[1] > 1:
            pred_m_vel = pred_mouth[:, 1:] - pred_mouth[:, :-1]
            gt_m_vel = gt_mouth[:, 1:] - gt_mouth[:, :-1]
            if mouth_speed_weight > 0.0:
                mouth_speed_loss = torch.nn.functional.mse_loss(
                    pred_m_vel, gt_m_vel
                )
            if out_motion.shape[1] > 2 and mouth_acc_weight > 0.0:
                pred_m_acc = pred_m_vel[:, 1:] - pred_m_vel[:, :-1]
                gt_m_acc = gt_m_vel[:, 1:] - gt_m_vel[:, :-1]
                mouth_acc_loss = torch.nn.functional.mse_loss(
                    pred_m_acc, gt_m_acc
                )

    pca_weight = float(getattr(cfg, "pca_weight", 0.0) or 0.0)
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
    return total_loss


def evaluate_ckpt(cfg, weight_path: str, split: str) -> Dict[str, float]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    json_path = _resolve_split_json(cfg, split)
    loader = get_single_dataset(cfg, json_path=json_path)
    cfg.identity_num = loader.dataset.get_identity_num()

    cfg.audio_encoder_feature_dim = get_audio_encoder_dim(cfg.audio_encoder_repo)

    model = EATalker(cfg).to(device)
    style_indices = getattr(cfg, "style_emb_init_indices", None)
    load_ckpt(model, weight_path, style_emb_init_indices=style_indices)
    model.eval()

    loss_module = EATalkerLoss(cfg).to(device)
    sums = MetricSums()

    def compute_metrics(
        loss_module: EATalkerLoss,
        predicted: torch.Tensor,
        target: torch.Tensor,
        template: torch.Tensor,
        annot_type: str,
    ):
        """Compute LVE/LVD/MVE/UFVE/FDD with the same definition as training."""
        device = predicted.device
        zero = torch.tensor(0.0, device=device)
        mouth_metric_L2 = zero
        mouth_metric_L2_norm = zero
        mve = zero
        ufve = zero
        fdd = zero

        try:
            mouth_metric_L2, mouth_metric_L2_norm = loss_module.mouth_metric(
                predicted.detach(), target.detach(), annot_type
            )
        except NotImplementedError:
            pass

        if hasattr(loss_module, "get_vertices_sequence"):
            try:
                pred_vertices = loss_module.get_vertices_sequence(
                    predicted.detach(), annot_type
                )
                target_vertices = loss_module.get_vertices_sequence(
                    target.detach(), annot_type
                )
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

                    # FDD（official / CodeTalker-style, ABS）：与 main/compute_objective_metrics.py 对齐
                    if template is not None and pred_upper.size(1) > 1:
                        # template: [B, 1, D] or [B, D]
                        if template.dim() == 3 and template.size(1) == 1:
                            template_flat = template[:, 0]
                        elif template.dim() == 2:
                            template_flat = template
                        else:
                            template_flat = template.reshape(template.size(0), -1)

                        template_vertices = None
                        if hasattr(loss_module, "get_vertices"):
                            template_vertices = loss_module.get_vertices(
                                template_flat.detach(), annot_type
                            )

                        if (
                            isinstance(template_vertices, torch.Tensor)
                            and template_vertices.numel() > 0
                        ):
                            template_vertices = template_vertices.to(
                                pred_vertices.device
                            ).unsqueeze(1)  # [B, 1, V, 3]
                            motion_pred = pred_vertices - template_vertices
                            motion_gt = target_vertices - template_vertices
                            motion_pred = motion_pred.index_select(2, upper_idx)
                            motion_gt = motion_gt.index_select(2, upper_idx)
                            energy_pred = motion_pred.pow(2).sum(dim=-1)  # [B,T,V]
                            energy_gt = motion_gt.pow(2).sum(dim=-1)
                            pred_motion_std = energy_pred.std(
                                dim=1, unbiased=False
                            ).mean(dim=-1)
                            gt_motion_std = energy_gt.std(
                                dim=1, unbiased=False
                            ).mean(dim=-1)
                            fdd = (gt_motion_std - pred_motion_std).abs().mean()
            except NotImplementedError:
                pass

        return mouth_metric_L2, mouth_metric_L2_norm, mve, ufve, fdd

    with torch.no_grad():
        for batch in loader:
            gt = batch["data"].to(device, non_blocking=True)
            template = batch["template"].to(device, non_blocking=True)
            identity = batch["id"].to(device, non_blocking=True)
            annot_type = batch["annot_type"][0]

            fps_tensor = batch.get("fps", None)
            fps_value: Optional[float] = None
            if fps_tensor is not None:
                if isinstance(fps_tensor, torch.Tensor):
                    fps_value = float(fps_tensor[0].item())
                else:
                    fps_value = float(fps_tensor[0])

            audio = None
            if getattr(cfg, "use_audio", True):
                audio_tensor = batch.get("audio", None)
                if isinstance(audio_tensor, torch.Tensor):
                    audio = audio_tensor.to(device, non_blocking=True)

            eeg = None
            if getattr(cfg, "use_eeg", True):
                eeg_tensor = batch.get("eeg", None)
                if isinstance(eeg_tensor, torch.Tensor):
                    eeg = eeg_tensor.to(device, non_blocking=True)

            out_motion, _out_pca, _gt_pca = model(
                audio=audio,
                eeg=eeg,
                template=template,
                face_motion=gt,
                style_idx=identity,
                annot_type=annot_type,
                fps=fps_value,
            )

            total_loss = _compute_total_loss(
                cfg=cfg,
                loss_module=loss_module,
                out_motion=out_motion,
                gt=gt,
                annot_type=annot_type,
            )

            lve, lvd, mve, ufve, fdd = compute_metrics(
                loss_module, out_motion, gt, template, annot_type
            )

            sums.add(
                total_loss=total_loss.item(),
                lve=lve.item(),
                lvd=lvd.item(),
                mve=mve.item(),
                ufve=ufve.item(),
                fdd=fdd.item(),
            )

    report = sums.mean()
    report["weight_path"] = weight_path
    report["split"] = split
    report["use_eeg"] = bool(getattr(cfg, "use_eeg", True))
    report["use_audio"] = bool(getattr(cfg, "use_audio", True))
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Offline evaluate a ckpt with current metric definitions."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/home/zym/EATalker/config/eatalker.yaml",
        help="Config yaml path",
    )
    parser.add_argument(
        "--weight_path", type=str, required=True, help="Checkpoint path"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("train", "val", "test"),
        help="Which split json to evaluate",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to save json report",
    )
    parser.add_argument(
        "--opts",
        nargs=argparse.REMAINDER,
        help="Optional KEY VALUE pairs to override config (same style as training)",
    )
    args = parser.parse_args()

    cfg = load_cfg_from_cfg_file(args.config)
    if args.opts:
        cfg = merge_cfg_from_list(cfg, args.opts)

    cfg.dataset = _as_list(getattr(cfg, "dataset", "EATalker"))
    cfg.demo_dataset = _as_list(getattr(cfg, "demo_dataset", "EATalker"))
    if isinstance(getattr(cfg, "duplicate_list", None), str):
        cfg.duplicate_list = [int(x) for x in cfg.duplicate_list.split(",") if x]

    report = evaluate_ckpt(cfg, args.weight_path, args.split)

    payload = json.dumps(report, indent=2, ensure_ascii=False)
    print(payload)
    if args.out:
        out_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(payload + "\n")
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
