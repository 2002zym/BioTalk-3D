#!/usr/bin/env python3
"""检查 51 维 ARKit blendshape 的列顺序是否与 inhouse_arkit.npz 基向量匹配。

功能概览：
1. 根据你提供的原始 CSV 列顺序，过滤掉非 blendshape 列，打印出 51 个通道名称。
2. 加载 `inhouse_arkit.npz` 中的均值和 51 个基向量，通过 `BlendShapeLoss` 复用现有逻辑。
3. 选择一个代表性帧（默认 L2 范数最大），输出该帧每个通道的系数、最大几何位移、位移方向，
   方便肉眼判断通道语义是否对齐。
4. 将该帧重建出的顶点保存成单帧 `.verts.npy` 文件，必要时可以单独渲染。

使用示例：
  PYTHONPATH=/home/zym python3 EATalker/tools/check_blendshape_order.py \
    --bs-npy /home/zym/EATalker/EATalker_data_template/EATalker/test/sentence8_liu2_happy_bs.npy
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from EATalker.loss.loss import BlendShapeLoss


RAW_CSV_COLUMN_ORDER = [
    "Timecode",
    "BlendshapeCount",
    "EyeBlinkLeft",
    "EyeLookDownLeft",
    "EyeLookInLeft",
    "EyeLookOutLeft",
    "EyeLookUpLeft",
    "EyeSquintLeft",
    "EyeWideLeft",
    "EyeBlinkRight",
    "EyeLookDownRight",
    "EyeLookInRight",
    "EyeLookOutRight",
    "EyeLookUpRight",
    "EyeSquintRight",
    "EyeWideRight",
    "JawForward",
    "JawRight",
    "JawLeft",
    "JawOpen",
    "MouthClose",
    "MouthFunnel",
    "MouthPucker",
    "MouthRight",
    "MouthLeft",
    "MouthSmileLeft",
    "MouthSmileRight",
    "MouthFrownLeft",
    "MouthFrownRight",
    "MouthDimpleLeft",
    "MouthDimpleRight",
    "MouthStretchLeft",
    "MouthStretchRight",
    "MouthRollLower",
    "MouthRollUpper",
    "MouthShrugLower",
    "MouthShrugUpper",
    "MouthPressLeft",
    "MouthPressRight",
    "MouthLowerDownLeft",
    "MouthLowerDownRight",
    "MouthUpperUpLeft",
    "MouthUpperUpRight",
    "BrowDownLeft",
    "BrowDownRight",
    "BrowInnerUp",
    "BrowOuterUpLeft",
    "BrowOuterUpRight",
    "CheekPuff",
    "CheekSquintLeft",
    "CheekSquintRight",
    "NoseSneerLeft",
    "NoseSneerRight",
    "TongueOut",
    "HeadYaw",
    "HeadPitch",
    "HeadRoll",
    "LeftEyeYaw",
    "LeftEyePitch",
    "LeftEyeRoll",
    "RightEyeYaw",
    "RightEyePitch",
    "RightEyeRoll",
]


EXCLUDE_NON_BLENDSHAPE = {
    "Timecode",
    "BlendshapeCount",
    "TongueOut",
    "HeadYaw",
    "HeadPitch",
    "HeadRoll",
    "LeftEyeYaw",
    "LeftEyePitch",
    "LeftEyeRoll",
    "RightEyeYaw",
    "RightEyePitch",
    "RightEyeRoll",
}


def load_bs_names() -> Sequence[str]:
    """读取原始 CSV 列顺序并剔除非 blendshape 列，得到 51 个通道名称。"""
    filtered = [
        name for name in RAW_CSV_COLUMN_ORDER if name not in EXCLUDE_NON_BLENDSHAPE
    ]
    if len(filtered) != 51:
        raise ValueError(
            f"Expected 51 blendshape channels after filtering, got {len(filtered)}. "
            "Please verify RAW_CSV_COLUMN_ORDER / EXCLUDE_NON_BLENDSHAPE."
        )
    return filtered


def pick_frame(blendshape_seq: np.ndarray, strategy: str) -> int:
    """根据策略选取一个代表性帧（默认选 L2 最大的帧）。"""
    if strategy == "max_l2":
        l2_norm = np.linalg.norm(blendshape_seq, axis=1)
        return int(np.argmax(l2_norm))
    if strategy == "middle":
        return int(len(blendshape_seq) // 2)
    raise ValueError(f"Unknown frame selection strategy: {strategy}")


def summarise_channel_effects(
    coeff: np.ndarray,
    basis: np.ndarray,
    channel_names: Sequence[str],
    top_k: int = 10,
) -> None:
    """打印每个通道产生的几何位移统计，便于观察语义是否对齐。

    coeff: shape (51,)  当前帧的 51 维系数。
    basis: shape (51, V, 3)  inhouse_arkit.npz 的 51 个基向量。
    """
    displacements = coeff[:, None, None] * basis  # (51, V, 3)
    disp_norm = np.linalg.norm(displacements, axis=2)  # (51, V)
    channel_norm = disp_norm.max(axis=1)  # max per channel
    channel_stats = []
    for idx, name in enumerate(channel_names):
        channel_stats.append(
            {
                "idx": idx,
                "name": name,
                "coeff": coeff[idx],
                "max_mm": float(channel_norm[idx]),
                "max_vertex": int(np.argmax(disp_norm[idx])),
                "xyz_mean": displacements[idx].mean(axis=0),
            }
        )

    print("\n=== Channel summary (sorted by |coeff|) ===")
    for item in sorted(channel_stats, key=lambda x: abs(x["coeff"]), reverse=True)[:top_k]:
        print(
            f"[{item['idx']:02d}] {item['name']:<24} "
            f"coeff={item['coeff']: .4f}  max_disp={item['max_mm']:.6f} "
            f"max_vertex={item['max_vertex']} "
            f"mean_xyz=({item['xyz_mean'][0]: .5f}, {item['xyz_mean'][1]: .5f}, {item['xyz_mean'][2]: .5f})"
        )

    print("\n=== Channels with largest geometric displacement ===")
    for item in sorted(channel_stats, key=lambda x: x["max_mm"], reverse=True)[:top_k]:
        print(
            f"[{item['idx']:02d}] {item['name']:<24} "
            f"coeff={item['coeff']: .4f}  max_disp={item['max_mm']:.6f} "
            f"max_vertex={item['max_vertex']} "
            f"mean_xyz=({item['xyz_mean'][0]: .5f}, {item['xyz_mean'][1]: .5f}, {item['xyz_mean'][2]: .5f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose blendshape ordering.")
    parser.add_argument(
        "--bs-npy",
        required=True,
        type=Path,
        help="Path to the 51-dim blendshape npy sequence.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help="Optional frame index to inspect (default selects frame with largest L2 norm).",
    )
    parser.add_argument(
        "--selection",
        type=str,
        default="max_l2",
        choices=("max_l2", "middle"),
        help="Heuristic used when --frame-index is not provided.",
    )
    parser.add_argument(
        "--resources-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "resources" / "binary_resources",
        help="inhouse_arkit.npz 与其辅助 .npy 文件所在目录。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many channels to display in the summaries.",
    )
    args = parser.parse_args()

    if not args.bs_npy.exists():
        raise FileNotFoundError(f"Blendshape file not found: {args.bs_npy}")

    inhouse_npz = args.resources_root / "inhouse_arkit.npz"
    mouth_idx_path = args.resources_root / "05_inhouse_arkit_mouth_idx.npy"
    if not inhouse_npz.exists():
        raise FileNotFoundError(f"inhouse_arkit.npz is missing: {inhouse_npz}")
    if not mouth_idx_path.exists():
        raise FileNotFoundError(f"05_inhouse_arkit_mouth_idx.npy is missing: {mouth_idx_path}")

    channel_names = list(load_bs_names())
    print("根据原始 CSV（过滤非 blendshape 列）推断的 51 个通道顺序：")
    for idx, name in enumerate(channel_names):
        print(f"  {idx:02d}: {name}")

    blendshape_seq = np.load(args.bs_npy).astype(np.float32)
    if blendshape_seq.ndim != 2 or blendshape_seq.shape[1] != 51:
        raise ValueError(f"Expected array of shape (T, 51), got {blendshape_seq.shape}")

    if args.frame_index is None:
        frame_idx = pick_frame(blendshape_seq, args.selection)
    else:
        frame_idx = args.frame_index
        if not (0 <= frame_idx < len(blendshape_seq)):
            raise IndexError(f"Frame index {frame_idx} outside valid range 0..{len(blendshape_seq)-1}")

    coeff = blendshape_seq[frame_idx]
    print(f"当前检查帧：{frame_idx} / {len(blendshape_seq)-1}")
    print(f"该帧系数的 L2 范数：{np.linalg.norm(coeff):.6f}")

    # Initialise BlendShapeLoss to reuse the inhouse ARKit bases.
    device = torch.device("cpu")
    blendshape_loss = BlendShapeLoss(
        mouth_indices_path=str(mouth_idx_path),
        blendshape_path=str(inhouse_npz),
        bs_beta=0.0,
    ).to(device)

    # Retrieve flattened basis vectors and reshape to (51, V, 3)
    basis = blendshape_loss.blend_shape.detach().cpu().numpy()
    num_vertices = basis.shape[1] // 3
    basis = basis.reshape(51, num_vertices, 3)

    summarise_channel_effects(coeff, basis, channel_names, top_k=args.top_k)

    # Optionally compute reconstructed vertices so the user can render them manually.
    coeff_tensor = torch.tensor(coeff[None], dtype=torch.float32, device=device)
    verts = blendshape_loss.get_vertices(coeff_tensor).detach().cpu().numpy()[0]
    ref_mean = blendshape_loss.mean_shape.detach().cpu().numpy().reshape(num_vertices, 3)
    displacement = np.linalg.norm(verts - ref_mean, axis=1)
    print("\n该帧重建顶点的位移统计：")
    print(f"  min={displacement.min():.6f}  max={displacement.max():.6f}  mean={displacement.mean():.6f}")

    out_dir = Path("/home/zym/EATalker/tool")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.bs_npy.stem}.frame{frame_idx}.verts.npy"
    np.save(out_path, verts)
    print(f"已将该帧重建顶点保存为：{out_path}")
    print("若需进一步验证，可将此单帧 npy 喂给渲染器或自定义调试脚本。")


if __name__ == "__main__":
    # Ensure PyTorch does not look for CUDA when not available.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    main()
