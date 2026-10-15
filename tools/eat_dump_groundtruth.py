#!/usr/bin/env python3
"""Dump ground-truth blendshape vertices for EATalker datasets."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch

from EATalker.dataset.dataset_config import dataset_config
from EATalker.loss.loss import EATalkerLoss
from EATalker.utils.config import load_cfg_from_cfg_file, merge_cfg_from_list

# 导出真值用于可视化的脚本
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='导出 Ground Truth 顶点序列')
    parser.add_argument(
        '--config',
        type=str,
        default='EATalker/config/eatalker.yaml',
        help='配置文件路径',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=('train', 'test'),
        help='选择导出的数据划分',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='experiments/eat_groundtruth/groundtruth.npz',
        help='输出 npz 文件路径',
    )
    parser.add_argument(
        '--audio-root',
        type=str,
        default=None,
        help='用于生成键值的音频根目录（默认自动推断）',
    )
    parser.add_argument(
        '--filter-audio',
        type=str,
        default=None,
        help='仅导出包含该子串的音频文件（可快速对比单个样本）',
    )
    parser.add_argument(
        '--key-mode',
        type=str,
        default='auto',
        choices=('auto', 'name', 'relative'),
        help='生成 npz 键的方式：auto=尽量相对路径，不可行则回退到文件名',
    )
    parser.add_argument(
        'opts',
        nargs=argparse.REMAINDER,
        help='额外的 KEY VALUE 配置覆盖',
    )
    return parser.parse_args()


def load_cfg(config_path: str, overrides: Iterable[str]) -> Dict:
    cfg = load_cfg_from_cfg_file(config_path)
    if overrides:
        cfg = merge_cfg_from_list(cfg, overrides)
    cfg.dataset = cfg.dataset.split(',') if isinstance(cfg.dataset, str) else cfg.dataset
    cfg.demo_dataset = cfg.demo_dataset.split(',') if isinstance(cfg.demo_dataset, str) else cfg.demo_dataset
    if isinstance(cfg.duplicate_list, str):
        cfg.duplicate_list = [int(i) for i in cfg.duplicate_list.split(',') if i.strip()]
    return cfg


def resolve_json_name(cfg, split: str) -> str:
    if split == 'train':
        return getattr(cfg, 'train_json_key', 'train.json')
    if split == 'val':
        return 'val.json'
    if split == 'test':
        return 'test.json'
    raise ValueError(f'未知划分: {split}')


def normalise_path(relative_path: str) -> str:
    return relative_path.replace('\\', '/')


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config, args.opts)

    loss_module = EATalkerLoss(cfg)

    results: Dict[str, Dict[str, np.ndarray]] = {}
    os.makedirs(Path(args.output).parent, exist_ok=True)

    split_json = resolve_json_name(cfg, args.split)

    for dataset_name in cfg.dataset:
        dirname = dataset_config[dataset_name]['dirname']
        base_dir = Path(cfg.data_root) / dirname
        json_path = base_dir / split_json
        if not json_path.exists():
            print(f'⚠️ 跳过 {json_path}（不存在）')
            continue

        with json_path.open('r') as f:
            payload = json.load(f)

        data_entries = payload.get('data', [])

        if args.audio_root:
            audio_root = Path(args.audio_root)
        else:
            if args.split == 'train':
                audio_root = base_dir / 'train'
            elif args.split == 'val':
                audio_root = base_dir / 'val'
            else:
                audio_root = base_dir / 'test'

        for item in data_entries:
            audio_rel = normalise_path(item['audio_path'])
            annot_rel = normalise_path(item['annot_path'])
            audio_path = base_dir / audio_rel
            annot_path = base_dir / annot_rel

            if args.filter_audio and args.filter_audio not in audio_path.name:
                continue

            if not audio_path.exists() or not annot_path.exists():
                print(f'⚠️ 缺少文件 {audio_path} 或 {annot_path}，已跳过')
                continue

            blendshape = np.load(annot_path).astype(np.float32)
            blendshape_tensor = torch.tensor(
                blendshape[None], dtype=torch.float32)
            try:
                vertices = loss_module.get_vertices_sequence(
                    blendshape_tensor, item['annot_type']).squeeze(0).numpy()
            except NotImplementedError:
                print(f'⚠️ annot_type={item["annot_type"]} 暂不支持顶点转换，跳过 {audio_path}')
                continue

            if args.key_mode == 'name':
                key = audio_path.name
            elif args.key_mode == 'relative':
                key = os.path.relpath(audio_path, audio_root).replace('\\', '/')
            else:  # auto
                try:
                    key = os.path.relpath(audio_path, audio_root).replace('\\', '/')
                except ValueError:
                    key = audio_path.name

            dataset_key = item.get('dataset', dataset_name)
            if key not in results:
                results[key] = {}
            results[key][dataset_key] = vertices

    if not results:
        raise RuntimeError('未导出任何样本，请检查筛选条件。')

    np.savez(args.output, **results)
    print(f'✅ Ground Truth 保存至 {args.output}')


if __name__ == '__main__':
    main()
