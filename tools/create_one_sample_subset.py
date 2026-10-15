#!/usr/bin/env python3
"""
Create a minimal dataset directory that only contains a single sample
from the main EATalker dataset. Useful for one-sample overfitting/debugging.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _find_sample(entries: List[Dict[str, Any]], sample_key: str) -> Dict[str, Any]:
    for entry in entries:
        audio_stem = Path(entry['audio_path']).stem
        annot_stem = Path(entry['annot_path']).stem
        if audio_stem == sample_key or annot_stem == sample_key:
            return entry
    raise ValueError(f'未在 train.json 中找到名为 {sample_key} 的样本')


def _copy_relative(base_dir: Path, rel_path: str, out_dir: Path) -> None:
    src = base_dir / rel_path
    if not src.exists():
        raise FileNotFoundError(f'缺少文件: {src}')
    dst = out_dir / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _build_json(info: Dict[str, Any], entry: Dict[str, Any], split: str) -> Dict[str, Any]:
    info_block = dict(info)
    info_block['split'] = split
    return {'info': info_block, 'data': [entry]}


def main() -> None:
    parser = argparse.ArgumentParser(description='生成单条样本数据集，便于过拟合实验')
    parser.add_argument('--base-dir',
                        default='/home/zym/EATalker/EATalker_data_template/EATalker',
                        help='原始 EATalker 数据集根目录')
    parser.add_argument('--sample',
                        required=True,
                        help='样本名（例如 sentence6_liu1_happy，不含路径）')
    parser.add_argument('--output-dir',
                        default='/home/zym/EATalker/EATalker_data_template/one_sample',
                        help='输出目录')
    parser.add_argument('--copy-to-test',
                        action='store_true',
                        help='是否在 test.json 中也包含该样本（便于验证）')
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    sample_key = Path(args.sample).stem

    train_json = base_dir / 'train.json'
    if not train_json.exists():
        raise FileNotFoundError(f'找不到 {train_json}')

    train_data = _load_json(train_json)
    entry = _find_sample(train_data['data'], sample_key)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # 复制相关文件
    for key in ('annot_path', 'audio_path', 'eeg_path'):
        rel = entry.get(key)
        if rel:
            _copy_relative(base_dir, rel, out_dir)

    # 拷贝模板
    template_src = base_dir / 'id_template.npy'
    if template_src.exists():
        shutil.copy2(template_src, out_dir / 'id_template.npy')

    # 生成 train.json
    train_out = _build_json(train_data['info'], entry, 'train')
    with (out_dir / 'train.json').open('w', encoding='utf-8') as f:
        json.dump(train_out, f, ensure_ascii=False, indent=2)

    if args.copy_to_test:
        test_out = _build_json(train_data['info'], entry, 'test')
        with (out_dir / 'test.json').open('w', encoding='utf-8') as f:
            json.dump(test_out, f, ensure_ascii=False, indent=2)
    else:
        # 写一个空的 test.json
        test_out = _build_json(train_data['info'], {}, 'test')
        test_out['data'] = []
        with (out_dir / 'test.json').open('w', encoding='utf-8') as f:
            json.dump(test_out, f, ensure_ascii=False, indent=2)

    print(f'✅ 单样本数据已生成: {out_dir}')


if __name__ == '__main__':
    main()
