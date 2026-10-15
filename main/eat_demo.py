import argparse
import os
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import soxr
import torch
from transformers import Wav2Vec2FeatureExtractor

from EATalker.dataset.dataset_config import dataset_config
from EATalker.loss.loss import EATalkerLoss
from EATalker.models.eatalker import EATalker
from EATalker.utils.config import load_cfg_from_cfg_file, merge_cfg_from_list
from EATalker.utils.utils import get_audio_encoder_dim, get_template_verts


def get_all_audios(audio_root: str) -> List[str]:
    wav_names = []
    for root, _, files in os.walk(audio_root):
        for fname in files:
            if fname.endswith(('.wav', '.mp3')):
                rel_path = os.path.relpath(os.path.join(root, fname),
                                           audio_root)
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
    """Limit waveform length to match训练时的≤20秒策略。"""
    max_len = int(max_seconds * sample_rate)
    if len(audio) <= max_len:
        return audio
    return audio[:max_len]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='EATalker 双模态推理脚本')
    parser.add_argument(
        '--config',
        type=str,
        default='EATalker/config/eatalker.yaml',
        help='配置文件路径')
    parser.add_argument(
        '--weight_path',
        type=str,
        required=True,
        help='待加载的模型检查点 (.pt)')
    parser.add_argument(
        '--condition_id',
        type=str,
        default='common',
        help='风格 ID 选择方式（同训练脚本）')
    parser.add_argument(
        '--use_eeg',
        action='store_true',
        help='显式启用 EEG → 音频双模态推理')
    parser.add_argument(
        '--pure_eeg',
        action='store_true',
        help='仅使用 EEG 进行推理（忽略音频分支）')
    parser.add_argument(
        '--opts',
        nargs=argparse.REMAINDER,
        help='额外的 KEY VALUE 配置覆盖，例如 TEST.test_wav_dir=/path')
    return parser.parse_args()


def find_eeg_path(wav_rel: str, wav_path: str, args) -> Optional[str]:
    base, _ = os.path.splitext(wav_path)
    primary = base + '_eeg.npy'
    if os.path.exists(primary):
        return primary
    eeg_root = getattr(args, 'test_eeg_dir', None)
    if eeg_root:
        rel_base, _ = os.path.splitext(wav_rel)
        candidate = os.path.join(eeg_root, rel_base + '_eeg.npy')
        if os.path.exists(candidate):
            return candidate
    return None


def load_eeg_tensor(eeg_path: str,
                    device: torch.device,
                    eeg_channels: int) -> torch.Tensor:
    eeg_np = np.load(eeg_path).astype(np.float32)
    if eeg_np.ndim == 1:
        eeg_np = eeg_np[None, :]
    if eeg_np.ndim != 2:
        raise ValueError(f'Unexpected EEG shape {eeg_np.shape} for {eeg_path}')
    if eeg_np.shape[1] == eeg_channels:
        eeg_np = eeg_np  # [T, C]
    elif eeg_np.shape[0] == eeg_channels:
        eeg_np = eeg_np.T  # [T, C]
    else:
        raise ValueError(
            f'EEG shape {eeg_np.shape} incompatible with eeg_channels={eeg_channels}'
        )
    eeg_tensor = torch.tensor(eeg_np[None], dtype=torch.float32, device=device)
    return eeg_tensor


def main():
    cli_args = parse_args()

    cfg = load_cfg_from_cfg_file(cli_args.config)
    if cli_args.opts:
        cfg = merge_cfg_from_list(cfg, cli_args.opts)

    cfg.condition_id = cli_args.condition_id
    cfg.dataset = cfg.dataset.split(',') if isinstance(cfg.dataset, str) else cfg.dataset
    cfg.demo_dataset = cfg.demo_dataset.split(',') if isinstance(cfg.demo_dataset, str) else cfg.demo_dataset
    if isinstance(cfg.duplicate_list, str):
        cfg.duplicate_list = [
            int(i) for i in cfg.duplicate_list.split(',') if i.strip()
        ]

    if cli_args.use_eeg:
        cfg.use_eeg = True
    cfg.pure_eeg = cli_args.pure_eeg

    args = cfg
    weight_path = cli_args.weight_path

    training_dataset_name_list = args.dataset
    demo_dataset_name_list = args.demo_dataset

    checkpoint = torch.load(weight_path, map_location='cpu')
    args.identity_num = len(checkpoint['decoder.learnable_style_emb.weight'])
    # 兼容 UniTalker：将 audio_feature_map 权重映射到 multimodal_feature_map
    renamed = {}
    for k, v in checkpoint.items():
        if k.startswith('decoder.audio_feature_map.'):
            renamed[k.replace('audio_feature_map', 'multimodal_feature_map')] = v
    checkpoint.update(renamed)

    args.audio_encoder_feature_dim = get_audio_encoder_dim(
        args.audio_encoder_repo)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EATalker(args)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    model.to(device)

    wav_names = get_all_audios(args.test_wav_dir)
    out_npz_path = args.test_out_path
    os.makedirs(os.path.dirname(out_npz_path), exist_ok=True)

    out_dict: Dict[str, Dict[str, np.ndarray]] = {}
    loss_module = EATalkerLoss(args).to(device)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        args.audio_encoder_repo,)
    template_idx_map = getattr(args, 'template_idx_map', {}) or {}
    condition_idx_map = getattr(args, 'condition_idx_map', {}) or {}

    with torch.no_grad():
        for wav_rel in wav_names:
            name_lower = wav_rel.lower()
            # 根据文件名推断身份：包含 pj -> 1，包含 liu -> 0，其他默认 0
            inferred_id = 1 if 'pj' in name_lower else 0
            out_dict[wav_rel] = {}
            wav_path = os.path.join(args.test_wav_dir, wav_rel)
            audio_wave, sr = sf.read(wav_path, always_2d=False)
            if audio_wave.ndim > 1:
                audio_wave = audio_wave.mean(axis=1)
            audio_wave = audio_wave.astype(np.float32)
            if sr != 16000:
                audio_wave = soxr.resample(audio_wave, sr, 16000)
                sr = 16000
            audio_wave = trim_audio_duration(audio_wave, sr)

            pure_eeg_mode = bool(getattr(args, 'pure_eeg', False))
            use_eeg_flag = bool(getattr(args, 'use_eeg', False))
            inference_mode = 'eeg' if pure_eeg_mode else (
                'dual' if use_eeg_flag else 'audio')

            eeg_tensor: Optional[torch.Tensor] = None
            if inference_mode in ('dual', 'eeg'):
                eeg_path = find_eeg_path(wav_rel, wav_path, args)
                if eeg_path is None:
                    if inference_mode == 'dual':
                        print(
                            f'⚠️ 未找到 EEG: {wav_rel}，回退到纯音频推理')
                        inference_mode = 'audio'
                    else:
                        raise FileNotFoundError(
                            f'纯 EEG 推理需要 EEG 文件，但未找到 {wav_rel}')
                else:
                    eeg_tensor = load_eeg_tensor(eeg_path, device,
                                                 getattr(args, 'eeg_channels',
                                                         0))

            if inference_mode == 'dual':
                processed_audio = np.squeeze(
                    processor(audio_wave, sampling_rate=sr).input_values).astype(
                        np.float32)
                audio_splits = [processed_audio]
            elif inference_mode == 'audio':
                audio_splits = split_long_audio(audio_wave, processor)
            else:
                audio_splits = [None]
            for dataset_name in demo_dataset_name_list:
                scale = dataset_config[dataset_name]['scale']
                annot_type = dataset_config[dataset_name]['annot_type']
                subjects = dataset_config[dataset_name]['subjects']
                # EATalker 单一 dataset 内包含多身份（liu/pj），默认按文件名自动推断；
                # 若用户想固定身份，请通过 --condition_id 显式指定。
                if dataset_name == 'EATalker':
                    template_idx = inferred_id
                else:
                    template_idx = int(template_idx_map.get(dataset_name,
                                                            inferred_id))

                template_np = get_template_verts(args.data_root, dataset_name,
                                                 template_idx)
                template = torch.tensor(template_np.reshape(1, -1),
                                        dtype=torch.float32,
                                        device=device)
                template = scale * template

                if args.condition_id == 'common':
                    if dataset_name == 'EATalker':
                        condition_idx = inferred_id % args.identity_num
                    else:
                        default_condition = int(
                            condition_idx_map.get(dataset_name, inferred_id))
                        condition_idx = default_condition % args.identity_num
                else:
                    try:
                        condition_idx = int(args.condition_id)
                    except ValueError:
                        condition_idx = template_idx % args.identity_num
                condition_id = torch.tensor([condition_idx],
                                            dtype=torch.long,
                                            device=device)

                fps = 25 if annot_type == 'BIWI_23370_vertices' else 30

                chunk_outputs: List[np.ndarray] = []
                for chunk in audio_splits:
                    if chunk is not None:
                        audio_tensor = torch.tensor(chunk[None],
                                                    dtype=torch.float32,
                                                    device=device)
                    else:
                        audio_tensor = None
                    if inference_mode == 'dual':
                        out, _, _ = model(audio=audio_tensor,
                                          eeg=eeg_tensor,
                                          template=template,
                                          face_motion=None,
                                          style_idx=condition_id,
                                          annot_type=annot_type,
                                          fps=fps,
                                          use_dual_modal=True)
                    elif inference_mode == 'audio':
                        out, _, _ = model(audio=audio_tensor,
                                          eeg=None,
                                          template=template,
                                          face_motion=None,
                                          style_idx=condition_id,
                                          annot_type=annot_type,
                                          fps=fps)
                    else:
                        out, _, _ = model(audio=None,
                                          eeg=eeg_tensor,
                                          template=template,
                                          face_motion=None,
                                          style_idx=condition_id,
                                          annot_type=annot_type,
                                          fps=fps,
                                          use_dual_modal=False)
                    chunk_outputs.append(out.cpu().numpy().squeeze(0))

                blendshape = merge_out_list(chunk_outputs, fps)
                bs_tensor = torch.tensor(blendshape[None],
                                         dtype=torch.float32,
                                         device=device)
                vertices = loss_module.get_vertices_sequence(bs_tensor,
                                                             annot_type)
                out_dict[wav_rel][dataset_name] = vertices.squeeze(0).cpu().numpy()

    print(f'save results to {out_npz_path}')
    np.savez(out_npz_path, **out_dict)


if __name__ == '__main__':
    main()
