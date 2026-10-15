import librosa
import numpy as np
import torch
from typing import Any
import os.path as osp

from .dataset_config import dataset_config


class DataItem:

    def __init__(self,
                 annot_path: str,
                 audio_path: str,
                 identity_idx: int,
                 annot_type: str = '',
                 dataset_name: str = '',
                 id_template: np.ndarray = None,
                 fps: float = 30,
                 processor=None,
                 eeg_path: str = None,
                 use_audio: bool = True,
                 use_eeg: bool = True,
                 mouth_gain: float = 1.0,
                 sample_id: str = '',
                 subject: str = '',
                 session: str = '',
                 sheet: str = '',
                 language: str = '',
                 timestamps=None,
                 blendshape_path: str = '',
                 metadata: dict = None,
                 cache_entry: dict = None) -> None:
        self.annot_path = annot_path
        self.audio_path = audio_path
        self.eeg_path = eeg_path
        self.identity_idx = identity_idx
        self.sample_id = sample_id or ''
        self.subject = subject or ''
        self.session = session or ''
        self.sheet = sheet or ''
        self.language = language or ''
        self.timestamps = timestamps if isinstance(timestamps, dict) else {}
        self.metadata = metadata if isinstance(metadata, dict) else {}
        self.cache_entry = cache_entry if isinstance(cache_entry, dict) else {}
        self.blendshape_path = blendshape_path if blendshape_path else annot_path
        self.fps = fps
        annot_array = np.load(annot_path)
        if self.fps == 60:
            annot_array = annot_array[::2]
            self.fps = 30
        elif self.fps == 100:
            annot_array = annot_array[::4]
            self.fps = 25
        elif self.fps != 30 and self.fps != 25:
            raise ValueError('wrong fps')
        scale = dataset_config[dataset_name]['scale']
        annot_array = self.scale_and_offset(annot_array, scale, 0.0)
        # Optional mouth gain to counteract small-amplitude BS (training-time only).
        if mouth_gain and mouth_gain != 1.0 and annot_array.shape[1] >= 51:
            d6_order = [
                'EyeBlinkLeft', 'EyeBlinkRight', 'EyeSquintLeft', 'EyeSquintRight',
                'EyeLookDownLeft', 'EyeLookDownRight', 'EyeLookInLeft', 'EyeLookInRight',
                'EyeWideLeft', 'EyeWideRight', 'EyeLookOutLeft', 'EyeLookOutRight',
                'EyeLookUpLeft', 'EyeLookUpRight', 'BrowDownLeft', 'BrowDownRight',
                'BrowInnerUp', 'BrowOuterUpLeft', 'BrowOuterUpRight', 'JawOpen',
                'MouthClose', 'JawLeft', 'JawRight', 'JawForward', 'MouthUpperUpLeft',
                'MouthUpperUpRight', 'MouthLowerDownLeft', 'MouthLowerDownRight',
                'MouthRollUpper', 'MouthRollLower', 'MouthSmileLeft', 'MouthSmileRight',
                'MouthDimpleLeft', 'MouthDimpleRight', 'MouthStretchLeft', 'MouthStretchRight',
                'MouthFrownLeft', 'MouthFrownRight', 'MouthPressLeft', 'MouthPressRight',
                'MouthPucker', 'MouthFunnel', 'MouthLeft', 'MouthRight',
                'MouthShrugLower', 'MouthShrugUpper', 'NoseSneerLeft', 'NoseSneerRight',
                'CheekPuff', 'CheekSquintLeft', 'CheekSquintRight'
            ]
            mouth_names = {
                'JawOpen', 'JawLeft', 'JawRight', 'JawForward', 'MouthClose',
                'MouthFunnel', 'MouthPucker', 'MouthLeft', 'MouthRight',
                'MouthSmileLeft', 'MouthSmileRight', 'MouthFrownLeft', 'MouthFrownRight',
                'MouthDimpleLeft', 'MouthDimpleRight', 'MouthStretchLeft', 'MouthStretchRight',
                'MouthRollUpper', 'MouthRollLower', 'MouthShrugLower', 'MouthShrugUpper',
                'MouthPressLeft', 'MouthPressRight', 'MouthUpperUpLeft', 'MouthUpperUpRight',
                'MouthLowerDownLeft', 'MouthLowerDownRight'
            }
            mouth_idx = [i for i, name in enumerate(d6_order) if name in mouth_names]
            annot_array[:, mouth_idx] = annot_array[:, mouth_idx] * float(mouth_gain)
        # 音频加载（可禁用）
        if use_audio:
            audio_array, sr = self.load_audio(audio_path)
            original_audio_array, annot_array = self.truncate(
                audio_array, sr, annot_array, self.fps)
            if processor is not None:
                processed_audio = np.squeeze(
                    processor(original_audio_array, sampling_rate=sr).input_values)
                processed_audio = processed_audio.astype(np.float32)
            else:
                processed_audio = original_audio_array.astype(np.float32)
        else:
            # 不读取音频，基于标注长度计算时长，提供占位音频张量
            annot_length = len(annot_array)
            duration = min(annot_length / self.fps, 20)
            # 将标注帧数裁剪到不超过 20 秒，保持与训练时长策略一致
            annot_length_cap = round(duration * self.fps)
            annot_array = annot_array[:annot_length_cap]
            original_audio_array = np.zeros(1, dtype=np.float32)
            processed_audio = np.zeros(1, dtype=np.float32)
            self.duration = duration

        # 转成 float32 张量，避免 DataLoader collate 在 numpy.float32 上报错
        # 必须使用 np.ascontiguousarray 确保内存连续，否则 DataLoader 可能会报错 "Trying to resize storage that is not resizable"
        self.annot_data = torch.tensor(
            np.ascontiguousarray(annot_array.reshape(len(annot_array), -1)),
            dtype=torch.float32)
        self.audio_data = torch.tensor(
            np.ascontiguousarray(processed_audio), dtype=torch.float32)
        self.original_audio_data = torch.tensor(
            np.ascontiguousarray(original_audio_array),
            dtype=torch.float32)

        # 预缓存特征（可选）：用于跳过每步在线编码，提速小样本训练
        self.audio_cached_feat = None
        self.eeg_cached_feat = None
        audio_cache_path = self.cache_entry.get('audio_feature_path', '')
        eeg_cache_path = self.cache_entry.get('eeg_feature_path', '')
        if isinstance(audio_cache_path, str) and audio_cache_path and osp.exists(
                audio_cache_path):
            audio_cached = np.load(audio_cache_path).astype(np.float32)
            self.audio_cached_feat = torch.tensor(
                np.ascontiguousarray(audio_cached), dtype=torch.float32)
        if isinstance(eeg_cache_path, str) and eeg_cache_path and osp.exists(
                eeg_cache_path):
            eeg_cached = np.load(eeg_cache_path).astype(np.float32)
            self.eeg_cached_feat = torch.tensor(
                np.ascontiguousarray(eeg_cached), dtype=torch.float32)

        # 加载EEG数据（如果提供）
        self.eeg_data = None
        if use_eeg and self.eeg_path is not None and osp.exists(self.eeg_path):
            # 加载已离线预处理的EEG数据（已在生成阶段完成滤波和归一化）
            eeg_array = np.load(self.eeg_path).astype(np.float32)
            self.eeg_data = torch.tensor(
                np.ascontiguousarray(eeg_array), dtype=torch.float32)
        self.annot_type = annot_type
        id_template_array = self.scale_and_offset(id_template, scale, 0.0)
        self.id_template = torch.tensor(
            id_template_array.reshape(1, -1), dtype=torch.float32)
        self.dataset_name = dataset_name
        self.data_dict = self.to_dict()
        return

    def load_audio(self, audio_path: str):
        if audio_path.endswith('.npy'):
            return np.load(audio_path), 16000
        else:
            return librosa.load(audio_path, sr=16000)

    def truncate(self, audio_array: np.ndarray, sr: int,
                 annot_data: np.ndarray, fps: int):
        audio_duration = len(audio_array) / sr
        annot_duration = len(annot_data) / fps
        duration = min(audio_duration, annot_duration, 20)
        audio_length = round(duration * sr)
        annot_length = round(duration * fps)
        self.duration = duration
        return audio_array[:audio_length], annot_data[:annot_length]

    def offset_id(self, offset: int):
        self.identity_idx = self.identity_idx + offset
        self.data_dict['id'] = self.identity_idx
        return

    def scale_and_offset(self,
                         data: np.ndarray,
                         scale: float = 1.0,
                         offset: np.ndarray = 0.0):
        return data * scale + offset

    def to_dict(self, ) -> Any:
        # 为了让 DataLoader 的 default_collate 正常工作，
        # 确保各字段都是可 collate 的类型；当某模态禁用时，提供空张量占位而不是 None。
        if self.eeg_data is None:
            eeg_field = torch.empty(0, dtype=torch.float32)
        else:
            eeg_field = self.eeg_data
        if self.audio_cached_feat is None:
            audio_cached_field = torch.empty(0, dtype=torch.float32)
        else:
            audio_cached_field = self.audio_cached_feat
        if self.eeg_cached_feat is None:
            eeg_cached_field = torch.empty(0, dtype=torch.float32)
        else:
            eeg_cached_field = self.eeg_cached_feat
        item = {
            'data': self.annot_data,
            'audio': self.audio_data,
            'audio_cached_feat': audio_cached_field,
            'original_audio': self.original_audio_data,
            'fps': self.fps,
            'id': self.identity_idx,
            'annot_path': self.annot_path,
            'blendshape_path': self.blendshape_path,
            'audio_path': self.audio_path,
            'annot_type': self.annot_type,
            'template': self.id_template,
            'dataset_name': self.dataset_name,
            'eeg': eeg_field,
            'eeg_cached_feat': eeg_cached_field,
            'eeg_path': self.eeg_path,
            'sample_id': self.sample_id,
            'subject': self.subject,
            'session': self.session,
            'sheet': self.sheet,
            'language': self.language,
            'timestamps': self.timestamps,
            'metadata': self.metadata,
        }
        return item


    def get_dict(self, ):
        return self.data_dict
