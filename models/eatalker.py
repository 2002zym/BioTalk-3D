"""
EATalker模型 - 集成传统EEGNet和音频的多模态3D面部动画生成

该模型在UniTalker的基础上进行扩展，增加了：
1. 传统EEGNet编码器：基于old_eeg_model.py的脑电信号特征提取
2. 多模态融合模块：融合音频和EEG特征
3. 双模态输入支持：同时处理音频和EEG数据

核心架构：
音频输入 → 音频编码器 → 
                    ↓
                    多模态融合 → 解码器 → 3D面部动画
                    ↑
EEG输入 → EEGNet编码器 →

基于传统EEGNet架构，保留了Conv2dWithConstraint和完整的块结构
"""

import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

from EATalker.dataset.dataset_config import dataset_config
from EATalker.models.wav2vec2 import Wav2Vec2Model
from EATalker.models.wavlm import WavLMModel
from EATalker.models.eeg_encoder import EEGEncoder  # 传统EEGNet编码器
from EATalker.models.multimodal_fusion import EfficientMultimodalFusion
from .base_model import BaseModel


class OutHead(BaseModel):
    """输出头 - 与原始UniTalker保持一致"""

    def __init__(self, args, out_dim: int):
        super().__init__()
        head_layer = []

        in_dim = args.decoder_dimension
        for i in range(args.headlayer - 1):
            head_layer.append(nn.Linear(in_dim, in_dim))
            head_layer.append(nn.Tanh())
        head_layer.append(nn.Linear(in_dim, out_dim))
        self.head_layer = nn.Sequential(*head_layer)

    def forward(self, x):
        return self.head_layer(x)


class EATalker(BaseModel):
    """
    EATalker主模型 - 支持传统EEGNet和音频双模态输入
    
    功能特点：
    1. 传统EEGNet编码器：基于old_eeg_model.py的完整架构
    2. 双模态编码：独立的音频和EEGNet编码器
    3. 智能融合：交叉注意力机制融合两种模态
    4. 灵活配置：支持纯音频、纯EEG、双模态三种模式
    5. 向后兼容：保持与原始UniTalker的接口兼容性
    
    EEGNet特点保留：
    - Conv2dWithConstraint：权重约束防止过拟合
    - 深度可分离卷积：高效的空间-时间特征提取
    - ELU激活函数：传统EEGNet标准配置
    - 平均池化：降采样同时保留重要信息
    
    使用模式：
    - 双模态模式：同时提供audio和eeg输入，获得最佳效果
    - 纯音频模式：只提供audio输入，退化为原始UniTalker行为
    - 纯EEG模式：只提供eeg输入，探索纯脑电驱动效果
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.identity_policy = str(
            getattr(args, "identity_policy", "per_id") or "per_id").lower()
        if self.identity_policy not in ("per_id", "global"):
            raise ValueError(
                f"Unsupported identity_policy={self.identity_policy}, expected per_id/global")
        
        # 音频编码器 - 复用UniTalker的音频处理
        if 'wav2vec2' in args.audio_encoder_repo:
            self.audio_encoder = Wav2Vec2Model.from_pretrained(args.audio_encoder_repo)
        elif 'wavlm' in args.audio_encoder_repo:
            self.audio_encoder = WavLMModel.from_pretrained(args.audio_encoder_repo)   
        else:
            raise ValueError(f"Unsupported audio_encoder_repo: {args.audio_encoder_repo}")
        
        # === Wav2Vec2/WavLM 冻结/微调策略 ===
        # - 默认：freeze_wav2vec=True 时完全冻结音频编码器
        # - 中间路线：仅微调 Transformer 的最后 N 层（wav2vec_train_last_n_layers>0）
        self.wav2vec_train_last_n_layers = int(
            getattr(args, 'wav2vec_train_last_n_layers', 0) or 0)
        self.wav2vec_frozen_to_eval = bool(
            getattr(args, 'wav2vec_frozen_to_eval', True))

        # CNN 前端默认冻结（与 UniTalker 训练习惯一致）
        if hasattr(self.audio_encoder, 'feature_extractor') and hasattr(
                self.audio_encoder.feature_extractor, '_freeze_parameters'):
            self.audio_encoder.feature_extractor._freeze_parameters()

        if self.wav2vec_train_last_n_layers > 0 and hasattr(
                self.audio_encoder, '_freeze_wav2vec2_except_last_n_layers'):
            self.audio_encoder._freeze_wav2vec2_except_last_n_layers(
                self.wav2vec_train_last_n_layers)
        elif hasattr(args, 'freeze_wav2vec') and args.freeze_wav2vec:
            self.audio_encoder._freeze_wav2vec2_parameters(True)
        
        # EEGNet编码器 - 基于传统EEGNet架构
        self.use_eeg = hasattr(args, 'use_eeg') and args.use_eeg
        if self.use_eeg:
            self.eeg_encoder = EEGEncoder(
                eeg_channels=getattr(args, 'eeg_channels', 64),
                eeg_length=getattr(args, 'eeg_length', 256),
                output_dim=getattr(args, 'eeg_output_dim', 128),
                dropout_rate=getattr(args, 'eeg_dropout', 0.5),
                kernel_length=getattr(args, 'eeg_kernel_length', 64),
                kernel_length2=getattr(args, 'eeg_kernel_length2', 16),
                F1=getattr(args, 'eeg_F1', 8),
                D=getattr(args, 'eeg_D', 2),
                F2=getattr(args, 'eeg_F2', 16),
                use_lstm=getattr(args, 'eeg_use_lstm', False),
                lstm_hidden=getattr(args, 'eeg_lstm_hidden', 64),
                freeze_eegnet=getattr(args, 'freeze_eegnet', False)
            )

            # 多模态融合模块
            self.multimodal_fusion = EfficientMultimodalFusion(
                audio_dim=self._get_audio_encoder_dim(),
                eeg_dim=getattr(args, 'eeg_output_dim', 128),
                hidden_dim=getattr(args, 'fusion_hidden_dim', 128),
                num_heads=getattr(args, 'fusion_num_heads', 2),
                alpha_mode=getattr(args, "fusion_alpha_mode", "scalar"),
                alpha_init=float(getattr(args, "fusion_alpha_init", 0.0)),
            )
            # 纯EEG路径的维度适配：将 EEGNet 输出映射到音频编码器维度，便于直接对接解码器
            self._audio_feature_dim = self._get_audio_encoder_dim()
            self._eeg_feature_dim = getattr(args, 'eeg_output_dim', 128)
            if self._eeg_feature_dim != self._audio_feature_dim:
                self.eeg_to_audio = nn.Linear(self._eeg_feature_dim,
                                              self._audio_feature_dim)
            else:
                # 占位映射，保持接口一致
                self.eeg_to_audio = nn.Identity()
        
        # 解码器 - 复用UniTalker的解码器
        if args.decoder_type == 'conv':
            from .eatalker_decoder import EaTalkerDecoderTCN as Decoder
        elif args.decoder_type == 'transformer':
            from .eatalker_decoder import EaTalkerDecoderTransformer as Decoder
        else:
            ValueError(f'Unknown decoder type: {args.decoder_type}')
        
        self.decoder = Decoder(args)
        
        # PCA层 - 与原始UniTalker保持一致
        if args.use_pca:
            pca_layer_dict = {}
            from models.pca import PCALayer
            for dataset_name in args.dataset:
                if dataset_config[dataset_name]['pca']:
                    annot_type = dataset_config[dataset_name]['annot_type']
                    dirname = dataset_config[dataset_name]['dirname']
                    pca_path = osp.join(args.data_root, dirname, 'pca.npz')
                    pca_layer_dict[annot_type] = PCALayer(pca_path)
            self.pca_layer_dict = nn.ModuleDict(pca_layer_dict)
        else:
            self.pca_layer_dict = {}

        # 输出头配置 - 与 UniTalker 保持一致以便复用预训练权重
        self.pca_dim = args.pca_dim
        self.use_pca = args.use_pca
        self.interpolate_pos = args.interpolate_pos

        out_head_dict = {}
        for dataset_name in args.dataset:
            annot_type = dataset_config[dataset_name]['annot_type']
            out_dim = dataset_config[dataset_name]['annot_dim']
            if (self.use_pca is True) and (annot_type in self.pca_layer_dict):
                pca_dim = self.pca_layer_dict[annot_type].pca_dim
                out_dim = min(self.pca_dim, out_dim, pca_dim)
            if args.headlayer == 1:
                out_projection = nn.Linear(args.decoder_dimension, out_dim)
                nn.init.constant_(out_projection.weight, 0)
                nn.init.constant_(out_projection.bias, 0)
            else:
                out_projection = OutHead(args, out_dim)
            out_head_dict[annot_type] = out_projection
        self.out_head_dict = nn.ModuleDict(out_head_dict)
        self.identity_num = args.identity_num

    def train(self, mode: bool = True):
        """Override train() to keep frozen audio encoder parts in eval() mode.

        Why: dropout/spec-augment randomness in frozen modules increases noise but
        cannot be compensated by parameter updates, often leading to conservative
        "mean" predictions on small datasets.
        """
        super().train(mode)
        if not mode:
            return self

        freeze_all = bool(getattr(self.args, 'freeze_wav2vec', False))
        last_n = int(getattr(self.args, 'wav2vec_train_last_n_layers', 0) or 0)
        frozen_to_eval = bool(getattr(self.args, 'wav2vec_frozen_to_eval', True))

        if frozen_to_eval and (freeze_all or last_n > 0):
            if hasattr(self.audio_encoder, '_apply_finetune_train_mode'):
                self.audio_encoder._apply_finetune_train_mode(
                    train_last_n_layers=last_n)
            else:
                self.audio_encoder.eval()
        return self
        
    def _get_audio_encoder_dim(self):
        """获取音频编码器输出维度"""
        feature_dim = getattr(self.args, 'audio_encoder_feature_dim', None)
        if feature_dim is not None:
            return feature_dim
        if 'wav2vec2' in self.args.audio_encoder_repo:
            return 768
        if 'wavlm' in self.args.audio_encoder_repo:
            return 768
        return 768

    def forward(self,
                audio=None,
                eeg=None,
                audio_cached_feat=None,
                eeg_cached_feat=None,
                template=None,
                face_motion=None,
                style_idx=None,
                annot_type: str = None,
                fps: float = None,
                use_dual_modal: bool = True):
        """
        前向传播 - 支持灵活的模态输入
        
        参数：
        - audio: [B, T_audio] 音频输入（可选）
        - eeg: [B, C, T_eeg] 或 [B, T_eeg, C] EEG输入（可选）
        - template: [B, V, 3] 面部模板
        - face_motion: [B, T, V, 3] 真实面部运动（训练时用）
        - style_idx: [B] 风格/身份索引
        - annot_type: 注释类型
        - fps: 帧率
        - use_dual_modal: 是否使用双模态融合
        
        返回：
        - out_motion: [B, T, V, 3] 预测的面部运动
        - out_pca: PCA空间中的预测（如果使用PCA）
        - gt_pca: PCA空间中的真实值（训练时）
        """
        
        # 确定输出帧数
        if face_motion is not None:
            frame_num = face_motion.shape[1]
        elif isinstance(audio_cached_feat, torch.Tensor) and audio_cached_feat.numel() > 0:
            frame_num = int(audio_cached_feat.shape[1])
        elif audio is not None:
            frame_num = round(audio.shape[-1] / 16000 * fps)
        elif isinstance(eeg_cached_feat, torch.Tensor) and eeg_cached_feat.numel() > 0:
            frame_num = int(eeg_cached_feat.shape[1])
        elif eeg is not None:
            # 如果没有音频，根据EEG长度估算帧数
            if len(eeg.shape) == 3:
                time_dim = eeg.shape[2] if eeg.shape[1] < eeg.shape[2] else eeg.shape[1]
            else:
                time_dim = eeg.shape[1]
            frame_num = min(time_dim // 2, 100)  # 保守估计
        else:
            raise ValueError("必须提供audio或eeg输入")
        
        has_audio = (audio is not None) or (isinstance(audio_cached_feat,
                                                       torch.Tensor)
                                            and audio_cached_feat.numel() > 0)
        has_eeg = (eeg is not None) or (isinstance(eeg_cached_feat,
                                                   torch.Tensor)
                                        and eeg_cached_feat.numel() > 0)

        # 特征提取和融合
        if self.use_eeg and use_dual_modal and has_audio and has_eeg:
            # 双模态模式 - 主要工作模式
            hidden_states = self._dual_modal_forward(
                audio,
                eeg,
                frame_num,
                audio_cached_feat=audio_cached_feat,
                eeg_cached_feat=eeg_cached_feat,
            )
            
        elif audio is not None or (
                isinstance(audio_cached_feat, torch.Tensor)
                and audio_cached_feat.numel() > 0):
            # 纯音频模式 - 兼容原始UniTalker
            hidden_states = self._audio_only_forward(
                audio, frame_num, audio_cached_feat=audio_cached_feat)
            
        elif (eeg is not None or (isinstance(eeg_cached_feat, torch.Tensor)
                                  and eeg_cached_feat.numel() > 0)) and self.use_eeg:
            # 纯EEG模式 - 探索性模式
            hidden_states = self._eeg_only_forward(
                eeg, frame_num, eeg_cached_feat=eeg_cached_feat)
            
        else:
            raise ValueError("输入配置不支持：需要audio和/或eeg输入")
        
        # 解码器处理
        style_idx = self._normalize_style_idx(style_idx)
        decoder_out = self.decoder(hidden_states, style_idx, frame_num)

        # 根据标注类型选择对应输出头
        if annot_type is None:
            annot_type = dataset_config[self.args.dataset[0]]['annot_type']

        if (not self.use_pca) or (annot_type not in self.pca_layer_dict):
            out_motion = self.out_head_dict[annot_type](decoder_out)
            out_motion = out_motion + template
            out_pca, gt_pca = None, None
        else:
            out_pca = self.out_head_dict[annot_type](decoder_out)
            out_motion = self.pca_layer_dict[annot_type].decode(
                out_pca, self.pca_dim)
            out_motion = out_motion + template
            if face_motion is not None:
                gt_pca = self.pca_layer_dict[annot_type].encode(
                    face_motion - template, self.pca_dim)
            else:
                gt_pca = None

        return out_motion, out_pca, gt_pca
    
    def _align_cached_features(self, features: torch.Tensor, frame_num: int):
        if not isinstance(features, torch.Tensor):
            return None
        if features.numel() == 0:
            return None
        if features.dim() == 2:
            features = features.unsqueeze(0)
        if features.dim() != 3:
            return None
        if features.shape[1] != frame_num:
            features = F.interpolate(
                features.transpose(1, 2),
                size=frame_num,
                mode='linear',
                align_corners=True).transpose(1, 2)
        return features

    def _normalize_style_idx(self, style_idx: torch.Tensor):
        if style_idx is None:
            return style_idx
        if self.identity_policy == "global":
            return torch.zeros_like(style_idx)
        if self.identity_num > 0:
            return style_idx % int(self.identity_num)
        return style_idx

    def _dual_modal_forward(
        self,
        audio: torch.Tensor,
        eeg: torch.Tensor,
        frame_num: int,
        audio_cached_feat: torch.Tensor = None,
        eeg_cached_feat: torch.Tensor = None,
    ):
        """
        双模态前向传播 - 核心创新
        
        处理流程：
        1. 分别提取音频和EEG特征
        2. 使用交叉注意力融合两种模态
        3. 输出增强的音频特征
        """
        # 音频特征提取
        audio_features = self._align_cached_features(audio_cached_feat,
                                                     frame_num)
        if audio_features is None:
            audio_features = self.audio_encoder(
                audio, frame_num=frame_num, interpolate_pos=self.interpolate_pos)
            audio_features = audio_features.last_hidden_state  # [B, T, audio_dim]
        
        # EEGNet特征提取
        eeg_features = self._align_cached_features(eeg_cached_feat, frame_num)
        if eeg_features is None:
            eeg_features = self.eeg_encoder(
                eeg, frame_num=frame_num, interpolate_pos=self.interpolate_pos)
            eeg_features = eeg_features.last_hidden_state  # [B, T, eeg_dim]
        
        # 多模态融合 - EEG信息增强音频特征
        fused_features = self.multimodal_fusion(audio_features, eeg_features)
        
        return fused_features
    
    def _audio_only_forward(self,
                            audio: torch.Tensor,
                            frame_num: int,
                            audio_cached_feat: torch.Tensor = None):
        """
        纯音频前向传播 - 兼容原始UniTalker
        """
        audio_features = self._align_cached_features(audio_cached_feat,
                                                     frame_num)
        if audio_features is not None:
            return audio_features
        audio_features = self.audio_encoder(
            audio, frame_num=frame_num, interpolate_pos=self.interpolate_pos)
        return audio_features.last_hidden_state
    
    def _eeg_only_forward(self,
                          eeg: torch.Tensor,
                          frame_num: int,
                          eeg_cached_feat: torch.Tensor = None):
        """
        纯EEG模式 - 探索性模式
        """
        feats = self._align_cached_features(eeg_cached_feat, frame_num)
        if feats is None:
            eeg_features = self.eeg_encoder(
                eeg, frame_num=frame_num, interpolate_pos=self.interpolate_pos)
            feats = eeg_features.last_hidden_state  # [B, T, eeg_dim]
        # 维度适配到解码器期望的音频特征维度（例如 1024）
        feats = self.eeg_to_audio(feats)
        return feats
    
    def get_modal_weights(self):
        """获取模态权重信息（用于分析）"""
        if not self.use_eeg:
            return {"audio": 1.0, "eeg": 0.0, "fusion": 0.0}
        alpha = 1.0
        if hasattr(self, "multimodal_fusion") and hasattr(
                self.multimodal_fusion, "alpha"):
            alpha_param = self.multimodal_fusion.alpha.detach()
            alpha = float(alpha_param.abs().mean().cpu().item())
        
        return {
            "audio": 1.0,
            "eeg": 1.0, 
            "fusion": 1.0,
            "alpha_abs_mean": alpha,
        }


# 测试代码
if __name__ == "__main__":
    import argparse
    
    # 创建测试参数
    args = argparse.Namespace()
    args.audio_encoder_repo = 'microsoft/wavlm-base-plus'
    args.freeze_wav2vec = False
    args.decoder_type = 'conv'
    args.decoder_dimension = 256
    args.period = 30
    args.headlayer = 1
    args.use_pca = True
    args.pca_dim = 512
    args.dataset = ['D0']
    args.data_root = '/tmp'
    args.identity_num = 10
    
    # EEGNet相关参数 - 基于传统架构
    args.use_eeg = True
    args.eeg_channels = 64
    args.eeg_length = 256
    args.eeg_output_dim = 128
    args.eeg_dropout = 0.5
    args.eeg_kernel_length = 64
    args.eeg_kernel_length2 = 16
    args.eeg_F1 = 8
    args.eeg_D = 2
    args.eeg_F2 = 16
    args.eeg_use_lstm = False
    args.eeg_lstm_hidden = 64
    args.freeze_eegnet = False
    args.fusion_hidden_dim = 128
    args.fusion_num_heads = 2
    
    # 创建模型
    model = EATalker(args)
    
    # 创建测试输入
    batch_size = 2
    audio = torch.randn(batch_size, 16000 * 2)  # 2秒音频
    eeg = torch.randn(batch_size, 64, 256)  # EEG数据
    template = torch.randn(batch_size, 5023, 3)  # FLAME模板
    style_idx = torch.randint(0, 10, (batch_size,))
    
    # 测试双模态
    with torch.no_grad():
        output_motion, output_pca, gt_pca = model(
            audio=audio,
            eeg=eeg,
            template=template,
            style_idx=style_idx,
            annot_type='flame_params',
            fps=30.0,
            use_dual_modal=True
        )
    
    print(f"双模态测试 - 输出形状: {output_motion.shape}")
    
    # 测试纯音频模式
    with torch.no_grad():
        output_motion_audio, _, _ = model(
            audio=audio,
            template=template,
            style_idx=style_idx,
            annot_type='flame_params',
            fps=30.0,
            use_dual_modal=False
        )
    
    print(f"纯音频测试 - 输出形状: {output_motion_audio.shape}")
    
    # 测试纯EEG模式
    with torch.no_grad():
        output_motion_eeg, _, _ = model(
            eeg=eeg,
            template=template,
            style_idx=style_idx,
            annot_type='flame_params',
            fps=30.0,
            use_dual_modal=False
        )
    
    print(f"纯EEG测试 - 输出形状: {output_motion_eeg.shape}")
    print("✅ 基于传统EEGNet的EATalker模型测试通过！")
