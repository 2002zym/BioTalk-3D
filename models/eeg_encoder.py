"""
EEG编码器模块 - 基于传统EEGNet架构
用于EATalker系统的脑电信号特征提取

该模块基于old_eeg_model.py中的传统EEGNet实现，专门为EATalker项目定制优化。
保留了EEGNet的核心架构：
1. 卷积约束层 (Conv2dWithConstraint)
2. 深度可分离卷积 (Depthwise + Separable)
3. 块结构 (InitialBlocks + ClassifierBlock)
4. ELU激活函数和平均池化

作者：基于传统EEGNet的EATalker专用版本
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import _addindent


class Conv2dWithConstraint(nn.Conv2d):
    """
    卷积约束层 - EEGNet的核心组件
    通过权重归一化限制卷积层权重的大小来防止过拟合，max_norm=1保证权重矩阵的范数不超过1
    """
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class EEGNetFeatureExtractor(nn.Module):
    """
    EEGNet特征提取器 - 基于传统EEGNet的核心架构
    
    保留了原始EEGNet的完整结构：
    1. Block1: 时间卷积 + 深度空间卷积 + 池化
    2. Block2: 可分离卷积 + 点态卷积 + 池化
    
    专为特征提取设计，去除了分类层，适配EATalker的序列输出需求
    """
    
    def __init__(self, 
                 channels=64,           # EEG通道数
                 samples=256,          # 时间采样点数  
                 dropoutRate=0.5,      # Dropout率
                 kernelLength=64,      # 第一卷积层核长度
                 kernelLength2=16,     # 第二卷积层核长度
                 F1=8,                 # 第一卷积层滤波器数
                 D=2,                  # 深度乘子
                 F2=16):               # 第二卷积层滤波器数
        super(EEGNetFeatureExtractor, self).__init__()
        
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.channels = channels
        self.samples = samples
        self.kernelLength = kernelLength
        self.kernelLength2 = kernelLength2
        self.dropoutRate = dropoutRate
        
        # Block 1: 时间卷积 + 深度空间卷积
        self.block1 = nn.Sequential(
            # 时间卷积：1x1卷积扩展特征维度
            nn.Conv2d(1, self.F1, (1, self.kernelLength), 
                     stride=1, padding=(0, self.kernelLength // 2), bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            
            # 深度空间卷积：学习空间关系，使用约束
            Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.channels, 1), 
                               max_norm=1, stride=1, padding=(0, 0),
                               groups=self.F1, bias=False),
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=4),
            nn.Dropout(p=dropoutRate)
        )
        
        # Block 2: 可分离卷积
        self.block2 = nn.Sequential(
            # 深度卷积（每个输入通道单独与对应的卷积核进行卷积，不进行通道间的信息融合。）
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, self.kernelLength2), 
                     stride=1, padding=(0, self.kernelLength2 // 2), 
                     bias=False, groups=self.F1 * self.D),
            # 点态卷积（使用 1x1 的卷积核，主要作用是融合通道信息或调整通道数量。）
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), 
                     groups=1, bias=False, stride=1),
            
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropoutRate)
        )
        
        # 计算输出尺寸
        self.block_output_size = self._calculate_output_size()
        
    def _calculate_output_size(self):
        """计算特征提取器的输出尺寸"""
        data = torch.rand(1, 1, self.channels, self.samples)
        self.eval()
        with torch.no_grad():
            out = self.blocks(data).shape
        return out[2:]
    
    def blocks(self, x):
        """前向传播通过两个块"""
        x = self.block1(x)
        x = self.block2(x)
        return x
    
    def get_output_channels(self):
        """获取输出通道数"""
        return self.F2
    
    def get_output_time_dim(self):
        """获取输出的时间维度大小"""
        return self.block_output_size[1]


class EEGNetEncoder(nn.Module):
    """
    基于传统EEGNet的EEG编码器
    
    特点：
    1. 保留传统EEGNet的所有核心组件
    2. 适配序列输出，支持可变长度
    3. 集成时序建模（LSTM）
    4. 输出维度可配置，兼容音频编码器
    """
    
    def __init__(self,
                 eeg_channels=64,          # EEG通道数
                 eeg_length=256,           # 输入序列长度
                 output_dim=128,           # 输出特征维度
                 dropout_rate=0.5,         # Dropout率
                 kernel_length=64,         # 第一卷积核长度
                 kernel_length2=16,        # 第二卷积核长度
                 F1=8,                     # 第一卷积滤波器数
                 D=2,                      # 深度乘子
                 F2=16,                    # 第二卷积滤波器数
                 use_lstm=False,            # 是否使用LSTM（先设成false）
                 lstm_hidden=64,           # LSTM隐藏层维度
                 lstm_layers=1):           # LSTM层数
        super(EEGNetEncoder, self).__init__()
        
        self.eeg_channels = eeg_channels
        self.eeg_length = eeg_length
        self.output_dim = output_dim
        self.use_lstm = use_lstm
        
        # EEGNet特征提取器 - 保留传统架构
        self.feature_extractor = EEGNetFeatureExtractor(
            channels=eeg_channels,
            samples=eeg_length,
            dropoutRate=dropout_rate,
            kernelLength=kernel_length,
            kernelLength2=kernel_length2,
            F1=F1,
            D=D,
            F2=F2
        )
        
        # 特征投影层
        self.feature_projection = nn.Sequential(
            nn.Flatten(start_dim=2),  # 保持batch和通道维度
            nn.Linear(self.feature_extractor.get_output_channels(), output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 时序建模 - 可选的LSTM
        if use_lstm:
            self.temporal_modeling = nn.LSTM(
                input_size=output_dim,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout_rate if lstm_layers > 1 else 0
            )
            # 双向LSTM输出维度调整
            self.lstm_projection = nn.Linear(lstm_hidden * 2, output_dim)
        
        # 输出归一化
        self.output_norm = nn.LayerNorm(output_dim)
        
    def forward(self, eeg_data, frame_num=None):
        """
        前向传播
        
        参数：
        - eeg_data: [B, C, T] 或 [B, T, C] EEG数据
        - frame_num: 目标输出帧数（用于与音频对齐）
        
        返回：
        - features: [B, T_out, output_dim] 特征序列
        """
        batch_size = eeg_data.shape[0]
        
        # 调整输入格式 - 确保是 [B, C, T]
        if len(eeg_data.shape) == 3:
            if eeg_data.shape[1] == self.eeg_channels:
                # 格式已经是 [B, C, T]
                pass
            elif eeg_data.shape[2] == self.eeg_channels:
                # 格式是 [B, T, C]，需要转置
                eeg_data = eeg_data.transpose(1, 2)
            else:
                raise ValueError(f"Unexpected EEG data shape: {eeg_data.shape}")
        else:
            raise ValueError(f"EEG data should be 3D, got {len(eeg_data.shape)}D")
        
        # 添加通道维度用于2D卷积 [B, 1, C, T]
        x = eeg_data.unsqueeze(1)
        
        # EEGNet特征提取
        features = self.feature_extractor.blocks(x)  # [B, F2, 1, T']
        
        # 移除空间维度 [B, F2, T']
        features = features.squeeze(2)
        
        # 转置用于时序处理 [B, T', F2]
        features = features.transpose(1, 2)
        
        # 特征投影 [B, T', output_dim]
        features = self.feature_projection(features)
        
        # 时序建模
        if self.use_lstm:
            features, _ = self.temporal_modeling(features)  # [B, T', lstm_hidden*2]
            features = self.lstm_projection(features)       # [B, T', output_dim]
        
        # 序列长度对齐
        if frame_num is not None and features.shape[1] != frame_num:
            features = self._align_sequence_length(features, frame_num)
        
        # 输出归一化
        features = self.output_norm(features)
        
        return features
    
    def _align_sequence_length(self, features, target_length):
        """对齐序列长度"""
        current_length = features.shape[1]
        
        if current_length == target_length:
            return features
        
        # 转置用于插值 [B, D, T]
        features = features.transpose(1, 2)
        
        # 线性插值
        aligned = F.interpolate(
            features,
            size=target_length,
            mode='linear',
            align_corners=True
        )
        
        # 转置回来 [B, T, D]
        return aligned.transpose(1, 2)


class EEGEncoder(nn.Module):
    """
    EEG编码器统一接口 - 适配EATalker项目
    
    功能：
    1. 提供与音频编码器一致的接口
    2. 支持不同的输入格式和配置
    3. 输出标准化的特征表示
    """
    
    def __init__(self,
                 eeg_channels=64,
                 eeg_length=256,
                 output_dim=128,
                 dropout_rate=0.5,
                 kernel_length=64,
                 kernel_length2=16,
                 F1=8,
                 D=2,
                 F2=16,
                 use_lstm=False,
                 lstm_hidden=64,
                 freeze_eegnet=False):
        super(EEGEncoder, self).__init__()
        
        self.eeg_channels = eeg_channels
        self.output_dim = output_dim
        
        # 核心EEGNet编码器
        self.eegnet = EEGNetEncoder(
            eeg_channels=eeg_channels,
            eeg_length=eeg_length,
            output_dim=output_dim,
            dropout_rate=dropout_rate,
            kernel_length=kernel_length,
            kernel_length2=kernel_length2,
            F1=F1,
            D=D,
            F2=F2,
            use_lstm=use_lstm,
            lstm_hidden=lstm_hidden
        )
        
        # 是否冻结EEGNet参数
        if freeze_eegnet:
            for param in self.eegnet.parameters():
                param.requires_grad = False
        
        # 输出归一化
        self.output_norm = nn.LayerNorm(output_dim)
        
    def forward(self, eeg_data, frame_num=None, interpolate_pos=False):
        """
        前向传播 - 统一接口
        
        参数：
        - eeg_data: [B, C, T] 或 [B, T, C] EEG数据
        - frame_num: 目标输出帧数
        - interpolate_pos: 位置插值标志（兼容音频编码器接口）
        
        返回：
        - output: SimpleNamespace 包含编码结果
            - last_hidden_state: [B, T_out, output_dim] 主要特征
            - hidden_states: 所有中间状态
        """
        from types import SimpleNamespace
        
        # 提取EEG特征
        eeg_features = self.eegnet(eeg_data, frame_num=frame_num)
        
        # 输出归一化
        eeg_features = self.output_norm(eeg_features)
        
        # 包装输出格式，与音频编码器保持一致
        output = SimpleNamespace()
        output.last_hidden_state = eeg_features
        output.hidden_states = [eeg_features]  # 兼容音频编码器接口
        
        return output
    
    def get_output_dim(self):
        """获取输出特征维度"""
        return self.output_dim


# 测试函数
def test_eegnet_encoder():
    """测试EEGNet编码器"""
    print("=== 测试传统EEGNet编码器 ===")
    
    # 测试参数
    batch_size = 2
    eeg_channels = 64
    eeg_length = 256
    output_dim = 128
    target_frames = 100
    
    # 创建模型
    model = EEGEncoder(
        eeg_channels=eeg_channels,
        eeg_length=eeg_length,
        output_dim=output_dim
    )
    
    # 创建测试输入
    test_input = torch.randn(batch_size, eeg_channels, eeg_length)
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(test_input, frame_num=target_frames)
    
    # 检查结果
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.last_hidden_state.shape}")
    print(f"期望输出形状: [{batch_size}, {target_frames}, {output_dim}]")
    
    # 验证输出形状
    assert output.last_hidden_state.shape == (batch_size, target_frames, output_dim), \
        f"输出形状不匹配！期望: [{batch_size}, {target_frames}, {output_dim}], 实际: {output.last_hidden_state.shape}"
    
    print("✅ 传统EEGNet编码器测试通过！\n")
    
    # 测试参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
    
    return True


if __name__ == "__main__":
    test_eegnet_encoder()