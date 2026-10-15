"""
紧凑多模态融合模块 (Compact Multimodal Fusion Module)
用于EATalker系统的音频-EEG特征融合

核心概念：
- 音频特征作为查询(Query)，EEG特征作为键(Key)和值(Value)
- 使用交叉注意力机制实现两种模态的信息融合
- 专为小数据集(40分钟)设计，参数极少，防止过拟合

作者：zym
"""

import torch
import torch.nn as nn


class CompactCrossAttention(nn.Module):
    """
    紧凑交叉注意力模块 - 核心融合机制
    
    工作原理：
    1. 将音频特征转换为"查询向量"(Query)，就像问问题的人
    2. 将EEG特征转换为"键向量"(Key)和"值向量"(Value)，就像被查询的知识库
    3. 通过注意力机制，让音频从EEG中选择最相关的信息
    
    设计特点：
    - 隐藏层维度极小(256维以内)，防止过拟合
    - 专为40分钟小数据集优化
    - 参数量比标准注意力减少约90%
    
    类比理解：就像老师(音频)从学生(EEG)那里获取最相关的反馈信息
    """
    
    def __init__(self, audio_dim: int, eeg_dim: int, num_heads: int = 4, dropout: float = 0.1):
        """
        初始化交叉注意力模块
        
        参数说明：
        - audio_dim: 音频特征的维度 (比如768维)
        - eeg_dim: EEG特征的维度 (比如128维)
        - num_heads: 注意力头的数量，这里用4个头，可以同时关注4个不同方面
        - dropout: 随机丢弃率，防止过拟合，这里设置为0.1(10%)
        
        关键设计选择：
        - hidden_dim取3个维度中的最小值，确保不会过大
        - 使用4个注意力头，在小数据集上表现良好
        - 加入dropout防止过拟合
        """
        super().__init__()
        
        # 选择最小的维度作为隐藏层维度，防止参数量过大
        # 比如音频768维，EEG 128维，那么隐藏层就是128维
        self.hidden_dim = min(256, audio_dim, eeg_dim)  # 紧凑的隐藏层维度
        
        # 投影层：将不同模态的特征映射到相同的隐藏空间
        # 就像翻译官，把英语和中文都翻译成共同的语言
        self.audio_query_proj = nn.Linear(audio_dim, self.hidden_dim)  # 音频→查询
        self.eeg_key_proj = nn.Linear(eeg_dim, self.hidden_dim)        # EEG→键
        self.eeg_value_proj = nn.Linear(eeg_dim, self.hidden_dim)      # EEG→值
        
        # 多头注意力机制 - 核心组件
        # 就像多个专家同时工作，每个专家关注不同的方面
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,    # 每个特征的维度
            num_heads=num_heads,          # 注意力头数
            dropout=dropout,              # 随机丢弃，防止过拟合
            batch_first=True,             # 输入格式：batch在前
            kdim=self.hidden_dim,         # 键的维度
            vdim=self.hidden_dim,         # 值的维度
        )
        
        # 正则化层，防止过拟合同时增加稳定性
        self.dropout = nn.Dropout(dropout)                    # 随机丢弃层
        self.output_proj = nn.Linear(self.hidden_dim, audio_dim)  # 投影回音频维度
        # 门控：抑制低置信度 EEG 注入，避免弱 EEG 拉低音频强基线
        self.gate_proj = nn.Linear(audio_dim * 2, audio_dim)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -1.5)
        
        self.audio_dim = audio_dim

    def forward(self, audio_features: torch.Tensor, eeg_features: torch.Tensor):
        """
        前向传播：音频查询EEG信息的核心流程
        
        参数详解：
        - audio_features: [B, T_audio, audio_dim] 音频特征
            B: 批次大小(batch size)，一次处理多少个样本
            T_audio: 音频序列的时间长度，比如100个时间步
            audio_dim: 每个时间步的特征维度，比如768维
        - eeg_features: [B, T_eeg, eeg_dim] EEG特征
            T_eeg: EEG序列的时间长度，可能与音频不同
            eeg_dim: EEG特征维度，比如128维
        
        返回值：
        - enhanced_audio: [B, T_audio, audio_dim] 增强后的音频特征
            融合了EEG信息的音频特征，维度保持不变
        
        处理流程：
        1. 特征投影：将音频和EEG映射到同一维度空间
        2. 交叉注意力：音频从EEG中选择最相关信息
        3. 残差连接：保留原始音频信息，避免信息丢失
        4. 层归一化：稳定训练过程
        
        实际例子：
        如果输入audio_features形状是[32, 100, 768] (32个样本，每个100个时间步，每步768维)
        eeg_features形状是[32, 80, 128] (32个样本，80个时间步，每步128维)
        那么输出就是[32, 100, 768]，音频序列长度保持不变，但融合了EEG信息
        """
        
        # 步骤1：将音频特征投影为查询向量
        # 就像把问题翻译成通用语言，便于理解
        query = self.audio_query_proj(audio_features)  # [B, T_audio, hidden_dim]
        
        # 步骤2：将EEG特征投影为键和值向量
        # 键：用于匹配查询的索引
        # 值：包含实际信息的特征
        key = self.eeg_key_proj(eeg_features)      # [B, T_eeg, hidden_dim]
        value = self.eeg_value_proj(eeg_features)  # [B, T_eeg, hidden_dim]
        
        # 步骤3：执行交叉注意力机制
        # 音频(作为查询)从EEG(作为键/值)中选择最相关信息
        attended, _ = self.attention(
            query=query,      # 查询：音频想要获取什么信息
            key=key,          # 键：EEG能提供什么信息
            value=value,      # 值：EEG实际包含的信息
            need_weights=False  # 不需要返回注意力权重，节省内存
        )
        
        # 步骤4：将注意力结果投影回原始音频维度
        delta = self.output_proj(attended)  # [B, T_audio, audio_dim]
        gate = torch.sigmoid(
            self.gate_proj(torch.cat([audio_features, delta], dim=-1)))
        gated_delta = delta * gate

        # 返回 EEG 注入增量（不在此处做 residual add），
        # 由上层融合模块按 h_audio + alpha * delta 组合。
        return self.dropout(gated_delta)


class EfficientMultimodalFusion(nn.Module):
    """
    高效多模态融合模块 - 完整融合流程
    
    设计目标：
    - 用最少参数实现音频和EEG的有效融合
    - 专为40分钟小数据集设计，防止过拟合
    - 计算效率高，推理速度快
    
    架构特点：
    1. 只使用一个交叉注意力层(而非多层堆叠)
    2. 隐藏层维度仅128维，参数量极小
    3. 只使用2个注意力头，减少计算量
    4. 简化的前馈网络，只有2个线性层
    
    整体流程：
    音频 → 交叉注意力融合EEG信息 → 前馈网络增强 → 输出增强音频特征
    
    参数量对比：
    - 标准Transformer：~100万参数
    - 本设计：~1万参数，减少了99%
    """
    
    def __init__(
        self,
        audio_dim: int,
        eeg_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 2,
        alpha_mode: str = "scalar",
        alpha_init: float = 0.0,
    ):
        """
        初始化高效融合模块
        
        参数说明：
        - audio_dim: 音频特征维度 (比如768维)
        - eeg_dim: EEG特征维度 (比如128维)  
        - hidden_dim: 隐藏层维度，默认128维，足够小防止过拟合
        - num_heads: 注意力头数，默认2个，比标准4个减少一半
        
        设计考虑：
        - hidden_dim=128：在小数据集上表现最佳，再大容易过拟合
        - num_heads=2：注意力头数减半，减少参数量但仍保持效果
        - 只用一个交叉注意力层：避免深度网络的过拟合风险
        """
        super().__init__()
        
        # 存储隐藏层维度，后续可能会用到
        self.hidden_dim = hidden_dim
        
        # 单个交叉注意力层 - 不进行深度堆叠
        # 深度网络在小数据集上容易过拟合，一层足够
        self.cross_attn = CompactCrossAttention(
            audio_dim=audio_dim,
            eeg_dim=eeg_dim,
            num_heads=num_heads,    # 使用较少的注意力头
            dropout=0.1              # 10%的随机丢弃率
        )
        
        # 微型前馈网络 - 只有2个线性层
        # 就像一个小型加工厂，简单但有效
        self.ffn = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),    # 降维到128维
            nn.ReLU(),                           # 非线性激活
            nn.Dropout(0.1),                     # 随机丢弃，防止过拟合
            nn.Linear(hidden_dim, audio_dim),    # 升维回原维度
            nn.Dropout(0.1),                     # 再次随机丢弃
        )
        
        # 前馈网络的层归一化（作用在 delta 支路）
        self.ffn_norm = nn.LayerNorm(audio_dim)

        alpha_mode = str(alpha_mode).lower().strip()
        if alpha_mode not in ("scalar", "vector"):
            raise ValueError(
                f"Unsupported alpha_mode={alpha_mode}, expected 'scalar' or 'vector'"
            )
        self.alpha_mode = alpha_mode
        if alpha_mode == "scalar":
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        else:
            self.alpha = nn.Parameter(
                torch.full((audio_dim,), float(alpha_init), dtype=torch.float32)
            )

    def forward(self, audio_features: torch.Tensor, eeg_features: torch.Tensor):
        """
        高效融合的前向传播流程
        
        参数详解：
        - audio_features: [B, T_audio, audio_dim] 输入音频特征
        - eeg_features: [B, T_eeg, eeg_dim] 输入EEG特征
        
        返回值：
        - fused_features: [B, T_audio, audio_dim] 融合后的音频特征
            保留了原始维度，但包含了EEG的互补信息
        
        处理步骤：
        1. 交叉注意力融合：音频主动从EEG获取相关信息
        2. 前馈网络增强：对融合后的特征进行非线性变换
        3. 残差连接：确保原始信息不会丢失
        
        设计优势：
        - 整个流程只包含2个主要操作，计算量极小
        - 所有操作都保持batch_first格式，内存访问高效
        - 残差连接让梯度流动更顺畅，训练更稳定
        
        实际例子：
        输入：audio_features=[32, 100, 768], eeg_features=[32, 80, 128]
        输出：fused_features=[32, 100, 768] - 维度不变，信息增强
        """
        
        # EEG 注入增量（cross-attn + delta 支路前馈）
        delta = self.cross_attn(audio_features, eeg_features)
        delta = self.ffn_norm(delta + self.ffn(delta))

        # 必须满足 alpha=0 时严格退化为 audio-only：output == audio_features
        if self.alpha_mode == "scalar":
            alpha = self.alpha
        else:
            alpha = self.alpha.view(1, 1, -1)
        final = audio_features + alpha * delta
        return final


# Utility function for temporal alignment
# 实用工具函数 - 时间对齐                                                                               
# 用于处理音频和EEG序列长度不一致的情况
def temporal_interpolate(features: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    Efficient temporal interpolation for sequence alignment
    时间序列插值对齐 - 解决序列长度不匹配问题                                                          

    使用场景：                                                                                          
    - 音频有100个时间步，EEG有80个时间步，需要统一到相同长度                                            
    - 或者反过来，EEG序列比音频长，需要压缩到相同长度                                                   

    技术原理：                                                                                          
    使用线性插值(linear interpolation)进行时间维度的缩放                                                
    - 拉伸：短序列变成长序列，通过插值增加采样点                                                        
    - 压缩：长序列变成短序列，通过采样减少数据点                                                        

    参数详解：                                                                                          
    - features: [B, T, D] 输入特征序列                                                                  
        B: 批次大小                                                                                     
        T: 原始时间长度                                                                                 
        D: 特征维度                                                                                     
    - target_length: 目标时间长度                                                                       
       正整数，表示想要对齐到的序列长度                                                                

    返回值：                                                                                            
    - interpolated: [B, target_length, D] 对齐后的特征序列                                             
       特征维度保持不变，时间维度变为target_length                                                     

    实际例子：                                                                                         
    输入：features=[32, 80, 128] (32个样本，80个时间步，128维特征)                                      
    target_length=100                                                                                   
    输出：[32, 100, 128] - 序列被拉伸到100个时间步                                                      

    注意事项：                                                                                          
    - 如果原始长度等于目标长度，直接返回原序列，不做处理                                                
    - 使用双线性插值，效果比最近邻插值更平滑                                                            
    - 保持batch_first格式，内存访问高效
    """
    
    B, T, D = features.shape
    
    if T == target_length:
        return features
    
    # Use PyTorch's built-in interpolation for efficiency
    # 步骤1：调整维度顺序用于插值                                                                   
    # PyTorch的插值函数需要 [B, D, T] 格式                                                              
    features = features.transpose(1, 2)  # [B, D, T]                                                   

    # 步骤2：执行线性插值                                                                              
     # mode='linear'：线性插值，结果平滑                                                                
    # align_corners=True：确保边界点对齐
    interpolated = torch.nn.functional.interpolate(
        features,
        size=target_length, # 目标长度
        mode='linear', # 线性插值
        align_corners=True
    )
    # 步骤3：恢复原始维度顺序 从 [B, D, target_length] 变回 [B, target_length, D]
    return interpolated.transpose(1, 2)  # B x target_length x D
