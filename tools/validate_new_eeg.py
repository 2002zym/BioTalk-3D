#!/usr/bin/env python3
"""
验证新预处理后的EEG数据质量
检查：
1. 数据统计信息（是否符合Robust Z-Score预期）
2. EEG GFP 与 音频包络的相关性
3. EEG GFP 与 Blendshape 动态的相关性
4. 可视化三者的时间对齐情况
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.stats import pearsonr
from scipy.io import wavfile

# ========== 配置 ==========
EEG_PATH = "/Users/zym/EATalker/sentence1_liu1_happy_eeg.npy"
AUDIO_PATH = "/Users/zym/EATalker/sentence1_liu1_happy.wav"
BS_PATH = "/Users/zym/EATalker/sentence1_liu1_happy_bs.npy"
OUTPUT_PATH = "/Users/zym/EATalker/zym_test/new_eeg_validation.png"
EEG_SFREQ = 1000  # EEG采样率

# ========== 加载数据 ==========
print("=" * 60)
print("加载数据...")
eeg_data = np.load(EEG_PATH)
bs_data = np.load(BS_PATH)
sr, audio = wavfile.read(AUDIO_PATH)
# 转换为浮点数并归一化
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0
elif audio.dtype == np.int32:
    audio = audio.astype(np.float32) / 2147483648.0
else:
    audio = audio.astype(np.float32)

print(f"EEG shape: {eeg_data.shape}")
print(f"Blendshape shape: {bs_data.shape}")
print(f"Audio shape: {audio.shape}, SR: {sr} Hz")

# ========== 1. EEG 统计检查 ==========
print("\n" + "=" * 60)
print("1. EEG 数据统计 (预期: 均值≈0, 标准差≈1, 范围在[-20,20])")
print("-" * 60)
print(f"  全局均值: {np.mean(eeg_data):.4f}")
print(f"  全局标准差: {np.std(eeg_data):.4f}")
print(f"  最小值: {np.min(eeg_data):.4f}")
print(f"  最大值: {np.max(eeg_data):.4f}")
print(f"  中位数: {np.median(eeg_data):.4f}")

# 检查是否有异常
if abs(np.mean(eeg_data)) > 1.0:
    print("  ⚠️ 警告: 均值偏离0较多")
if np.std(eeg_data) < 0.5 or np.std(eeg_data) > 5.0:
    print("  ⚠️ 警告: 标准差异常")
if np.max(np.abs(eeg_data)) > 20.0:
    print("  ⚠️ 警告: 存在超出[-20, 20]的极端值")

# ========== 2. 计算 GFP ==========
print("\n" + "=" * 60)
print("2. 计算 Global Field Power (GFP)...")

# 检测数据格式并调整 [C, T] 或 [T, C]
if eeg_data.shape[0] > eeg_data.shape[1]:
    # 数据是 [T, C] 格式，需要转置
    print(f"  检测到数据格式: [Time={eeg_data.shape[0]}, Channels={eeg_data.shape[1]}]")
    eeg_transposed = eeg_data.T  # 转为 [C, T]
else:
    # 数据已经是 [C, T] 格式
    print(f"  检测到数据格式: [Channels={eeg_data.shape[0]}, Time={eeg_data.shape[1]}]")
    eeg_transposed = eeg_data

# GFP = 各通道的标准差（反映全脑活动强度）
# 沿通道轴计算标准差
gfp = np.std(eeg_transposed, axis=0)  # [T,]
print(f"  GFP shape: {gfp.shape} (时间点数)")
print(f"  GFP 均值: {np.mean(gfp):.4f}")
print(f"  GFP 标准差: {np.std(gfp):.4f}")
print(f"  GFP 变异系数 (CV): {np.std(gfp) / np.mean(gfp) * 100:.2f}%")

if np.std(gfp) / np.mean(gfp) < 0.1:
    print("  ⚠️ 警告: GFP变异系数过低，信号可能过于平坦")
else:
    print("  ✅ GFP 有明显的时间动态变化")

# ========== 3. 计算音频包络 ==========
print("\n" + "=" * 60)
print("3. 计算音频包络...")
# 使用Hilbert变换提取包络
analytic_signal = hilbert(audio)
audio_envelope = np.abs(analytic_signal)
# 平滑处理
window_size = int(sr * 0.05)  # 50ms窗口
audio_envelope_smooth = np.convolve(audio_envelope, np.ones(window_size)/window_size, mode='same')
print(f"  音频时长: {len(audio)/sr:.2f} 秒")

# ========== 4. 时间对齐 ==========
print("\n" + "=" * 60)
print("4. 时间对齐...")

# 创建统一的时间轴（以秒为单位）
eeg_duration = len(gfp) / EEG_SFREQ
audio_duration = len(audio) / sr
bs_fps = 30  # 假设30fps
bs_duration = len(bs_data) / bs_fps

print(f"  EEG时长: {eeg_duration:.2f}s, 音频时长: {audio_duration:.2f}s, BS时长: {bs_duration:.2f}s")

# 重采样到相同的时间分辨率（100Hz，方便计算）
target_fps = 100
target_samples = int(min(eeg_duration, audio_duration) * target_fps)

# 重采样 GFP
gfp_time = np.linspace(0, eeg_duration, len(gfp))
target_time = np.linspace(0, eeg_duration, target_samples)
gfp_resampled = np.interp(target_time, gfp_time, gfp)

# 重采样音频包络
audio_time = np.linspace(0, audio_duration, len(audio_envelope_smooth))
audio_resampled = np.interp(target_time, audio_time, audio_envelope_smooth)

# 重采样 Blendshape (使用嘴部相关的几个维度的能量)
# ARKit 中嘴部相关: jawOpen(17), mouthClose(25), mouthPucker(32), mouthSmile_L(35), mouthSmile_R(36)
mouth_indices = [17, 25, 32, 35, 36] if bs_data.shape[-1] >= 37 else list(range(min(5, bs_data.shape[-1])))
bs_mouth = bs_data[:, mouth_indices] if len(bs_data.shape) == 2 else bs_data[..., mouth_indices]
bs_energy = np.sqrt(np.sum(bs_mouth**2, axis=-1))
bs_time = np.linspace(0, bs_duration, len(bs_energy))
bs_resampled = np.interp(target_time, bs_time, bs_energy)

# ========== 5. 相关性分析 ==========
print("\n" + "=" * 60)
print("5. 相关性分析...")

# GFP vs Audio
corr_gfp_audio, p_gfp_audio = pearsonr(gfp_resampled, audio_resampled)
print(f"  EEG GFP vs 音频包络: r = {corr_gfp_audio:.4f}, p = {p_gfp_audio:.2e}")

# GFP vs Blendshape
corr_gfp_bs, p_gfp_bs = pearsonr(gfp_resampled, bs_resampled)
print(f"  EEG GFP vs 嘴部BS: r = {corr_gfp_bs:.4f}, p = {p_gfp_bs:.2e}")

# Audio vs Blendshape
corr_audio_bs, p_audio_bs = pearsonr(audio_resampled, bs_resampled)
print(f"  音频包络 vs 嘴部BS: r = {corr_audio_bs:.4f}, p = {p_audio_bs:.2e}")

# 评估
print("\n" + "-" * 60)
if abs(corr_gfp_audio) > 0.3:
    print("  ✅ EEG与音频有明显相关性！肌电信号可能被正确保留。")
elif abs(corr_gfp_audio) > 0.1:
    print("  ⚠️ EEG与音频有弱相关性，可能存在但不强。")
else:
    print("  ❌ EEG与音频几乎无相关性，需要进一步检查。")

# ========== 6. 可视化 ==========
print("\n" + "=" * 60)
print("6. 生成可视化...")

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
fig.suptitle('EEG/Audio/Blendshape Dynamics Comparison', fontsize=14)

# Plot 1: EEG GFP
ax1 = axes[0]
ax1.plot(target_time, gfp_resampled, 'b-', linewidth=0.8, alpha=0.8)
ax1.set_ylabel('EEG GFP\n(Std across channels)')
ax1.set_title(f'EEG Global Field Power (CV={np.std(gfp)/np.mean(gfp)*100:.1f}%)')
ax1.grid(True, alpha=0.3)

# Plot 2: Audio Envelope
ax2 = axes[1]
ax2.plot(target_time, audio_resampled, 'g-', linewidth=0.8, alpha=0.8)
ax2.set_ylabel('Audio Envelope')
ax2.set_title('Audio Envelope (Hilbert Transform)')
ax2.grid(True, alpha=0.3)

# Plot 3: Blendshape Energy
ax3 = axes[2]
ax3.plot(target_time, bs_resampled, 'r-', linewidth=0.8, alpha=0.8)
ax3.set_ylabel('Mouth BS Energy')
ax3.set_title('Blendshape Mouth Energy (RMS of jaw/mouth params)')
ax3.grid(True, alpha=0.3)

# Plot 4: Overlay (Normalized)
ax4 = axes[3]
# Normalize to [0, 1] for comparison
gfp_norm = (gfp_resampled - gfp_resampled.min()) / (gfp_resampled.max() - gfp_resampled.min() + 1e-8)
audio_norm = (audio_resampled - audio_resampled.min()) / (audio_resampled.max() - audio_resampled.min() + 1e-8)
bs_norm = (bs_resampled - bs_resampled.min()) / (bs_resampled.max() - bs_resampled.min() + 1e-8)

ax4.plot(target_time, gfp_norm, 'b-', linewidth=1, alpha=0.7, label=f'EEG GFP (r={corr_gfp_audio:.3f})')
ax4.plot(target_time, audio_norm, 'g-', linewidth=1, alpha=0.7, label='Audio')
ax4.plot(target_time, bs_norm, 'r-', linewidth=1, alpha=0.7, label=f'Mouth BS (r={corr_gfp_bs:.3f})')
ax4.set_xlabel('Time (seconds)')
ax4.set_ylabel('Normalized Signal')
ax4.set_title('Overlay: Normalized Signals Comparison')
ax4.legend(loc='upper right')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
print(f"  图表已保存到: {OUTPUT_PATH}")

# ========== 7. 结论 ==========
print("\n" + "=" * 60)
print("7. 结论")
print("=" * 60)
print(f"  数据形状: EEG [{eeg_data.shape}], Audio [{audio.shape}], BS [{bs_data.shape}]")
print(f"  EEG统计: 均值={np.mean(eeg_data):.3f}, 标准差={np.std(eeg_data):.3f}")
print(f"  GFP变异系数: {np.std(gfp)/np.mean(gfp)*100:.1f}%")
print(f"  相关性: EEG-Audio={corr_gfp_audio:.3f}, EEG-BS={corr_gfp_bs:.3f}, Audio-BS={corr_audio_bs:.3f}")

if np.std(gfp)/np.mean(gfp) > 0.2 and abs(corr_gfp_audio) > 0.1:
    print("\n  🎉 数据看起来良好！EEG信号有动态变化，与音频有一定相关性。")
else:
    print("\n  ⚠️ 需要进一步检查数据质量或预处理流程。")

print("=" * 60)
