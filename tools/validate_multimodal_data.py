#!/usr/bin/env python3
"""
多模态数据验证脚本
===================
验证 Audio, Blendshape, EEG 三个模态的数据是否正确对齐和处理。

检查项：
1. 时长对齐（三模态时长是否一致）
2. EEG 预处理质量（50Hz是否去除，归一化范围）
3. Blendshape 数据有效性（形状、范围、动态）
"""
import argparse
import numpy as np
import wave
from pathlib import Path


def load_audio_duration(wav_path):
    """获取音频时长"""
    with wave.open(str(wav_path), 'rb') as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        duration = n_frames / sr
    return duration, sr, n_frames


def analyze_eeg(eeg_data, sfreq=1000):
    """分析 EEG 数据质量"""
    # 确保格式为 [Time, Channels] 或 [Channels, Time]
    if eeg_data.shape[0] > eeg_data.shape[1]:
        # [Time, Channels] -> [Channels, Time]
        eeg_data = eeg_data.T
    
    n_ch, n_t = eeg_data.shape
    duration = n_t / sfreq
    
    # 1. 基本统计
    stats = {
        'shape': eeg_data.shape,
        'duration': duration,
        'min': float(np.min(eeg_data)),
        'max': float(np.max(eeg_data)),
        'mean': float(np.mean(eeg_data)),
        'std': float(np.std(eeg_data)),
    }
    
    # 2. 检查 50Hz 噪声
    def check_50hz(data, fs):
        n = len(data)
        fft_val = np.fft.rfft(data)
        psd = np.abs(fft_val)**2 / n
        freqs = np.fft.rfftfreq(n, d=1.0/fs)
        
        idx_49_5 = np.argmin(np.abs(freqs - 49.5))
        idx_50_5 = np.argmin(np.abs(freqs - 50.5))
        idx_40 = np.argmin(np.abs(freqs - 40))
        idx_60 = np.argmin(np.abs(freqs - 60))
        
        peak_power = np.sum(psd[idx_49_5:idx_50_5])
        total_power = np.sum(psd[idx_40:idx_60])
        background = total_power - peak_power
        
        ratio = peak_power / (background + 1e-9)
        return ratio
    
    ratios = [check_50hz(eeg_data[i], sfreq) for i in range(min(5, n_ch))]
    stats['50hz_ratio'] = float(np.mean(ratios))
    
    # 3. GFP 动态
    gfp = np.std(eeg_data, axis=0)
    gfp_cv = np.std(gfp) / (np.mean(gfp) + 1e-9)
    stats['gfp_cv'] = float(gfp_cv)
    
    # 4. 归一化检查
    stats['in_range_20'] = float(np.mean(np.abs(eeg_data) <= 20.0) * 100)
    
    return stats


def analyze_blendshape(bs_data, fps=60):
    """分析 Blendshape 数据质量"""
    # 确保格式为 [Time, Channels]
    if bs_data.shape[1] > bs_data.shape[0]:
        bs_data = bs_data.T
    
    n_frames, n_dims = bs_data.shape
    duration = n_frames / fps
    
    stats = {
        'shape': bs_data.shape,
        'duration': duration,
        'n_frames': n_frames,
        'n_dims': n_dims,
        'min': float(np.min(bs_data)),
        'max': float(np.max(bs_data)),
        'mean': float(np.mean(bs_data)),
    }
    
    # 检查值域 (ARKit blendshapes should be in [0, 1])
    stats['in_range_01'] = float(np.mean((bs_data >= 0) & (bs_data <= 1)) * 100)
    
    # 检查动态变化
    velocity = np.diff(bs_data, axis=0)
    stats['mean_velocity'] = float(np.mean(np.abs(velocity)))
    stats['max_velocity'] = float(np.max(np.abs(velocity)))
    
    # 检查是否有全零维度
    zero_dims = np.sum(np.all(bs_data == 0, axis=0))
    stats['zero_dims'] = int(zero_dims)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Multimodal Data Validation')
    parser.add_argument('--audio', required=True, help='Audio WAV file')
    parser.add_argument('--bs', required=True, help='Blendshape NPY file')
    parser.add_argument('--eeg', required=True, help='EEG NPY file')
    parser.add_argument('--eeg-sfreq', type=int, default=1000, help='EEG sampling rate')
    parser.add_argument('--bs-fps', type=int, default=60, help='Blendshape FPS')
    args = parser.parse_args()
    
    print("=" * 60)
    print("多模态数据验证报告")
    print("=" * 60)
    
    # 1. Load and analyze audio
    print("\n📢 [1/3] 音频分析")
    print("-" * 40)
    audio_duration, audio_sr, audio_frames = load_audio_duration(args.audio)
    print(f"文件: {Path(args.audio).name}")
    print(f"采样率: {audio_sr} Hz")
    print(f"帧数: {audio_frames}")
    print(f"时长: {audio_duration:.3f} 秒")
    
    # 2. Load and analyze blendshape
    print("\n😊 [2/3] Blendshape 分析")
    print("-" * 40)
    bs_data = np.load(args.bs)
    bs_stats = analyze_blendshape(bs_data, args.bs_fps)
    print(f"文件: {Path(args.bs).name}")
    print(f"形状: {bs_stats['shape']} (帧数 x 维度)")
    print(f"时长: {bs_stats['duration']:.3f} 秒 (@ {args.bs_fps} FPS)")
    print(f"值域: [{bs_stats['min']:.4f}, {bs_stats['max']:.4f}]")
    print(f"在 [0,1] 范围内: {bs_stats['in_range_01']:.1f}%")
    print(f"平均速度: {bs_stats['mean_velocity']:.4f}")
    print(f"全零维度: {bs_stats['zero_dims']} / {bs_stats['n_dims']}")
    
    # 3. Load and analyze EEG
    print("\n🧠 [3/3] EEG 分析")
    print("-" * 40)
    eeg_data = np.load(args.eeg)
    eeg_stats = analyze_eeg(eeg_data, args.eeg_sfreq)
    print(f"文件: {Path(args.eeg).name}")
    print(f"形状: {eeg_stats['shape']} (通道 x 采样点)")
    print(f"时长: {eeg_stats['duration']:.3f} 秒 (@ {args.eeg_sfreq} Hz)")
    print(f"值域: [{eeg_stats['min']:.3f}, {eeg_stats['max']:.3f}]")
    print(f"均值: {eeg_stats['mean']:.3f}, 标准差: {eeg_stats['std']:.3f}")
    print(f"50Hz 浓度比: {eeg_stats['50hz_ratio']:.2f} {'✅ 已滤除' if eeg_stats['50hz_ratio'] < 1.0 else '⚠️ 可能残留'}")
    print(f"GFP CV: {eeg_stats['gfp_cv']:.4f}")
    print(f"在 [-20,20] 范围内: {eeg_stats['in_range_20']:.1f}%")
    
    # 4. Alignment check
    print("\n⏱️ [4/4] 时长对齐检查")
    print("-" * 40)
    
    durations = {
        'Audio': audio_duration,
        'Blendshape': bs_stats['duration'],
        'EEG': eeg_stats['duration']
    }
    
    for name, dur in durations.items():
        print(f"{name:12s}: {dur:.3f} 秒")
    
    # Calculate max difference
    dur_list = list(durations.values())
    max_diff = max(dur_list) - min(dur_list)
    
    print(f"\n最大时长差异: {max_diff:.3f} 秒 ({max_diff*1000:.1f} ms)")
    
    if max_diff < 0.05:
        print("✅ 时长对齐良好 (差异 < 50ms)")
    elif max_diff < 0.2:
        print("⚠️ 时长略有差异 (50-200ms)，可能需要检查")
    else:
        print("❌ 时长差异较大 (>200ms)，请检查数据处理流程")
    
    # Final summary
    print("\n" + "=" * 60)
    print("📊 验证总结")
    print("=" * 60)
    
    issues = []
    
    # Check EEG
    if eeg_stats['50hz_ratio'] >= 1.0:
        issues.append("EEG 50Hz 噪声可能未完全滤除")
    if eeg_stats['in_range_20'] < 99.0:
        issues.append(f"EEG 有 {100-eeg_stats['in_range_20']:.1f}% 数据超出 [-20,20] 范围")
    
    # Check Blendshape
    if bs_stats['n_dims'] != 51:
        issues.append(f"Blendshape 维度为 {bs_stats['n_dims']}，期望 51")
    if bs_stats['in_range_01'] < 95.0:
        issues.append(f"Blendshape 有 {100-bs_stats['in_range_01']:.1f}% 数据超出 [0,1] 范围")
    if bs_stats['zero_dims'] > 5:
        issues.append(f"Blendshape 有 {bs_stats['zero_dims']} 个全零维度")
    
    # Check alignment
    if max_diff >= 0.2:
        issues.append(f"时长差异过大: {max_diff:.3f}秒")
    
    if issues:
        print("⚠️ 发现以下问题:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    else:
        print("✅ 所有检查通过！数据处理正确。")


if __name__ == '__main__':
    main()
