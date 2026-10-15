#!/usr/bin/env python3
"""
EEG 质量检测脚本 v5 (支持滤波对比 + 可视化)
==============================================
检测 EEG 数据中的 50Hz 工频噪声，并评估信号的动态特性。

用法：
  python check_eeg_quality_v5.py --eeg xxx.npy --wav xxx.wav
  python check_eeg_quality_v5.py --eeg xxx.npy --wav xxx.wav --filter  # 先滤波再分析
"""
import argparse
import numpy as np
import wave


def parse_args():
    parser = argparse.ArgumentParser(description='EEG Quality Check with optional filtering')
    parser.add_argument('--eeg', required=True, help='EEG npy path')
    parser.add_argument('--wav', required=True, help='WAV path')
    parser.add_argument('--fs-eeg', type=int, default=1000, help='EEG Sampling Rate')
    parser.add_argument('--filter', action='store_true', help='Apply 50Hz notch filter before analysis')
    parser.add_argument('--output', default=None, help='Output figure path (auto-generated if not specified)')
    return parser.parse_args()


def notch_filter(data, fs=1000, freq=50.0, Q=60.0):
    """应用 50Hz 陷波滤波器"""
    from scipy.signal import iirnotch, filtfilt
    b, a = iirnotch(freq, Q, fs=fs)
    return filtfilt(b, a, data, axis=-1)


def analyze_50hz(data, fs):
    """计算 50Hz 浓度比"""
    n = len(data)
    fft_val = np.fft.rfft(data)
    psd = np.abs(fft_val)**2 / n
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    
    # Define narrow band around 50Hz (49.5 - 50.5 Hz)
    idx_49_5 = np.argmin(np.abs(freqs - 49.5))
    idx_50_5 = np.argmin(np.abs(freqs - 50.5))
    peak_power = np.sum(psd[idx_49_5:idx_50_5])

    # Define Broad band (40-60 Hz, excluding peak)
    idx_40 = np.argmin(np.abs(freqs - 40))
    idx_60 = np.argmin(np.abs(freqs - 60))
    total_power_band = np.sum(psd[idx_40:idx_60])
    background_power = total_power_band - peak_power
    
    concentration_ratio = peak_power / (background_power + 1e-9)
    return concentration_ratio, freqs, psd


def analyze_eeg(eeg, fs_eeg, label=""):
    """分析 EEG 信号质量"""
    n_ch, n_t = eeg.shape
    
    print(f"\n{'='*50}")
    print(f"Analysis: {label}")
    print(f"{'='*50}")
    
    # 1. Check Line Noise Concentration
    ratios = []
    all_psd = []
    for i in range(min(5, n_ch)):
        r, freqs, psd = analyze_50hz(eeg[i], fs_eeg)
        ratios.append(r)
        all_psd.append(psd)
    avg_ratio = np.mean(ratios)
    mean_psd = np.mean(all_psd, axis=0)
    
    print(f"50Hz Concentration Ratio: {avg_ratio:.2f}")
    if avg_ratio > 10.0:
        print("=> High confidence: LINE NOISE. (Signal is concentrated in <1Hz band)")
    elif avg_ratio > 1.0:
        print("=> Moderate noise. Some 50Hz component remains.")
    else:
        print("=> CLEAN! Possible Bio-signal. (Signal is spread across band)")

    # 2. Check GFP (Global Field Power) Dynamics
    gfp = np.std(eeg, axis=0)
    gfp_mean = np.mean(gfp)
    gfp_std = np.std(gfp)
    cv = gfp_std / (gfp_mean + 1e-9)  # Coefficient of Variation
    
    print(f"GFP Coeff of Variation (CV): {cv:.4f}")
    if cv < 0.1:
        print("=> GFP is nearly CONSTANT. Dominated by steady noise (e.g. 50Hz hum).")
        print("   Real speech EMG should have high variance (bursts).")
    else:
        print("=> GFP is dynamic. Likely contains speech-correlated signal.")
    
    return avg_ratio, cv, freqs, mean_psd, gfp


def plot_comparison(eeg_raw, eeg_filtered, fs_eeg, results_raw, results_filtered, output_path):
    """生成对比图"""
    import matplotlib.pyplot as plt
    
    ratio_raw, cv_raw, freqs_raw, psd_raw, gfp_raw = results_raw
    ratio_filtered, cv_filtered, freqs_filtered, psd_filtered, gfp_filtered = results_filtered
    
    n_t = eeg_raw.shape[1]
    t = np.arange(n_t) / fs_eeg
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('EEG Quality Check: Raw vs Filtered (50Hz Notch, Q=60)', fontsize=14, fontweight='bold')
    
    # Subplot 1: EEG Time Domain (Channel 0)
    ax1 = axes[0]
    ax1.plot(t, eeg_raw[0], color='red', linewidth=0.5, alpha=0.7, label='Raw')
    ax1.plot(t, eeg_filtered[0], color='blue', linewidth=0.5, alpha=0.7, label='Filtered')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Plot 1: EEG Time Domain (Channel 0)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: PSD Comparison
    ax2 = axes[1]
    ax2.semilogy(freqs_raw, psd_raw, color='red', linewidth=1.5, alpha=0.8, label=f'Raw (50Hz Ratio={ratio_raw:.2f})')
    ax2.semilogy(freqs_filtered, psd_filtered, color='blue', linewidth=1.5, alpha=0.8, label=f'Filtered (50Hz Ratio={ratio_filtered:.2f})')
    ax2.axvline(50, color='green', linestyle='--', linewidth=1, label='50Hz')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power (log scale)')
    ax2.set_title('Plot 2: Power Spectral Density (PSD) - 50Hz Notch Effect')
    ax2.set_xlim(0, 150)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: GFP Dynamics
    ax3 = axes[2]
    gfp_raw_norm = gfp_raw / (gfp_raw.max() + 1e-9)
    gfp_filtered_norm = gfp_filtered / (gfp_filtered.max() + 1e-9)
    ax3.plot(t, gfp_raw_norm, color='red', linewidth=1, alpha=0.7, label=f'Raw (CV={cv_raw:.4f})')
    ax3.plot(t, gfp_filtered_norm, color='blue', linewidth=1, alpha=0.7, label=f'Filtered (CV={cv_filtered:.4f})')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Normalized GFP')
    ax3.set_title('Plot 3: Global Field Power (GFP) Dynamics')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nFigure saved to: {output_path}")


def main():
    args = parse_args()
    eeg_raw = np.load(args.eeg).astype(np.float32)
    if eeg_raw.shape[0] > eeg_raw.shape[1]:
        eeg_raw = eeg_raw.T
    n_ch, n_t = eeg_raw.shape
    print(f"EEG shape: {n_ch} channels x {n_t} samples")
    
    if args.filter:
        # 对比模式：先分析原始，再分析滤波后
        print("\n" + "="*60)
        print("Comparison Mode: Raw EEG vs Filtered EEG")
        print("="*60)
        
        results_raw = analyze_eeg(eeg_raw, args.fs_eeg, label="Raw EEG")
        
        eeg_filtered = notch_filter(eeg_raw, fs=args.fs_eeg, freq=50.0, Q=60.0)
        results_filtered = analyze_eeg(eeg_filtered, args.fs_eeg, label="Filtered EEG (50Hz Notch, Q=60)")
        
        ratio_raw, cv_raw = results_raw[0], results_raw[1]
        ratio_filtered, cv_filtered = results_filtered[0], results_filtered[1]
        
        print("\n" + "="*60)
        print("Summary")
        print("="*60)
        print(f"50Hz Concentration Ratio: {ratio_raw:.2f} -> {ratio_filtered:.2f} (Reduced {(1 - ratio_filtered/ratio_raw)*100:.1f}%)")
        print(f"GFP CV: {cv_raw:.4f} -> {cv_filtered:.4f}")
        
        if ratio_filtered < 1.0:
            print("\n✅ Filter successful! 50Hz noise effectively suppressed.")
        else:
            print("\n⚠️ Residual noise remains. Check filter parameters.")
        
        # Generate figure
        if args.output:
            output_path = args.output
        else:
            output_path = args.eeg.replace('.npy', '_quality_check.png')
        plot_comparison(eeg_raw, eeg_filtered, args.fs_eeg, results_raw, results_filtered, output_path)
    else:
        # 仅分析原始
        analyze_eeg(eeg_raw, args.fs_eeg, label="Raw EEG")


if __name__ == '__main__':
    main()


