#!/usr/bin/env python3
import numpy as np
import sys

def analyze_50hz(data, fs):
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
    return concentration_ratio

def notch_filter(data, fs, freq=50.0, Q=60.0):
    try:
        from scipy.signal import filtfilt, iirnotch
        print("✅ Scipy found, using iirnotch.")
        b, a = iirnotch(freq, Q, fs=fs)
        if data.shape[0] > data.shape[1]: 
             return filtfilt(b, a, data, axis=-1)
        else:
             return filtfilt(b, a, data, axis=-1)
    except ImportError:
        print("❌ ERROR: Scipy not found! The filter cannot run.")
        print("Please install scipy: pip install scipy")
        sys.exit(1)

def main():
    eeg_path = '/Users/zym/EATalker/zym_test/sentence5_liu1_happy_eeg.npy'
    print(f"Loading {eeg_path}...")
    try:
        eeg = np.load(eeg_path)
    except FileNotFoundError:
        print("File not found.")
        return

    if eeg.shape[0] > eeg.shape[1]: eeg = eeg.T
    n_ch, n_t = eeg.shape
    fs = 1000

    print(f"Data shape: {eeg.shape}")

    # 1. Analyze BEFORE
    ratios_before = []
    for i in range(min(5, n_ch)):
        r = analyze_50hz(eeg[i], fs)
        ratios_before.append(r)
    avg_before = np.mean(ratios_before)
    print(f"\n[BEFORE FILTER] 50Hz Concentration Ratio: {avg_before:.2f}")

    # 2. Apply Filter
    print(f"\nApplying Notch Filter (50Hz, Q=60)...")
    eeg_filtered = notch_filter(eeg, fs, freq=50.0, Q=60.0)

    # 3. Analyze AFTER
    ratios_after = []
    for i in range(min(5, n_ch)):
        r = analyze_50hz(eeg_filtered[i], fs)
        ratios_after.append(r)
    avg_after = np.mean(ratios_after)
    print(f"[AFTER FILTER]  50Hz Concentration Ratio: {avg_after:.2f}")

    # 4. Conclusion
    if avg_before > 10 and avg_after < 2:
        print("\n✅ SUCCESS: Noise removed! Signal is now clean.")
        reduction = avg_before / avg_after
        print(f"Noise reduction factor: {reduction:.1f}x")
    elif avg_after >= 2:
        print("\n⚠️  WARNING: Significant 50Hz energy remains. Filter might be too weak or mismatched.")
    else:
        print("\nℹ️  Note: Baseline noise was already low.")

if __name__ == '__main__':
    main()
