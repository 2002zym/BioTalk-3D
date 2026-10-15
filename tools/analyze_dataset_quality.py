#!/usr/bin/env python3
import json
import os
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import welch

def analyze_50hz(data, fs):
    # Simple PSD-based 50Hz detector
    # Returns concentration ratio (Peak power / Background power)
    n = len(data)
    fft_val = np.fft.rfft(data)
    psd = np.abs(fft_val)**2 / n
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    
    idx_49_5 = np.argmin(np.abs(freqs - 49.5))
    idx_50_5 = np.argmin(np.abs(freqs - 50.5))
    peak_power = np.sum(psd[idx_49_5:idx_50_5])

    idx_40 = np.argmin(np.abs(freqs - 40))
    idx_60 = np.argmin(np.abs(freqs - 60))
    total_power_band = np.sum(psd[idx_40:idx_60])
    background_power = total_power_band - peak_power
    
    return peak_power / (background_power + 1e-9)

def analyze_audio(path):
    try:
        sr, y = wavfile.read(path)
        # scipy reads as int16 usually, scale to float -1..1
        if y.dtype == np.int16:
            y = y.astype(np.float32) / 32768.0
        elif y.dtype == np.int32:
            y = y.astype(np.float32) / 2147483648.0
            
        if len(y.shape) > 1: y = y[:, 0] # mono
        rms = np.sqrt(np.mean(y**2))
        db = 20 * np.log10(rms + 1e-9)
        # Check clipping
        clipping_ratio = np.sum(np.abs(y) > 0.99) / len(y)
        return sr, db, clipping_ratio
    except Exception as e:
        print(f"Error reading audio {path}: {e}")
        return 0, -100, 0

def analyze_blendshape(path):
    try:
        bs = np.load(path) # shape: (frames, 52) usually
        # Check activity (Dynamic Range)
        if bs.shape[0] < 2: return 0, 0, 0
        
        # Velocity
        vel = np.diff(bs, axis=0)
        mean_vel = np.mean(np.abs(vel))
        
        # Acceleration (Jitter)
        acc = np.diff(vel, axis=0)
        mean_acc = np.mean(np.abs(acc))
        
        # Max amplitude (Expression intensity)
        max_amp = np.max(bs)
        
        return mean_vel, mean_acc, max_amp
    except:
        return 0, 0, 0

def main():
    root = "/Users/zym/EATalker_data_template/EATalker"
    train_json = os.path.join(root, "train.json")
    
    with open(train_json) as f:
        data = json.load(f)['data']
        
    print(f"Total samples: {len(data)}")
    
    # Random sample of 20
    indices = np.random.choice(len(data), 20, replace=False)
    
    metrics = {
        'audio_db': [],
        'audio_clip': [],
        'eeg_50hz': [],
        'eeg_dead': [],
        'bs_vel': [],
        'bs_acc': [],
        'bs_amp': []
    }
    
    for idx in indices:
        item = data[idx]
        # Paths
        aud_path = os.path.join(root, item['audio_path'])
        eeg_path = os.path.join(root, item['eeg_path'])
        bs_path = os.path.join(root, item['annot_path'])
        
        # Audio
        sr, db, clip = analyze_audio(aud_path)
        metrics['audio_db'].append(db)
        metrics['audio_clip'].append(clip)
        
        # EEG
        try:
            eeg = np.load(eeg_path)
            if eeg.shape[0] > eeg.shape[1]: eeg = eeg.T
            # Check 50Hz on first 5 channels
            ratios = [analyze_50hz(eeg[i], 1000) for i in range(min(5, eeg.shape[0]))]
            metrics['eeg_50hz'].append(np.mean(ratios))
            # Check dead channels
            stds = np.std(eeg, axis=1)
            metrics['eeg_dead'].append(np.sum(stds < 1e-6))
        except:
            pass
            
        # Blendshape
        vel, acc, amp = analyze_blendshape(bs_path)
        metrics['bs_vel'].append(vel)
        metrics['bs_acc'].append(acc)
        metrics['bs_amp'].append(amp)

    print("\n=== Dataset Quality Report (Sample N=20) ===")
    
    print("\n[Audio]")
    print(f"  Mean Volume: {np.mean(metrics['audio_db']):.1f} dB (Should be > -40)")
    print(f"  Samples with Clipping: {np.sum(np.array(metrics['audio_clip']) > 0)}")
    
    print("\n[EEG]")
    mean_50hz = np.mean(metrics['eeg_50hz'])
    print(f"  Mean 50Hz Noise Ratio: {mean_50hz:.2f}")
    if mean_50hz > 2.0:
        print("  ❌ SEVERE Line Noise detected across dataset.")
    else:
        print("  ✅ Line Noise controlled.")
        
    print(f"  Dead Channels (Mean): {np.mean(metrics['eeg_dead']):.1f} per sample")

    print("\n[Blendshapes (GT)]")
    print(f"  Mean Velocity (Dynamics): {np.mean(metrics['bs_vel']):.4f}")
    print(f"  Mean Acceleration (Jitter): {np.mean(metrics['bs_acc']):.4f}")
    print(f"  Max Amplitude (Expressiveness): {np.mean(metrics['bs_amp']):.2f}")
    
    if np.mean(metrics['bs_acc']) > 0.05: # Heuristic threshold
        print("  ⚠️  High Jitter in Ground Truth detected.")
    if np.mean(metrics['bs_amp']) < 0.3:
        print("  ⚠️  Low Expressiveness (Under-acting).")

if __name__ == "__main__":
    main()
