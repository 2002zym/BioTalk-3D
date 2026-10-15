#!/usr/bin/env python3
"""
EEG 数值分布诊断 (Histogram Check)
================================
绘制 EEG 数据的直方图，检查是否存在异常值导致的动态范围压缩。
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eeg', required=True, help='EEG npy path')
    parser.add_argument('--output', default='eeg_value_dist.png', help='Output plot path')
    args = parser.parse_args()
    
    # 1. Load Data
    eeg = np.load(args.eeg).astype(np.float32)
    # Ensure [Channels, Time]
    if eeg.shape[0] > eeg.shape[1]: 
        eeg = eeg.T
        
    print(f"Data shape: {eeg.shape}")
    print(f"Min: {eeg.min():.6f}")
    print(f"Max: {eeg.max():.6f}")
    print(f"Mean: {eeg.mean():.6f}")
    print(f"Std: {eeg.std():.6f}")
    
    # 2. Check for outliers
    threshold = eeg.mean() + 3 * eeg.std()
    outliers = eeg[np.abs(eeg) > threshold]
    print(f"Outliers (> 3std): {len(outliers)} / {eeg.size} ({len(outliers)/eeg.size*100:.2f}%)")

    # 3. Plot Histogram
    plt.figure(figsize=(10, 6))
    
    # Plot 1: Full Range
    plt.subplot(2, 1, 1)
    plt.hist(eeg.flatten(), bins=100, log=True, color='blue', alpha=0.7)
    plt.title(f'All Values Distribution (Log Scale)\nMin: {eeg.min():.2f}, Max: {eeg.max():.2f}')
    plt.xlabel('Value')
    plt.ylabel('Count (Log)')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Central Range (zoom in)
    plt.subplot(2, 1, 2)
    # Clip to 3 std dev for better visualization of the "body"
    limit = eeg.std() * 3
    data_clipped = eeg.flatten()
    data_clipped = data_clipped[np.abs(data_clipped) < limit]
    
    plt.hist(data_clipped, bins=100, color='green', alpha=0.7)
    plt.title(f'Central 3-Sigma Range Distribution\nRange: [-{limit:.2f}, +{limit:.2f}]')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Figure saved to: {args.output}")

if __name__ == '__main__':
    main()
