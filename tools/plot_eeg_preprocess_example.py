#!/usr/bin/env python3
"""
Generate an EEG preprocessing example figure for paper use:
  1) raw
  2) 1 Hz high-pass + 50 Hz notch
  3) robust z-score
Plus an inset PSD plot (before/after notch) around 50 Hz.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mne
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.signal import butter, filtfilt, iirnotch, welch


DEFAULT_SET = "/home/zym/EaTalker_Dataset_v2/lc/EEG/recon_eeg/lc1.set"
DEFAULT_OUT = "/home/zym/EATalker/doc/figures/eeg_preprocess_example.pdf"


def highpass(data: np.ndarray, fs: float, cutoff: float = 1.0, order: int = 5) -> np.ndarray:
    nyq = fs * 0.5
    b, a = butter(order, cutoff / nyq, btype="high", analog=False)
    return filtfilt(b, a, data, axis=-1)


def notch50(data: np.ndarray, fs: float, freq: float = 50.0, q: float = 60.0) -> np.ndarray:
    b, a = iirnotch(freq, q, fs=fs)
    return filtfilt(b, a, data, axis=-1)


def robust_zscore(data: np.ndarray) -> np.ndarray:
    med = np.median(data, axis=-1, keepdims=True)
    q75 = np.percentile(data, 75, axis=-1, keepdims=True)
    q25 = np.percentile(data, 25, axis=-1, keepdims=True)
    iqr = q75 - q25
    iqr[iqr == 0] = 1.0
    scale = iqr / 1.349
    out = (data - med) / scale
    return np.clip(out, -20.0, 20.0)


def pick_sentence_segment(raw: mne.io.BaseRaw, sentence_id: int) -> tuple[int, int]:
    onsets = np.asarray(raw.annotations.onset, dtype=np.float64)
    i = 2 * (sentence_id - 1)
    if i + 1 >= len(onsets):
        raise ValueError(
            f"sentence_id={sentence_id} needs markers [{i},{i + 1}], but only {len(onsets)} markers found."
        )
    sfreq = float(raw.info["sfreq"])
    start = int(round(onsets[i] * sfreq))
    end = int(round(onsets[i + 1] * sfreq))
    if end <= start:
        raise ValueError(f"Invalid marker pair for sentence_id={sentence_id}: start={start}, end={end}")
    return start, end


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set-path", type=str, default=DEFAULT_SET)
    parser.add_argument("--sentence-id", type=int, default=77, help="1-based sentence index in marker pairs")
    parser.add_argument(
        "--wave-channel",
        type=str,
        default="FPZ",
        help="Single channel for waveform rows, e.g., FT7",
    )
    parser.add_argument(
        "--psd-channel",
        type=str,
        default="auto",
        help="Channel for PSD: 'same' (use wave channel), channel name, or 'auto' for strongest 50Hz attenuation.",
    )
    parser.add_argument("--zoom-start-sec", type=float, default=1.2, help="Zoom window start time (s)")
    parser.add_argument("--zoom-dur-sec", type=float, default=1.6, help="Zoom window duration (s)")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_path = Path(args.set_path)
    out_path = Path(args.out)

    raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    all_ch = raw.info["ch_names"]

    ch_to_idx = {c.upper(): i for i, c in enumerate(all_ch)}
    wave_name = args.wave_channel.strip().upper()
    if wave_name not in ch_to_idx:
        raise ValueError(f"Wave channel not found: {wave_name}. Available examples: {all_ch[:15]}")
    wave_idx = ch_to_idx[wave_name]

    start, end = pick_sentence_segment(raw, args.sentence_id)
    data = raw.get_data()[:, start:end]  # raw signal

    filtered = notch50(highpass(data, fs=sfreq, cutoff=1.0, order=5), fs=sfreq, freq=50.0, q=60.0)
    normalized = robust_zscore(filtered)

    t = np.arange(data.shape[1], dtype=np.float64) / sfreq

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 7.4), sharex=True, constrained_layout=True)
    color_main = "#1f77b4"

    for ax, sig, title, ylabel in [
        (axes[0], data, "Raw EEG", "Amplitude"),
        (axes[1], filtered, "After 1 Hz high-pass + 50 Hz notch", "Amplitude"),
        (axes[2], normalized, "After robust z-score (clipped to [-20, 20])", "Normalized"),
    ]:
        ax.plot(t, sig[wave_idx], lw=1.0, color=color_main, alpha=0.95, label=all_ch[wave_idx])
        ax.set_title(title, loc="left", pad=6)
        ax.set_ylabel(ylabel)
        ax.grid(True, ls="--", lw=0.5, alpha=0.35)

    z0 = float(max(0.0, min(args.zoom_start_sec, t[-1])))
    z1 = float(max(z0 + 0.5, min(z0 + args.zoom_dur_sec, t[-1])))
    for ax in axes:
        ax.axvspan(z0, z1, color="#f2b179", alpha=0.18, lw=0)

    axes[0].legend(loc="upper right", frameon=True, framealpha=0.9, title=f"Channel: {all_ch[wave_idx]}")
    axes[2].set_xlabel("Time (s)")

    # Add zoom inset on raw waveform to make morphology visible.
    inz = inset_axes(axes[0], width="37%", height="52%", loc="lower right", borderpad=1.0)
    mask_zoom = (t >= z0) & (t <= z1)
    inz.plot(t[mask_zoom], data[wave_idx][mask_zoom], lw=1.1, color=color_main)
    inz.set_title("Zoom", fontsize=9)
    inz.grid(True, ls=":", lw=0.4, alpha=0.35)
    inz.tick_params(axis="both", labelsize=8)

    # PSD inset: same channel by default; can use auto strongest 50Hz attenuation.
    psd_mode = args.psd_channel.strip().upper()
    if psd_mode == "AUTO":
        best_idx = 0
        best_red = -np.inf
        for i in range(data.shape[0]):
            f_tmp, p0_tmp = welch(data[i], fs=sfreq, nperseg=min(4096, data.shape[1]))
            _, p1_tmp = welch(filtered[i], fs=sfreq, nperseg=min(4096, data.shape[1]))
            i50_tmp = int(np.argmin(np.abs(f_tmp - 50.0)))
            red_tmp = 10.0 * np.log10((p0_tmp[i50_tmp] + 1e-12) / (p1_tmp[i50_tmp] + 1e-12))
            if red_tmp > best_red:
                best_red = red_tmp
                best_idx = i
        psd_idx = best_idx
    elif psd_mode == "SAME":
        psd_idx = wave_idx
    else:
        psd_name = psd_mode
        if psd_name not in ch_to_idx:
            raise ValueError(f"PSD channel not found: {psd_name}")
        psd_idx = ch_to_idx[psd_name]

    f_raw, p_raw = welch(data[psd_idx], fs=sfreq, nperseg=min(4096, data.shape[1]))
    f_flt, p_flt = welch(filtered[psd_idx], fs=sfreq, nperseg=min(4096, data.shape[1]))
    band = (f_raw >= 1.0) & (f_raw <= 100.0)
    p_raw_db = 10.0 * np.log10(p_raw + 1e-12)
    p_flt_db = 10.0 * np.log10(p_flt + 1e-12)
    i50 = int(np.argmin(np.abs(f_raw - 50.0)))
    reduce_db = 10.0 * np.log10((p_raw[i50] + 1e-12) / (p_flt[i50] + 1e-12))

    ins = inset_axes(axes[2], width="42%", height="55%", loc="upper right", borderpad=1.1)
    ins.plot(
        f_raw[band],
        p_raw_db[band],
        lw=1.8,
        color="black",
        linestyle="--",
        alpha=0.9,
        label="Raw PSD",
        zorder=3,
    )
    ins.plot(
        f_flt[band],
        p_flt_db[band],
        lw=1.8,
        color="#2ca02c",
        linestyle="-",
        alpha=0.9,
        label="Filtered PSD",
        zorder=4,
    )
    ins.axvline(50.0, color="red", ls="--", lw=1.0)
    ins.set_title(f"PSD ({all_ch[psd_idx]}), notch@50Hz", fontsize=9)
    ins.set_xlabel("Hz", fontsize=8)
    ins.set_ylabel("PSD (dB/Hz)", fontsize=8)
    ins.tick_params(axis="both", labelsize=8)
    ins.grid(True, ls=":", lw=0.4, alpha=0.35)
    ins.legend(fontsize=7, loc="upper right")
    ins.text(
        0.03,
        0.05,
        f"50Hz attenuation: {reduce_db:.1f} dB",
        transform=ins.transAxes,
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.9),
    )

    duration = (end - start) / sfreq
    fig.suptitle(
        f"EEG Preprocessing Example | {set_path.name}, sentence={args.sentence_id}, "
        f"wave={all_ch[wave_idx]}, PSD={all_ch[psd_idx]}, duration={duration:.3f}s",
        fontsize=12.5,
        y=1.01,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=450, bbox_inches="tight")
    print(f"[saved] {out_path}")
    print(
        f"[info] set={set_path}, sentence_id={args.sentence_id}, wave_channel={all_ch[wave_idx]}, "
        f"psd_channel={all_ch[psd_idx]}, sfreq={sfreq}, notch_50Hz_atten={reduce_db:.2f} dB"
    )


if __name__ == "__main__":
    main()
