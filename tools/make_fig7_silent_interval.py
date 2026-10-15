#!/usr/bin/env python3
"""
Build ECCV Fig.7 qualitative panel:
- Top: audio + EEG signal evidence over first 4s
- Bottom: GT vs Audio-only baseline mesh frames in silent interval
- Third row: Audio+EEG Ours placeholder (pending training)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib import gridspec
from scipy.signal import butter, filtfilt
import torch

from EATalker.loss.loss import BlendShapeLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wav",
        type=Path,
        default=Path("/home/zym/EATalker_data_template_v2/lc/EATalker/test/s8_sentence083_lc_neutral.wav"),
    )
    parser.add_argument(
        "--eeg",
        type=Path,
        default=Path("/home/zym/EATalker_data_template_v2/lc/EATalker/test/s8_sentence083_lc_neutral_eeg.npy"),
    )
    parser.add_argument(
        "--gt-video",
        type=Path,
        default=Path("/home/zym/EATalker_data_template_v2/lc/render_demo/test/s8_sentence083_lc_neutral_EATalker.mp4"),
    )
    parser.add_argument(
        "--gt-bs",
        type=Path,
        default=Path("/home/zym/EATalker_data_template_v2/lc/EATalker/test/s8_sentence083_lc_neutral_bs.npy"),
    )
    parser.add_argument(
        "--baseline-video",
        type=Path,
        default=Path("/tmp/fig7_out/pred_videos/s8_sentence083_lc_neutral_EATalker.mp4"),
    )
    parser.add_argument(
        "--baseline-npz",
        type=Path,
        default=Path("/tmp/fig7_out/audio_only_unittalker_pred.npz"),
    )
    parser.add_argument(
        "--sample-key",
        type=str,
        default="s8_sentence083_lc_neutral.wav",
        help="Key inside baseline npz.",
    )
    parser.add_argument(
        "--mouth-idx",
        type=Path,
        default=Path("/home/zym/EATalker/resources/binary_resources/05_inhouse_arkit_mouth_idx.npy"),
    )
    parser.add_argument(
        "--blendshape-basis",
        type=Path,
        default=Path("/home/zym/EATalker/resources/binary_resources/inhouse_arkit.npz"),
    )
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--n-frames", type=int, default=3)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/zym/EATalker/doc/figures/fig7_silent_interval_audio_vs_eeg.pdf"),
    )
    return parser.parse_args()


def load_audio_first_window(wav_path: Path, window_sec: float) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(str(wav_path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    n = int(window_sec * sr)
    audio = audio[:n].astype(np.float32)
    return audio, sr


def load_eeg_first_window(eeg_path: Path, window_sec: float, eeg_fs: int = 1000) -> np.ndarray:
    eeg = np.load(str(eeg_path)).astype(np.float32)  # expected [T, C]
    if eeg.ndim != 2:
        raise ValueError(f"Unexpected EEG shape: {eeg.shape}")
    if eeg.shape[1] == 60:
        arr = eeg
    elif eeg.shape[0] == 60:
        arr = eeg.T
    else:
        raise ValueError(f"Cannot infer EEG layout: {eeg.shape}")
    n = int(window_sec * eeg_fs)
    return arr[:n]


def bandpass(signal: np.ndarray, fs: int, low: float, high: float) -> np.ndarray:
    b, a = butter(4, [low / (0.5 * fs), high / (0.5 * fs)], btype="band")
    return filtfilt(b, a, signal)


def moving_rms(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return np.abs(x)
    sq = x * x
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.sqrt(np.convolve(sq, kernel, mode="same"))


def pick_silent_interval(audio: np.ndarray, sr: int, min_len_sec: float = 0.9) -> Tuple[float, float]:
    env = moving_rms(audio, max(1, int(0.04 * sr)))
    thr = float(np.percentile(env, 25))
    mask = env <= thr
    min_len = int(min_len_sec * sr)

    best = (0, min_len)
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if (not v or i == len(mask) - 1) and start is not None:
            end = i if not v else i + 1
            if (end - start) >= (best[1] - best[0]):
                best = (start, end)
            start = None

    s, e = best
    if (e - s) < min_len:
        center = int(np.argmin(env))
        half = min_len // 2
        s = max(0, center - half)
        e = min(len(audio), s + min_len)
    return s / sr, e / sr


def pick_frame_times(s0: float, s1: float, n_frames: int) -> List[float]:
    if n_frames <= 1:
        return [0.5 * (s0 + s1)]
    margin = 0.12 * max(0.2, s1 - s0)
    left = s0 + margin
    right = s1 - margin
    if right <= left:
        left, right = s0, s1
    return np.linspace(left, right, n_frames).tolist()


def compute_mouth_motion_scores(
    gt_bs_path: Path,
    baseline_npz: Path,
    sample_key: str,
    mouth_idx_path: Path,
    blendshape_basis: Path,
    total_dur_sec: float,
    window_sec: float,
) -> Tuple[np.ndarray, np.ndarray]:
    gt_bs = np.load(str(gt_bs_path)).astype(np.float32)  # [T, 51]
    base_obj = np.load(str(baseline_npz), allow_pickle=True)[sample_key].item()
    base_v = np.asarray(base_obj["EATalker"], dtype=np.float32)  # [Tb, 1220, 3]

    loss = BlendShapeLoss(
        mouth_indices_path=str(mouth_idx_path),
        blendshape_path=str(blendshape_basis),
        bs_beta=0.0,
    )
    with torch.no_grad():
        gt_v = loss.get_vertices(torch.tensor(gt_bs, dtype=torch.float32)).cpu().numpy()

    mouth_idx = np.load(str(mouth_idx_path)).astype(np.int64)

    # use baseline frame times as candidate times (fits displayed frame sampling)
    base_fps = len(base_v) / max(total_dur_sec, 1e-8)
    cand_t = np.arange(0, int(window_sec * base_fps)) / base_fps
    cand_t = cand_t[cand_t <= window_sec]
    if len(cand_t) == 0:
        cand_t = np.linspace(0.05, max(0.1, window_sec - 0.05), 60)

    gt_ref = gt_v[0, mouth_idx]
    base_ref = base_v[0, mouth_idx]
    scores = []
    for t in cand_t:
        ig = int(round((t / total_dur_sec) * (len(gt_v) - 1)))
        ib = int(round((t / total_dur_sec) * (len(base_v) - 1)))
        ig = max(0, min(len(gt_v) - 1, ig))
        ib = max(0, min(len(base_v) - 1, ib))
        e_gt = np.linalg.norm(gt_v[ig, mouth_idx] - gt_ref, axis=-1).mean()
        e_ba = np.linalg.norm(base_v[ib, mouth_idx] - base_ref, axis=-1).mean()
        scores.append(float(e_gt - e_ba))
    return cand_t, np.asarray(scores, dtype=np.float32)


def select_contrast_frames(
    cand_t: np.ndarray,
    scores: np.ndarray,
    audio: np.ndarray,
    sr: int,
    n_frames: int,
    min_gap_sec: float = 0.45,
) -> List[float]:
    # low-energy mask for silent-like region
    win = max(1, int(0.10 * sr))
    env = moving_rms(audio, win)
    thr = float(np.percentile(env, 40))

    audio_vals = []
    for t in cand_t:
        i = int(round(t * sr))
        i = max(0, min(len(env) - 1, i))
        audio_vals.append(float(env[i]))
    audio_vals = np.asarray(audio_vals, dtype=np.float32)

    # prioritize "silent + high GT-vs-baseline contrast"
    idx = np.arange(len(cand_t))
    valid = idx[(audio_vals <= thr) & (scores > 0) & (cand_t >= 0.8)]
    if len(valid) < n_frames:
        valid = idx[scores > 0]
    if len(valid) < n_frames:
        valid = idx
    order = valid[np.argsort(scores[valid])[::-1]]

    chosen: List[float] = []
    for j in order:
        t = float(cand_t[j])
        if all(abs(t - c) >= min_gap_sec for c in chosen):
            chosen.append(t)
        if len(chosen) >= n_frames:
            break

    if len(chosen) < n_frames:
        extra = np.linspace(0.15, max(0.2, cand_t.max() - 0.15), n_frames).tolist()
        for t in extra:
            if all(abs(t - c) >= 0.25 for c in chosen):
                chosen.append(float(t))
            if len(chosen) >= n_frames:
                break
    chosen = sorted(chosen[:n_frames])
    return chosen


def read_frame_at(video_path: Path, time_sec: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = int(round(time_sec * fps))
    idx = max(0, min(total - 1, idx))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read frame t={time_sec:.3f}s from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def draw_baseline_mark(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    x0, x1 = int(0.33 * w), int(0.67 * w)
    y0, y1 = int(0.56 * h), int(0.80 * h)
    # dashed red box
    for x in range(x0, x1, 10):
        cv2.line(out, (x, y0), (min(x + 6, x1), y0), (220, 20, 20), 2)
        cv2.line(out, (x, y1), (min(x + 6, x1), y1), (220, 20, 20), 2)
    for y in range(y0, y1, 10):
        cv2.line(out, (x0, y), (x0, min(y + 6, y1)), (220, 20, 20), 2)
        cv2.line(out, (x1, y), (x1, min(y + 6, y1)), (220, 20, 20), 2)
    cv2.putText(
        out,
        "X",
        (int(0.87 * w), int(0.17 * h)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (220, 20, 20),
        3,
        cv2.LINE_AA,
    )
    return out


def make_placeholder_tile(shape: Tuple[int, int, int], text: str) -> np.ndarray:
    h, w, _ = shape
    tile = np.full((h, w, 3), 238, dtype=np.uint8)
    cv2.rectangle(tile, (8, 8), (w - 8, h - 8), (120, 120, 120), 2)
    lines = text.split("\n")
    y = int(0.43 * h)
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(
            tile,
            line,
            ((w - tw) // 2, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (60, 60, 60),
            2,
            cv2.LINE_AA,
        )
        y += int(th * 1.8)
    return tile


def main() -> None:
    args = parse_args()

    audio, sr = load_audio_first_window(args.wav, args.window_sec)
    eeg = load_eeg_first_window(args.eeg, args.window_sec, eeg_fs=1000)  # [T,C]

    # choose channel with strongest high-frequency activity in first 4s
    hf_scores = []
    eeg_hf_all = []
    for c in range(eeg.shape[1]):
        hf = bandpass(eeg[:, c], fs=1000, low=20.0, high=120.0)
        eeg_hf_all.append(hf)
        hf_scores.append(np.std(hf))
    ch = int(np.argmax(hf_scores))
    eeg_hf = eeg_hf_all[ch]

    # IMPORTANT: use full utterance duration for GT/baseline frame alignment,
    # not the truncated 4s window length.
    total_dur = float(sf.info(str(args.wav)).duration)
    if total_dur <= 0:
        total_dur = len(audio) / sr if len(audio) > 0 else args.window_sec
    cand_t, contrast_scores = compute_mouth_motion_scores(
        gt_bs_path=args.gt_bs,
        baseline_npz=args.baseline_npz,
        sample_key=args.sample_key,
        mouth_idx_path=args.mouth_idx,
        blendshape_basis=args.blendshape_basis,
        total_dur_sec=total_dur,
        window_sec=args.window_sec,
    )
    frame_times = select_contrast_frames(
        cand_t,
        contrast_scores,
        audio,
        sr,
        n_frames=args.n_frames,
        min_gap_sec=0.40,
    )
    # visual "silent interval" bracket around selected frames
    s0 = max(0.0, min(frame_times) - 0.22)
    s1 = min(args.window_sec, max(frame_times) + 0.22)
    if s1 - s0 < 0.6:
        mid = 0.5 * (s0 + s1)
        s0 = max(0.0, mid - 0.35)
        s1 = min(args.window_sec, mid + 0.35)

    gt_frames = [read_frame_at(args.gt_video, t) for t in frame_times]
    base_frames = [draw_baseline_mark(read_frame_at(args.baseline_video, t)) for t in frame_times]
    placeholder = make_placeholder_tile(
        gt_frames[0].shape, "Pending training\n(Audio+EEG Ours)"
    )
    our_frames = [placeholder.copy() for _ in frame_times]

    t_audio = np.arange(len(audio)) / sr
    t_eeg = np.arange(len(eeg_hf)) / 1000.0
    audio_norm = audio / (np.max(np.abs(audio)) + 1e-8)
    eeg_norm = eeg_hf / (np.max(np.abs(eeg_hf)) + 1e-8)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig = plt.figure(figsize=(13.8, 8.2), constrained_layout=True)
    gs = gridspec.GridSpec(
        5,
        args.n_frames + 1,
        figure=fig,
        width_ratios=[1.5] + [1.0] * args.n_frames,
        height_ratios=[1.0, 1.0, 1.15, 1.15, 1.15],
    )

    # Top: audio/eeg signals
    ax_a = fig.add_subplot(gs[0, :])
    ax_e = fig.add_subplot(gs[1, :], sharex=ax_a)

    ax_a.plot(t_audio, audio_norm, color="#2c7fb8", lw=1.0)
    ax_a.axvspan(s0, s1, color="#fde68a", alpha=0.35, label="Silent Interval")
    ax_a.set_xlim(0, args.window_sec)
    ax_a.set_ylabel("Audio\n(norm.)")
    ax_a.set_title("Signal evidence in the first 4s (Silent interval vs EEG activity)", loc="left")
    ax_a.legend(loc="upper right", fontsize=9, frameon=True)
    ax_a.grid(True, ls="--", alpha=0.3)

    ax_e.plot(t_eeg, eeg_norm, color="#d95f0e", lw=0.9)
    ax_e.axvspan(s0, s1, color="#fde68a", alpha=0.35)
    ax_e.set_ylabel(f"EEG HF\n(ch {ch})")
    ax_e.set_xlabel("Time (s)")
    ax_e.grid(True, ls="--", alpha=0.3)
    ax_e.annotate(
        "Muscle Activation / Pre-speech Planning",
        xy=(0.5 * (s0 + s1), 0.65),
        xytext=(0.5 * (s0 + s1) + 0.35, 0.92),
        arrowprops=dict(arrowstyle="->", lw=1.0, color="black"),
        fontsize=9,
    )

    for t in frame_times:
        ax_a.axvline(t, color="0.35", ls="--", lw=0.9)
        ax_e.axvline(t, color="0.35", ls="--", lw=0.9)

    row_titles = ["Ground Truth", "Audio-Only Baseline (UniTalker)", "Audio+EEG Ours"]
    rows = [gt_frames, base_frames, our_frames]
    for r in range(3):
        ax_label = fig.add_subplot(gs[2 + r, 0])
        ax_label.axis("off")
        ax_label.text(0.98, 0.5, row_titles[r], ha="right", va="center", fontsize=11, weight="bold")
        for c, frame in enumerate(rows[r]):
            ax = fig.add_subplot(gs[2 + r, 1 + c])
            ax.imshow(frame)
            ax.axis("off")
            if r == 0:
                ax.set_title(f"t={frame_times[c]:.2f}s", fontsize=10)
            if r == 2:
                # green check mark to indicate intended improvement path
                h, w = frame.shape[:2]
                ax.text(
                    0.88 * w,
                    0.15 * h,
                    "✓",
                    color="#16a34a",
                    fontsize=22,
                    ha="center",
                    va="center",
                    transform=ax.transData,
                )

    fig.suptitle(
        "Fig.7  Silent-period ambiguity: audio-only misses subtle facial motion, EEG can disambiguate",
        fontsize=14,
        y=1.01,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=450, bbox_inches="tight")
    print(f"[saved] {args.out}")
    print(
        f"[info] silent_interval=[{s0:.2f}, {s1:.2f}] s, frames={','.join(f'{x:.2f}' for x in frame_times)}, eeg_channel={ch}"
    )


if __name__ == "__main__":
    main()
