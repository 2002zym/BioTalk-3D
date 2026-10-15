#!/usr/bin/env python3
"""
Verify that the first EEG marker aligns with the first sentence's audio start.

For每个被试：
1. 读取原始 CNT 文件，定位第一个 marker（默认优先 type '100'，否则取首个）。
2. 读取音频会话起始时间与句子相对时间（“标签 1.txt”）。
3. 计算 marker 的绝对时间以及首句音频的绝对时间。
4. 输出两者相差多少秒，用于检测配置是否合理。
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, Tuple

import mne
import numpy as np
import pytz

FPS = 60  # 默认视频/时间码帧率

PERSON_CONFIG = {
    'liu1': {
        'cnt': "/home/zym/EATalker/EATalker_dataset/eeg数据/eeg数据/liu1-10.39.12.551.cnt",
        'audio_start_time': "10:38:35:16.995",
        'time_file': "/home/zym/EATalker/EATalker_dataset/audio_liu1/标签 1.txt",
    },
    'liu2': {
        'cnt': "/home/zym/EATalker/EATalker_dataset/eeg数据/eeg数据/liu2-10.54.15.070.cnt",
        'audio_start_time': "10:53:38:51.926",
        'time_file': "/home/zym/EATalker/EATalker_dataset/audio_liu2/标签 1.txt",
    },
    'pj1': {
        'cnt': "/home/zym/EATalker/EATalker_dataset/eeg数据/eeg数据/pj1-22.13.40.109.cnt",
        'audio_start_time': "22:13:04:12.344",
        'time_file': "/home/zym/EATalker/EATalker_dataset/audio_pj1/标签 1.txt",
    },
    'pj2': {
        'cnt': "/home/zym/EATalker/EATalker_dataset/eeg数据/eeg数据/pj2-21.51.37.281.cnt",
        'audio_start_time': "21:51:00:17.838",
        'time_file': "/home/zym/EATalker/EATalker_dataset/audio_pj2/标签 1.txt",
    },
}


def parse_timecode(ts: str) -> float:
    """
    Parse HH:MM:SS:FF(.mmm) into seconds.
    """
    if not ts:
        return 0.0
    parts = ts.split(':')
    if len(parts) < 4:
        raise ValueError(f"时间码格式不正确: {ts}")
    hours, minutes, seconds = map(int, parts[:3])
    tail = parts[3]
    millis = 0.0
    if '.' in tail:
        frames_str, frac_str = tail.split('.', 1)
        millis = int(frac_str) / 1000.0
    else:
        frames_str = tail
    frames = int(frames_str)
    total = hours * 3600 + minutes * 60 + seconds
    if frames >= FPS:
        total += frames / 1000.0
    else:
        total += frames / FPS
    total += millis
    return total


def parse_timefile(path: Path) -> Dict[int, Tuple[float, float]]:
    """
    读取“标签 1.txt”，返回 {audio_id: (start, end)}。
    """
    result: Dict[int, Tuple[float, float]] = {}
    with path.open('r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    for line in lines[1:]:
        parts = line.split('\t')
        if len(parts) < 3:
            continue
        start = float(parts[0])
        end = float(parts[1])
        audio_id = int(parts[2])
        result[audio_id] = (start, end)
    if not result:
        raise RuntimeError(f"时间文件 {path} 中没有有效条目")
    return result


def normalize_meas_date(meas_date):
    """Convert meas_date to timezone-aware datetime (UTC)."""
    if meas_date is None:
        return None
    if isinstance(meas_date, dt.datetime):
        return meas_date if meas_date.tzinfo else meas_date.replace(
            tzinfo=dt.timezone.utc)
    if isinstance(meas_date, tuple) and len(meas_date) == 2:
        sec, usec = meas_date
        return dt.datetime.fromtimestamp(
            sec, tz=dt.timezone.utc) + dt.timedelta(microseconds=usec)
    try:
        base = np.datetime64(meas_date).astype('datetime64[us]').astype(
            'int64') / 1e6
        return dt.datetime.fromtimestamp(base, tz=dt.timezone.utc)
    except Exception:
        return None


def combine_with_reference(date_ref: dt.date,
                           time_of_day: dt.time) -> dt.datetime:
    """
    Combine date + time, adjusting for wrap-around (±12h) if necessary.
    """
    candidate = dt.datetime.combine(date_ref, time_of_day)
    return candidate


def extract_first_marker(raw: mne.io.BaseRaw, preferred_type: str = '100'):
    """
    Return (description, latency_samples, latency_seconds).
    """
    annotations = raw.annotations
    if len(annotations) == 0:
        raise RuntimeError("EEG 文件无 annotations，无法定位 marker")
    sfreq = float(raw.info['sfreq'])
    desc = annotations.description
    onset = annotations.onset

    # 先找指定 type
    idx = None
    if preferred_type is not None:
        for i, d in enumerate(desc):
            if d == preferred_type:
                idx = i
                break
    if idx is None:
        idx = 0
    latency_seconds = onset[idx]
    latency_samples = int(round(latency_seconds * sfreq))
    return desc[idx], latency_samples, latency_seconds


def main():
    parser = argparse.ArgumentParser(
        description="验证 EEG marker 与音频起点的对齐情况")
    parser.add_argument('--person',
                        type=str,
                        choices=PERSON_CONFIG.keys(),
                        default=None,
                        help="指定被试（默认全部）")
    args = parser.parse_args()

    selected = [args.person] if args.person else PERSON_CONFIG.keys()
    cn_tz = pytz.timezone('Asia/Shanghai')

    for person in selected:
        cfg = PERSON_CONFIG[person]
        cnt_path = Path(cfg['cnt'])
        if not cnt_path.exists():
            print(f"[{person}] CNT 文件不存在: {cnt_path}")
            continue
        raw = mne.io.read_raw_cnt(str(cnt_path), preload=False, verbose=False)
        sfreq = float(raw.info['sfreq'])
        meas_date = normalize_meas_date(raw.info.get('meas_date'))
        if meas_date is None:
            print(f"[{person}] meas_date 缺失，无法计算绝对时间，跳过")
            continue

        marker_type, latency_samples, latency_seconds = extract_first_marker(
            raw, preferred_type='100')
        marker_time = meas_date + dt.timedelta(seconds=latency_seconds)
        marker_bj = marker_time.astimezone(cn_tz)

        audio_start_seconds = parse_timecode(cfg['audio_start_time'])
        marker_local_date = marker_time.astimezone(cn_tz).date()
        base_dt = (dt.datetime.combine(marker_local_date, dt.time(0, 0)) +
                   dt.timedelta(seconds=audio_start_seconds))
        audio_start_local = cn_tz.localize(base_dt)
        base_time = audio_start_local.astimezone(dt.timezone.utc)

        time_file = Path(cfg['time_file'])
        audio_times = parse_timefile(time_file)
        first_id = min(audio_times.keys())
        first_start, _ = audio_times[first_id]
        first_start_local = audio_start_local + dt.timedelta(seconds=first_start)
        expected_first = first_start_local.astimezone(dt.timezone.utc)

        delta = (marker_time - expected_first).total_seconds()

        print(f"\n[{person}]")
        print(f"  CNT: {cnt_path.name}, sfreq={sfreq:.1f}Hz")
        print(f"  第一个 marker type: {marker_type}, "
              f"latency={latency_seconds:.3f}s, samples={latency_samples}")
        print(f"  第一个 marker 北京时间: {marker_bj.strftime('%Y-%m-%d %H:%M:%S.%f')}")
        print(f"  音频会话起点(绝对): {audio_start_local.strftime('%Y-%m-%d %H:%M:%S.%f')}")
        print(f"  首句 ID: {first_id}, 相对起点: {first_start:.3f}s")
        print(f"  首句绝对时间: {first_start_local.strftime('%Y-%m-%d %H:%M:%S.%f')}")
        print(f"  marker 与首句时间差: {delta:+.3f}s "
              f"({'marker 在首句之前' if delta < 0 else 'marker 在首句之后'})")


if __name__ == "__main__":
    main()
