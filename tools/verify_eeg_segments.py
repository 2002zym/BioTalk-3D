#!/usr/bin/env python3
"""
Verify EEG absolute start/end against an utterance's audio absolute times by
anchoring on 一个已知对齐到某句音频起点（通常是句子 1）的 marker 事件。

Inputs:
  --audio      Path to utterance WAV，例如
               /home/zym/EATalker/EATalker_data_template/EATalker/audio/sentence1_liu1_happy.wav
  --eeg-raw    Path to original EEG record (SET/CNT)
  --time-table 会话的 '标签 1.txt'（包含每条音频相对 start/end 秒）
  --audio-start-time  音频会话的本地起始时刻（必填，用于设定 marker 的真实本地时刻）
  --event-index Marker 对应的事件索引（默认 1 → “第二个 marker”）
  --anchor-audio-id 将哪条音频视作 marker 对应的句子（默认 1，即第一条句子）

Outputs:
  - sentence1/2 的音频绝对起止时刻
  - sentence1/2 的 EEG 绝对起止时刻（基于对齐偏移模拟裁剪结果）
"""

import argparse
import os
import re
from datetime import datetime, timedelta

import mne


def normalise_timecode_delimiters(timecode: str) -> str:
    """允许使用冒号或点号分隔，最多前三个分隔符会被视作冒号。"""
    if ':' in timecode:
        return timecode
    parts = timecode.split('.')
    if len(parts) >= 4:
        core = ':'.join(parts[:4])
        if len(parts) == 4:
            return core
        tail = '.'.join(parts[4:])
        return f'{core}.{tail}'
    return timecode


def parse_timecode_to_seconds(timecode: str, fps: float = 60.0) -> float:
    """解析 HH:MM:SS:FF.mmm 或 HH:MM:SS:FFmmm 到秒。"""
    normalised = normalise_timecode_delimiters(timecode)
    m = re.match(r'^(\d+):(\d+):(\d+):(\d+)(?:[.](\d+))?$', normalised)
    if not m:
        raise ValueError(f"无法解析时间码: {timecode}")
    h, mi, s, frame_part, frac_part = m.groups()
    hours = int(h)
    minutes = int(mi)
    seconds = int(s)
    frames = int(frame_part)
    total = hours * 3600 + minutes * 60 + seconds
    if frac_part is not None:
        # 标准格式 FF.mmm
        total += frames / fps
        total += int(frac_part) / 1000.0
    else:
        # 无小数部分：若数值明显大于 fps，则视为毫秒，否则视为帧
        if frames >= fps:
            total += frames / 1000.0
        else:
            total += frames / fps
    return total


def seconds_to_time(sec: float):
    sec = sec % 86400
    hours = int(sec // 3600)
    minutes = int((sec % 3600) // 60)
    seconds_float = sec % 60
    seconds_int = int(seconds_float)
    micro = int(round((seconds_float - seconds_int) * 1_000_000))
    if micro >= 1_000_000:
        micro -= 1_000_000
        seconds_int += 1
    if seconds_int >= 60:
        seconds_int -= 60
        minutes += 1
    if minutes >= 60:
        minutes -= 60
        hours = (hours + 1) % 24
    return datetime(2000, 1, 1, hours, minutes, seconds_int, micro).time()


def read_time_table(table_path: str, audio_id: int):
    with open(table_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    for line in lines[1:]:  # skip header
        parts = line.split('\t')
        if len(parts) >= 3 and int(parts[2]) == audio_id:
            start = float(parts[0])
            end = float(parts[1])
            return start, end
    raise ValueError(f'audio_id {audio_id} not found in {table_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio', required=True)
    ap.add_argument('--eeg-raw', required=True)
    ap.add_argument('--time-table', required=True)
    ap.add_argument('--audio-start-time', required=True,
                    help='整段会话的音频起始本地时刻（例如 10:38:35:16.995）')
    ap.add_argument('--event-index', type=int, default=1,
                    help='用于对齐的 marker 事件索引（默认 1 → 第二个事件）')
    ap.add_argument('--anchor-audio-id', type=int, default=1,
                    help='marker 对应的句子编号（默认 1，sentence1）')
    args = ap.parse_args()

    base = os.path.basename(args.audio)
    m = re.match(r'^sentence(\d+)_([a-z0-9]+)_(\w+)\.wav$', base)
    if not m:
        raise SystemExit(f'无法解析音频文件名: {base}')
    audio_id = int(m.group(1))

    raw = mne.io.read_raw_eeglab(args.eeg_raw, preload=False)
    sfreq = float(raw.info['sfreq'])

    events, _ = mne.events_from_annotations(raw)
    if len(events) <= args.event_index:
        raise SystemExit(f'事件索引 {args.event_index} 超出范围（共 {len(events)} 个事件）')
    marker_onset_sec = float(events[args.event_index, 0]) / sfreq

    rel_start, rel_end = read_time_table(args.time_table, audio_id)
    anchor_rel_start, _ = read_time_table(args.time_table, args.anchor_audio_id)

    base_date = datetime.today().date()
    audio_start_seconds = parse_timecode_to_seconds(args.audio_start_time)
    audio_start_time_obj = seconds_to_time(audio_start_seconds)
    audio_session_start = datetime.combine(base_date, audio_start_time_obj)

    marker_local = audio_session_start + timedelta(seconds=anchor_rel_start)
    eeg_start_local = marker_local - timedelta(seconds=marker_onset_sec)

    alignment_offset = marker_onset_sec - anchor_rel_start

    print(f'EEG sfreq        : {sfreq:.1f} Hz')
    print(f'Event index      : {args.event_index} (onset {marker_onset_sec:.3f}s)')
    print(f'Anchor audio id  : {args.anchor_audio_id}')
    print(f'Anchor rel start : {anchor_rel_start:.3f}s')
    print(f'Marker (local)   : {marker_local.strftime("%H:%M:%S.%f")[:-3]}')
    print(f'Alignment offset : {alignment_offset:+.6f}s (marker_onset - anchor_rel_start)')
    print('---')

    target_ids = [1, 2]
    results = {}

    for tid in target_ids:
        try:
            start_rel, end_rel = read_time_table(args.time_table, tid)
        except ValueError:
            raise SystemExit(f'标签文件中找不到 sentence{tid}')
        audio_start = audio_session_start + timedelta(seconds=start_rel)
        audio_end = audio_session_start + timedelta(seconds=end_rel)
        eeg_start = eeg_start_local + timedelta(seconds=start_rel + alignment_offset)
        eeg_end = eeg_start_local + timedelta(seconds=end_rel + alignment_offset)
        results[tid] = {
            'audio_start': audio_start,
            'audio_end': audio_end,
            'eeg_start': eeg_start,
            'eeg_end': eeg_end,
        }

    for tid in target_ids:
        info = results[tid]
        prefix = f'sentence{tid}'
        print(f'{prefix} audio start : {info["audio_start"].strftime("%H:%M:%S.%f")[:-3]}')
        print(f'{prefix} audio end   : {info["audio_end"].strftime("%H:%M:%S.%f")[:-3]}')
        print(f'{prefix} EEG start   : {info["eeg_start"].strftime("%H:%M:%S.%f")[:-3]}')
        print(f'{prefix} EEG end     : {info["eeg_end"].strftime("%H:%M:%S.%f")[:-3]}')
        if tid != target_ids[-1]:
            print('---')


if __name__ == '__main__':
    main()
