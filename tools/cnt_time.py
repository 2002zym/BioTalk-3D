#!/usr/bin/env python3
"""
Inspect CNT events and markers.

This script prints:
- Total number of annotations (markers)
- Count of each annotation type (string description)
- Whether the first marker's type equals "100"
- Absolute Beijing time of the first marker (if meas_date is available)
- Code/type mapping from events_from_annotations
"""

from collections import Counter
import datetime
from pathlib import Path

import mne
import numpy as np
import pytz


def normalize_meas_date(meas_date):
    """Return timezone-aware datetime from various meas_date formats."""
    if meas_date is None:
        return None
    if isinstance(meas_date, datetime.datetime):
        return meas_date if meas_date.tzinfo else meas_date.replace(
            tzinfo=datetime.timezone.utc)
    if isinstance(meas_date, tuple) and len(meas_date) == 2:
        sec, usec = meas_date
        return datetime.datetime.fromtimestamp(
            sec, tz=datetime.timezone.utc) + datetime.timedelta(
                microseconds=usec)
    try:
        base = np.datetime64(meas_date).astype('datetime64[us]').astype(
            'int64') / 1e6
        return datetime.datetime.fromtimestamp(
            base, tz=datetime.timezone.utc)
    except Exception:
        return None


def main():
    cnt_path = Path(
        "/home/zym/EATalker/EATalker_dataset/eeg数据/eeg数据/liu1-10.39.12.551.cnt"
    )
    raw = mne.io.read_raw_cnt(cnt_path, preload=False, verbose=False)
    sfreq = float(raw.info['sfreq'])
    meas_date = normalize_meas_date(raw.info.get('meas_date'))
    cn_tz = pytz.timezone('Asia/Shanghai')

    annotations = raw.annotations
    print(f"总标注/事件条数: {len(annotations)}")
    if len(annotations):
        counts = Counter(annotations.description)
        print("各 type 计数:", counts)
        first_type = annotations.description[0]
        print("第一个标注 type:", first_type, "是否为 '100':", first_type == '100')
    else:
        print("⚠️ 没有 annotations，无法判断 marker 类型。")

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    print(f"events_from_annotations 事件数: {len(events)}")
    if len(events):
        inv_map = {code: desc for desc, code in event_id.items()}
        first_evt = events[0]
        first_code = int(first_evt[2])
        first_type = inv_map.get(first_code, str(first_code))
        print("第一个事件 type:", first_type, "code:", first_code,
              "是否为 '100':", first_type == '100')

        if meas_date is not None:
            latency = int(first_evt[0])
            event_time = meas_date + datetime.timedelta(seconds=latency /
                                                        sfreq)
            bj_time = event_time.astimezone(cn_tz)
            print("第一个事件 北京时间:", bj_time.strftime("%Y-%m-%d %H:%M:%S.%f"))
        else:
            print("meas_date 缺失，无法给出绝对时间（可用 latency/sfreq 表示相对时间）")

        codes, counts = np.unique(events[:, 2], return_counts=True)
        print("事件码集合:", codes.tolist())
        print("各事件码计数:",
              {int(code): int(cnt) for code, cnt in zip(codes, counts)})
    else:
        print("⚠️ events_from_annotations 未解析到事件。")


if __name__ == "__main__":
    main()
