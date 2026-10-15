import mne
import pandas as pd

# 读取脑电数据
raw = mne.io.read_raw_cnt(r'E:\EEG+Audio to Face\2025_7_3采集\eeg数据\eeg数据\liu1-10.39.12.551.cnt', preload=False)

# 获取基本信息
print("=== 脑电数据基本信息 ===")
print(f"采样率: {raw.info['sfreq']} Hz")
print(f"通道数量: {len(raw.info['ch_names'])}")
print(f"通道名称: {raw.info['ch_names']}")
print(f"数据时长: {raw.times[-1]:.2f} 秒")
print(f"测量日期: {raw.info['meas_date']}")

# 获取事件信息
events, event_id = mne.events_from_annotations(raw)
print(f"\n=== 事件信息 ===")
print(f"事件数量: {len(events)}")
print(f"事件类型: {event_id}")

# 将所有事件转换为真实时间
events_df = pd.DataFrame(events, columns=['onset', '0', 'id'])
events_df['time (UTC)'] = raw.info['meas_date'] + pd.to_timedelta(events_df['onset']/raw.info['sfreq'], unit='s')

print(f"\n=== 事件详情 ===")
print(events_df[['id', 'time (UTC)']])

# 统计事件类型
print(f"\n=== 事件统计 ===")
event_counts = events_df['id'].value_counts().sort_index()
for event_id, count in event_counts.items():
    print(f"事件ID {event_id}: {count} 次")

# 查看第一个和最后一个事件的时间
print(f"\n=== 时间范围 ===")
print(f"第一个事件: {events_df['time (UTC)'].iloc[0]}")
print(f"最后一个事件: {events_df['time (UTC)'].iloc[-1]}")
print(f"总时长: {(events_df['time (UTC)'].iloc[-1] - events_df['time (UTC)'].iloc[0]).total_seconds():.2f} 秒")