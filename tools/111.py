import mne
import datetime
import pytz
# 读取 .cnt 文件
raw = mne.io.read_raw_cnt("/home/zym/EATalker/EATalker_dataset/eeg数据/eeg数据/liu1-10.39.12.551.cnt", preload=False)

# 提取事件信息
events, event_id = mne.events_from_annotations(raw)

sfreq = raw.info['sfreq']           # 采样率
start_time = raw.info['meas_date']  # 系统绝对时间（datetime 对象）

# 遍历所有事件
for e in events:
    latency = e[0]  # 采样点编号
    event_time = start_time + datetime.timedelta(seconds=latency / sfreq)
    beijing_time = event_time.astimezone(pytz.timezone('Asia/Shanghai'))
    print("Recording start (Beijing Time):", beijing_time.strftime("%Y-%m-%d %H:%M:%S.%f"))