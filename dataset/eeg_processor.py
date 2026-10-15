"""
EEG数据处理器 - 用于EATalker项目的脑电信号预处理

功能：
1. 读取EEGLAB预处理后的SET文件
2. 转换为模型输入格式 [B, C, T] 或 [B, T, C]
3. 根据音频时间信息进行精确分段
4. 保存为标准numpy格式供训练使用

作者：基于预处理SET文件的EEG处理器
"""

import numpy as np
import mne
from pathlib import Path
from typing import List, Tuple, Optional
try:
    import pandas as pd  # noqa: F401
except ImportError:  # pandas 仅在个别分析流程中使用，若缺失可跳过
    pd = None
from datetime import datetime, timedelta
import logging
import h5py  # 添加h5py支持MATLAB v7.3文件
import os  # 添加os模块用于路径操作

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EEGProcessor:
    """
    EEG数据处理器 - 处理EEGLAB预处理后的数据

    处理流程：
    1. 读取SET文件（已预处理：电极定位、重参考）
    2. 获取数据和时间信息
    3. 转换为模型输入格式
    4. 根据音频时间分段
    5. 保存为.npy文件
    """

    def __init__(self,
                 sfreq: float = 1000.0,  # 原始采样率
                 eeg_channels: int = 60,
                 time_window: float = 2.0,
                 overlap: float = 0.5,
                 target_sfreq: float = 1000.0):  # 目标采样率 (1000Hz 保留完整 500Hz 带宽)
        self.sfreq = sfreq
        self.eeg_channels = eeg_channels
        self.time_window = time_window
        self.overlap = overlap
        self.target_sfreq = target_sfreq
        self.samples_per_window = int(time_window * sfreq)
        self.step_size = int(self.samples_per_window * (1 - overlap))

    def load_set_file(self, set_file_path: str) -> Tuple[np.ndarray, dict]:
        """
        读取EEGLAB预处理后的SET文件

        Args:
            set_file_path: SET文件路径

        Returns:
            data: EEG数据 [channels, time_points]
            info: 数据信息字典
        """
        logger.info(f"读取SET文件: {set_file_path}")

        try:
            # 首先尝试使用mne的标准方法读取
            raw = mne.io.read_raw_eeglab(set_file_path, preload=True)

            # 获取数据
            data = raw.get_data()  # [channels, time_points]

            # 获取信息
            info = {
                'sfreq': raw.info['sfreq'],
                'ch_names': raw.info['ch_names'],
                'n_channels': len(raw.info['ch_names']),
                'duration': raw.times[-1],
                'meas_date': raw.info['meas_date']
            }

        except Exception as e:
            logger.warning(f"标准EEGLAB读取失败: {e}")
            logger.info("尝试使用HDF5读取MATLAB v7.3格式...")

            # 尝试使用HDF5读取MATLAB v7.3格式
            try:
                data, info = self._load_matlab_v73(set_file_path)
            except Exception as e2:
                logger.error(f"HDF5读取也失败: {e2}")
                raise e

        logger.info(f"原始数据形状: {data.shape}")
        logger.info(f"采样率: {info['sfreq']} Hz")
        logger.info(f"通道数: {info['n_channels']}")
        logger.info(f"时长: {info['duration']:.2f} 秒")

        return data, info

    def _load_matlab_v73(self, set_file_path: str) -> Tuple[np.ndarray, dict]:
        """
        使用HDF5读取MATLAB v7.3格式的SET文件（EEGLAB格式）
        处理SET+FDT文件对，其中SET包含元数据，FDT包含实际EEG数据
        """
        logger.info(f"使用HDF5读取MATLAB v7.3 SET文件: {set_file_path}")

        with h5py.File(set_file_path, 'r') as f:
            logger.info(f"SET文件键: {list(f.keys())}")

            # 读取基本参数
            if 'srate' not in f:
                raise ValueError("SET文件中找不到采样率(srate)")
            sfreq = float(f['srate'][0, 0])
            logger.info(f"采样率: {sfreq} Hz")

            if 'nbchan' not in f:
                raise ValueError("SET文件中找不到通道数(nbchan)")
            n_channels = int(f['nbchan'][0, 0])
            logger.info(f"通道数: {n_channels}")

            if 'pnts' not in f:
                raise ValueError("SET文件中找不到采样点数(pnts)")
            n_samples = int(f['pnts'][0, 0])
            logger.info(f"采样点数: {n_samples}")

            # 获取FDT文件名
            if 'data' not in f:
                raise ValueError("SET文件中找不到data字段")

            data_ref = f['data']
            if isinstance(data_ref, h5py.Dataset):
                # data字段应该是字符引用，指向FDT文件名
                fdt_filename_bytes = data_ref[...]
                if fdt_filename_bytes.dtype == np.uint16:
                    # 将uint16数组转换为字符串
                    fdt_filename = ''.join([chr(x) for x in fdt_filename_bytes.flatten() if x != 0])
                    logger.info(f"FDT文件名: {fdt_filename}")
                else:
                    raise ValueError(f"未预期的data数据类型: {fdt_filename_bytes.dtype}")
            else:
                raise ValueError("data字段不是Dataset类型")

            # 构建FDT文件路径
            set_dir = os.path.dirname(set_file_path)
            fdt_path = os.path.join(set_dir, fdt_filename)

            if not os.path.exists(fdt_path):
                raise ValueError(f"FDT文件不存在: {fdt_path}")

            logger.info(f"读取FDT文件: {fdt_path}")

            # 读取FDT二进制数据
            expected_size = n_channels * n_samples * 4  # float32 = 4字节
            actual_size = os.path.getsize(fdt_path)

            if actual_size != expected_size:
                logger.warning(f"FDT文件大小不匹配: 期望{expected_size}, 实际{actual_size}")
                # 尝试根据实际大小调整
                n_samples_actual = actual_size // (n_channels * 4)
                logger.warning(f"调整采样点数为: {n_samples_actual}")
                n_samples = n_samples_actual

            # 读取EEG数据
            with open(fdt_path, 'rb') as fdt_file:
                eeg_data = np.frombuffer(fdt_file.read(), dtype=np.float32)

            # reshape为 [通道, 时间] 格式
            eeg_data = eeg_data.reshape(n_samples, n_channels).T  # 转置为 [通道, 时间]

            logger.info(f"EEG数据形状: {eeg_data.shape}")
            logger.info(f"数据范围: [{np.min(eeg_data):.3f}, {np.max(eeg_data):.3f}]")
            logger.info(f"数据均值: {np.mean(eeg_data):.3f}")

            # 读取通道名称
            ch_names = []
            if 'chanlocs' in f:
                try:
                    chanlocs = f['chanlocs']
                    if isinstance(chanlocs, h5py.Group):
                        # 检查是否有labels字段
                        if 'labels' in chanlocs:
                            labels = chanlocs['labels']
                            # 处理通道标签（可能是引用结构）
                            for i in range(n_channels):
                                try:
                                    if str(i) in labels:
                                        label_ref = labels[str(i)]
                                        if isinstance(label_ref, h5py.Dataset):
                                            label_data = label_ref[...]
                                            if label_data.dtype == np.uint16:
                                                label_str = ''.join([chr(x) for x in label_data.flatten() if x != 0])
                                                ch_names.append(label_str)
                                            else:
                                                ch_names.append(str(label_data))
                                except Exception as e:
                                    logger.warning(f"读取通道{i}标签失败: {e}")
                                    ch_names.append(f'CH{i+1}')
                        else:
                            logger.warning("chanlocs中没有labels字段")
                    else:
                        logger.warning("chanlocs不是Group类型")
                except Exception as e:
                    logger.warning(f"读取通道位置信息失败: {e}")

            if not ch_names:
                ch_names = [f'CH{i+1}' for i in range(n_channels)]

            logger.info(f"通道名称: {ch_names[:5]}...")  # 显示前5个

            # 计算记录时长
            duration = n_samples / sfreq

            info = {
                'sfreq': sfreq,
                'ch_names': ch_names,
                'n_channels': n_channels,
                'duration': duration,
                'n_samples': n_samples,
                'meas_date': None,
                'fdt_filename': fdt_filename
            }

            return eeg_data, info

    def _butter_highpass_filter(self, data, cutoff=0.5, fs=1000.0, order=5):
        """高通滤波器，去除直流漂移"""
        from scipy.signal import butter, filtfilt
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return filtfilt(b, a, data, axis=-1)

    def _notch_filter(self, data, freq=50.0, fs=1000.0, Q=60.0):
        """50Hz 陷波滤波器"""
        from scipy.signal import iirnotch, filtfilt
        b, a = iirnotch(freq, Q, fs=fs)
        return filtfilt(b, a, data, axis=-1)

    def preprocess_data(self, data: np.ndarray, original_sfreq: float) -> np.ndarray:
        """
        预处理EEG数据
        1. 1.0Hz 高通滤波 (去基线)
        2. 50Hz 陷波滤波 (去工频)
        3. Per-Channel Robust Z-Score 归一化
        4. (可选) 重采样到 target_sfreq (默认 1000Hz 不降采样)
        """
        logger.info("开始EEG数据预处理...")
        
        # 0. 严格通道校验 [C, T]
        if data.shape[0] != self.eeg_channels:
            raise ValueError(
                f"EEG channel mismatch: expected {self.eeg_channels}, got {data.shape[0]}"
            )

        # 1. 滤波处理
        logger.info(f"应用滤波: Highpass(1.0Hz) + Notch(50Hz, Q=60)")
        data = self._butter_highpass_filter(data, cutoff=1.0, fs=original_sfreq)
        data = self._notch_filter(data, freq=50.0, fs=original_sfreq, Q=60.0)

        # 2. Per-Channel Robust Z-Score 归一化
        logger.info("应用 Per-Channel Robust Z-Score 归一化...")
        median = np.median(data, axis=-1, keepdims=True)
        q75 = np.percentile(data, 75, axis=-1, keepdims=True)
        q25 = np.percentile(data, 25, axis=-1, keepdims=True)
        iqr = q75 - q25
        iqr[iqr == 0] = 1.0
        scale = iqr / 1.349
        data = (data - median) / scale
        
        # 3. 软截断
        data = np.clip(data, -20.0, 20.0)

        # 4. 重采样 (Resample)
        if self.target_sfreq != original_sfreq:
            logger.info(f"重采样: {original_sfreq}Hz -> {self.target_sfreq}Hz")
            from scipy.signal import resample
            num_samples = int(data.shape[1] * self.target_sfreq / original_sfreq)
            data = resample(data, num_samples, axis=-1)

        logger.info(f"预处理完成: Shape={data.shape}, Mean={np.mean(data):.3f}")
        return data.astype(np.float32)

    def convert_to_model_format(self, data: np.ndarray, target_format: str = 'BCT') -> np.ndarray:
        """
        转换数据为模型输入格式

        Args:
            data: EEG数据 [channels, time_points]
            target_format: 目标格式 'BCT' [batch, channels, time] 或 'BTC' [batch, time, channels]

        Returns:
            formatted_data: 格式化后的数据
        """
        logger.info(f"转换为模型格式: {target_format}")

        # 添加batch维度
        data = np.expand_dims(data, axis=0)  # [1, channels, time_points]

        if target_format == 'BTC':
            data = np.transpose(data, (0, 2, 1))  # [1, time_points, channels]

        return data

    def segment_by_audio_time(self,
                            eeg_data: np.ndarray,
                            audio_start_times: List[str],
                            audio_end_times: List[str],
                            eeg_start_time_utc: datetime,
                            output_dir: str,
                            subject_name: str) -> List[str]:
        """
        根据音频时间信息分段EEG数据
        """
        logger.info(f"根据音频时间分段EEG数据: {len(audio_start_times)} 个片段")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        eeg_file_paths = []

        # 使用处理后的实际长度计算时长，此时数据采样率已是 target_sfreq
        total_time_points = eeg_data.shape[1]
        total_duration = total_time_points / self.target_sfreq

        # EEG开始时间（本地时间）
        eeg_start_time_local = eeg_start_time_utc + timedelta(hours=8)

        for i, (start_time_str, end_time_str) in enumerate(zip(audio_start_times, audio_end_times)):

            try:
                # 解析时间字符串（格式：时:分:秒:帧.毫秒）
                start_time = self._parse_timecode(start_time_str)
                end_time = self._parse_timecode(end_time_str)

                # 计算相对于EEG recording的时间偏移
                start_offset = (start_time - eeg_start_time_local).total_seconds()
                end_offset = (end_time - eeg_start_time_local).total_seconds()

                # 检查时间范围
                if start_offset < 0 or end_offset > total_duration:
                    logger.warning(f"第{i+1}个片段时间超出EEG数据范围，跳过")
                    continue

                # 转换为样本点索引 (使用 target_sfreq，因为数据已被重采样)
                start_sample = int(start_offset * self.target_sfreq)
                end_sample = int(end_offset * self.target_sfreq)

                # 提取EEG片段
                eeg_segment = eeg_data[:, start_sample:end_sample]

                # 保存为.npy文件
                output_filename = f"sentence_{i+1:03d}_{subject_name}_eeg.npy"
                output_path = output_dir / output_filename

                np.save(output_path, eeg_segment)
                eeg_file_paths.append(str(output_path))

                logger.info(f"保存EEG片段 {i+1}: {output_filename} - 形状: {eeg_segment.shape}")

            except Exception as e:
                logger.error(f"处理第{i+1}个片段时出错: {str(e)}")
                continue

        logger.info(f"成功分段 {len(eeg_file_paths)} 个EEG数据")
        return eeg_file_paths

    def _parse_timecode(self, timecode_str: str) -> datetime:
        """
        解析时间码字符串（格式：时:分:秒:帧.毫秒）

        Args:
            timecode_str: 时间码字符串，如 "10:38:35:16.995"

        Returns:
            datetime对象
        """
        # 假设输入的是本地时间字符串
        # 格式：时:分:秒:帧.毫秒 -> 转换为时:分:秒.毫秒
        parts = timecode_str.split(':')
        if len(parts) == 4:  # 时:分:秒:帧.毫秒
            hours, minutes, seconds, frame_ms = parts
            # 移除帧信息，只保留毫秒
            seconds = f"{seconds}.{frame_ms.split('.')[1]}"
        else:
            hours, minutes, seconds = parts

        time_str = f"{hours}:{minutes}:{seconds}"

        # 创建一个datetime对象（日期不重要，只关心时间）
        base_date = datetime(2024, 1, 1)
        time_obj = datetime.strptime(time_str, "%H:%M:%S.%f").time()

        return datetime.combine(base_date, time_obj)

    def process_single_subject(self,
                             set_file_path: str,
                             audio_timing_file: str,
                             output_dir: str,
                             subject_name: str,
                             eeg_start_time_utc_str: str) -> List[str]:
        """
        处理单个被试的完整EEG数据

        Args:
            set_file_path: SET文件路径
            audio_timing_file: 音频时间信息文件路径
            output_dir: 输出目录
            subject_name: 被试名称
            eeg_start_time_utc_str: EEG recording开始时间（UTC时间字符串）

        Returns:
            eeg_file_paths: 生成的EEG文件路径列表
        """
        logger.info(f"开始处理被试: {subject_name}")

        # 1. 读取SET文件
        eeg_data, info = self.load_set_file(set_file_path)

        # 2. 预处理数据
        eeg_data = self.preprocess_data(eeg_data, info['sfreq'])

        # 3. 读取音频时间信息
        audio_start_times, audio_end_times = self._load_audio_timing(audio_timing_file)

        # 4. 解析EEG开始时间
        eeg_start_time_utc = datetime.fromisoformat(eeg_start_time_utc_str.replace('Z', '+00:00'))

        # 5. 分段EEG数据
        eeg_file_paths = self.segment_by_audio_time(
            eeg_data=eeg_data,
            audio_start_times=audio_start_times,
            audio_end_times=audio_end_times,
            eeg_start_time_utc=eeg_start_time_utc,
            output_dir=output_dir,
            subject_name=subject_name
        )

        logger.info(f"被试 {subject_name} 处理完成，生成 {len(eeg_file_paths)} 个EEG文件")
        return eeg_file_paths

    def validate_output_format(self, data: np.ndarray, expected_format: str = 'BCT') -> bool:
        """
        验证输出格式是否满足EEG编码器要求

        Args:
            data: 输出数据
            expected_format: 期望格式 'BCT' 或 'BTC'

        Returns:
            bool: 是否满足要求
        """
        logger.info("验证输出格式...")

        # 检查维度
        if len(data.shape) != 3:
            logger.error(f"数据维度错误: 期望3维，实际{len(data.shape)}维")
            return False

        # 检查格式
        if expected_format == 'BCT':
            batch, channels, time = data.shape
            if channels != self.eeg_channels:
                logger.error(f"通道数错误: 期望{self.eeg_channels}，实际{channels}")
                return False
        elif expected_format == 'BTC':
            batch, time, channels = data.shape
            if channels != self.eeg_channels:
                logger.error(f"通道数错误: 期望{self.eeg_channels}，实际{channels}")
                return False

        # 检查数值范围 (Robust Z-Score 后范围约 [-20, 20])
        if np.max(np.abs(data)) > 20.0:
            logger.warning(f"数值范围超出[-20, 20]: 最大绝对值{np.max(np.abs(data))}")

        # 检查是否有NaN或Inf
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            logger.error("数据包含NaN或Inf值")
            return False

        logger.info(f"格式验证通过: 形状{data.shape}, 范围[{np.min(data):.3f}, {np.max(data):.3f}]")
        return True

    def _load_audio_timing(self, timing_file_path: str) -> Tuple[List[str], List[str]]:
        """
        加载音频时间信息文件

        Args:
            timing_file_path: 时间信息文件路径

        Returns:
            start_times: 开始时间列表
            end_times: 结束时间列表
        """
        start_times = []
        end_times = []

        with open(timing_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and '\t' in line:
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        start_time = parts[0]
                        end_time = parts[1]
                        start_times.append(start_time)
                        end_times.append(end_time)

        logger.info(f"加载音频时间信息: {len(start_times)} 个片段")
        return start_times, end_times


def main():
    """主函数 - 测试EEG处理器"""

    # 配置参数
    processor = EEGProcessor(
        sfreq=1000.0,      # 原始采样率
        target_sfreq=100.0,  # 目标采样率
        eeg_channels=64,    # EEG通道数
        time_window=2.0,    # 时间窗口
        overlap=0.5        # 重叠比例
    )

    # 示例：处理单个被试
    set_file_path = "path/to/your/set_file.set"
    audio_timing_file = "path/to/your/timing_file.txt"
    output_dir = "path/to/output/directory"
    subject_name = "liu1"
    eeg_start_time_utc = "2024-07-03T10:39:12.551Z"

    if Path(set_file_path).exists():
        eeg_files = processor.process_single_subject(
            set_file_path=set_file_path,
            audio_timing_file=audio_timing_file,
            output_dir=output_dir,
            subject_name=subject_name,
            eeg_start_time_utc_str=eeg_start_time_utc
        )
        print(f"生成了 {len(eeg_files)} 个EEG文件")
    else:
        print("请设置正确的文件路径进行测试")


if __name__ == "__main__":
    main()
