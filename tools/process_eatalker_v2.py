#!/usr/bin/env python3
"""
Process EaTalker_Dataset_v2 subjects into EATalker-compatible files.

Stage A:
- Build aligned triplets (audio / blendshape / eeg) from raw sessions.
- Output base assets and all_full.json for manual quality review.

Stage B:
- Remove dirty samples by sample-id.
- Re-split remaining samples with per-session, per-emotion 8:1:1.
- Emit train/val/test JSON and split folders.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import subprocess
import sys
import wave
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple
import xml.etree.ElementTree as ET

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


D6_ORDER = [
    "EyeBlinkLeft",
    "EyeBlinkRight",
    "EyeSquintLeft",
    "EyeSquintRight",
    "EyeLookDownLeft",
    "EyeLookDownRight",
    "EyeLookInLeft",
    "EyeLookInRight",
    "EyeWideLeft",
    "EyeWideRight",
    "EyeLookOutLeft",
    "EyeLookOutRight",
    "EyeLookUpLeft",
    "EyeLookUpRight",
    "BrowDownLeft",
    "BrowDownRight",
    "BrowInnerUp",
    "BrowOuterUpLeft",
    "BrowOuterUpRight",
    "JawOpen",
    "MouthClose",
    "JawLeft",
    "JawRight",
    "JawForward",
    "MouthUpperUpLeft",
    "MouthUpperUpRight",
    "MouthLowerDownLeft",
    "MouthLowerDownRight",
    "MouthRollUpper",
    "MouthRollLower",
    "MouthSmileLeft",
    "MouthSmileRight",
    "MouthDimpleLeft",
    "MouthDimpleRight",
    "MouthStretchLeft",
    "MouthStretchRight",
    "MouthFrownLeft",
    "MouthFrownRight",
    "MouthPressLeft",
    "MouthPressRight",
    "MouthPucker",
    "MouthFunnel",
    "MouthLeft",
    "MouthRight",
    "MouthShrugLower",
    "MouthShrugUpper",
    "NoseSneerLeft",
    "NoseSneerRight",
    "CheekPuff",
    "CheekSquintLeft",
    "CheekSquintRight",
]

EMOTION_CN_TO_EN = {
    "高兴": "happy",
    "生气": "angry",
    "伤心": "sad",
    "中性": "neutral",
    "恐惧": "scared",
}

XLSX_NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "p": "http://schemas.openxmlformats.org/package/2006/relationships",
}

DEFAULT_SESSION_SHEETS = ("session1", "session2", "session3", "session4")

VALID_EEG_PREPROCESS = {"strict", "none"}
VALID_TIMECODE_REPAIR = {"stitch", "none"}
VALID_TIMECODE_MODE = {"raw", "ff_only", "robust", "adaptive"}


@dataclass
class SessionInfo:
    sheet_name: str
    audio_dir_name: str
    eeg_index: int
    slate_take: int


def _parse_timecode_parts(timecode: str) -> Tuple[int, int, int, int, int | None]:
    m = re.match(r"^(\d+):(\d+):(\d+):(\d+)(?:\.(\d+))?$", timecode.strip())
    if not m:
        raise ValueError(f"Invalid timecode: {timecode}")
    hh, mm, ss, ff, msec = m.groups()
    return int(hh), int(mm), int(ss), int(ff), (int(msec) if msec is not None else None)


def parse_timecode_to_seconds(
    timecode: str, fps: float = 60.0, include_msec: bool = True
) -> float:
    """
    Parse HH:MM:SS:FF(.mmm) to seconds.
    """
    hh, mm, ss, ff, msec = _parse_timecode_parts(timecode)
    out = hh * 3600 + mm * 60 + ss + ff / float(fps)
    if include_msec and msec is not None:
        # ARKit CSV stores time as HH:MM:SS:FF.mmm, where `.mmm` is fractional frame.
        out += (msec / 1000.0) / float(fps)
    return out


def build_timecode_seconds_sequence(
    timecodes: Sequence[str], fps: float = 60.0, mode: str = "robust"
) -> List[float]:
    """
    Convert a timecode sequence to seconds with robust handling of unstable `.mmm`.

    Modes:
      - raw: use HH:MM:SS:FF.mmm directly
      - ff_only: ignore `.mmm`, use HH:MM:SS:FF only
      - robust: per-sample choose between raw / ff_only to keep a stable monotonic timeline
      - adaptive: prefer raw on 30fps-like steps (~33.3ms), robust on stable 60fps steps
    """
    mode_norm = (mode or "robust").strip().lower()
    if mode_norm not in VALID_TIMECODE_MODE:
        raise ValueError(f"Unsupported timecode mode: {mode}")
    if not timecodes:
        return []

    base = [parse_timecode_to_seconds(tc, fps=fps, include_msec=False) for tc in timecodes]
    raw = [parse_timecode_to_seconds(tc, fps=fps, include_msec=True) for tc in timecodes]

    if mode_norm == "raw":
        return raw
    if mode_norm == "ff_only":
        return base

    # Robust mode: keep monotonicity and realistic frame cadence while preserving ff jumps.
    exp_dt = 1.0 / float(fps)
    max_reasonable_dt = 4.5 * exp_dt
    robust_out: List[float] = [raw[0]]
    for i in range(1, len(timecodes)):
        prev = robust_out[-1]
        cand_raw = raw[i]
        cand_base = base[i]
        dt_raw = cand_raw - prev
        dt_base = cand_base - prev

        raw_ok = dt_raw > 0 and dt_raw <= max_reasonable_dt
        base_ok = dt_base > 0 and dt_base <= max_reasonable_dt

        if raw_ok and base_ok:
            # Both are plausible: pick the one closer to expected cadence.
            chosen = cand_raw if abs(dt_raw - exp_dt) <= abs(dt_base - exp_dt) else cand_base
        elif raw_ok:
            chosen = cand_raw
        elif base_ok:
            chosen = cand_base
        else:
            # Outlier path: suppress extreme jumps while keeping monotonicity.
            if dt_base > 0:
                chosen = prev + min(dt_base, max_reasonable_dt)
            else:
                chosen = prev + exp_dt

        robust_out.append(chosen)
    if mode_norm == "robust":
        return robust_out

    # Adaptive mode:
    # - 30fps-like step (~33.3ms): trust raw
    # - stable 60fps step (~16.7ms): trust robust
    # - fallback: trust raw, then rely on stitch to patch rare outliers
    raw_arr = np.asarray(raw, dtype=np.float64)
    robust_arr = np.asarray(robust_out, dtype=np.float64)
    raw_dt = np.diff(raw_arr)
    robust_dt = np.diff(robust_arr)
    tol_60 = 0.003
    tol_30 = 0.004

    out: List[float] = [float(raw_arr[0])]
    for dt_raw, dt_robust in zip(raw_dt, robust_dt):
        raw_is_30 = abs(dt_raw - 2.0 * exp_dt) <= tol_30
        robust_is_60 = abs(dt_robust - exp_dt) <= tol_60
        chosen_dt = float(dt_raw if (raw_is_30 or not robust_is_60) else dt_robust)
        if chosen_dt <= 0:
            chosen_dt = float(exp_dt)
        out.append(out[-1] + chosen_dt)
    return out


def summarize_timecode_sequence(seconds: Sequence[float], fps: float = 60.0) -> Dict[str, float | int]:
    if len(seconds) < 2:
        return {
            "count": int(len(seconds)),
            "exp_dt_ms": float(1000.0 / float(fps)),
            "mean_dt_ms": 0.0,
            "std_dt_ms": 0.0,
            "max_dt_ms": 0.0,
            "min_dt_ms": 0.0,
            "neg_count": 0,
            "gt_40ms_count": 0,
        }
    arr = np.asarray(seconds, dtype=np.float64)
    dts = np.diff(arr)
    return {
        "count": int(len(seconds)),
        "exp_dt_ms": float(1000.0 / float(fps)),
        "mean_dt_ms": float(np.mean(dts) * 1000.0),
        "std_dt_ms": float(np.std(dts) * 1000.0),
        "max_dt_ms": float(np.max(dts) * 1000.0),
        "min_dt_ms": float(np.min(dts) * 1000.0),
        "neg_count": int(np.sum(dts < 0)),
        "gt_40ms_count": int(np.sum(dts > 0.04)),
    }


def repair_timecode_jump_stitch(
    seconds: Sequence[float],
    fps: float = 60.0,
    dt_threshold_sec: float = 0.04,
    min_run_len: int = 3,
) -> Tuple[List[float], Dict[str, object]]:
    """
    Repair large-delta runs by stitching timeline cadence back to expected frame step.
    """
    if len(seconds) < 2:
        return list(seconds), {
            "applied": False,
            "jump_idx": None,
            "jump_size_ms": 0.0,
            "applied_offset_ms": 0.0,
            "segments": [],
        }

    arr = np.asarray(seconds, dtype=np.float64)
    dts = np.diff(arr)
    exp_dt = 1.0 / float(fps)

    runs: List[Tuple[int, int]] = []
    start = None
    for i, dt in enumerate(dts):
        if dt > dt_threshold_sec:
            if start is None:
                start = i
        else:
            if start is not None:
                runs.append((start, i - 1))
                start = None
    if start is not None:
        runs.append((start, len(dts) - 1))

    runs = [(s, e) for (s, e) in runs if (e - s + 1) >= int(min_run_len)]
    if not runs:
        return list(arr.tolist()), {
            "applied": False,
            "jump_idx": None,
            "jump_size_ms": 0.0,
            "applied_offset_ms": 0.0,
            "segments": [],
        }

    repair_idx: Set[int] = set()
    segments: List[Dict[str, float | int]] = []
    for s, e in runs:
        repair_idx.update(range(s, e + 1))
        seg = dts[s : e + 1]
        segments.append(
            {
                "start_idx": int(s),
                "end_idx": int(e),
                "length": int(e - s + 1),
                "mean_dt_ms": float(np.mean(seg) * 1000.0),
                "max_dt_ms": float(np.max(seg) * 1000.0),
            }
        )

    out = [float(arr[0])]
    applied_offset = 0.0
    for i, dt in enumerate(dts):
        new_dt = float(exp_dt if i in repair_idx else dt)
        if new_dt <= 0:
            new_dt = float(exp_dt)
        applied_offset += float(dt - new_dt)
        out.append(out[-1] + new_dt)

    first_idx = runs[0][0]
    return out, {
        "applied": True,
        "jump_idx": int(first_idx),
        "jump_size_ms": float(dts[first_idx] * 1000.0),
        "applied_offset_ms": float(applied_offset * 1000.0),
        "segments": segments,
    }


def build_marker_pairs(onsets: Sequence[float]) -> List[Tuple[float, float]]:
    if len(onsets) % 2 != 0:
        raise ValueError(f"Marker count must be even, got {len(onsets)}")
    pairs: List[Tuple[float, float]] = []
    for i in range(0, len(onsets), 2):
        s = float(onsets[i])
        e = float(onsets[i + 1])
        if e <= s:
            raise ValueError(f"Invalid marker pair: {s} -> {e}")
        pairs.append((s, e))
    return pairs


def split_counts_8_1_1(n: int) -> Tuple[int, int, int]:
    """
    Return per-group split sizes for train/val/test.
    """
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 0, 1
    # n >= 3
    train_n = int(n * 0.8)
    val_n = int(n * 0.1)
    test_n = n - train_n - val_n
    if val_n == 0:
        val_n = 1
        test_n = max(1, n - train_n - val_n)
    while train_n + val_n + test_n > n:
        if train_n > 1:
            train_n -= 1
        elif test_n > 1:
            test_n -= 1
        else:
            break
    while train_n + val_n + test_n < n:
        train_n += 1
    return train_n, val_n, test_n


def parse_dirty_samples(raw_text: str) -> Set[str]:
    tokens = re.split(r"[\s,;]+", raw_text.strip())
    cleaned = {t.strip() for t in tokens if t.strip()}
    return cleaned


def normalize_eeg_preprocess_mode(mode: str) -> str:
    m = (mode or "strict").strip().lower()
    if m not in VALID_EEG_PREPROCESS:
        raise ValueError(f"Unsupported eeg preprocess mode: {mode}")
    return m


def normalize_timecode_repair_mode(mode: str) -> str:
    m = (mode or "stitch").strip().lower()
    if m not in VALID_TIMECODE_REPAIR:
        raise ValueError(f"Unsupported timecode repair mode: {mode}")
    return m


def normalize_timecode_mode(mode: str) -> str:
    m = (mode or "robust").strip().lower()
    if m not in VALID_TIMECODE_MODE:
        raise ValueError(f"Unsupported timecode mode: {mode}")
    return m


def discover_audio_base_dir(subject_root: Path) -> Path:
    for name in ("audio", "Audio"):
        p = subject_root / name
        if p.is_dir():
            return p
    raise FileNotFoundError(f"Neither audio/ nor Audio/ exists under {subject_root}")


def discover_slate_session_dirs(audio_base_dir: Path) -> List[Tuple[int, str]]:
    """
    Discover session directories as [(slate_take, dir_name)] sorted by slate_take.
    """
    out: List[Tuple[int, str]] = []
    pat = re.compile(r"^\d{8}_MySlate_(\d+)$")
    for p in sorted(audio_base_dir.iterdir()):
        if not p.is_dir():
            continue
        m = pat.match(p.name)
        if not m:
            continue
        out.append((int(m.group(1)), p.name))
    out = sorted(out, key=lambda x: x[0])
    return out


def compute_end_pad_seconds(
    sample_id: str, end_pad_ms: int, selected_samples: Set[str]
) -> float:
    if end_pad_ms <= 0:
        return 0.0
    if not selected_samples:
        return 0.0
    return float(end_pad_ms) / 1000.0 if sample_id in selected_samples else 0.0


def resolve_session_end_pad_ms(
    sample_id: str,
    session_name: str,
    default_end_pad_ms: int,
    selected_samples: Set[str],
    session_overrides: Dict[str, int] | None = None,
) -> int:
    if sample_id not in selected_samples:
        return 0
    default_ms = max(0, int(default_end_pad_ms))
    overrides = session_overrides or {}
    if session_name in overrides:
        return max(0, int(overrides[session_name]))
    return default_ms


def parse_session_pad_overrides(raw_text: str) -> Dict[str, int]:
    """
    Parse JSON map like {"s14": 600, "14": 600} into normalized {"s14": 600}.
    """
    text = (raw_text or "").strip()
    if not text:
        return {}
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Session pad overrides must be a JSON object.")
    out: Dict[str, int] = {}
    for k, v in parsed.items():
        key = str(k).strip()
        if not key:
            continue
        if not key.startswith("s"):
            key = f"s{int(key)}"
        out[key] = max(0, int(v))
    return out


def _to_audio_float(sig: np.ndarray) -> np.ndarray:
    out = np.asarray(sig, dtype=np.float32)
    if np.issubdtype(sig.dtype, np.integer):
        out = out / 32768.0
    return out


def compute_effective_tail_pad_ms(
    signal: np.ndarray | Sequence[float],
    sample_rate: int,
    base_end_sample: int,
    max_end_sample: int,
    alpha: float = 0.25,
    abs_floor: float = 0.015,
    frame_ms: int = 10,
    min_keep_ms: int = 40,
) -> int:
    """
    Adaptive trim inside [base_end_sample, max_end_sample].
    Keep to the last active frame + min_keep_ms; return pad in milliseconds.
    """
    if sample_rate <= 0:
        return 0
    x = np.asarray(signal)
    n = len(x)
    if n <= 0:
        return 0
    b = max(0, min(int(base_end_sample), n))
    m = max(b, min(int(max_end_sample), n))
    if m <= b:
        return 0

    x_f = _to_audio_float(x)
    # Use up to last 1s of body to set per-sample threshold.
    body_start = max(0, b - int(sample_rate))
    body = x_f[body_start:b]
    if len(body) == 0:
        return 0
    body_rms = float(np.sqrt(np.mean(body**2)))
    threshold = max(float(abs_floor), float(alpha) * body_rms)

    added = x_f[b:m]
    frame = max(1, int(round(sample_rate * float(frame_ms) / 1000.0)))
    min_keep_samples = max(0, int(round(sample_rate * float(min_keep_ms) / 1000.0)))

    frame_count = len(added) // frame
    if frame_count <= 0:
        return 0
    framed = added[: frame_count * frame].reshape(frame_count, frame)
    frame_rms = np.sqrt(np.mean(framed**2, axis=1))
    active_idx = np.where(frame_rms >= threshold)[0]
    if len(active_idx) == 0:
        return 0
    last_active = int(active_idx[-1])
    keep_samples = min(len(added), (last_active + 1) * frame + min_keep_samples)
    return max(0, int(round(keep_samples * 1000.0 / float(sample_rate))))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: Sequence[str]) -> None:
    subprocess.run(cmd, check=True)


def parse_cnt_header_start(cnt_path: Path) -> datetime:
    """
    Parse local start datetime from CNT header with `strings`.
    """
    out = subprocess.check_output(
        ["strings", "-n", "6", str(cnt_path)], text=True, errors="ignore"
    )
    date_hits = re.findall(r"\b\d{2}/\d{2}/\d{4}\b", out)
    time_hits = re.findall(r"\b\d{2}:\d{2}:\d{2}\b", out)
    if not date_hits or not time_hits:
        raise ValueError(f"Unable to parse CNT header time from {cnt_path}")
    date_str = date_hits[0]
    start_time_str = time_hits[0]
    return datetime.strptime(f"{date_str} {start_time_str}", "%m/%d/%Y %H:%M:%S")


def to_local_datetime(
    session_date: datetime.date, timecode: str, add_hours: int, fps: float = 60.0
) -> datetime:
    sec = parse_timecode_to_seconds(timecode, fps=fps, include_msec=True)
    midnight = datetime.combine(session_date, datetime.min.time())
    return midnight + timedelta(seconds=sec) + timedelta(hours=add_hours)


def ffmpeg_extract_session_wav(mov_path: Path, wav_path: Path) -> None:
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(mov_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "48000",
            str(wav_path),
            "-loglevel",
            "error",
        ]
    )


def read_wav_pcm16_mono(wav_path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(wav_path), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        fr = wf.getframerate()
        nframes = wf.getnframes()
        if channels != 1 or sampwidth != 2:
            raise ValueError(f"Require mono 16-bit wav, got channels={channels}, width={sampwidth}")
        data = wf.readframes(nframes)
    arr = np.frombuffer(data, dtype="<i2").copy()
    return arr, fr


def write_wav_pcm16_mono(wav_path: Path, signal: np.ndarray, sample_rate: int) -> None:
    sig = np.asarray(signal, dtype=np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(sig.tobytes())


def resample_to_expected_frames(values: np.ndarray, expected_frames: int) -> np.ndarray:
    if expected_frames <= 0:
        return values.astype(np.float32)
    src_len = len(values)
    if src_len == 0:
        return values.astype(np.float32)
    if src_len == expected_frames:
        return values.astype(np.float32)
    x = np.linspace(0.0, 1.0, src_len, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, expected_frames, dtype=np.float64)
    out = np.empty((expected_frames, values.shape[1]), dtype=np.float32)
    for d in range(values.shape[1]):
        out[:, d] = np.interp(x_new, x, values[:, d].astype(np.float64))
    return out


def parse_xlsx_labels(xlsx_path: Path) -> Dict[str, Dict[int, Dict[str, str]]]:
    """
    Parse XLSX using stdlib xml parser (no pandas/openpyxl dependency).
    Returns:
      {sheet_name: {sentence_id: {"text":..., "emotion_cn":..., "emotion":...}}}
    """
    with zipfile.ZipFile(xlsx_path, "r") as zf:
        shared: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", XLSX_NS):
                texts = [t.text or "" for t in si.findall(".//a:t", XLSX_NS)]
                shared.append("".join(texts))

        wb_root = ET.fromstring(zf.read("xl/workbook.xml"))
        rel_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rel_root.findall("p:Relationship", XLSX_NS)
        }

        def col_to_idx(col: str) -> int:
            idx = 0
            for ch in col:
                idx = idx * 26 + (ord(ch) - 64)
            return idx

        out: Dict[str, Dict[int, Dict[str, str]]] = {}
        for sheet in wb_root.findall("a:sheets/a:sheet", XLSX_NS):
            name = sheet.attrib.get("name", "")
            rid = sheet.attrib.get(
                "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
            )
            target = rel_map.get(rid, "")
            if not target.startswith("worksheets/"):
                continue
            ws_root = ET.fromstring(zf.read("xl/" + target))

            rows: List[List[str]] = []
            for row in ws_root.findall("a:sheetData/a:row", XLSX_NS):
                vals: List[str] = []
                for cell in row.findall("a:c", XLSX_NS):
                    ref = cell.attrib.get("r", "")
                    m = re.match(r"([A-Z]+)", ref)
                    ci = col_to_idx(m.group(1)) if m else (len(vals) + 1)
                    while len(vals) < ci - 1:
                        vals.append("")
                    t = cell.attrib.get("t")
                    v = cell.find("a:v", XLSX_NS)
                    val = ""
                    if v is not None and v.text is not None:
                        if t == "s":
                            sid = int(v.text)
                            val = shared[sid] if 0 <= sid < len(shared) else ""
                        else:
                            val = v.text
                    is_elem = cell.find("a:is", XLSX_NS)
                    if is_elem is not None:
                        ts = [tt.text or "" for tt in is_elem.findall(".//a:t", XLSX_NS)]
                        if ts:
                            val = "".join(ts)
                    vals.append(val)
                if any(x.strip() for x in vals):
                    rows.append(vals)
            if not rows:
                continue
            # header: 序号,句子,情绪标签,来源
            data = {}
            for row in rows[1:]:
                if len(row) < 3:
                    continue
                sid_raw = row[0].strip()
                if not sid_raw.isdigit():
                    continue
                sid = int(sid_raw)
                text = row[1].strip()
                emo_cn = row[2].strip()
                emo_en = EMOTION_CN_TO_EN.get(emo_cn, emo_cn.lower())
                data[sid] = {"text": text, "emotion_cn": emo_cn, "emotion": emo_en}
            out[name] = data
    return out


def build_split_json(
    split_name: str,
    entries: Sequence[dict],
    subject_name: str,
    sample_rate: int = 48000,
    fps: int = 60,
) -> dict:
    return {
        "info": {
            "id_list": [subject_name],
            "data_source": "EATalker",
            "total_duration": float(sum(e["duration"] for e in entries)),
            "fps": fps,
            "audio_sample_rate": sample_rate,
            "split": split_name,
        },
        "data": list(entries),
    }


class LCDataProcessor:
    def __init__(
        self,
        raw_root: Path,
        out_root: Path,
        labels_xlsx: Path,
        subject: str = "lc",
        identity_id: int = 0,
        end_pad_ms: int = 0,
        end_pad_samples: Set[str] | None = None,
        eeg_preprocess: str = "strict",
        timecode_mode: str = "robust",
        timecode_repair: str = "stitch",
    ):
        self.raw_root = raw_root
        self.out_root = out_root
        self.labels_xlsx = labels_xlsx
        self.subject = subject
        self.identity_id = identity_id
        self.end_pad_ms = int(end_pad_ms)
        self.end_pad_samples: Set[str] = set(end_pad_samples or set())
        self.eeg_preprocess = normalize_eeg_preprocess_mode(eeg_preprocess)
        self.timecode_mode = normalize_timecode_mode(timecode_mode)
        self.timecode_repair = normalize_timecode_repair_mode(timecode_repair)
        self.subject_root = self.raw_root / subject
        self.audio_base_dir = discover_audio_base_dir(self.subject_root)
        self.dataset_root = self.out_root / subject / "EATalker"
        self.audio_root = self.dataset_root / "audio"
        self.bs_root = self.dataset_root / "blendshapes"
        self.eeg_root = self.dataset_root / "eeg"
        self.cache_root = self.dataset_root / "_cache"
        self._session_infos_cache = self._discover_session_infos()

    def _session_infos(self) -> List[SessionInfo]:
        return list(self._session_infos_cache)

    def _discover_session_infos(self) -> List[SessionInfo]:
        sessions = discover_slate_session_dirs(self.audio_base_dir)
        if len(sessions) != len(DEFAULT_SESSION_SHEETS):
            raise ValueError(
                f"{self.subject}: expected {len(DEFAULT_SESSION_SHEETS)} sessions, "
                f"found {len(sessions)} under {self.audio_base_dir}"
            )
        out: List[SessionInfo] = []
        for idx, (slate_take, dir_name) in enumerate(sessions, start=1):
            out.append(
                SessionInfo(
                    sheet_name=DEFAULT_SESSION_SHEETS[idx - 1],
                    audio_dir_name=dir_name,
                    eeg_index=idx,
                    slate_take=slate_take,
                )
            )
        return out

    def _preprocess_eeg(self, eeg_data: np.ndarray, sfreq: float) -> np.ndarray:
        if self.eeg_preprocess == "none":
            return eeg_data.astype(np.float32)
        try:
            from dataset.eeg_processor import EEGProcessor
        except ModuleNotFoundError:
            from EATalker.dataset.eeg_processor import EEGProcessor

        processor = EEGProcessor(
            sfreq=float(sfreq),
            eeg_channels=60,
            target_sfreq=float(sfreq),
        )
        return processor.preprocess_data(eeg_data, float(sfreq)).astype(np.float32)

    def _read_video_fps(self, audio_dir: Path) -> float:
        vm_path = audio_dir / "video_metadata.json"
        if not vm_path.exists():
            return 60.0
        try:
            vm = json.loads(vm_path.read_text(encoding="utf-8"))
        except Exception:
            return 60.0
        fps = vm.get("FrameRate")
        try:
            fps_f = float(fps)
        except (TypeError, ValueError):
            return 60.0
        return fps_f if fps_f > 0 else 60.0

    def _load_bs_session(
        self,
        bs_csv_path: Path,
        session_date: datetime.date,
        timecode_fps: float = 60.0,
        session_tag: str = "",
    ) -> Tuple[np.ndarray, np.ndarray]:
        timecodes: List[str] = []
        rows: List[List[float]] = []
        with open(bs_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing = [c for c in D6_ORDER if c not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"BS CSV missing columns: {missing}")
            for row in reader:
                timecodes.append(row["Timecode"])
                rows.append([float(row[c]) if row[c] else 0.0 for c in D6_ORDER])
        secs = build_timecode_seconds_sequence(
            timecodes,
            fps=timecode_fps,
            mode=self.timecode_mode,
        )
        pre_check = summarize_timecode_sequence(secs, fps=timecode_fps)
        repair_diag: Dict[str, object] = {
            "applied": False,
            "jump_idx": None,
            "jump_size_ms": 0.0,
            "applied_offset_ms": 0.0,
            "segments": [],
        }
        repair_min_run_len = 1 if self.timecode_mode == "adaptive" else 3
        if self.timecode_repair == "stitch":
            secs, repair_diag = repair_timecode_jump_stitch(
                secs,
                fps=timecode_fps,
                min_run_len=repair_min_run_len,
            )
        post_check = summarize_timecode_sequence(secs, fps=timecode_fps)
        if session_tag:
            self.cache_root.mkdir(parents=True, exist_ok=True)
            diag_path = self.cache_root / f"timecode_diag_{session_tag}.json"
            diag_payload = {
                "session": session_tag,
                "fps": float(timecode_fps),
                "timecode_mode": self.timecode_mode,
                "timecode_repair": self.timecode_repair,
                "timecode_repair_min_run_len": int(repair_min_run_len),
                "pre_check": pre_check,
                "repair": repair_diag,
                "post_check": post_check,
            }
            diag_path.write_text(
                json.dumps(diag_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        midnight = datetime.combine(session_date, datetime.min.time()) + timedelta(hours=8)
        times = [(midnight + timedelta(seconds=s)).timestamp() for s in secs]
        return np.asarray(times, dtype=np.float64), np.asarray(rows, dtype=np.float32)

    def _build_stage_a_records(self) -> List[dict]:
        import mne

        labels = parse_xlsx_labels(self.labels_xlsx)
        records: List[dict] = []

        clear_dir(self.audio_root)
        clear_dir(self.bs_root)
        clear_dir(self.eeg_root)
        clear_dir(self.cache_root)

        for cfg in self._session_infos():
            audio_dir = self.audio_base_dir / cfg.audio_dir_name
            cnt_path = self.subject_root / "EEG" / "raw_EEG" / f"{self.subject}{cfg.eeg_index}.cnt"
            set_path = self.subject_root / "EEG" / "recon_eeg" / f"{self.subject}{cfg.eeg_index}.set"
            take_path = audio_dir / "take.json"
            mov_path = audio_dir / f"MySlate_{cfg.slate_take}_iPhone.mov"
            bs_csv_path = audio_dir / f"MySlate_{cfg.slate_take}_iPhone.csv"
            wav_cache = self.cache_root / f"{cfg.audio_dir_name}.wav"

            if cfg.sheet_name not in labels:
                raise FileNotFoundError(f"Missing sheet {cfg.sheet_name} in {self.labels_xlsx}")
            label_table = labels[cfg.sheet_name]

            cnt_start_local = parse_cnt_header_start(cnt_path)
            session_date = cnt_start_local.date()

            take = json.loads(take_path.read_text(encoding="utf-8"))
            tc_fps = self._read_video_fps(audio_dir)
            audio_start_local = to_local_datetime(
                session_date, take["startTimecode"], add_hours=8, fps=tc_fps
            )
            audio_end_local = to_local_datetime(
                session_date, take["endTimecode"], add_hours=8, fps=tc_fps
            )

            ffmpeg_extract_session_wav(mov_path, wav_cache)
            wav_data, wav_sr = read_wav_pcm16_mono(wav_cache)
            wav_duration = len(wav_data) / wav_sr

            bs_tc_fps = tc_fps
            bs_times, bs_values = self._load_bs_session(
                bs_csv_path,
                session_date,
                timecode_fps=bs_tc_fps,
                session_tag=f"s{cfg.slate_take}",
            )

            raw_set = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)
            eeg_data = raw_set.get_data().astype(np.float32)
            if eeg_data.shape[0] != 60:
                raise ValueError(f"{set_path} channel mismatch: expected 60, got {eeg_data.shape[0]}")
            sfreq = float(raw_set.info["sfreq"])
            eeg_data = self._preprocess_eeg(eeg_data, sfreq)
            pairs = build_marker_pairs(raw_set.annotations.onset)
            if len(pairs) != 125:
                raise ValueError(f"{set_path} marker pairs != 125: {len(pairs)}")

            print(
                f"[{cfg.audio_dir_name}] markers={len(pairs)} "
                f"audio_start={audio_start_local.strftime('%H:%M:%S')} "
                f"audio_end={audio_end_local.strftime('%H:%M:%S')} "
                f"cnt_start={cnt_start_local.strftime('%H:%M:%S')}"
            )

            for idx, (onset_s, onset_e) in enumerate(pairs):
                local_sid = idx + 1
                label = label_table.get(local_sid)
                if label is None:
                    raise ValueError(f"Sheet {cfg.sheet_name} missing sentence id {local_sid}")
                emotion = label["emotion"]
                text = label["text"]
                sample_id = f"s{cfg.slate_take}_sentence{local_sid:03d}"

                end_pad_sec = compute_end_pad_seconds(
                    sample_id=sample_id,
                    end_pad_ms=self.end_pad_ms,
                    selected_samples=self.end_pad_samples,
                )
                if idx + 1 < len(pairs):
                    next_start = pairs[idx + 1][0]
                    max_pad = max(0.0, next_start - onset_e - 1e-4)
                    end_pad_sec = min(end_pad_sec, max_pad)

                marker_start_abs = cnt_start_local + timedelta(seconds=onset_s)
                marker_end_abs = cnt_start_local + timedelta(seconds=onset_e + end_pad_sec)

                audio_rel_s = (marker_start_abs - audio_start_local).total_seconds()
                audio_rel_e = (marker_end_abs - audio_start_local).total_seconds()
                if audio_rel_e <= audio_rel_s:
                    continue

                audio_rel_s = max(0.0, audio_rel_s)
                audio_rel_e = min(wav_duration, audio_rel_e)
                if audio_rel_e - audio_rel_s < 0.2:
                    continue

                start_idx = int(round(audio_rel_s * wav_sr))
                end_idx = int(round(audio_rel_e * wav_sr))
                start_idx = max(0, min(start_idx, len(wav_data)))
                end_idx = max(0, min(end_idx, len(wav_data)))
                if end_idx <= start_idx:
                    continue
                seg_audio = wav_data[start_idx:end_idx]

                eeg_start = int(round(onset_s * sfreq))
                eeg_end = int(round((onset_e + end_pad_sec) * sfreq))
                eeg_start = max(0, min(eeg_start, eeg_data.shape[1]))
                eeg_end = max(0, min(eeg_end, eeg_data.shape[1]))
                if eeg_end <= eeg_start:
                    continue
                seg_eeg = eeg_data[:, eeg_start:eeg_end].T.astype(np.float32)

                start_ts = marker_start_abs.timestamp()
                end_ts = marker_end_abs.timestamp()
                tol = 1.0 / 120.0
                mask = (bs_times >= start_ts - tol) & (bs_times <= end_ts + tol)
                seg_bs = bs_values[mask]
                if len(seg_bs) == 0:
                    continue
                expected_frames = max(1, int(round((audio_rel_e - audio_rel_s) * 60.0)))
                seg_bs = resample_to_expected_frames(seg_bs, expected_frames)

                stem = f"{sample_id}_{self.subject}_{emotion}"
                audio_name = f"{stem}.wav"
                bs_name = f"{stem}_bs.npy"
                eeg_name = f"{stem}_eeg.npy"

                audio_out = self.audio_root / audio_name
                bs_out = self.bs_root / bs_name
                eeg_out = self.eeg_root / eeg_name

                write_wav_pcm16_mono(audio_out, seg_audio, wav_sr)
                np.save(bs_out, seg_bs.astype(np.float32))
                np.save(eeg_out, seg_eeg)

                duration = len(seg_audio) / wav_sr
                records.append(
                    {
                        "sample_id": sample_id,
                        "session": f"s{cfg.slate_take}",
                        "sheet": cfg.sheet_name,
                        "local_sentence_id": local_sid,
                        "emotion": emotion,
                        "text": text,
                        "duration": float(duration),
                        "end_pad_ms": int(round(end_pad_sec * 1000.0)),
                        "id": self.identity_id,
                        "fps": 60,
                        "dataset": "EATalker",
                        "annot_type": "inhouse_blendshape_weight",
                        "annot_path": f"blendshapes/{bs_name}",
                        "audio_path": f"audio/{audio_name}",
                        "eeg_path": f"eeg/{eeg_name}",
                    }
                )
        return records

    def build_all(self) -> None:
        ensure_dir(self.dataset_root)
        records = self._build_stage_a_records()
        records = sorted(records, key=lambda x: (x["session"], x["local_sentence_id"]))

        all_full = build_split_json("all_full", records, self.subject, sample_rate=48000, fps=60)
        all_full["info"]["eeg_preprocess"] = self.eeg_preprocess
        all_full["info"]["eeg_preprocess_source"] = (
            "dataset.eeg_processor.preprocess_data"
            if self.eeg_preprocess == "strict"
            else "none"
        )
        with open(self.dataset_root / "all_full.json", "w", encoding="utf-8") as f:
            json.dump(all_full, f, indent=2, ensure_ascii=False)

        with open(self.dataset_root / "review_manifest.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "sample_id",
                    "session",
                    "local_sentence_id",
                    "emotion",
                    "duration",
                    "drop_flag",
                    "drop_reason",
                    "text",
                ]
            )
            for r in records:
                writer.writerow(
                    [
                        r["sample_id"],
                        r["session"],
                        r["local_sentence_id"],
                        r["emotion"],
                        f"{r['duration']:.6f}",
                        0,
                        "",
                        r["text"],
                    ]
                )

        id_template = np.zeros((1, 51), dtype=np.float32)
        np.save(self.dataset_root / "id_template.npy", id_template)
        print(f"Stage A done: {len(records)} samples -> {self.dataset_root}")

    def _session_from_name(self, session_name: str) -> SessionInfo:
        try:
            slate_take = int(session_name.lstrip("s"))
        except ValueError as exc:
            raise ValueError(f"Invalid session name: {session_name}") from exc
        for cfg in self._session_infos():
            if cfg.slate_take == slate_take:
                return cfg
        raise ValueError(f"Session {session_name} is not in SESSION_PLAN")

    def _sync_selected_to_splits(self, rows: Sequence[dict]) -> None:
        for split_name in ["train", "val", "test"]:
            split_dir = self.dataset_root / split_name
            if not split_dir.exists():
                continue
            for r in rows:
                for key in ["audio_path", "annot_path", "eeg_path"]:
                    rel = Path(r[key])
                    src = self.dataset_root / rel
                    dst = split_dir / rel.name
                    if dst.exists():
                        try:
                            if src.samefile(dst):
                                continue
                        except FileNotFoundError:
                            pass
                        shutil.copy2(src, dst)

    def _update_duration_in_json(self, json_path: Path, duration_map: Dict[str, float]) -> None:
        if not json_path.exists():
            return
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        changed = False
        for row in payload.get("data", []):
            sid = row.get("sample_id")
            if sid in duration_map:
                row["duration"] = float(duration_map[sid])
                changed = True
        if changed:
            payload["info"]["total_duration"] = float(
                sum(float(x["duration"]) for x in payload.get("data", []))
            )
            json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def secondary_pad_selected(
        self,
        selected_samples: Set[str],
        end_pad_ms: int,
        session_pad_overrides: Dict[str, int] | None = None,
        tail_adaptive: bool = False,
        tail_alpha: float = 0.25,
        tail_abs_floor: float = 0.015,
        tail_frame_ms: int = 10,
        tail_min_keep_ms: int = 40,
    ) -> None:
        if not selected_samples:
            raise ValueError("No selected samples provided for secondary processing.")
        src_json = self.dataset_root / "all_full.json"
        if not src_json.exists():
            raise FileNotFoundError(f"Missing {src_json}. Run build-all first.")

        all_full = json.loads(src_json.read_text(encoding="utf-8"))
        rows = all_full.get("data", [])
        row_map = {r["sample_id"]: r for r in rows}
        missing = sorted([s for s in selected_samples if s not in row_map])
        if missing:
            raise ValueError(f"Selected samples not found in all_full.json: {missing[:20]}")

        labels = parse_xlsx_labels(self.labels_xlsx)
        cache_by_session: Dict[str, dict] = {}
        changed_rows: List[dict] = []
        duration_map: Dict[str, float] = {}
        end_pad_set = set(selected_samples)
        session_pad_overrides = dict(session_pad_overrides or {})

        import mne

        def build_session_cache(session_name: str) -> dict:
            if session_name in cache_by_session:
                return cache_by_session[session_name]
            cfg = self._session_from_name(session_name)
            audio_dir = self.audio_base_dir / cfg.audio_dir_name
            set_path = self.subject_root / "EEG" / "recon_eeg" / f"{self.subject}{cfg.eeg_index}.set"
            take_path = audio_dir / "take.json"
            mov_path = audio_dir / f"MySlate_{cfg.slate_take}_iPhone.mov"
            bs_csv_path = audio_dir / f"MySlate_{cfg.slate_take}_iPhone.csv"
            wav_cache = self.cache_root / f"{cfg.audio_dir_name}_secondary.wav"

            cnt_start_local = parse_cnt_header_start(
                self.subject_root / "EEG" / "raw_EEG" / f"{self.subject}{cfg.eeg_index}.cnt"
            )
            session_date = cnt_start_local.date()
            take = json.loads(take_path.read_text(encoding="utf-8"))
            tc_fps = self._read_video_fps(audio_dir)
            audio_start_local = to_local_datetime(
                session_date, take["startTimecode"], add_hours=8, fps=tc_fps
            )

            ffmpeg_extract_session_wav(mov_path, wav_cache)
            wav_data, wav_sr = read_wav_pcm16_mono(wav_cache)
            wav_duration = len(wav_data) / wav_sr
            bs_tc_fps = tc_fps
            bs_times, bs_values = self._load_bs_session(
                bs_csv_path,
                session_date,
                timecode_fps=bs_tc_fps,
                session_tag=f"s{cfg.slate_take}",
            )

            raw_set = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)
            eeg_data = raw_set.get_data().astype(np.float32)
            if eeg_data.shape[0] != 60:
                raise ValueError(f"{set_path} channel mismatch: expected 60, got {eeg_data.shape[0]}")
            sfreq = float(raw_set.info["sfreq"])
            eeg_data = self._preprocess_eeg(eeg_data, sfreq)
            pairs = build_marker_pairs(raw_set.annotations.onset)
            if len(pairs) != 125:
                raise ValueError(f"{set_path} marker pairs != 125: {len(pairs)}")
            if cfg.sheet_name not in labels:
                raise ValueError(f"Missing sheet {cfg.sheet_name} in {self.labels_xlsx}")
            cache = {
                "cfg": cfg,
                "cnt_start_local": cnt_start_local,
                "audio_start_local": audio_start_local,
                "wav_data": wav_data,
                "wav_sr": wav_sr,
                "wav_duration": wav_duration,
                "bs_times": bs_times,
                "bs_values": bs_values,
                "eeg_data": eeg_data,
                "sfreq": sfreq,
                "pairs": pairs,
            }
            cache_by_session[session_name] = cache
            return cache

        for sid in sorted(selected_samples):
            r = row_map[sid]
            cache = build_session_cache(r["session"])
            local_sid = int(r["local_sentence_id"])
            idx = local_sid - 1
            onset_s, onset_e = cache["pairs"][idx]

            resolved_pad_ms = resolve_session_end_pad_ms(
                sample_id=sid,
                session_name=r["session"],
                default_end_pad_ms=end_pad_ms,
                selected_samples=end_pad_set,
                session_overrides=session_pad_overrides,
            )
            end_pad_sec = float(resolved_pad_ms) / 1000.0
            if idx + 1 < len(cache["pairs"]):
                next_start = cache["pairs"][idx + 1][0]
                max_pad = max(0.0, next_start - onset_e - 1e-4)
                end_pad_sec = min(end_pad_sec, max_pad)

            marker_start_abs = cache["cnt_start_local"] + timedelta(seconds=onset_s)
            marker_end_abs = cache["cnt_start_local"] + timedelta(seconds=onset_e + end_pad_sec)

            audio_rel_s = (marker_start_abs - cache["audio_start_local"]).total_seconds()
            audio_rel_e = (marker_end_abs - cache["audio_start_local"]).total_seconds()
            audio_rel_s = max(0.0, audio_rel_s)
            audio_rel_e = min(cache["wav_duration"], audio_rel_e)
            if audio_rel_e <= audio_rel_s:
                raise ValueError(f"Invalid secondary audio window for {sid}: {audio_rel_s} -> {audio_rel_e}")

            start_idx = int(round(audio_rel_s * cache["wav_sr"]))
            end_idx = int(round(audio_rel_e * cache["wav_sr"]))
            base_marker_end_abs = cache["cnt_start_local"] + timedelta(seconds=onset_e)
            base_audio_rel_e = (base_marker_end_abs - cache["audio_start_local"]).total_seconds()
            base_audio_rel_e = max(0.0, min(base_audio_rel_e, cache["wav_duration"]))
            base_end_idx = int(round(base_audio_rel_e * cache["wav_sr"]))

            if tail_adaptive and end_idx > base_end_idx:
                adaptive_ms = compute_effective_tail_pad_ms(
                    signal=cache["wav_data"],
                    sample_rate=int(cache["wav_sr"]),
                    base_end_sample=base_end_idx,
                    max_end_sample=end_idx,
                    alpha=tail_alpha,
                    abs_floor=tail_abs_floor,
                    frame_ms=tail_frame_ms,
                    min_keep_ms=tail_min_keep_ms,
                )
                adaptive_end_idx = base_end_idx + int(round(adaptive_ms * cache["wav_sr"] / 1000.0))
                end_idx = max(base_end_idx, min(end_idx, adaptive_end_idx))
                audio_rel_e = float(end_idx) / float(cache["wav_sr"])
                end_pad_sec = max(0.0, audio_rel_e - base_audio_rel_e)

            marker_end_abs = cache["cnt_start_local"] + timedelta(seconds=onset_e + end_pad_sec)
            seg_audio = cache["wav_data"][start_idx:end_idx]
            if len(seg_audio) == 0:
                raise ValueError(f"Empty secondary audio segment for {sid}")

            eeg_start = int(round(onset_s * cache["sfreq"]))
            eeg_end = int(round((onset_e + end_pad_sec) * cache["sfreq"]))
            eeg_start = max(0, min(eeg_start, cache["eeg_data"].shape[1]))
            eeg_end = max(0, min(eeg_end, cache["eeg_data"].shape[1]))
            if eeg_end <= eeg_start:
                raise ValueError(f"Invalid secondary eeg window for {sid}: {eeg_start} -> {eeg_end}")
            seg_eeg = cache["eeg_data"][:, eeg_start:eeg_end].T.astype(np.float32)

            start_ts = marker_start_abs.timestamp()
            end_ts = marker_end_abs.timestamp()
            tol = 1.0 / 120.0
            mask = (cache["bs_times"] >= start_ts - tol) & (cache["bs_times"] <= end_ts + tol)
            seg_bs = cache["bs_values"][mask]
            if len(seg_bs) == 0:
                raise ValueError(f"Empty secondary blendshape segment for {sid}")
            expected_frames = max(1, int(round((audio_rel_e - audio_rel_s) * 60.0)))
            seg_bs = resample_to_expected_frames(seg_bs, expected_frames)

            audio_out = self.dataset_root / r["audio_path"]
            bs_out = self.dataset_root / r["annot_path"]
            eeg_out = self.dataset_root / r["eeg_path"]
            write_wav_pcm16_mono(audio_out, seg_audio, int(cache["wav_sr"]))
            np.save(bs_out, seg_bs.astype(np.float32))
            np.save(eeg_out, seg_eeg.astype(np.float32))

            duration = len(seg_audio) / cache["wav_sr"]
            r["duration"] = float(duration)
            r["end_pad_ms"] = int(round(end_pad_sec * 1000.0))
            duration_map[sid] = float(duration)
            changed_rows.append(r)

        all_full["info"]["total_duration"] = float(
            sum(float(x["duration"]) for x in all_full.get("data", []))
        )
        src_json.write_text(json.dumps(all_full, indent=2, ensure_ascii=False), encoding="utf-8")
        self._sync_selected_to_splits(changed_rows)
        for json_name in ["train.json", "val.json", "test.json", "all.json"]:
            self._update_duration_in_json(self.dataset_root / json_name, duration_map)
        print(f"Secondary processing done: {len(changed_rows)} samples updated.")

    def split_after_dirty(
        self, dirty_samples: Set[str], seed: int = 42, make_copy: bool = False
    ) -> None:
        src_json = self.dataset_root / "all_full.json"
        if not src_json.exists():
            raise FileNotFoundError(f"Missing {src_json}. Run build-all first.")
        all_full = json.loads(src_json.read_text(encoding="utf-8"))
        records = [r for r in all_full["data"] if r["sample_id"] not in dirty_samples]
        if not records:
            raise ValueError("No samples left after dirty filtering.")

        for split in ["train", "val", "test"]:
            clear_dir(self.dataset_root / split)

        grouped: Dict[Tuple[str, str], List[dict]] = {}
        for r in records:
            grouped.setdefault((r["session"], r["emotion"]), []).append(r)

        rng = random.Random(seed)
        split_map: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
        for key, group in grouped.items():
            group = sorted(group, key=lambda x: x["sample_id"])
            rng.shuffle(group)
            tn, vn, sn = split_counts_8_1_1(len(group))
            train_g = group[:tn]
            val_g = group[tn : tn + vn]
            test_g = group[tn + vn : tn + vn + sn]
            split_map["train"].extend(train_g)
            split_map["val"].extend(val_g)
            split_map["test"].extend(test_g)
            print(f"group={key} n={len(group)} -> train={len(train_g)} val={len(val_g)} test={len(test_g)}")

        def link_or_copy(src: Path, dst: Path) -> None:
            if dst.exists():
                return
            if make_copy:
                shutil.copy2(src, dst)
                return
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)

        def build_entries(split_name: str, rows: Sequence[dict]) -> List[dict]:
            out: List[dict] = []
            split_dir = self.dataset_root / split_name
            for r in rows:
                src_audio = self.dataset_root / r["audio_path"]
                src_bs = self.dataset_root / r["annot_path"]
                src_eeg = self.dataset_root / r["eeg_path"]
                dst_audio = split_dir / src_audio.name
                dst_bs = split_dir / src_bs.name
                dst_eeg = split_dir / src_eeg.name
                link_or_copy(src_audio, dst_audio)
                link_or_copy(src_bs, dst_bs)
                link_or_copy(src_eeg, dst_eeg)
                out.append(
                    {
                        "annot_type": "inhouse_blendshape_weight",
                        "annot_path": f"{split_name}/{dst_bs.name}",
                        "audio_path": f"{split_name}/{dst_audio.name}",
                        "id": r["id"],
                        "fps": 60,
                        "dataset": "EATalker",
                        "emotion": r["emotion"],
                        "duration": r["duration"],
                        "eeg_path": f"{split_name}/{dst_eeg.name}",
                        "sample_id": r["sample_id"],
                        "session": r["session"],
                        "local_sentence_id": r["local_sentence_id"],
                        "text": r["text"],
                    }
                )
            return out

        train_entries = build_entries("train", split_map["train"])
        val_entries = build_entries("val", split_map["val"])
        test_entries = build_entries("test", split_map["test"])

        for split_name, entries in [
            ("train", train_entries),
            ("val", val_entries),
            ("test", test_entries),
        ]:
            payload = build_split_json(split_name, entries, self.subject, sample_rate=48000, fps=60)
            with open(self.dataset_root / f"{split_name}.json", "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

        all_entries = train_entries + val_entries + test_entries
        all_payload = build_split_json("all", all_entries, self.subject, sample_rate=48000, fps=60)
        with open(self.dataset_root / "all.json", "w", encoding="utf-8") as f:
            json.dump(all_payload, f, indent=2, ensure_ascii=False)

        with open(self.dataset_root / "dirty_samples_applied.txt", "w", encoding="utf-8") as f:
            for s in sorted(dirty_samples):
                f.write(f"{s}\n")

        print(
            f"Stage B done: kept={len(all_entries)} "
            f"train={len(train_entries)} val={len(val_entries)} test={len(test_entries)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process EaTalker_Dataset_v2 subjects")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_all = sub.add_parser("build-all", help="Stage A: build full aligned dataset")
    p_all.add_argument("--raw-root", default="/home/zym/EaTalker_Dataset_v2")
    p_all.add_argument("--out-root", default="/home/zym/EATalker_data_template_v2")
    p_all.add_argument("--labels-xlsx", default="/home/zym/EaTalker_Dataset_v2/情感文本朗读素材.xlsx")
    p_all.add_argument("--subject", default="lc")
    p_all.add_argument("--identity-id", type=int, default=0)
    p_all.add_argument("--end-pad-ms", type=int, default=0)
    p_all.add_argument("--pad-file", default="")
    p_all.add_argument("--pad-samples", default="")
    p_all.add_argument(
        "--eeg-preprocess",
        default="strict",
        choices=sorted(VALID_EEG_PREPROCESS),
        help="EEG preprocessing mode before segmentation",
    )
    p_all.add_argument(
        "--timecode-mode",
        default="robust",
        choices=sorted(VALID_TIMECODE_MODE),
        help="Timecode parse mode for blendshape timeline",
    )
    p_all.add_argument(
        "--timecode-repair",
        default="stitch",
        choices=sorted(VALID_TIMECODE_REPAIR),
        help="Timecode repair mode for blendshape timeline",
    )

    p_split = sub.add_parser("split", help="Stage B: remove dirty and split 8:1:1")
    p_split.add_argument("--raw-root", default="/home/zym/EaTalker_Dataset_v2")
    p_split.add_argument("--out-root", default="/home/zym/EATalker_data_template_v2")
    p_split.add_argument("--labels-xlsx", default="/home/zym/EaTalker_Dataset_v2/情感文本朗读素材.xlsx")
    p_split.add_argument("--subject", default="lc")
    p_split.add_argument("--identity-id", type=int, default=0)
    p_split.add_argument("--seed", type=int, default=42)
    p_split.add_argument("--dirty-file", default="")
    p_split.add_argument("--dirty-samples", default="")
    p_split.add_argument("--copy", action="store_true", help="Copy split files instead of hardlinks")
    p_split.add_argument(
        "--eeg-preprocess",
        default="strict",
        choices=sorted(VALID_EEG_PREPROCESS),
        help="Kept for CLI consistency; split step does not reprocess EEG",
    )
    p_split.add_argument(
        "--timecode-mode",
        default="robust",
        choices=sorted(VALID_TIMECODE_MODE),
        help="Kept for CLI consistency; split step does not reprocess timelines",
    )
    p_split.add_argument(
        "--timecode-repair",
        default="stitch",
        choices=sorted(VALID_TIMECODE_REPAIR),
        help="Kept for CLI consistency; split step does not reprocess timelines",
    )

    p_secondary = sub.add_parser(
        "secondary-pad-selected",
        help="Rebuild audio/bs/eeg only for selected sample IDs with end padding",
    )
    p_secondary.add_argument("--raw-root", default="/home/zym/EaTalker_Dataset_v2")
    p_secondary.add_argument("--out-root", default="/home/zym/EATalker_data_template_v2")
    p_secondary.add_argument("--labels-xlsx", default="/home/zym/EaTalker_Dataset_v2/情感文本朗读素材.xlsx")
    p_secondary.add_argument("--subject", default="lc")
    p_secondary.add_argument("--identity-id", type=int, default=0)
    p_secondary.add_argument("--end-pad-ms", type=int, default=120)
    p_secondary.add_argument("--pad-file", default="")
    p_secondary.add_argument("--pad-samples", default="")
    p_secondary.add_argument(
        "--session-pad-file",
        default="",
        help="Optional JSON file mapping session->max end_pad_ms, e.g. {\"s14\": 600}",
    )
    p_secondary.add_argument(
        "--session-pad-json",
        default="",
        help="Inline JSON mapping session->max end_pad_ms",
    )
    p_secondary.add_argument(
        "--tail-adaptive",
        action="store_true",
        help="Enable per-sample adaptive tail trim within the max pad window",
    )
    p_secondary.add_argument("--tail-alpha", type=float, default=0.25)
    p_secondary.add_argument("--tail-abs-floor", type=float, default=0.015)
    p_secondary.add_argument("--tail-frame-ms", type=int, default=10)
    p_secondary.add_argument("--tail-min-keep-ms", type=int, default=40)
    p_secondary.add_argument(
        "--eeg-preprocess",
        default="strict",
        choices=sorted(VALID_EEG_PREPROCESS),
        help="EEG preprocessing mode before secondary segmentation",
    )
    p_secondary.add_argument(
        "--timecode-mode",
        default="robust",
        choices=sorted(VALID_TIMECODE_MODE),
        help="Timecode parse mode for secondary BS timeline rebuild",
    )
    p_secondary.add_argument(
        "--timecode-repair",
        default="stitch",
        choices=sorted(VALID_TIMECODE_REPAIR),
        help="Timecode repair mode for secondary BS timeline rebuild",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pad_samples: Set[str] = set()
    if hasattr(args, "pad_file") and args.pad_file:
        pad_samples |= parse_dirty_samples(Path(args.pad_file).read_text(encoding="utf-8"))
    if hasattr(args, "pad_samples") and args.pad_samples:
        pad_samples |= parse_dirty_samples(args.pad_samples)

    processor = LCDataProcessor(
        raw_root=Path(args.raw_root),
        out_root=Path(args.out_root),
        labels_xlsx=Path(args.labels_xlsx),
        subject=args.subject,
        identity_id=args.identity_id,
        end_pad_ms=getattr(args, "end_pad_ms", 0),
        end_pad_samples=pad_samples,
        eeg_preprocess=getattr(args, "eeg_preprocess", "strict"),
        timecode_mode=getattr(args, "timecode_mode", "robust"),
        timecode_repair=getattr(args, "timecode_repair", "stitch"),
    )

    if args.cmd == "build-all":
        processor.build_all()
        return

    if args.cmd == "split":
        dirty: Set[str] = set()
        if args.dirty_file:
            text = Path(args.dirty_file).read_text(encoding="utf-8")
            dirty |= parse_dirty_samples(text)
        if args.dirty_samples:
            dirty |= parse_dirty_samples(args.dirty_samples)
        processor.split_after_dirty(dirty_samples=dirty, seed=args.seed, make_copy=args.copy)
        return

    if args.cmd == "secondary-pad-selected":
        selected = set(pad_samples)
        session_pad_overrides: Dict[str, int] = {}
        if args.session_pad_file:
            session_pad_overrides.update(
                parse_session_pad_overrides(
                    Path(args.session_pad_file).read_text(encoding="utf-8")
                )
            )
        if args.session_pad_json:
            session_pad_overrides.update(parse_session_pad_overrides(args.session_pad_json))
        processor.secondary_pad_selected(
            selected_samples=selected,
            end_pad_ms=args.end_pad_ms,
            session_pad_overrides=session_pad_overrides,
            tail_adaptive=bool(args.tail_adaptive),
            tail_alpha=float(args.tail_alpha),
            tail_abs_floor=float(args.tail_abs_floor),
            tail_frame_ms=int(args.tail_frame_ms),
            tail_min_keep_ms=int(args.tail_min_keep_ms),
        )
        return

    raise ValueError(f"Unsupported command: {args.cmd}")


if __name__ == "__main__":
    main()
