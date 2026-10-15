#!/usr/bin/env python3
"""Build reproducible BioTalk-3D protocol splits for EATalker."""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


DEFAULT_SUBJECTS = ("lc", "zym", "yrx", "lws", "pj")
TARGET_FPS = 30
SOURCE_FPS = 60
EEG_SAMPLE_RATE_HZ = 1000


def infer_language(text: str) -> str:
    """Infer language tag from transcript text."""
    if not isinstance(text, str):
        return "unknown"
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", text))
    has_latin = bool(re.search(r"[A-Za-z]", text))
    if has_cjk and has_latin:
        return "mix"
    if has_cjk:
        return "zh"
    if has_latin:
        return "en"
    return "unknown"


def normalize_sheet_name(raw: str) -> str:
    """Normalize sheet/session marker to session1..session4."""
    if not isinstance(raw, str):
        return "unknown"
    token = raw.strip().lower()
    m = re.fullmatch(r"session([1-4])", token)
    if m:
        return f"session{m.group(1)}"
    m = re.fullmatch(r"s(\d+)", token)
    if m:
        idx = ((int(m.group(1)) - 1) % 4) + 1
        return f"session{idx}"
    return "unknown"


def _safe_load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _as_abs(root: Path, value: str) -> str:
    p = Path(value)
    if p.is_absolute():
        return str(p)
    return str((root / p).resolve())


def _sample_id_from_entry(entry: dict) -> str:
    sample_id = entry.get("sample_id")
    if isinstance(sample_id, str) and sample_id:
        return sample_id
    annot_path = Path(entry.get("annot_path", ""))
    stem = annot_path.stem
    if stem.endswith("_bs"):
        stem = stem[:-3]
    return stem


@dataclass
class SubjectData:
    train: List[dict]
    val: List[dict]
    test: List[dict]
    all_full: List[dict]


def _build_subject_meta_map(eat_root: Path, subject: str, subject_id: int) -> Dict[str, dict]:
    all_full = _safe_load_json(eat_root / "all_full.json")
    meta_map: Dict[str, dict] = {}
    for item in all_full.get("data", []):
        sample_id = _sample_id_from_entry(item)
        session = item.get("session", "")
        sheet = normalize_sheet_name(item.get("sheet", session))
        language = item.get("language", infer_language(item.get("text", "")))
        meta_map[sample_id] = {
            "sample_id": sample_id,
            "subject": subject,
            "id": int(item.get("id", subject_id)),
            "session": session,
            "sheet": sheet,
            "language": language,
            "timestamps": {
                "duration_sec": float(item.get("duration", 0.0)),
                "source_fps": int(item.get("fps", SOURCE_FPS)),
                "target_fps": TARGET_FPS,
                "eeg_sample_rate_hz": EEG_SAMPLE_RATE_HZ,
            },
            "text": item.get("text", ""),
            "emotion": item.get("emotion", ""),
        }
    return meta_map


def _convert_entry(entry: dict, eat_root: Path, subject: str, subject_id: int,
                   meta_map: Dict[str, dict]) -> dict:
    sample_id = _sample_id_from_entry(entry)
    meta = meta_map.get(sample_id, {})
    session = meta.get("session", entry.get("session", ""))
    sheet = meta.get("sheet", normalize_sheet_name(session))
    language = meta.get("language", infer_language(entry.get("text", "")))
    duration = float(entry.get("duration", meta.get("timestamps", {}).get("duration_sec", 0.0)))
    fps = int(entry.get("fps", SOURCE_FPS))
    result = {
        "annot_path": _as_abs(eat_root, entry["annot_path"]),
        "blendshape_path": _as_abs(eat_root, entry.get("blendshape_path", entry["annot_path"])),
        "audio_path": _as_abs(eat_root, entry["audio_path"]),
        "eeg_path": _as_abs(eat_root, entry["eeg_path"]),
        "id": int(entry.get("id", subject_id)),
        "annot_type": entry.get("annot_type", "inhouse_blendshape_weight"),
        "dataset": entry.get("dataset", "EATalker"),
        "duration": duration,
        "fps": fps,
        "sample_id": sample_id,
        "subject": subject,
        "session": session,
        "sheet": sheet,
        "language": language,
        "timestamps": {
            "duration_sec": duration,
            "source_fps": fps,
            "target_fps": TARGET_FPS,
            "eeg_sample_rate_hz": EEG_SAMPLE_RATE_HZ,
        },
        "text": meta.get("text", entry.get("text", "")),
        "emotion": meta.get("emotion", entry.get("emotion", "")),
    }
    return result


def _load_subject_data(source_root: Path, subject: str, subject_id: int) -> SubjectData:
    eat_root = source_root / subject / "EATalker"
    meta_map = _build_subject_meta_map(eat_root, subject, subject_id)
    converted_splits = {}
    for split_name in ("train", "val", "test"):
        payload = _safe_load_json(eat_root / f"{split_name}.json")
        converted = [
            _convert_entry(item, eat_root, subject, subject_id, meta_map)
            for item in payload.get("data", [])
        ]
        converted_splits[split_name] = converted

    all_full_payload = _safe_load_json(eat_root / "all_full.json")
    all_full = [
        _convert_entry(item, eat_root, subject, subject_id, meta_map)
        for item in all_full_payload.get("data", [])
    ]
    return SubjectData(
        train=converted_splits["train"],
        val=converted_splits["val"],
        test=converted_splits["test"],
        all_full=all_full,
    )


def _build_info(id_list: Sequence[str], split_name: str, data: Sequence[dict], seed: int,
                protocol: str) -> dict:
    total_duration = sum(float(item.get("duration", 0.0)) for item in data)
    return {
        "id_list": list(id_list),
        "data_source": "BioTalk-3D",
        "split": split_name,
        "num_samples": len(data),
        "total_duration": round(total_duration, 3),
        "source_fps": SOURCE_FPS,
        "target_fps": TARGET_FPS,
        "audio_sample_rate": 16000,
        "eeg_sample_rate_hz": EEG_SAMPLE_RATE_HZ,
        "seed": seed,
        "protocol": protocol,
    }


def _write_json(path: Path, payload: dict) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_bundle(out_dir: Path,
                  id_template: np.ndarray,
                  id_list: Sequence[str],
                  train: List[dict],
                  val: List[dict],
                  test: List[dict],
                  seed: int,
                  protocol: str,
                  tag: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "id_template.npy", id_template.astype(np.float32))

    train_payload = {"info": _build_info(id_list, "train", train, seed, protocol), "data": train}
    val_payload = {"info": _build_info(id_list, "val", val, seed, protocol), "data": val}
    test_payload = {"info": _build_info(id_list, "test", test, seed, protocol), "data": test}
    _write_json(out_dir / "train.json", train_payload)
    _write_json(out_dir / "val.json", val_payload)
    _write_json(out_dir / "test.json", test_payload)

    summary = {
        "tag": tag,
        "protocol": protocol,
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "seed": seed,
        "source_fps": SOURCE_FPS,
        "target_fps": TARGET_FPS,
        "eeg_sample_rate_hz": EEG_SAMPLE_RATE_HZ,
    }
    _write_json(out_dir / "meta.json", summary)
    return summary


def _sorted(items: Iterable[dict]) -> List[dict]:
    return sorted(items, key=lambda x: (x.get("subject", ""), x.get("sample_id", "")))


def _build_global_id_template(source_root: Path, subjects: Sequence[str]) -> np.ndarray:
    rows = []
    for subject in subjects:
        path = source_root / subject / "EATalker" / "id_template.npy"
        arr = np.load(path)
        if arr.ndim == 1:
            rows.append(arr.astype(np.float32))
        else:
            rows.append(arr[0].astype(np.float32))
    return np.stack(rows, axis=0)


def _split_zh_train_val(records: Sequence[dict], seed: int) -> Tuple[List[dict], List[dict]]:
    pool = [x for x in records if x.get("language") == "zh"]
    pool = _sorted(pool)
    rng = random.Random(seed)
    rng.shuffle(pool)
    val_n = max(1, int(round(len(pool) * 0.1)))
    val_set = {x["sample_id"] for x in pool[:val_n]}
    val = [x for x in pool if x["sample_id"] in val_set]
    train = [x for x in pool if x["sample_id"] not in val_set]
    return _sorted(train), _sorted(val)


def build_splits(source_root: Path, out_root: Path, subjects: Sequence[str], seed: int) -> None:
    id_map = {name: idx for idx, name in enumerate(subjects)}
    subject_data = {
        name: _load_subject_data(source_root, name, id_map[name])
        for name in subjects
    }
    id_template = _build_global_id_template(source_root, subjects)
    id_list = list(subjects)
    summaries = []

    # P1
    for subject in subjects:
        bundle = subject_data[subject]
        out_dir = out_root / "P1" / subject / "EATalker"
        summaries.append(
            _write_bundle(
                out_dir=out_dir,
                id_template=id_template,
                id_list=id_list,
                train=_sorted(bundle.train),
                val=_sorted(bundle.val),
                test=_sorted(bundle.test),
                seed=seed,
                protocol="P1",
                tag=f"P1_{subject}",
            ))

    # P2 LOSO
    for held_out in subjects:
        train = []
        val = []
        for subject in subjects:
            if subject == held_out:
                continue
            train.extend(subject_data[subject].train)
            val.extend(subject_data[subject].val)
        test = list(subject_data[held_out].test)
        out_dir = out_root / "P2" / f"loso_{held_out}" / "EATalker"
        summaries.append(
            _write_bundle(
                out_dir=out_dir,
                id_template=id_template,
                id_list=id_list,
                train=_sorted(train),
                val=_sorted(val),
                test=_sorted(test),
                seed=seed,
                protocol="P2",
                tag=f"P2_loso_{held_out}",
            ))

    # P3 session_holdout_s4
    all_records = []
    for subject in subjects:
        all_records.extend(subject_data[subject].all_full)
    all_records = _sorted(all_records)

    train_s = [x for x in all_records if x.get("sheet") in ("session1", "session2")]
    val_s = [x for x in all_records if x.get("sheet") == "session3"]
    test_s = [x for x in all_records if x.get("sheet") == "session4"]
    summaries.append(
        _write_bundle(
            out_dir=out_root / "P3" / "session_holdout_s4" / "EATalker",
            id_template=id_template,
            id_list=id_list,
            train=_sorted(train_s),
            val=_sorted(val_s),
            test=_sorted(test_s),
            seed=seed,
            protocol="P3",
            tag="P3_session_holdout_s4",
        ))

    # P3 language_zh_to_en
    zh_train, zh_val = _split_zh_train_val(all_records, seed=seed)
    en_test = _sorted([x for x in all_records if x.get("language") == "en"])
    summaries.append(
        _write_bundle(
            out_dir=out_root / "P3" / "language_zh_to_en" / "EATalker",
            id_template=id_template,
            id_list=id_list,
            train=zh_train,
            val=zh_val,
            test=en_test,
            seed=seed,
            protocol="P3",
            tag="P3_language_zh_to_en",
        ))

    _write_json(out_root / "summary.json", {"seed": seed, "splits": summaries})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BioTalk protocol splits")
    parser.add_argument("--source-root",
                        type=Path,
                        default=Path("/home/zym/EATalker_data_template_v2"))
    parser.add_argument("--out-root",
                        type=Path,
                        default=Path("/home/zym/EATalker/splits/biotalk"))
    parser.add_argument("--subjects",
                        type=str,
                        default=",".join(DEFAULT_SUBJECTS))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    build_splits(args.source_root, args.out_root, subjects, seed=args.seed)
    print(f"Saved splits to {args.out_root}")


if __name__ == "__main__":
    main()
