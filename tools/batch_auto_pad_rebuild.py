#!/usr/bin/env python3
"""
One-click batch pipeline for EaTalker_Dataset_v2 subjects.

Per subject:
1) build-all (no pad)
2) auto-detect suspect clips by tail energy
3) secondary-pad-selected with selective end-pad on suspects
4) split train/val/test
5) verify basic audio/BS/EEG alignment and export secondary-processed lists
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import wave
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


def parse_subjects_arg(subjects_arg: str) -> List[str]:
    if not subjects_arg.strip():
        return []
    out = []
    for x in subjects_arg.split(","):
        t = x.strip()
        if t:
            out.append(t)
    return out


def discover_subjects(raw_root: Path) -> List[str]:
    subjects: List[str] = []
    for p in sorted(raw_root.iterdir()):
        if not p.is_dir():
            continue
        has_audio = (p / "audio").is_dir() or (p / "Audio").is_dir()
        if has_audio and (p / "EEG").is_dir():
            subjects.append(p.name)
    return subjects


def should_mark_suspect(
    end_rms: float,
    prev_rms: float,
    end_rms_threshold: float,
    prev_rms_threshold: float,
) -> bool:
    return end_rms > end_rms_threshold and prev_rms > prev_rms_threshold


def run_cmd(cmd: Sequence[str]) -> None:
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def read_tail_rms(wav_path: Path, window_sec: float = 0.04) -> Tuple[float, float]:
    with wave.open(str(wav_path), "rb") as wf:
        nframes = wf.getnframes()
        fr = wf.getframerate()
        sig = np.frombuffer(wf.readframes(nframes), dtype="<i2").astype(np.float32) / 32768.0
    w = max(1, int(round(window_sec * fr)))
    end = sig[-w:]
    prev = sig[-2 * w : -w] if len(sig) >= 2 * w else sig[:w]
    end_rms = float(np.sqrt(np.mean(end**2)))
    prev_rms = float(np.sqrt(np.mean(prev**2)))
    return end_rms, prev_rms


def detect_suspects(
    dataset_root: Path,
    end_rms_threshold: float,
    prev_rms_threshold: float,
) -> Tuple[List[str], List[dict]]:
    all_full_path = dataset_root / "all_full.json"
    payload = json.loads(all_full_path.read_text(encoding="utf-8"))
    records = payload["data"]

    suspects: List[str] = []
    metrics: List[dict] = []
    for r in records:
        wav_path = dataset_root / r["audio_path"]
        end_rms, prev_rms = read_tail_rms(wav_path)
        suspect = should_mark_suspect(
            end_rms=end_rms,
            prev_rms=prev_rms,
            end_rms_threshold=end_rms_threshold,
            prev_rms_threshold=prev_rms_threshold,
        )
        if suspect:
            suspects.append(r["sample_id"])
        metrics.append(
            {
                "sample_id": r["sample_id"],
                "session": r["session"],
                "emotion": r["emotion"],
                "duration": r["duration"],
                "end_rms": end_rms,
                "prev_rms": prev_rms,
                "is_suspect": int(suspect),
            }
        )

    suspects = sorted(set(suspects))
    metrics = sorted(metrics, key=lambda x: x["end_rms"], reverse=True)
    return suspects, metrics


def write_metrics_csv(metrics: Sequence[dict], out_csv: Path) -> None:
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "session",
                "emotion",
                "duration",
                "end_rms",
                "prev_rms",
                "is_suspect",
            ],
        )
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)


def verify_alignment(dataset_root: Path) -> Dict[str, float]:
    payload = json.loads((dataset_root / "all_full.json").read_text(encoding="utf-8"))
    bad_bs = 0
    bad_eeg = 0
    audio_bs_diff_ms: List[float] = []
    audio_eeg_diff_ms: List[float] = []
    for r in payload["data"]:
        duration = float(r["duration"])
        expected = max(1, int(round(duration * 60.0)))
        bs = np.load(dataset_root / r["annot_path"])
        eeg = np.load(dataset_root / r["eeg_path"])
        bs_duration = float(len(bs)) / 60.0
        eeg_duration = float(eeg.shape[0]) / 1000.0
        audio_bs_diff_ms.append(abs(duration - bs_duration) * 1000.0)
        audio_eeg_diff_ms.append(abs(duration - eeg_duration) * 1000.0)
        if abs(len(bs) - expected) > 1:
            bad_bs += 1
        if eeg.ndim != 2 or eeg.shape[1] != 60 or eeg.shape[0] <= 0:
            bad_eeg += 1
    bs_arr = np.asarray(audio_bs_diff_ms, dtype=np.float64)
    eeg_arr = np.asarray(audio_eeg_diff_ms, dtype=np.float64)
    return {
        "bad_bs": float(bad_bs),
        "bad_eeg": float(bad_eeg),
        "audio_bs_p95_ms": float(np.percentile(bs_arr, 95)) if len(bs_arr) else 0.0,
        "audio_bs_max_ms": float(np.max(bs_arr)) if len(bs_arr) else 0.0,
        "audio_eeg_max_ms": float(np.max(eeg_arr)) if len(eeg_arr) else 0.0,
    }


def read_secondary_rows(dataset_root: Path, subject: str) -> List[dict]:
    payload = json.loads((dataset_root / "all_full.json").read_text(encoding="utf-8"))
    metrics_rows = {}
    metrics_csv = dataset_root / "auto_pad_metrics.csv"
    if metrics_csv.exists():
        with open(metrics_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sid = row.get("sample_id", "")
                if sid:
                    metrics_rows[sid] = row

    out: List[dict] = []
    for r in payload["data"]:
        end_pad_ms = int(r.get("end_pad_ms", 0))
        if end_pad_ms <= 0:
            continue
        sid = r["sample_id"]
        m = metrics_rows.get(sid, {})
        out.append(
            {
                "subject": subject,
                "sample_id": sid,
                "session": r.get("session", ""),
                "emotion": r.get("emotion", ""),
                "end_pad_ms": int(end_pad_ms),
                "end_rms": float(m.get("end_rms", 0.0) or 0.0),
                "prev_rms": float(m.get("prev_rms", 0.0) or 0.0),
                "reason": "tail_rms_high",
            }
        )
    out.sort(key=lambda x: x["sample_id"])
    return out


def write_secondary_rows_csv(rows: Sequence[dict], out_csv: Path) -> None:
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subject",
                "sample_id",
                "session",
                "emotion",
                "end_pad_ms",
                "end_rms",
                "prev_rms",
                "reason",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_global_secondary_md(all_rows: Sequence[dict], out_md: Path) -> None:
    by_subject: Dict[str, List[dict]] = {}
    for row in all_rows:
        by_subject.setdefault(row["subject"], []).append(row)

    lines: List[str] = []
    lines.append("# Secondary Processed Samples")
    lines.append("")
    lines.append(f"Total samples: {len(all_rows)}")
    lines.append("")
    for subject in sorted(by_subject.keys()):
        rows = sorted(by_subject[subject], key=lambda x: x["sample_id"])
        lines.append(f"## {subject}")
        lines.append("")
        lines.append(f"Count: {len(rows)}")
        lines.append("")
        lines.append("| sample_id | session | emotion | end_pad_ms | end_rms | prev_rms | reason |")
        lines.append("|---|---|---|---:|---:|---:|---|")
        for r in rows:
            lines.append(
                f"| {r['sample_id']} | {r['session']} | {r['emotion']} | "
                f"{r['end_pad_ms']} | {r['end_rms']:.6f} | {r['prev_rms']:.6f} | {r['reason']} |"
            )
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def resolve_dirty_file(dirty_root: Path | None, subject: str) -> str:
    if dirty_root is None:
        return ""
    candidate = dirty_root / f"{subject}.txt"
    return str(candidate) if candidate.exists() else ""


def build_subject(
    project_root: Path,
    raw_root: Path,
    out_root: Path,
    labels_xlsx: Path,
    subject: str,
    identity_id: int,
    end_pad_ms: int,
    end_rms_threshold: float,
    prev_rms_threshold: float,
    seed: int,
    dirty_file: str,
    eeg_preprocess: str,
    timecode_mode: str,
    timecode_repair: str,
) -> dict:
    script = project_root / "tools" / "process_eatalker_v2.py"
    dataset_root = out_root / subject / "EATalker"

    base_cmd = [
        sys.executable,
        str(script),
        "build-all",
        "--raw-root",
        str(raw_root),
        "--out-root",
        str(out_root),
        "--labels-xlsx",
        str(labels_xlsx),
        "--subject",
        subject,
        "--identity-id",
        str(identity_id),
        "--end-pad-ms",
        "0",
        "--eeg-preprocess",
        str(eeg_preprocess),
        "--timecode-mode",
        str(timecode_mode),
        "--timecode-repair",
        str(timecode_repair),
    ]
    run_cmd(base_cmd)

    suspects, metrics = detect_suspects(
        dataset_root=dataset_root,
        end_rms_threshold=end_rms_threshold,
        prev_rms_threshold=prev_rms_threshold,
    )
    suspects_path = dataset_root / "auto_pad_samples.txt"
    suspects_path.write_text("\n".join(suspects) + ("\n" if suspects else ""), encoding="utf-8")
    write_metrics_csv(metrics, dataset_root / "auto_pad_metrics.csv")

    secondary_rows: List[dict] = []
    rebuilt = False
    if suspects and end_pad_ms > 0:
        rebuild_cmd = [
            sys.executable,
            str(script),
            "secondary-pad-selected",
            "--raw-root",
            str(raw_root),
            "--out-root",
            str(out_root),
            "--labels-xlsx",
            str(labels_xlsx),
            "--subject",
            subject,
            "--identity-id",
            str(identity_id),
            "--end-pad-ms",
            str(end_pad_ms),
            "--pad-file",
            str(suspects_path),
            "--eeg-preprocess",
            str(eeg_preprocess),
            "--timecode-mode",
            str(timecode_mode),
            "--timecode-repair",
            str(timecode_repair),
        ]
        run_cmd(rebuild_cmd)
        rebuilt = True

    split_cmd = [
        sys.executable,
        str(script),
        "split",
        "--raw-root",
        str(raw_root),
        "--out-root",
        str(out_root),
        "--labels-xlsx",
        str(labels_xlsx),
        "--subject",
        subject,
        "--identity-id",
        str(identity_id),
        "--seed",
        str(seed),
        "--eeg-preprocess",
        str(eeg_preprocess),
        "--timecode-mode",
        str(timecode_mode),
        "--timecode-repair",
        str(timecode_repair),
    ]
    if dirty_file:
        split_cmd += ["--dirty-file", dirty_file]
    run_cmd(split_cmd)

    secondary_rows = read_secondary_rows(dataset_root, subject)
    write_secondary_rows_csv(secondary_rows, dataset_root / "secondary_processed_list.csv")

    verify = verify_alignment(dataset_root)
    all_full = json.loads((dataset_root / "all_full.json").read_text(encoding="utf-8"))
    split_train = json.loads((dataset_root / "train.json").read_text(encoding="utf-8"))
    split_val = json.loads((dataset_root / "val.json").read_text(encoding="utf-8"))
    split_test = json.loads((dataset_root / "test.json").read_text(encoding="utf-8"))
    end_pad_samples = sorted(
        [r["sample_id"] for r in all_full["data"] if int(r.get("end_pad_ms", 0)) > 0]
    )
    secondary_match = sorted(suspects) == end_pad_samples
    return {
        "subject": subject,
        "identity_id": identity_id,
        "eeg_preprocess": eeg_preprocess,
        "timecode_mode": timecode_mode,
        "timecode_repair": timecode_repair,
        "suspect_count": len(suspects),
        "rebuilt_with_pad": rebuilt,
        "end_pad_ms": end_pad_ms if rebuilt else 0,
        "secondary_list_count": len(secondary_rows),
        "secondary_samples_match_suspects": bool(secondary_match),
        "bad_bs": int(verify["bad_bs"]),
        "bad_eeg": int(verify["bad_eeg"]),
        "audio_bs_p95_ms": float(verify["audio_bs_p95_ms"]),
        "audio_bs_max_ms": float(verify["audio_bs_max_ms"]),
        "audio_eeg_max_ms": float(verify["audio_eeg_max_ms"]),
        "all_full_count": len(all_full["data"]),
        "train_count": len(split_train["data"]),
        "val_count": len(split_val["data"]),
        "test_count": len(split_test["data"]),
        "dirty_file": dirty_file,
        "secondary_processed_csv": str(dataset_root / "secondary_processed_list.csv"),
        "auto_pad_samples": str(suspects_path),
        "secondary_rows": secondary_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch auto-pad rebuild for EaTalker_Dataset_v2")
    parser.add_argument("--project-root", default="/home/zym/EATalker")
    parser.add_argument("--raw-root", default="/home/zym/EaTalker_Dataset_v2")
    parser.add_argument("--out-root", default="/home/zym/EATalker_data_template_v2")
    parser.add_argument("--labels-xlsx", default="/home/zym/EaTalker_Dataset_v2/情感文本朗读素材.xlsx")
    parser.add_argument("--subjects", default="", help="Comma-separated subject list, e.g. lc,wy,zf")
    parser.add_argument("--identity-start", type=int, default=0)
    parser.add_argument("--end-pad-ms", type=int, default=120)
    parser.add_argument("--end-rms-threshold", type=float, default=0.05)
    parser.add_argument("--prev-rms-threshold", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dirty-root", default="", help="Optional dir containing {subject}.txt")
    parser.add_argument("--eeg-preprocess", default="strict", choices=("strict", "none"))
    parser.add_argument(
        "--timecode-mode",
        default="robust",
        choices=("raw", "ff_only", "robust", "adaptive"),
    )
    parser.add_argument("--timecode-repair", default="stitch", choices=("stitch", "none"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    labels_xlsx = Path(args.labels_xlsx)
    dirty_root = Path(args.dirty_root) if args.dirty_root else None

    subjects = parse_subjects_arg(args.subjects)
    if not subjects:
        subjects = discover_subjects(raw_root)
    if not subjects:
        raise ValueError(f"No subjects found under {raw_root}")

    print("Subjects:", subjects)
    summary: List[dict] = []
    global_secondary_rows: List[dict] = []
    for i, subject in enumerate(subjects):
        identity_id = args.identity_start + i
        print(f"\n===== Subject {subject} (identity_id={identity_id}) =====")
        dirty_file = resolve_dirty_file(dirty_root, subject)
        result = build_subject(
            project_root=project_root,
            raw_root=raw_root,
            out_root=out_root,
            labels_xlsx=labels_xlsx,
            subject=subject,
            identity_id=identity_id,
            end_pad_ms=args.end_pad_ms,
            end_rms_threshold=args.end_rms_threshold,
            prev_rms_threshold=args.prev_rms_threshold,
            seed=args.seed,
            dirty_file=dirty_file,
            eeg_preprocess=args.eeg_preprocess,
            timecode_mode=args.timecode_mode,
            timecode_repair=args.timecode_repair,
        )
        summary.append(result)
        global_secondary_rows.extend(result.pop("secondary_rows", []))
        print("Subject done:", json.dumps(result, ensure_ascii=False))

    summary_path = out_root / "auto_pad_batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    global_csv = out_root / "secondary_processed_list_all.csv"
    global_md = out_root / "secondary_processed_list_all.md"
    write_secondary_rows_csv(global_secondary_rows, global_csv)
    write_global_secondary_md(global_secondary_rows, global_md)
    print("\nBatch done.")
    print("Summary:", summary_path)
    print("Global secondary CSV:", global_csv)
    print("Global secondary MD:", global_md)


if __name__ == "__main__":
    main()
