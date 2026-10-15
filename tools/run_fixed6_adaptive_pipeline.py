#!/usr/bin/env python3
"""
Run adaptive secondary padding for 6 problematic sessions and rerender affected videos only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence


FIXED6_PLAN: Dict[str, Dict[str, object]] = {
    "pj": {
        "identity_id": 4,
        "session_pad_ms": {"s17": 500, "s19": 400},
    },
    "lws": {
        "identity_id": 3,
        "session_pad_ms": {"s13": 500, "s14": 600, "s15": 500},
    },
    "yrx": {
        "identity_id": 2,
        "session_pad_ms": {"s11": 500},
    },
}


def run_cmd(cmd: Sequence[str], env: Dict[str, str] | None = None) -> None:
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Adaptive fixed-6 secondary pad + targeted rerender")
    ap.add_argument("--project-root", default="/home/zym/EATalker")
    ap.add_argument("--raw-root", default="/home/zym/EaTalker_Dataset_v2")
    ap.add_argument("--out-root", default="/home/zym/EATalker_data_template_v2")
    ap.add_argument("--labels-xlsx", default="/home/zym/EaTalker_Dataset_v2/情感文本朗读素材.xlsx")
    ap.add_argument("--subjects", default="pj,lws,yrx")
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--decimate", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tail-alpha", type=float, default=0.25)
    ap.add_argument("--tail-abs-floor", type=float, default=0.015)
    ap.add_argument("--tail-frame-ms", type=int, default=10)
    ap.add_argument("--tail-min-keep-ms", type=int, default=40)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    labels_xlsx = Path(args.labels_xlsx)
    process_script = project_root / "tools" / "process_eatalker_v2.py"

    subjects = [x.strip() for x in args.subjects.split(",") if x.strip()]
    for s in subjects:
        if s not in FIXED6_PLAN:
            raise ValueError(f"{s} is not in FIXED6_PLAN")

    global_rows: List[dict] = []
    summary: List[dict] = []

    render_env = os.environ.copy()
    render_env["PYTHONPATH"] = str(project_root.parent)

    for subject in subjects:
        plan = FIXED6_PLAN[subject]
        identity_id = int(plan["identity_id"])
        session_pad_ms = dict(plan["session_pad_ms"])
        dataset_root = out_root / subject / "EATalker"
        all_full_path = dataset_root / "all_full.json"
        if not all_full_path.exists():
            raise FileNotFoundError(f"Missing {all_full_path}")

        all_full = load_json(all_full_path)
        rows = all_full["data"]
        target_rows = [r for r in rows if r.get("session") in session_pad_ms]
        target_ids = sorted({r["sample_id"] for r in target_rows})
        if not target_ids:
            raise ValueError(f"{subject}: no target samples found for sessions {sorted(session_pad_ms)}")

        pad_file = dataset_root / "fixed6_target_samples.txt"
        pad_file.write_text("\n".join(target_ids) + "\n", encoding="utf-8")
        session_pad_file = dataset_root / "fixed6_session_pad_map.json"
        write_json(session_pad_file, session_pad_ms)

        default_pad = int(max(session_pad_ms.values()))
        secondary_cmd = [
            sys.executable,
            str(process_script),
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
            str(default_pad),
            "--pad-file",
            str(pad_file),
            "--session-pad-file",
            str(session_pad_file),
            "--tail-adaptive",
            "--tail-alpha",
            str(args.tail_alpha),
            "--tail-abs-floor",
            str(args.tail_abs_floor),
            "--tail-frame-ms",
            str(args.tail_frame_ms),
            "--tail-min-keep-ms",
            str(args.tail_min_keep_ms),
            "--eeg-preprocess",
            "strict",
            "--timecode-mode",
            "adaptive",
            "--timecode-repair",
            "stitch",
        ]
        if not args.dry_run:
            run_cmd(secondary_cmd)

        # refresh all_full and build per-sample list
        all_full_after = load_json(all_full_path)
        row_map = {r["sample_id"]: r for r in all_full_after["data"]}
        split_map: Dict[str, str] = {}
        for split in ("train", "val", "test"):
            split_json = load_json(dataset_root / f"{split}.json")
            for r in split_json["data"]:
                split_map[r["sample_id"]] = split

        subject_rows: List[dict] = []
        for sid in target_ids:
            r = row_map[sid]
            rec = {
                "subject": subject,
                "sample_id": sid,
                "session": r["session"],
                "split": split_map.get(sid, ""),
                "effective_end_pad_ms": int(r.get("end_pad_ms", 0)),
                "audio_path": r["audio_path"],
                "annot_path": r["annot_path"],
            }
            subject_rows.append(rec)
            global_rows.append(rec)

        subject_csv = dataset_root / "fixed6_adaptive_processed_list.csv"
        with open(subject_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "subject",
                    "sample_id",
                    "session",
                    "split",
                    "effective_end_pad_ms",
                    "audio_path",
                    "annot_path",
                ],
            )
            writer.writeheader()
            writer.writerows(subject_rows)

        # targeted rerender only for affected samples
        temp_root = Path("/tmp/fixed6_rerender") / subject
        render_root = out_root / subject / "render_demo"
        npz_root = render_root / "npz"
        npz_root.mkdir(parents=True, exist_ok=True)

        rendered = 0
        for split in ("train", "val", "test"):
            split_rows = [r for r in subject_rows if r["split"] == split]
            if not split_rows:
                continue
            temp_split = temp_root / split
            clear_dir(temp_split)
            for r in split_rows:
                src_wav = dataset_root / r["audio_path"]
                src_bs = dataset_root / r["annot_path"]
                shutil.copy2(src_wav, temp_split / src_wav.name)
                shutil.copy2(src_bs, temp_split / src_bs.name)

            render_cmd = [
                sys.executable,
                "-m",
                "EATalker.tools.render_testset_gt",
                "--test-dir",
                str(temp_split),
                "--out-npz",
                str(npz_root / f"{split}_gt_vertices_fixed6_adaptive.npz"),
                "--out-videos",
                str(render_root / split),
                "--device",
                args.device,
                "--decimate",
                str(args.decimate),
                "--img-size",
                str(args.img_size),
            ]
            if not args.dry_run:
                run_cmd(render_cmd, env=render_env)
            rendered += len(split_rows)

        summary.append(
            {
                "subject": subject,
                "target_sample_count": len(target_ids),
                "processed_csv": str(subject_csv),
                "session_pad_ms": session_pad_ms,
                "tail_adaptive": {
                    "alpha": float(args.tail_alpha),
                    "abs_floor": float(args.tail_abs_floor),
                    "frame_ms": int(args.tail_frame_ms),
                    "min_keep_ms": int(args.tail_min_keep_ms),
                },
                "rendered_sample_count": rendered,
            }
        )

    # global artifacts
    global_csv = out_root / "fixed6_adaptive_processed_list_all.csv"
    with open(global_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subject",
                "sample_id",
                "session",
                "split",
                "effective_end_pad_ms",
                "audio_path",
                "annot_path",
            ],
        )
        writer.writeheader()
        writer.writerows(global_rows)

    summary_path = out_root / "fixed6_adaptive_summary.json"
    write_json(summary_path, {"subjects": summary})
    print("Done.")
    print("Summary:", summary_path)
    print("Global list:", global_csv)


if __name__ == "__main__":
    main()
