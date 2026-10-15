#!/usr/bin/env python3
"""Cleanup non-best checkpoints for completed BioTalk runs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def cleanup_once(run_dir: Path, dry_run: bool = False) -> dict:
    removed_count = 0
    removed_bytes = 0
    skipped_dirs = 0
    cleaned_dirs = 0

    for d in sorted(run_dir.glob("*/*")):
        if not d.is_dir():
            continue
        best_file = d / "best_ckpt.txt"
        if not best_file.exists():
            skipped_dirs += 1
            continue
        best_txt = best_file.read_text(encoding="utf-8").strip()
        if not best_txt:
            skipped_dirs += 1
            continue
        best_name = Path(best_txt).name
        ckpts = list(d.glob("eatalker_*.pt"))
        if not ckpts:
            skipped_dirs += 1
            continue
        cleaned_dirs += 1
        for ckpt in ckpts:
            if ckpt.name == best_name:
                continue
            try:
                size = ckpt.stat().st_size
            except FileNotFoundError:
                continue
            if not dry_run:
                try:
                    ckpt.unlink()
                except FileNotFoundError:
                    continue
            removed_count += 1
            removed_bytes += size

    return {
        "removed_count": removed_count,
        "removed_gb": round(removed_bytes / (1024 ** 3), 3),
        "cleaned_dirs": cleaned_dirs,
        "skipped_dirs": skipped_dirs,
        "dry_run": dry_run,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cleanup non-best checkpoints")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help=".../runs/biotalk/<run_id>/main",
    )
    parser.add_argument("--interval-sec", type=int, default=600)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.watch:
        while True:
            stat = cleanup_once(args.run_dir, dry_run=args.dry_run)
            stat["ts"] = int(time.time())
            print(json.dumps(stat, ensure_ascii=False), flush=True)
            time.sleep(max(30, args.interval_sec))
    else:
        stat = cleanup_once(args.run_dir, dry_run=args.dry_run)
        print(json.dumps(stat, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
