#!/usr/bin/env python3
"""Storage guard for BioTalk runs: prune stale ckpts/caches safely."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cleanup stale BioTalk artifacts")
    parser.add_argument("--runs-root",
                        type=str,
                        default="/home/zym/experiments/runs/biotalk")
    parser.add_argument("--cache-root",
                        type=str,
                        action="append",
                        default=[],
                        help="Cache roots to prune unreferenced .npy files")
    parser.add_argument("--active-manifest",
                        type=str,
                        action="append",
                        default=[],
                        help="Manifest files whose entries should be preserved")
    parser.add_argument("--current-run-dir",
                        type=str,
                        default="",
                        help="Protect this running directory from deletion")
    parser.add_argument("--home-threshold-gb", type=float, default=120.0)
    parser.add_argument("--mnt-threshold-gb", type=float, default=180.0)
    parser.add_argument("--cache-min-age-min", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _free_gb(path: str) -> float:
    stat = shutil.disk_usage(path)
    return float(stat.free) / (1024.0**3)


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _read_best_ckpt(best_file: Path) -> str:
    try:
        txt = best_file.read_text(encoding="utf-8").strip()
    except Exception:
        return ""
    if not txt:
        return ""
    return Path(txt).name


def _delete_file(path: Path, dry_run: bool) -> int:
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return 0
    if not dry_run:
        try:
            path.unlink()
        except FileNotFoundError:
            return 0
    return int(size)


def _prune_ckpts(runs_root: Path, current_run_dir: Path | None,
                 dry_run: bool) -> Dict[str, float]:
    removed = 0
    removed_bytes = 0
    scanned_dirs = 0
    for best_file in runs_root.rglob("best_ckpt.txt"):
        ckpt_dir = best_file.parent
        if current_run_dir is not None and _is_under(ckpt_dir, current_run_dir):
            continue
        scanned_dirs += 1
        keep_best = _read_best_ckpt(best_file)
        ckpts = sorted(ckpt_dir.glob("eatalker_*.pt"))
        if not ckpts:
            continue
        latest = max(ckpts, key=lambda p: p.stat().st_mtime)
        keep_names = {latest.name}
        if keep_best:
            keep_names.add(keep_best)
        for ckpt in ckpts:
            if ckpt.name in keep_names:
                continue
            removed += 1
            removed_bytes += _delete_file(ckpt, dry_run=dry_run)
    return {
        "scanned_dirs": float(scanned_dirs),
        "removed_ckpts": float(removed),
        "removed_ckpt_gb": round(removed_bytes / (1024.0**3), 4),
    }


def _collect_referenced_paths(manifests: Iterable[Path]) -> Set[Path]:
    refs: Set[Path] = set()
    for path in manifests:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        entries = payload.get("entries", payload if isinstance(payload, dict)
                              else {})
        if not isinstance(entries, dict):
            continue
        for item in entries.values():
            if not isinstance(item, dict):
                continue
            for key in ("audio_feature_path", "eeg_feature_path"):
                p = item.get(key, "")
                if isinstance(p, str) and p:
                    refs.add(Path(p).resolve())
    return refs


def _prune_cache(cache_roots: Iterable[Path], referenced: Set[Path],
                 min_age_sec: float, dry_run: bool) -> Dict[str, float]:
    now = time.time()
    removed = 0
    removed_bytes = 0
    scanned = 0
    for root in cache_roots:
        if not root.exists():
            continue
        for npy in root.rglob("*.npy"):
            scanned += 1
            rp = npy.resolve()
            if rp in referenced:
                continue
            age = now - npy.stat().st_mtime
            if age < min_age_sec:
                continue
            removed += 1
            removed_bytes += _delete_file(npy, dry_run=dry_run)
        if not dry_run:
            for d in sorted(root.rglob("*"), reverse=True):
                if d.is_dir():
                    try:
                        d.rmdir()
                    except OSError:
                        pass
    return {
        "scanned_cache_files": float(scanned),
        "removed_cache_files": float(removed),
        "removed_cache_gb": round(removed_bytes / (1024.0**3), 4),
    }


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root).resolve()
    current_run_dir = Path(
        args.current_run_dir).resolve() if args.current_run_dir else None
    cache_roots = [Path(p).resolve() for p in args.cache_root]
    manifests = [Path(p).resolve() for p in args.active_manifest]

    free_home = _free_gb("/home")
    free_mnt = _free_gb("/mnt")
    low_space = (free_home < float(args.home_threshold_gb)) or (
        free_mnt < float(args.mnt_threshold_gb))

    ckpt_stat = _prune_ckpts(runs_root,
                             current_run_dir=current_run_dir,
                             dry_run=bool(args.dry_run))
    cache_stat = {
        "scanned_cache_files": 0.0,
        "removed_cache_files": 0.0,
        "removed_cache_gb": 0.0,
    }
    if low_space and cache_roots:
        refs = _collect_referenced_paths(manifests)
        cache_stat = _prune_cache(
            cache_roots=cache_roots,
            referenced=refs,
            min_age_sec=float(args.cache_min_age_min) * 60.0,
            dry_run=bool(args.dry_run),
        )

    result = {
        "free_home_gb_before": round(free_home, 3),
        "free_mnt_gb_before": round(free_mnt, 3),
        "low_space": int(low_space),
        "dry_run": int(bool(args.dry_run)),
        **ckpt_stat,
        **cache_stat,
        "free_home_gb_after": round(_free_gb("/home"), 3),
        "free_mnt_gb_after": round(_free_gb("/mnt"), 3),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

