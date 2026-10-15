#!/usr/bin/env python3
"""Run BioTalk metric sanity checks on one split and persist logs/artifacts."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List


def _run_and_tee(cmd: List[str], log_path: Path, env: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"\n[{datetime.now().isoformat()}] CMD: {' '.join(shlex.quote(c) for c in cmd)}\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
        code = proc.wait()
        if code != 0:
            raise RuntimeError(f"Command failed with code={code}: {' '.join(cmd)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BioTalk metric sanity checks")
    parser.add_argument("--config", type=str, default="/home/zym/EATalker/config/eatalker.yaml")
    parser.add_argument(
        "--weight-path",
        type=str,
        default="/home/zym/pretrained_models/UniTalker-L-D0-D7.pt",
    )
    parser.add_argument(
        "--split-json",
        type=str,
        default="/home/zym/EATalker/splits/biotalk/P1/lc/EATalker/test.json",
    )
    parser.add_argument("--protocol", type=str, default="P1_sanity_lc")
    parser.add_argument("--baseline", type=str, default="audio+eeg")
    parser.add_argument("--out-dir", type=str, default="/home/zym/experiments/runs/biotalk/sanity_metrics")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--run-normal-eval", action="store_true")
    parser.add_argument("--eeg-shift-ms", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = "/home/zym" + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")

    eval_py = "/home/zym/EATalker/tools/eval_biotalk_protocol.py"
    py = sys.executable

    sanity_cmd = [
        py,
        eval_py,
        "--config",
        args.config,
        "--weight-path",
        args.weight_path,
        "--split-json",
        args.split_json,
        "--protocol",
        args.protocol,
        "--baseline",
        args.baseline,
        "--phase",
        "test",
        "--use-eeg",
        "--sanity-checks",
        "--sanity-pred-eq-gt",
        "--metrics-csv",
        str(out_dir / "metrics_sanity.csv"),
        "--per-sample-csv",
        str(out_dir / "per_sample_sanity.csv"),
        "--out-json",
        str(out_dir / "summary_sanity.json"),
        "--eeg-shift-ms",
        str(args.eeg_shift_ms),
    ]
    if args.max_samples > 0:
        sanity_cmd += ["--max-samples", str(args.max_samples)]
    sanity_cmd += ["--opts", "workers", "0"]
    _run_and_tee(sanity_cmd, out_dir / "sanity_check.log", env=env)

    if args.run_normal_eval:
        normal_cmd = [
            py,
            eval_py,
            "--config",
            args.config,
            "--weight-path",
            args.weight_path,
            "--split-json",
            args.split_json,
            "--protocol",
            args.protocol,
            "--baseline",
            args.baseline,
            "--phase",
            "test",
            "--use-eeg",
            "--sanity-checks",
            "--metrics-csv",
            str(out_dir / "metrics_normal.csv"),
            "--per-sample-csv",
            str(out_dir / "per_sample_normal.csv"),
            "--out-json",
            str(out_dir / "summary_normal.json"),
            "--eeg-shift-ms",
            str(args.eeg_shift_ms),
        ]
        if args.max_samples > 0:
            normal_cmd += ["--max-samples", str(args.max_samples)]
        normal_cmd += ["--opts", "workers", "0"]
        _run_and_tee(normal_cmd, out_dir / "sanity_check.log", env=env)

    print(f"[done] sanity artifacts at {out_dir}")


if __name__ == "__main__":
    main()
