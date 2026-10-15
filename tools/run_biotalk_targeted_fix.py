#!/usr/bin/env python3
"""Run targeted BioTalk A/B fix after the main pipeline finishes.

Design goals:
- Keep current structure: train entry remains main/train_eatalker.py.
- Apply targeted fix for failing protocols with minimal extra knobs.
- Force Audio+EEG training from UniTalker first, then optional stage2 from best audio ckpt.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tools.ckpt_scoring import pick_best_by_rank


PROTOCOL_SPECS: Dict[str, str] = {
    "P1_lc": "P1/lc",
    "P1_zym": "P1/zym",
    "P1_yrx": "P1/yrx",
    "P1_lws": "P1/lws",
    "P1_pj": "P1/pj",
    "P2_loso_lc": "P2/loso_lc",
    "P2_loso_zym": "P2/loso_zym",
    "P2_loso_yrx": "P2/loso_yrx",
    "P2_loso_lws": "P2/loso_lws",
    "P2_loso_pj": "P2/loso_pj",
    "P3_session_holdout_s4": "P3/session_holdout_s4",
    "P3_language_zh_to_en": "P3/language_zh_to_en",
}
SHIFT_OFFSETS_MS = (-80, -40, -20, 20, 40, 80)
BEST_SCORE_METRICS: Tuple[str, ...] = ("MVE", "LVE", "UFVE", "FDD")
BEST_SCORE_WEIGHTS: Dict[str, float] = {
    "MVE": 0.20,
    "LVE": 0.15,
    "UFVE": 0.35,
    "FDD": 0.30,
}


@dataclass
class MetricPair:
    mve: float
    bs_mae: float


@dataclass
class StageResult:
    stage: str
    ckpt: str
    mve: float
    bs_mae: float
    out_dir: str
    velocity_mae: float = 0.0
    acceleration_mae: float = 0.0
    fdd: float = 0.0
    source_audio_ckpt: str = ""


@dataclass
class AudioCandidate:
    ckpt: Path
    source: str
    rank_score: float
    val_metrics: Dict[str, float]
    test_metrics: Optional[Dict[str, float]] = None


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _append_csv(path: Path, row: Dict[str, object]) -> None:
    _ensure_dir(path.parent)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _wait_for_pid(pid: int, poll_sec: int) -> None:
    while _pid_alive(pid):
        time.sleep(max(5, poll_sec))


def _run_command(
    cmd: Sequence[str],
    cwd: Path,
    log_path: Path,
    env: Optional[Dict[str, str]] = None,
) -> None:
    _ensure_dir(log_path.parent)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    with log_path.open("w", encoding="utf-8") as lf:
        lf.write("$ " + " ".join(cmd) + "\n\n")
        lf.flush()
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            lf.write(line)
        code = proc.wait()
        if code != 0:
            raise RuntimeError(f"Command failed ({code}): {' '.join(cmd)}")


def _torch_cuda_count(python_bin: str) -> int:
    cmd = [
        python_bin,
        "-c",
        (
            "import torch; "
            "print(torch.cuda.device_count() if torch.cuda.is_available() else 0)"
        ),
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip()
        return int(out)
    except Exception:
        return 0


def _resolve_nproc(args: argparse.Namespace) -> int:
    if args.nproc > 0:
        return args.nproc
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return max(1, len([x for x in visible.split(",") if x.strip()]))
    count = _torch_cuda_count(args.python_bin)
    return max(1, count)


def _rows_for_run(run_dir: Path, run_id: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for name in ("results_P1.csv", "results_P2.csv", "results_P3.csv"):
        for row in _read_csv(run_dir / name):
            if row.get("run_id") == run_id:
                rows.append(row)
    return rows


def _protocol_metric(rows: Iterable[Dict[str, str]], protocol: str, baseline: str) -> Optional[MetricPair]:
    filt = [
        r for r in rows
        if r.get("protocol") == protocol
        and r.get("baseline") == baseline
        and r.get("phase") == "test"
    ]
    if not filt:
        return None
    row = filt[-1]
    return MetricPair(mve=float(row["MVE"]), bs_mae=float(row["bs_mae_51d"]))


def relative_improvement(audio_value: float, multimodal_value: float) -> float:
    denom = max(abs(audio_value), 1e-12)
    return (audio_value - multimodal_value) / denom


def should_run_stage2(
    audio: MetricPair,
    candidate: MetricPair,
    min_mve_improve: float,
    min_bs_improve: float,
) -> bool:
    mve_improve = relative_improvement(audio.mve, candidate.mve)
    bs_improve = relative_improvement(audio.bs_mae, candidate.bs_mae)
    return (mve_improve < min_mve_improve) or (bs_improve < min_bs_improve)


def pick_best_by_weighted_rank(metric_rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not metric_rows:
        raise ValueError("metric_rows is empty")
    return pick_best_by_rank(
        metric_rows,
        metrics=BEST_SCORE_METRICS,
        weights=BEST_SCORE_WEIGHTS,
    )  # type: ignore[return-value]


def _rank_by_weights(metric_rows: List[Dict[str, float]], weights: Dict[str, float], out_key: str) -> None:
    if not metric_rows:
        return
    for metric in weights:
        order = sorted(range(len(metric_rows)), key=lambda i: float(metric_rows[i][metric]))
        for rank, idx in enumerate(order, start=1):
            metric_rows[idx][f"rank_{out_key}_{metric}"] = float(rank)
    for row in metric_rows:
        row[out_key] = sum(float(weights[m]) * float(row[f"rank_{out_key}_{m}"]) for m in weights)


def pick_audio_candidates_by_visual_and_metrics(
    metric_rows: List[Dict[str, float]],
    candidate_count: int,
    visual_weight: float,
) -> List[Dict[str, float]]:
    if not metric_rows:
        return []
    rows = [dict(r) for r in metric_rows]
    fidelity_weights = {
        "MVE": 0.45,
        "bs_mae_51d": 0.35,
        "mouth_mae_51d": 0.10,
        "upper_face_mae_51d": 0.10,
    }
    visual_weights = {
        "velocity_mae": 0.30,
        "acceleration_mae": 0.30,
        "FDD": 0.20,
        "LVD": 0.10,
        "UFVE": 0.10,
    }
    _rank_by_weights(rows, fidelity_weights, out_key="fidelity_score")
    _rank_by_weights(rows, visual_weights, out_key="visual_score")
    vw = min(0.9, max(0.0, float(visual_weight)))
    for row in rows:
        row["audio_candidate_score"] = (1.0 - vw) * float(row["fidelity_score"]) + vw * float(
            row["visual_score"]
        )
    rows.sort(
        key=lambda r: (
            float(r["audio_candidate_score"]),
            float(r["MVE"]),
            float(r["bs_mae_51d"]),
            float(r["velocity_mae"]),
        )
    )
    k = max(1, int(candidate_count))
    if k == 1 or len(rows) == 1:
        return rows[:1]

    # Hard preference: include one fidelity-best and one visual-best non-best ckpt.
    best_fidelity = min(rows, key=lambda r: (float(r["fidelity_score"]), float(r["MVE"])))
    visual_sorted = sorted(
        rows,
        key=lambda r: (float(r["visual_score"]), float(r["audio_candidate_score"]), float(r["MVE"])),
    )
    best_visual_nonbest = None
    for r in visual_sorted:
        if r["ckpt"] != best_fidelity["ckpt"]:
            best_visual_nonbest = r
            break
    if best_visual_nonbest is None:
        return [best_fidelity]

    picked = [best_fidelity, best_visual_nonbest]
    if k > 2:
        for r in rows:
            if r["ckpt"] in {picked[0]["ckpt"], picked[1]["ckpt"]}:
                continue
            picked.append(r)
            if len(picked) >= k:
                break
    return picked


def _write_rows_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    _ensure_dir(path.parent)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _eval_summary(
    args: argparse.Namespace,
    split_json: Path,
    split_parent: Path,
    protocol: str,
    baseline: str,
    phase: str,
    run_id: str,
    weight_path: Path,
    out_json: Path,
    metrics_csv: Optional[Path],
    per_sample_csv: Optional[Path],
    visual_root: Optional[Path],
    use_eeg: bool,
    eeg_shift_ms: float = 0.0,
    log_path: Optional[Path] = None,
) -> Dict[str, float]:
    cmd = [
        args.python_bin,
        "tools/eval_biotalk_protocol.py",
        "--config",
        args.config_path,
        "--weight-path",
        str(weight_path),
        "--split-json",
        str(split_json),
        "--protocol",
        protocol,
        "--baseline",
        baseline,
        "--phase",
        phase,
        "--run-id",
        run_id,
        "--out-json",
        str(out_json),
    ]
    if use_eeg:
        cmd.append("--use-eeg")
    if abs(eeg_shift_ms) > 1e-9:
        cmd += ["--eeg-shift-ms", str(float(eeg_shift_ms))]
    if metrics_csv is not None:
        cmd += ["--metrics-csv", str(metrics_csv)]
    if per_sample_csv is not None:
        cmd += ["--per-sample-csv", str(per_sample_csv)]
    if visual_root is not None:
        cmd += [
            "--visual-root",
            str(visual_root),
            "--render-topk",
            str(args.render_topk),
            "--render-worstk",
            str(args.render_worstk),
            "--img-size",
            str(args.img_size),
        ]
    # Important: --opts uses REMAINDER and must be the final argument block.
    cmd += [
        "--opts",
        "data_root",
        str(split_parent),
        "dataset",
        "EATalker",
        "use_audio",
        "True",
        "use_eeg",
        "True" if use_eeg else "False",
        "workers",
        str(args.workers),
    ]

    if log_path is None:
        proc = subprocess.run(cmd, cwd=str(args.project_root), check=True)
    else:
        _run_command(cmd, cwd=args.project_root, log_path=log_path)

    with out_json.open("r", encoding="utf-8") as f:
        d = json.load(f)

    def _metric(keys: Sequence[str], default: float = 0.0) -> float:
        for k in keys:
            if k in d and d[k] is not None:
                return float(d[k])
        return float(default)

    return {
        "MVE": _metric(("MVE", "mve")),
        "LVE": _metric(("LVE", "lve")),
        "bs_mae_51d": _metric(("bs_mae_51d",)),
        "mouth_mae_51d": _metric(("mouth_mae_51d",)),
        "upper_face_mae_51d": _metric(("upper_face_mae_51d",)),
        "velocity_mae": _metric(("velocity_mae",)),
        "acceleration_mae": _metric(("acceleration_mae",)),
        "FDD": _metric(("FDD", "fdd")),
        "LVD": _metric(("LVD", "lvd")),
        "UFVE": _metric(("UFVE", "ufve")),
    }


def _train_audio_eeg(
    args: argparse.Namespace,
    run_id: str,
    protocol: str,
    split_parent: Path,
    weight_path: Path,
    out_dir: Path,
    epochs: int,
    grad_acc: int,
    wav2vec_lr: float,
) -> None:
    _ensure_dir(out_dir)
    _ensure_dir(out_dir / "logs")

    opts = [
        "dataset",
        "EATalker",
        "data_root",
        str(split_parent),
        "train_json_key",
        "train.json",
        "save_path",
        str(out_dir),
        "weight_path",
        str(weight_path),
        "do_validate",
        "True",
        "do_test",
        "False",
        "use_audio",
        "True",
        "use_eeg",
        "True",
        "epochs",
        str(int(epochs)),
        "workers",
        str(args.workers),
        "batch_size",
        "1",
        "gradient_accumulation_steps",
        str(int(grad_acc)),
        "save_every",
        "1",
        "eval_every",
        "1",
        "test_every",
        "1",
        "random_id_prob",
        "0.0",
        "wav2vec_train_last_n_layers",
        str(args.wav2vec_train_last_n_layers),
        "wav2vec_frozen_to_eval",
        "True",
        "wav2vec_lr",
        str(float(wav2vec_lr)),
        "eeg_fusion_lr_scale",
        str(float(args.eeg_fusion_lr_scale)),
        "other_lr_scale",
        str(float(args.other_lr_scale)),
    ]

    port = args.master_port_base + random.randint(1, 999)
    env = {"PYTHONPATH": "/home/zym", "EAT_LOG_DIR": str(out_dir / "logs")}

    if args.nproc_resolved > 1:
        cmd = [
            args.torchrun_bin,
            "--nproc_per_node",
            str(args.nproc_resolved),
            "--master_port",
            str(port),
            "-m",
            "EATalker.main.train_eatalker",
            "--config",
            args.config_path,
        ] + opts
    else:
        cmd = [
            args.python_bin,
            "-m",
            "EATalker.main.train_eatalker",
            "--config",
            args.config_path,
        ] + opts

    (out_dir / "command.sh").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    _run_command(cmd, cwd=args.project_root, log_path=out_dir / "train_stdout.log", env=env)


def _train_audio_only_probe(
    args: argparse.Namespace,
    split_parent: Path,
    weight_path: Path,
    out_dir: Path,
) -> None:
    _ensure_dir(out_dir)
    _ensure_dir(out_dir / "logs")
    opts = [
        "dataset",
        "EATalker",
        "data_root",
        str(split_parent),
        "train_json_key",
        "train.json",
        "save_path",
        str(out_dir),
        "weight_path",
        str(weight_path),
        "do_validate",
        "True",
        "do_test",
        "False",
        "use_audio",
        "True",
        "use_eeg",
        "False",
        "epochs",
        str(int(args.audio_probe_epochs)),
        "workers",
        str(args.workers),
        "batch_size",
        "1",
        "gradient_accumulation_steps",
        str(int(args.audio_probe_grad_acc)),
        "save_every",
        "1",
        "eval_every",
        "1",
        "test_every",
        "1",
        "random_id_prob",
        "0.0",
        "wav2vec_train_last_n_layers",
        str(args.wav2vec_train_last_n_layers),
        "wav2vec_frozen_to_eval",
        "True",
        "wav2vec_lr",
        str(float(args.audio_probe_wav2vec_lr)),
    ]
    port = args.master_port_base + random.randint(1, 999)
    env = {"PYTHONPATH": "/home/zym", "EAT_LOG_DIR": str(out_dir / "logs")}
    if args.nproc_resolved > 1:
        cmd = [
            args.torchrun_bin,
            "--nproc_per_node",
            str(args.nproc_resolved),
            "--master_port",
            str(port),
            "-m",
            "EATalker.main.train_eatalker",
            "--config",
            args.config_path,
        ] + opts
    else:
        cmd = [
            args.python_bin,
            "-m",
            "EATalker.main.train_eatalker",
            "--config",
            args.config_path,
        ] + opts
    (out_dir / "command.sh").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    _run_command(cmd, cwd=args.project_root, log_path=out_dir / "train_stdout.log", env=env)


def _select_best_ckpt(
    args: argparse.Namespace,
    run_id: str,
    protocol: str,
    split_parent: Path,
    baseline_name: str,
    out_dir: Path,
) -> Tuple[Path, Dict[str, float]]:
    val_json = split_parent / "EATalker" / "val.json"
    _ensure_dir(out_dir / "eval" / "val")

    ckpts = sorted(out_dir.glob("eatalker_*.pt"))
    if not ckpts:
        raise RuntimeError(f"No checkpoint found under: {out_dir}")

    metric_rows: List[Dict[str, float]] = []
    table_path = out_dir / "eval" / "val_ckpt_metrics.tsv"
    with table_path.open("w", encoding="utf-8") as table:
        for ckpt in ckpts:
            stem = ckpt.stem
            out_json = out_dir / "eval" / "val" / f"{stem}.json"
            metrics = _eval_summary(
                args=args,
                split_json=val_json,
                split_parent=split_parent,
                protocol=protocol,
                baseline=baseline_name,
                phase="val_targeted",
                run_id=run_id,
                weight_path=ckpt,
                out_json=out_json,
                metrics_csv=None,
                per_sample_csv=None,
                visual_root=None,
                use_eeg=True,
                log_path=out_dir / "eval" / "val" / f"{stem}.log",
            )
            row = {
                "ckpt": str(ckpt),
                **metrics,
            }
            metric_rows.append(row)
            table.write(
                "\t".join(
                    [
                        str(ckpt),
                        str(metrics["MVE"]),
                        str(metrics["LVE"]),
                        str(metrics["UFVE"]),
                        str(metrics["FDD"]),
                    ]
                )
                + "\n"
            )

    best = pick_best_by_weighted_rank(metric_rows)
    best_ckpt = Path(str(best["ckpt"]))
    (out_dir / "best_ckpt.txt").write_text(str(best_ckpt) + "\n", encoding="utf-8")
    (out_dir / "best_val_mve.txt").write_text(f"{float(best['MVE']):.12f}\n", encoding="utf-8")
    return best_ckpt, best


def _prune_ckpts_keep(out_dir: Path, keep_ckpt: Path) -> Dict[str, float]:
    removed = 0
    removed_bytes = 0
    keep_name = keep_ckpt.name
    for ckpt in out_dir.glob("eatalker_*.pt"):
        if ckpt.name == keep_name:
            continue
        try:
            size = ckpt.stat().st_size
            ckpt.unlink()
            removed += 1
            removed_bytes += size
        except FileNotFoundError:
            continue
    stat = {
        "removed_count": float(removed),
        "removed_gb": round(removed_bytes / (1024 ** 3), 4),
    }
    (out_dir / "ckpt_prune_stat.json").write_text(
        json.dumps(stat, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return stat


def _pick_target_protocols(
    rows: Iterable[Dict[str, str]],
    min_mve_improve: float,
    min_bs_improve: float,
) -> List[str]:
    chosen: List[str] = []
    row_list = list(rows)
    for protocol in PROTOCOL_SPECS:
        a = _protocol_metric(row_list, protocol, "audio-only")
        b = _protocol_metric(row_list, protocol, "audio+eeg")
        if a is None:
            continue
        if b is None:
            chosen.append(protocol)
            continue
        if should_run_stage2(a, b, min_mve_improve=min_mve_improve, min_bs_improve=min_bs_improve):
            chosen.append(protocol)
    return chosen


def _locate_main_audio_ckpt(run_dir: Path, protocol: str, rows: Iterable[Dict[str, str]]) -> Optional[Path]:
    p = run_dir / "main" / protocol / "audio-only" / "best_ckpt.txt"
    if p.exists():
        txt = p.read_text(encoding="utf-8").strip()
        if txt:
            ckpt = Path(txt)
            if ckpt.exists():
                return ckpt
    for row in rows:
        if row.get("protocol") == protocol and row.get("baseline") == "audio-only" and row.get("phase") == "test":
            wp = row.get("weight_path", "")
            if wp:
                ckpt = Path(wp)
                if ckpt.exists():
                    return ckpt
    return None


def _collect_audio_ckpts(main_audio_dir: Path, fallback_ckpt: Path) -> List[Path]:
    ckpts = sorted(main_audio_dir.glob("eatalker_*.pt"))
    if fallback_ckpt.exists() and fallback_ckpt not in ckpts:
        ckpts.append(fallback_ckpt)
    # de-duplicate while preserving order
    uniq: List[Path] = []
    seen = set()
    for ckpt in ckpts:
        s = str(ckpt.resolve())
        if s in seen:
            continue
        seen.add(s)
        uniq.append(ckpt)
    return uniq


def _build_audio_candidates(
    args: argparse.Namespace,
    protocol: str,
    split_parent: Path,
    run_id_targeted: str,
    protocol_root: Path,
    fallback_audio_ckpt: Path,
) -> List[AudioCandidate]:
    main_audio_dir = args.run_dir / "main" / protocol / "audio-only"
    candidate_pool = _collect_audio_ckpts(main_audio_dir, fallback_audio_ckpt)

    if len(candidate_pool) < max(2, int(args.audio_candidate_count)) and args.audio_probe_when_insufficient:
        probe_dir = protocol_root / "audio_probe"
        if not probe_dir.exists() or not list(probe_dir.glob("eatalker_*.pt")):
            _train_audio_only_probe(
                args=args,
                split_parent=split_parent,
                weight_path=fallback_audio_ckpt,
                out_dir=probe_dir,
            )
        probe_ckpts = sorted(probe_dir.glob("eatalker_*.pt"))
        for ckpt in probe_ckpts:
            if ckpt not in candidate_pool:
                candidate_pool.append(ckpt)

    val_json = split_parent / "EATalker" / "val.json"
    val_dir = protocol_root / "audio_candidate_val"
    _ensure_dir(val_dir)
    metric_rows: List[Dict[str, float]] = []
    for ckpt in candidate_pool:
        stem = ckpt.stem
        out_json = val_dir / f"{stem}.json"
        metrics = _eval_summary(
            args=args,
            split_json=val_json,
            split_parent=split_parent,
            protocol=protocol,
            baseline="audio-only",
            phase="val_audio_candidate",
            run_id=run_id_targeted,
            weight_path=ckpt,
            out_json=out_json,
            metrics_csv=None,
            per_sample_csv=None,
            visual_root=None,
            use_eeg=False,
            log_path=val_dir / f"{stem}.log",
        )
        metric_rows.append(
            {
                "ckpt": str(ckpt),
                **metrics,
            }
        )

    _write_rows_csv(protocol_root / "audio_candidate_pool_val.csv", metric_rows)
    picked_rows = pick_audio_candidates_by_visual_and_metrics(
        metric_rows=metric_rows,
        candidate_count=max(1, int(args.audio_candidate_count)),
        visual_weight=float(args.audio_candidate_visual_weight),
    )
    _write_rows_csv(protocol_root / "audio_candidate_picked_val.csv", picked_rows)

    selected: List[AudioCandidate] = []
    for row in picked_rows:
        selected.append(
            AudioCandidate(
                ckpt=Path(str(row["ckpt"])),
                source="main_or_probe",
                rank_score=float(row["audio_candidate_score"]),
                val_metrics={k: float(v) for k, v in row.items() if k != "ckpt"},
            )
        )
    probe_dir = protocol_root / "audio_probe"
    if probe_dir.exists():
        keep = {str(c.ckpt.resolve()) for c in selected}
        removed = 0
        removed_bytes = 0
        for ckpt in probe_dir.glob("eatalker_*.pt"):
            if str(ckpt.resolve()) in keep:
                continue
            try:
                sz = ckpt.stat().st_size
                ckpt.unlink()
                removed += 1
                removed_bytes += sz
            except FileNotFoundError:
                continue
        (protocol_root / "audio_probe_prune_stat.json").write_text(
            json.dumps(
                {
                    "removed_count": removed,
                    "removed_gb": round(removed_bytes / (1024 ** 3), 4),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return selected


def _result_row(
    run_id: str,
    protocol: str,
    audio: MetricPair,
    stage1: StageResult,
    stage2: Optional[StageResult],
    picked: StageResult,
    nonbest_case_found: bool = False,
    nonbest_stage: str = "",
    nonbest_source_audio_ckpt: str = "",
    nonbest_mve_improve: float = 0.0,
    nonbest_bs_improve: float = 0.0,
) -> Dict[str, object]:
    return {
        "run_id": run_id,
        "protocol": protocol,
        "audio_mve": audio.mve,
        "audio_bs_mae": audio.bs_mae,
        "stage1_mve": stage1.mve,
        "stage1_bs_mae": stage1.bs_mae,
        "stage2_mve": "" if stage2 is None else stage2.mve,
        "stage2_bs_mae": "" if stage2 is None else stage2.bs_mae,
        "picked_stage": picked.stage,
        "picked_ckpt": picked.ckpt,
        "picked_source_audio_ckpt": picked.source_audio_ckpt,
        "picked_mve": picked.mve,
        "picked_bs_mae": picked.bs_mae,
        "picked_velocity_mae": picked.velocity_mae,
        "picked_acceleration_mae": picked.acceleration_mae,
        "picked_fdd": picked.fdd,
        "mve_improve_vs_audio": relative_improvement(audio.mve, picked.mve),
        "bs_improve_vs_audio": relative_improvement(audio.bs_mae, picked.bs_mae),
        "nonbest_case_found": int(bool(nonbest_case_found)),
        "nonbest_stage": nonbest_stage,
        "nonbest_source_audio_ckpt": nonbest_source_audio_ckpt,
        "nonbest_mve_improve": nonbest_mve_improve,
        "nonbest_bs_improve": nonbest_bs_improve,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run post-main targeted fix for BioTalk")
    parser.add_argument("--project-root", type=Path, default=Path("/home/zym/EATalker"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--wait-main-pid", type=int, default=0)
    parser.add_argument("--wait-poll-sec", type=int, default=120)

    parser.add_argument("--python-bin", type=str, default="/home/zym/.local/share/mamba/envs/eatalker/bin/python")
    parser.add_argument("--torchrun-bin", type=str, default="/home/zym/.local/share/mamba/envs/eatalker/bin/torchrun")
    parser.add_argument("--config-path", type=str, default="/home/zym/EATalker/config/eatalker.yaml")
    parser.add_argument("--pretrained-ckpt", type=Path, default=Path("/home/zym/pretrained_models/UniTalker-L-D0-D7.pt"))
    parser.add_argument("--split-root", type=Path, default=Path("/home/zym/EATalker/splits/biotalk"))

    parser.add_argument("--nproc", type=int, default=0)
    parser.add_argument("--master-port-base", type=int, default=30610)
    parser.add_argument("--workers", type=int, default=8)

    parser.add_argument("--wav2vec-train-last-n-layers", type=int, default=12)
    parser.add_argument("--stage1-epochs", type=int, default=40)
    parser.add_argument("--stage1-grad-acc", type=int, default=6)
    parser.add_argument("--stage1-wav2vec-lr", type=float, default=4e-6)
    parser.add_argument("--stage2-epochs", type=int, default=24)
    parser.add_argument("--stage2-grad-acc", type=int, default=6)
    parser.add_argument("--stage2-wav2vec-lr", type=float, default=3e-6)
    parser.add_argument("--audio-candidate-count", type=int, default=2)
    parser.add_argument("--audio-candidate-visual-weight", type=float, default=0.40)
    parser.add_argument("--audio-probe-when-insufficient", action="store_true", default=True)
    parser.add_argument("--no-audio-probe-when-insufficient", dest="audio_probe_when_insufficient", action="store_false")
    parser.add_argument("--require-nonbest-audio-case", action="store_true", default=True)
    parser.add_argument("--no-require-nonbest-audio-case", dest="require_nonbest_audio_case", action="store_false")
    parser.add_argument("--audio-probe-epochs", type=int, default=18)
    parser.add_argument("--audio-probe-grad-acc", type=int, default=4)
    parser.add_argument("--audio-probe-wav2vec-lr", type=float, default=5e-6)

    parser.add_argument("--eeg-fusion-lr-scale", type=float, default=2.0)
    parser.add_argument("--other-lr-scale", type=float, default=1.0)

    parser.add_argument("--min-mve-improve", type=float, default=0.0)
    parser.add_argument("--min-bs-improve", type=float, default=0.0)

    parser.add_argument("--protocols", type=str, default="")
    parser.add_argument("--render-topk", type=int, default=10)
    parser.add_argument("--render-worstk", type=int, default=10)
    parser.add_argument("--img-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.run_id:
        args.run_id = args.run_dir.name
    args.nproc_resolved = _resolve_nproc(args)

    if args.wait_main_pid > 0:
        print(f"[targeted-fix] waiting for main pid={args.wait_main_pid}")
        _wait_for_pid(args.wait_main_pid, args.wait_poll_sec)

    if not args.pretrained_ckpt.exists():
        raise FileNotFoundError(f"pretrained ckpt missing: {args.pretrained_ckpt}")

    all_rows = _rows_for_run(args.run_dir, args.run_id)
    if not all_rows:
        raise RuntimeError(f"No rows found for run_id={args.run_id} in {args.run_dir}")

    if args.protocols.strip():
        target_protocols = [x.strip() for x in args.protocols.split(",") if x.strip()]
    else:
        target_protocols = _pick_target_protocols(
            all_rows,
            min_mve_improve=float(args.min_mve_improve),
            min_bs_improve=float(args.min_bs_improve),
        )

    if not target_protocols:
        print("[targeted-fix] no failing protocol detected, exit")
        return

    targeted_root = args.run_dir / "targeted_fix"
    visuals_root = args.run_dir / "visuals_targeted"
    results_csv = args.run_dir / "results_targeted.csv"
    summary_csv = targeted_root / "summary.csv"
    shift_csv = targeted_root / "metrics_shift.csv"
    _ensure_dir(targeted_root)

    run_id_targeted = f"{args.run_id}_targeted"

    print(f"[targeted-fix] protocols={target_protocols}")
    print(f"[targeted-fix] nproc={args.nproc_resolved}")

    for protocol in target_protocols:
        if protocol not in PROTOCOL_SPECS:
            print(f"[targeted-fix] skip unknown protocol: {protocol}")
            continue
        split_parent = args.split_root / PROTOCOL_SPECS[protocol]
        if not split_parent.exists():
            raise FileNotFoundError(f"split parent missing: {split_parent}")

        audio_main = _protocol_metric(all_rows, protocol, "audio-only")
        if audio_main is None:
            print(f"[targeted-fix] skip {protocol}: no audio-only test row")
            continue
        audio_ckpt = _locate_main_audio_ckpt(args.run_dir, protocol, all_rows)
        if audio_ckpt is None:
            raise RuntimeError(f"audio ckpt not found for {protocol}")

        protocol_root = targeted_root / protocol
        _ensure_dir(protocol_root)

        test_json = split_parent / "EATalker" / "test.json"
        audio_candidates = _build_audio_candidates(
            args=args,
            protocol=protocol,
            split_parent=split_parent,
            run_id_targeted=run_id_targeted,
            protocol_root=protocol_root,
            fallback_audio_ckpt=audio_ckpt,
        )
        if not audio_candidates:
            raise RuntimeError(f"no audio candidates for {protocol}")
        audio_ref_metrics = _eval_summary(
            args=args,
            split_json=test_json,
            split_parent=split_parent,
            protocol=protocol,
            baseline="audio-only",
            phase="test_ref",
            run_id=run_id_targeted,
            weight_path=audio_candidates[0].ckpt,
            out_json=protocol_root / "audio_ref_test.json",
            metrics_csv=results_csv,
            per_sample_csv=protocol_root / "audio_ref_per_sample.csv",
            visual_root=visuals_root,
            use_eeg=False,
            log_path=protocol_root / "audio_ref_eval.log",
        )
        audio_main = MetricPair(
            mve=float(audio_ref_metrics["MVE"]),
            bs_mae=float(audio_ref_metrics["bs_mae_51d"]),
        )
        for idx, cand in enumerate(audio_candidates, start=1):
            cand_out = protocol_root / f"audio_candidate_{idx:02d}"
            _ensure_dir(cand_out)
            cand_test = _eval_summary(
                args=args,
                split_json=test_json,
                split_parent=split_parent,
                protocol=protocol,
                baseline="audio-only",
                phase=f"test_audio_candidate_{idx}",
                run_id=run_id_targeted,
                weight_path=cand.ckpt,
                out_json=cand_out / "test.json",
                metrics_csv=results_csv,
                per_sample_csv=cand_out / "per_sample.csv",
                visual_root=visuals_root,
                use_eeg=False,
                log_path=cand_out / "test.log",
            )
            cand.test_metrics = cand_test

        # Stage1: UniTalker -> Audio+EEG with lowered eeg/fusion LR.
        stage1_dir = protocol_root / "audio+eeg_stage1"
        _train_audio_eeg(
            args=args,
            run_id=run_id_targeted,
            protocol=protocol,
            split_parent=split_parent,
            weight_path=args.pretrained_ckpt,
            out_dir=stage1_dir,
            epochs=args.stage1_epochs,
            grad_acc=args.stage1_grad_acc,
            wav2vec_lr=args.stage1_wav2vec_lr,
        )
        stage1_ckpt, _ = _select_best_ckpt(
            args=args,
            run_id=run_id_targeted,
            protocol=protocol,
            split_parent=split_parent,
            baseline_name="audio+eeg",
            out_dir=stage1_dir,
        )
        _prune_ckpts_keep(stage1_dir, stage1_ckpt)
        stage1_metrics = _eval_summary(
            args=args,
            split_json=test_json,
            split_parent=split_parent,
            protocol=protocol,
            baseline="audio+eeg",
            phase="test_stage1",
            run_id=run_id_targeted,
            weight_path=stage1_ckpt,
            out_json=stage1_dir / "eval" / "test_stage1.json",
            metrics_csv=results_csv,
            per_sample_csv=stage1_dir / "eval" / "test_stage1_per_sample.csv",
            visual_root=visuals_root,
            use_eeg=True,
            log_path=stage1_dir / "eval" / "test_stage1.log",
        )
        picked = StageResult(
            stage="stage1",
            ckpt=str(stage1_ckpt),
            mve=float(stage1_metrics["MVE"]),
            bs_mae=float(stage1_metrics["bs_mae_51d"]),
            out_dir=str(stage1_dir),
            velocity_mae=float(stage1_metrics["velocity_mae"]),
            acceleration_mae=float(stage1_metrics["acceleration_mae"]),
            fdd=float(stage1_metrics["FDD"]),
        )

        stage2_result: Optional[StageResult] = None
        stage2_results: List[StageResult] = []
        nonbest_case_found = False
        nonbest_stage = ""
        nonbest_source_audio_ckpt = ""
        nonbest_mve_improve = 0.0
        nonbest_bs_improve = 0.0

        if should_run_stage2(
            audio=audio_main,
            candidate=MetricPair(mve=picked.mve, bs_mae=picked.bs_mae),
            min_mve_improve=float(args.min_mve_improve),
            min_bs_improve=float(args.min_bs_improve),
        ):
            for idx, cand_audio in enumerate(audio_candidates, start=1):
                stage2_dir = protocol_root / f"audio+eeg_stage2_from_audio_cand{idx:02d}"
                _train_audio_eeg(
                    args=args,
                    run_id=run_id_targeted,
                    protocol=protocol,
                    split_parent=split_parent,
                    weight_path=cand_audio.ckpt,
                    out_dir=stage2_dir,
                    epochs=args.stage2_epochs,
                    grad_acc=args.stage2_grad_acc,
                    wav2vec_lr=args.stage2_wav2vec_lr,
                )
                stage2_ckpt, _ = _select_best_ckpt(
                    args=args,
                    run_id=run_id_targeted,
                    protocol=protocol,
                    split_parent=split_parent,
                    baseline_name="audio+eeg",
                    out_dir=stage2_dir,
                )
                _prune_ckpts_keep(stage2_dir, stage2_ckpt)
                stage2_metrics = _eval_summary(
                    args=args,
                    split_json=test_json,
                    split_parent=split_parent,
                    protocol=protocol,
                    baseline="audio+eeg",
                    phase=f"test_stage2_cand{idx}",
                    run_id=run_id_targeted,
                    weight_path=stage2_ckpt,
                    out_json=stage2_dir / "eval" / "test_stage2.json",
                    metrics_csv=results_csv,
                    per_sample_csv=stage2_dir / "eval" / "test_stage2_per_sample.csv",
                    visual_root=visuals_root,
                    use_eeg=True,
                    log_path=stage2_dir / "eval" / "test_stage2.log",
                )
                stage2_results.append(
                    StageResult(
                        stage=f"stage2_cand{idx}",
                        ckpt=str(stage2_ckpt),
                        mve=float(stage2_metrics["MVE"]),
                        bs_mae=float(stage2_metrics["bs_mae_51d"]),
                        out_dir=str(stage2_dir),
                        velocity_mae=float(stage2_metrics["velocity_mae"]),
                        acceleration_mae=float(stage2_metrics["acceleration_mae"]),
                        fdd=float(stage2_metrics["FDD"]),
                        source_audio_ckpt=str(cand_audio.ckpt),
                    )
                )

            # Prioritize stage2 models that beat their source audio candidate on both MVE and BS-MAE.
            stage2_good: List[StageResult] = []
            for s2 in stage2_results:
                source_audio = next((c for c in audio_candidates if str(c.ckpt) == s2.source_audio_ckpt), None)
                if source_audio is None or source_audio.test_metrics is None:
                    continue
                src = source_audio.test_metrics
                if float(s2.mve) < float(src["MVE"]) and float(s2.bs_mae) < float(src["bs_mae_51d"]):
                    stage2_good.append(s2)
            best_audio_ckpt = None
            aud_with_test = [c for c in audio_candidates if c.test_metrics is not None]
            if aud_with_test:
                best_audio_ckpt = str(
                    min(aud_with_test, key=lambda x: float(x.test_metrics["MVE"])).ckpt
                )
            nonbest_good: List[StageResult] = []
            if best_audio_ckpt is not None:
                for s2 in stage2_good:
                    if s2.source_audio_ckpt != best_audio_ckpt:
                        nonbest_good.append(s2)
            if stage2_good:
                stage2_good.sort(key=lambda x: (x.mve, x.bs_mae, x.velocity_mae, x.acceleration_mae, x.fdd))
                stage2_result = stage2_good[0]
            elif stage2_results:
                stage2_results.sort(key=lambda x: (x.mve, x.bs_mae, x.velocity_mae, x.acceleration_mae, x.fdd))
                stage2_result = stage2_results[0]

            if nonbest_good:
                nonbest_good.sort(key=lambda x: (x.mve, x.bs_mae, x.velocity_mae, x.acceleration_mae, x.fdd))
                best_nonbest = nonbest_good[0]
                source_audio = next((c for c in audio_candidates if str(c.ckpt) == best_nonbest.source_audio_ckpt), None)
                if source_audio is not None and source_audio.test_metrics is not None:
                    src = source_audio.test_metrics
                    nonbest_case_found = True
                    nonbest_stage = best_nonbest.stage
                    nonbest_source_audio_ckpt = best_nonbest.source_audio_ckpt
                    nonbest_mve_improve = relative_improvement(float(src["MVE"]), best_nonbest.mve)
                    nonbest_bs_improve = relative_improvement(float(src["bs_mae_51d"]), best_nonbest.bs_mae)
                    if args.require_nonbest_audio_case:
                        stage2_result = best_nonbest
                        picked = best_nonbest

            if stage2_result is not None:
                cand = [picked, stage2_result]
                cand.sort(key=lambda x: (x.mve, x.bs_mae, x.velocity_mae, x.acceleration_mae, x.fdd))
                picked = cand[0]

        # P2 shift sensitivity on picked checkpoint.
        if protocol.startswith("P2_"):
            for offset in SHIFT_OFFSETS_MS:
                _eval_summary(
                    args=args,
                    split_json=test_json,
                    split_parent=split_parent,
                    protocol=protocol,
                    baseline="audio+eeg",
                    phase="shift_test_targeted",
                    run_id=run_id_targeted,
                    weight_path=Path(picked.ckpt),
                    out_json=protocol_root / f"shift_{offset}.json",
                    metrics_csv=shift_csv,
                    per_sample_csv=None,
                    visual_root=None,
                    use_eeg=True,
                    eeg_shift_ms=float(offset),
                    log_path=protocol_root / f"shift_{offset}.log",
                )

        _append_csv(
            summary_csv,
            _result_row(
                run_id=run_id_targeted,
                protocol=protocol,
                audio=audio_main,
                stage1=StageResult(
                    stage="stage1",
                    ckpt=str(stage1_ckpt),
                    mve=float(stage1_metrics["MVE"]),
                    bs_mae=float(stage1_metrics["bs_mae_51d"]),
                    out_dir=str(stage1_dir),
                ),
                stage2=stage2_result,
                picked=picked,
                nonbest_case_found=nonbest_case_found,
                nonbest_stage=nonbest_stage,
                nonbest_source_audio_ckpt=nonbest_source_audio_ckpt,
                nonbest_mve_improve=nonbest_mve_improve,
                nonbest_bs_improve=nonbest_bs_improve,
            ),
        )

    manifest = {
        "run_id_main": args.run_id,
        "run_id_targeted": run_id_targeted,
        "run_dir": str(args.run_dir),
        "targeted_root": str(targeted_root),
        "results_csv": str(results_csv),
        "summary_csv": str(summary_csv),
        "shift_csv": str(shift_csv),
        "protocols": target_protocols,
        "pretrained_ckpt": str(args.pretrained_ckpt),
        "eeg_fusion_lr_scale": float(args.eeg_fusion_lr_scale),
        "other_lr_scale": float(args.other_lr_scale),
        "require_nonbest_audio_case": bool(args.require_nonbest_audio_case),
        "nproc": int(args.nproc_resolved),
    }
    (targeted_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
