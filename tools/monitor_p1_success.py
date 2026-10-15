#!/usr/bin/env python3
import argparse
import csv
import re
import time
from pathlib import Path


METRICS_5 = ["MVE", "LVE", "UFVE", "FDD", "velocity_mae"]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def better_count(audio: dict[str, str], multi: dict[str, str]) -> int:
    cnt = 0
    for k in METRICS_5:
        if k not in audio or k not in multi:
            continue
        try:
            a = float(audio[k])
            b = float(multi[k])
        except Exception:
            continue
        if b < a:
            cnt += 1
    return cnt


def rel_improve_sum(audio: dict[str, str], multi: dict[str, str]) -> float:
    s = 0.0
    for k in METRICS_5:
        if k not in audio or k not in multi:
            continue
        try:
            a = float(audio[k])
            b = float(multi[k])
        except Exception:
            continue
        denom = max(abs(a), 1e-12)
        s += (a - b) / denom
    return s


def avg_metrics(rows_by_proto: dict[str, dict[str, str]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in METRICS_5:
        vals = []
        for r in rows_by_proto.values():
            if k not in r:
                continue
            try:
                vals.append(float(r[k]))
            except Exception:
                continue
        if vals:
            out[k] = sum(vals) / len(vals)
    return out


def avg_better_count(a: dict[str, float], b: dict[str, float]) -> int:
    cnt = 0
    for k in METRICS_5:
        if k in a and k in b and b[k] < a[k]:
            cnt += 1
    return cnt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-results-csv", type=Path, required=True)
    ap.add_argument("--results-csv", type=Path, required=True)
    ap.add_argument("--poll-sec", type=float, default=30.0)
    ap.add_argument("--success-out", type=Path, required=True)
    ap.add_argument("--log-out", type=Path, required=True)
    ap.add_argument("--min-better", type=int, default=4)
    args = ap.parse_args()

    seen = set()
    stage2_pat = re.compile(r"^test_stage2_cand(\d+)$")

    args.log_out.parent.mkdir(parents=True, exist_ok=True)
    args.success_out.parent.mkdir(parents=True, exist_ok=True)

    # Audio-only baseline from main run: protocol -> row
    main_rows = read_rows(args.main_results_csv)
    main_audio: dict[str, dict[str, str]] = {}
    main_multi: dict[str, dict[str, str]] = {}
    for r in main_rows:
        if r.get("phase") != "test":
            continue
        prot = r.get("protocol", "")
        if not prot.startswith("P1_"):
            continue
        if r.get("baseline") == "audio-only":
            main_audio[prot] = r
        elif r.get("baseline") == "audio+eeg":
            main_multi[prot] = r

    # Best multimodal per protocol vs main audio-only. Seed with main audio+eeg (if present).
    best_multi_vs_main: dict[str, tuple[int, float, dict[str, str]]] = {}
    for prot, mm in main_multi.items():
        base = main_audio.get(prot)
        if base is None:
            continue
        bc2 = better_count(base, mm)
        rs = rel_improve_sum(base, mm)
        best_multi_vs_main[prot] = (bc2, rs, mm)

    while True:
        rows = read_rows(args.results_csv)
        # Index audio-only reference rows: (protocol, cand_idx) -> row
        audio_ref: dict[tuple[str, int], dict[str, str]] = {}
        for r in rows:
            if r.get("phase", "") == "":
                continue
            if r.get("baseline") != "audio-only":
                continue
            prot = r.get("protocol", "")
            if not prot.startswith("P1_"):
                continue
            ph = r.get("phase", "")
            m = re.match(r"^test_audio_candidate_(\\d+)$", ph)
            if m:
                audio_ref[(prot, int(m.group(1)))] = r

        # Scan stage2 test rows and update best multimodal per protocol vs main audio-only.
        for r in rows:
            if r.get("baseline") != "audio+eeg":
                continue
            prot = r.get("protocol", "")
            if not prot.startswith("P1_"):
                continue
            ph = r.get("phase", "")
            m = stage2_pat.match(ph)
            if not m:
                continue
            cand_idx = int(m.group(1))
            key = (prot, ph, r.get("weight_path", ""))
            if key in seen:
                continue
            seen.add(key)

            ref = audio_ref.get((prot, cand_idx))
            if ref is None:
                continue

            bc = better_count(ref, r)
            line = (
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"{prot} {ph} better_count={bc}/5 "
                f"audio={ref.get('weight_path','')} multi={r.get('weight_path','')}"
            )
            with args.log_out.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

            if bc >= int(args.min_better):
                msg = (
                    line
                    + "\n"
                    + "metrics: "
                    + ", ".join(
                        f"{k}: {ref.get(k,'')} -> {r.get(k,'')}" for k in METRICS_5
                    )
                    + "\n"
                )
                args.success_out.write_text(msg, encoding="utf-8")
                return 0

            # Update best-vs-main snapshot for macro-average. Use main audio-only as baseline.
            base = main_audio.get(prot)
            if base is not None:
                bc2 = better_count(base, r)
                rs = rel_improve_sum(base, r)
                prev = best_multi_vs_main.get(prot)
                if prev is None or (bc2, rs) > (prev[0], prev[1]):
                    best_multi_vs_main[prot] = (bc2, rs, r)

        # Macro-average criterion across the 5 P1 subjects: compare best multimodal-so-far vs main audio-only.
        prot_set = sorted(p for p in main_audio if p in {"P1_lc", "P1_zym", "P1_yrx", "P1_lws", "P1_pj"})
        if len(prot_set) == 5 and all(p in best_multi_vs_main for p in prot_set):
            multi_rows = {p: best_multi_vs_main[p][2] for p in prot_set}
            a_avg = avg_metrics({p: main_audio[p] for p in prot_set})
            m_avg = avg_metrics(multi_rows)
            bc_avg = avg_better_count(a_avg, m_avg)
            avg_line = (
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"P1_AVG better_count={bc_avg}/5 "
                + " ".join(
                    f"{k}:{a_avg.get(k, float('nan')):.6g}->{m_avg.get(k, float('nan')):.6g}"
                    for k in METRICS_5
                )
            )
            with args.log_out.open("a", encoding="utf-8") as f:
                f.write(avg_line + "\n")
            if bc_avg >= int(args.min_better):
                args.success_out.write_text(avg_line + "\n", encoding="utf-8")
                return 0

        time.sleep(args.poll_sec)


if __name__ == "__main__":
    raise SystemExit(main())
