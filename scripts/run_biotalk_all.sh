#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/zym/EATalker}"
PYTHON_BIN="${PYTHON_BIN:-/home/zym/.local/share/mamba/envs/eatalker/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/zym/.local/share/mamba/envs/eatalker/bin/torchrun}"
CONFIG_PATH="${CONFIG_PATH:-/home/zym/EATalker/config/eatalker.yaml}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-/home/zym/pretrained_models/UniTalker-L-D0-D7.pt}"
SOURCE_ROOT="${SOURCE_ROOT:-/home/zym/EATalker_data_template_v2}"
SPLIT_ROOT="${SPLIT_ROOT:-/home/zym/EATalker/splits/biotalk}"
RUN_ROOT="${RUN_ROOT:-/home/zym/experiments/runs/biotalk}"
SEED="${SEED:-42}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

EPOCHS_MAIN="${EPOCHS_MAIN:-40}"
GRAD_ACC_MAIN="${GRAD_ACC_MAIN:-4}"
WAV2VEC_LR_MAIN="${WAV2VEC_LR_MAIN:-0.000005}"
WAV2VEC_LR_RETRY="${WAV2VEC_LR_RETRY:-}"
EPOCHS_RETRY="${EPOCHS_RETRY:-60}"
GRAD_ACC_RETRY="${GRAD_ACC_RETRY:-6}"
SAVE_EVERY_MAIN="${SAVE_EVERY_MAIN:-2}"
SAVE_EVERY_RETRY="${SAVE_EVERY_RETRY:-2}"
EVAL_EVERY_MAIN="${EVAL_EVERY_MAIN:-1}"
EVAL_EVERY_RETRY="${EVAL_EVERY_RETRY:-1}"
WORKERS="${WORKERS:-8}"
IMG_SIZE="${IMG_SIZE:-256}"
RENDER_TOPK="${RENDER_TOPK:-10}"
RENDER_WORSTK="${RENDER_WORSTK:-10}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29610}"
PROTOCOLS_ONLY="${PROTOCOLS_ONLY:-}"

TARGET_MVE_IMPROVE="${TARGET_MVE_IMPROVE:-0.05}"
TARGET_BSMAE_IMPROVE="${TARGET_BSMAE_IMPROVE:-0.03}"

mkdir -p "${RUN_ROOT}"
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
mkdir -p "${RUN_DIR}"
export PYTHONPATH="/home/zym"
if [[ -z "${WAV2VEC_LR_RETRY}" ]]; then
  WAV2VEC_LR_RETRY="$("${PYTHON_BIN}" - <<PY
print(float("${WAV2VEC_LR_MAIN}") * 0.8)
PY
)"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "python binary not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "config not found: ${CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -f "${PRETRAINED_CKPT}" ]]; then
  echo "pretrained ckpt not found: ${PRETRAINED_CKPT}" >&2
  exit 1
fi
if [[ ! -d "${SOURCE_ROOT}" ]]; then
  echo "source root not found: ${SOURCE_ROOT}" >&2
  exit 1
fi

if [[ -n "${NPROC_OVERRIDE:-}" ]]; then
  NPROC="${NPROC_OVERRIDE}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _gpu_arr <<<"${CUDA_VISIBLE_DEVICES}"
  NPROC="${#_gpu_arr[@]}"
else
  # 自动探测可用 GPU 数，避免在未设置 CUDA_VISIBLE_DEVICES 时退化为单卡
  NPROC="$("${PYTHON_BIN}" - <<'PY'
import os
count = 0
try:
    import torch
    if torch.cuda.is_available():
        count = int(torch.cuda.device_count())
except Exception:
    count = 0
if count <= 0:
    try:
        import subprocess
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL)
        lines = [x for x in out.splitlines() if x.strip().startswith("GPU ")]
        count = len(lines)
    except Exception:
        count = 0
print(max(1, count))
PY
)"
fi
if [[ "${NPROC}" -lt 1 ]]; then
  NPROC=1
fi

echo "RUN_ID=${RUN_ID}"
echo "NPROC=${NPROC}"

echo "${SEED}" > "${RUN_DIR}/seed.txt"
echo "${PRETRAINED_CKPT}" > "${RUN_DIR}/pretrained_ckpt.txt"
git -C "${PROJECT_ROOT}" rev-parse HEAD > "${RUN_DIR}/git_commit.txt"

"${PYTHON_BIN}" - <<PY > "${RUN_DIR}/env.txt"
import json, os, platform, sys
payload = {
  "python": sys.version,
  "platform": platform.platform(),
  "hostname": platform.node(),
  "cwd": os.getcwd(),
  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
}
try:
  import torch
  payload["torch"] = torch.__version__
  payload["cuda_available"] = bool(torch.cuda.is_available())
  payload["cuda_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
except Exception as exc:
  payload["torch_error"] = str(exc)
print(json.dumps(payload, indent=2, ensure_ascii=False))
PY

cd "${PROJECT_ROOT}"

echo "[1/9] Build protocol splits"
"${PYTHON_BIN}" tools/build_biotalk_splits.py \
  --source-root "${SOURCE_ROOT}" \
  --out-root "${SPLIT_ROOT}" \
  --seed "${SEED}"

echo "[2/9] Sanity check one sample"
"${PYTHON_BIN}" tools/sanity_check_biotalk.py \
  --split-json "${SPLIT_ROOT}/P1/zym/EATalker/train.json" \
  --seed "${SEED}"

RESULTS_P1="${RUN_DIR}/results_P1.csv"
RESULTS_P2="${RUN_DIR}/results_P2.csv"
RESULTS_P3="${RUN_DIR}/results_P3.csv"
SHIFT_CSV="${RUN_DIR}/metrics_shift.csv"

all_protocols=(
  "P1_lc"
  "P1_zym"
  "P1_yrx"
  "P1_lws"
  "P1_pj"
  "P2_loso_lc"
  "P2_loso_zym"
  "P2_loso_yrx"
  "P2_loso_lws"
  "P2_loso_pj"
  "P3_session_holdout_s4"
  "P3_language_zh_to_en"
)

split_parent_for_protocol() {
  local protocol="$1"
  case "${protocol}" in
    P1_lc) echo "${SPLIT_ROOT}/P1/lc" ;;
    P1_zym) echo "${SPLIT_ROOT}/P1/zym" ;;
    P1_yrx) echo "${SPLIT_ROOT}/P1/yrx" ;;
    P1_lws) echo "${SPLIT_ROOT}/P1/lws" ;;
    P1_pj) echo "${SPLIT_ROOT}/P1/pj" ;;
    P2_loso_lc) echo "${SPLIT_ROOT}/P2/loso_lc" ;;
    P2_loso_zym) echo "${SPLIT_ROOT}/P2/loso_zym" ;;
    P2_loso_yrx) echo "${SPLIT_ROOT}/P2/loso_yrx" ;;
    P2_loso_lws) echo "${SPLIT_ROOT}/P2/loso_lws" ;;
    P2_loso_pj) echo "${SPLIT_ROOT}/P2/loso_pj" ;;
    P3_session_holdout_s4) echo "${SPLIT_ROOT}/P3/session_holdout_s4" ;;
    P3_language_zh_to_en) echo "${SPLIT_ROOT}/P3/language_zh_to_en" ;;
    *)
      echo "unknown protocol: ${protocol}" >&2
      exit 1
      ;;
  esac
}

selected_protocols=()
if [[ -n "${PROTOCOLS_ONLY}" ]]; then
  IFS=',' read -r -a req_protocols <<<"${PROTOCOLS_ONLY}"
  for protocol in "${req_protocols[@]}"; do
    protocol="${protocol//[[:space:]]/}"
    if [[ -z "${protocol}" ]]; then
      continue
    fi
    found=0
    for known in "${all_protocols[@]}"; do
      if [[ "${known}" == "${protocol}" ]]; then
        found=1
        break
      fi
    done
    if [[ "${found}" -ne 1 ]]; then
      echo "invalid protocol in PROTOCOLS_ONLY: ${protocol}" >&2
      exit 1
    fi
    selected_protocols+=("${protocol}")
  done
else
  selected_protocols=("${all_protocols[@]}")
fi

protocols=()
for protocol in "${selected_protocols[@]}"; do
  protocols+=("${protocol}|$(split_parent_for_protocol "${protocol}")")
done

result_csv_for_protocol() {
  local protocol="$1"
  if [[ "${protocol}" == P1_* ]]; then
    echo "${RESULTS_P1}"
    return
  fi
  if [[ "${protocol}" == P2_* ]]; then
    echo "${RESULTS_P2}"
    return
  fi
  echo "${RESULTS_P3}"
}

train_one() {
  local protocol="$1"
  local split_parent="$2"
  local baseline="$3"
  local use_eeg="$4"
  local epochs="$5"
  local grad_acc="$6"
  local wav2vec_lr="$7"
  local run_group="$8"
  local save_every="$9"
  local eval_every="${10}"

  local out_dir="${RUN_DIR}/${run_group}/${protocol}/${baseline}"
  mkdir -p "${out_dir}/logs" "${out_dir}/eval"

  local opts=(
    "dataset" "EATalker"
    "data_root" "${split_parent}"
    "train_json_key" "train.json"
    "save_path" "${out_dir}"
    "weight_path" "${PRETRAINED_CKPT}"
    "do_validate" "True"
    "do_test" "False"
    "use_audio" "True"
    "use_eeg" "${use_eeg}"
    "epochs" "${epochs}"
    "workers" "${WORKERS}"
    "batch_size" "1"
    "gradient_accumulation_steps" "${grad_acc}"
    "save_every" "${save_every}"
    "eval_every" "${eval_every}"
    "test_every" "${eval_every}"
    "random_id_prob" "0.0"
    "wav2vec_train_last_n_layers" "12"
    "wav2vec_frozen_to_eval" "True"
    "wav2vec_lr" "${wav2vec_lr}"
  )

  printf '%s\n' "${opts[@]}" > "${out_dir}/opts.txt"
  cp "${CONFIG_PATH}" "${out_dir}/config_copy.yaml"
  {
    echo "PYTHONPATH=/home/zym"
    echo "EAT_LOG_DIR=${out_dir}/logs"
    if [[ "${NPROC}" -gt 1 ]]; then
      echo "${TORCHRUN_BIN} --nproc_per_node=${NPROC} --master_port=$((MASTER_PORT_BASE + RANDOM % 1000)) -m EATalker.main.train_eatalker --config ${CONFIG_PATH} ${opts[*]}"
    else
      echo "${PYTHON_BIN} -m EATalker.main.train_eatalker --config ${CONFIG_PATH} ${opts[*]}"
    fi
  } > "${out_dir}/command.sh"
  chmod +x "${out_dir}/command.sh"

  if [[ "${NPROC}" -gt 1 ]]; then
    PYTHONPATH=/home/zym EAT_LOG_DIR="${out_dir}/logs" \
      "${TORCHRUN_BIN}" --nproc_per_node="${NPROC}" --master_port="$((MASTER_PORT_BASE + RANDOM % 1000))" \
      -m EATalker.main.train_eatalker --config "${CONFIG_PATH}" "${opts[@]}" \
      2>&1 | tee "${out_dir}/train_stdout.log"
  else
    PYTHONPATH=/home/zym EAT_LOG_DIR="${out_dir}/logs" \
      "${PYTHON_BIN}" -m EATalker.main.train_eatalker --config "${CONFIG_PATH}" "${opts[@]}" \
      2>&1 | tee "${out_dir}/train_stdout.log"
  fi

  echo "${out_dir}"
}

prune_ckpts_keep() {
  local out_dir="$1"
  local keep_ckpt="$2"
  if [[ -z "${keep_ckpt}" ]]; then
    return
  fi
  shopt -s nullglob
  for ckpt in "${out_dir}"/eatalker_*.pt; do
    if [[ "${ckpt}" != "${keep_ckpt}" ]]; then
      rm -f "${ckpt}"
    fi
  done
  shopt -u nullglob
}

select_best_ckpt() {
  local protocol="$1"
  local split_parent="$2"
  local baseline="$3"
  local use_eeg="$4"
  local run_group="$5"
  local run_dir="${RUN_DIR}/${run_group}/${protocol}/${baseline}"
  local val_json="${split_parent}/EATalker/val.json"
  local best_ckpt=""
  local best_mve="999999"
  mkdir -p "${run_dir}/eval/val"
  local metric_table="${run_dir}/eval/val_ckpt_metrics.tsv"
  : > "${metric_table}"

  shopt -s nullglob
  local ckpts=("${run_dir}"/eatalker_*.pt)
  shopt -u nullglob
  if [[ "${#ckpts[@]}" -eq 0 ]]; then
    echo "No checkpoints found in ${run_dir}" >&2
    exit 1
  fi

  for ckpt in "${ckpts[@]}"; do
    local stem
    stem="$(basename "${ckpt}" .pt)"
    local out_json="${run_dir}/eval/val/${stem}.json"
    local extra_eval_args=()
    if [[ "${use_eeg}" == "True" ]]; then
      extra_eval_args+=(--use-eeg)
    fi
    "${PYTHON_BIN}" tools/eval_biotalk_protocol.py \
      --config "${CONFIG_PATH}" \
      --weight-path "${ckpt}" \
      --split-json "${val_json}" \
      --protocol "${protocol}" \
      --baseline "${baseline}" \
      --phase "val" \
      --run-id "${RUN_ID}" \
      --out-json "${out_json}" \
      "${extra_eval_args[@]}" \
      --opts data_root "${split_parent}" dataset EATalker use_audio True use_eeg "${use_eeg}" workers "${WORKERS}" \
      > /dev/null

    "${PYTHON_BIN}" - <<PY >> "${metric_table}"
import json
with open("${out_json}", "r", encoding="utf-8") as f:
    d = json.load(f)
print("\\t".join([
    "${ckpt}",
    str(float(d["MVE"])),
    str(float(d["LVE"])),
    str(float(d["UFVE"])),
    str(float(d["FDD"]))
]))
PY
  done
  read -r best_ckpt best_mve <<<"$("${PYTHON_BIN}" - <<PY
import math
from tools.ckpt_scoring import pick_best_by_rank
rows = []
with open("${metric_table}", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        ckpt, mve, lve, ufve, fdd = line.split("\\t")
        rows.append({
            "ckpt": ckpt,
            "MVE": float(mve),
            "LVE": float(lve),
            "UFVE": float(ufve),
            "FDD": float(fdd),
        })

if not rows:
    raise SystemExit("no val rows")

best = pick_best_by_rank(
    rows,
    metrics=("MVE", "LVE", "UFVE", "FDD"),
    weights={"MVE": 0.20, "LVE": 0.15, "UFVE": 0.35, "FDD": 0.30},
)
print(best["ckpt"], best["MVE"])
PY
)"
  if [[ -z "${best_ckpt}" ]]; then
    echo "Failed to select best checkpoint for ${protocol}/${baseline}" >&2
    exit 1
  fi
  echo "${best_ckpt}" > "${run_dir}/best_ckpt.txt"
  echo "${best_mve}" > "${run_dir}/best_val_mve.txt"
  echo "${best_ckpt}"
}

eval_test_and_visualize() {
  local protocol="$1"
  local split_parent="$2"
  local baseline="$3"
  local use_eeg="$4"
  local best_ckpt="$5"
  local run_group="$6"

  local run_dir="${RUN_DIR}/${run_group}/${protocol}/${baseline}"
  local test_json="${split_parent}/EATalker/test.json"
  local result_csv
  result_csv="$(result_csv_for_protocol "${protocol}")"
  local per_sample="${run_dir}/eval/test_per_sample.csv"
  local out_json="${run_dir}/eval/test_summary.json"
  local extra_eval_args=()
  if [[ "${use_eeg}" == "True" ]]; then
    extra_eval_args+=(--use-eeg)
  fi

  "${PYTHON_BIN}" tools/eval_biotalk_protocol.py \
    --config "${CONFIG_PATH}" \
    --weight-path "${best_ckpt}" \
    --split-json "${test_json}" \
    --protocol "${protocol}" \
    --baseline "${baseline}" \
    --phase "test" \
    --run-id "${RUN_ID}" \
    --metrics-csv "${result_csv}" \
    --per-sample-csv "${per_sample}" \
    --out-json "${out_json}" \
    --visual-root "${RUN_DIR}/visuals" \
    --render-topk "${RENDER_TOPK}" \
    --render-worstk "${RENDER_WORSTK}" \
    --img-size "${IMG_SIZE}" \
    "${extra_eval_args[@]}" \
    --opts data_root "${split_parent}" dataset EATalker use_audio True use_eeg "${use_eeg}" workers "${WORKERS}" \
    | tee "${run_dir}/eval/test_eval_stdout.log"
}

run_shift_eval_for_p2_b() {
  local protocol="$1"
  local split_parent="$2"
  local best_ckpt="$3"
  local run_group="$4"
  local run_dir="${RUN_DIR}/${run_group}/${protocol}/audio+eeg"
  local test_json="${split_parent}/EATalker/test.json"
  local offsets=(-80 -40 -20 20 40 80)
  for offset in "${offsets[@]}"; do
    "${PYTHON_BIN}" tools/eval_biotalk_protocol.py \
      --config "${CONFIG_PATH}" \
      --weight-path "${best_ckpt}" \
      --split-json "${test_json}" \
      --protocol "${protocol}" \
      --baseline "audio+eeg" \
      --phase "shift_test" \
      --run-id "${RUN_ID}" \
      --eeg-shift-ms "${offset}" \
      --use-eeg \
      --metrics-csv "${SHIFT_CSV}" \
      --opts data_root "${split_parent}" dataset EATalker use_audio True use_eeg True workers "${WORKERS}" \
      > "${run_dir}/eval/shift_${offset}.log"
  done
}

run_protocol_pair() {
  local protocol="$1"
  local split_parent="$2"
  local run_group="$3"
  echo "[train] ${protocol} / audio-only"
  train_one "${protocol}" "${split_parent}" "audio-only" "False" "${EPOCHS_MAIN}" "${GRAD_ACC_MAIN}" "${WAV2VEC_LR_MAIN}" "${run_group}" "${SAVE_EVERY_MAIN}" "${EVAL_EVERY_MAIN}" >/dev/null
  local best_a
  best_a="$(select_best_ckpt "${protocol}" "${split_parent}" "audio-only" "False" "${run_group}")"
  prune_ckpts_keep "${RUN_DIR}/${run_group}/${protocol}/audio-only" "${best_a}"
  eval_test_and_visualize "${protocol}" "${split_parent}" "audio-only" "False" "${best_a}" "${run_group}"

  echo "[train] ${protocol} / audio+eeg"
  train_one "${protocol}" "${split_parent}" "audio+eeg" "True" "${EPOCHS_MAIN}" "${GRAD_ACC_MAIN}" "${WAV2VEC_LR_MAIN}" "${run_group}" "${SAVE_EVERY_MAIN}" "${EVAL_EVERY_MAIN}" >/dev/null
  local best_b
  best_b="$(select_best_ckpt "${protocol}" "${split_parent}" "audio+eeg" "True" "${run_group}")"
  prune_ckpts_keep "${RUN_DIR}/${run_group}/${protocol}/audio+eeg" "${best_b}"
  eval_test_and_visualize "${protocol}" "${split_parent}" "audio+eeg" "True" "${best_b}" "${run_group}"

  if [[ "${protocol}" == P2_* ]]; then
    run_shift_eval_for_p2_b "${protocol}" "${split_parent}" "${best_b}" "${run_group}"
  fi
}

echo "[3/9] Train/Eval all protocols (A/B)"
for spec in "${protocols[@]}"; do
  IFS='|' read -r protocol split_parent <<<"${spec}"
  run_protocol_pair "${protocol}" "${split_parent}" "main"
done

has_p2=0
has_p3=0
for spec in "${protocols[@]}"; do
  IFS='|' read -r protocol _split_parent <<<"${spec}"
  if [[ "${protocol}" == P2_* ]]; then
    has_p2=1
  fi
  if [[ "${protocol}" == P3_* ]]; then
    has_p3=1
  fi
done

if [[ "${has_p2}" -eq 1 ]]; then
echo "[4/9] Quality gate for P2"
read -r MVE_IMPROVE BSMAE_IMPROVE <<<"$("${PYTHON_BIN}" - <<PY
import csv
from statistics import mean

rows = list(csv.DictReader(open("${RESULTS_P2}", "r", encoding="utf-8")))
rows = [r for r in rows if r.get("run_id") == "${RUN_ID}" and r.get("phase") == "test"]
a = [r for r in rows if r.get("baseline") == "audio-only"]
b = [r for r in rows if r.get("baseline") == "audio+eeg"]
if not a or not b:
    print("0.0 0.0")
    raise SystemExit
mve_a = mean(float(r["MVE"]) for r in a)
mve_b = mean(float(r["MVE"]) for r in b)
bs_a = mean(float(r["bs_mae_51d"]) for r in a)
bs_b = mean(float(r["bs_mae_51d"]) for r in b)
mve_improve = (mve_a - mve_b) / max(mve_a, 1e-12)
bs_improve = (bs_a - bs_b) / max(bs_a, 1e-12)
print(f"{mve_improve:.8f} {bs_improve:.8f}")
PY
)"

echo "P2 improvement: MVE=${MVE_IMPROVE}, BS-MAE=${BSMAE_IMPROVE}"

need_retry=0
if awk "BEGIN {exit !(${MVE_IMPROVE} < ${TARGET_MVE_IMPROVE})}"; then
  need_retry=1
fi
if awk "BEGIN {exit !(${BSMAE_IMPROVE} < ${TARGET_BSMAE_IMPROVE})}"; then
  need_retry=1
fi

if [[ "${need_retry}" -eq 1 ]]; then
  echo "[5/9] Trigger one retry for P2 audio+eeg"
  RETRY_DIR="${RUN_DIR}/ablation_retry"
  mkdir -p "${RETRY_DIR}"
  cp "${RESULTS_P2}" "${RETRY_DIR}/results_P2_before_retry.csv"
  for spec in "${protocols[@]}"; do
    IFS='|' read -r protocol split_parent <<<"${spec}"
    if [[ "${protocol}" != P2_* ]]; then
      continue
    fi
    echo "[retry-train] ${protocol} / audio+eeg"
    train_one "${protocol}" "${split_parent}" "audio+eeg" "True" "${EPOCHS_RETRY}" "${GRAD_ACC_RETRY}" "${WAV2VEC_LR_RETRY}" "retry1" "${SAVE_EVERY_RETRY}" "${EVAL_EVERY_RETRY}" >/dev/null
    best_b="$(select_best_ckpt "${protocol}" "${split_parent}" "audio+eeg" "True" "retry1")"
    prune_ckpts_keep "${RUN_DIR}/retry1/${protocol}/audio+eeg" "${best_b}"

    "${PYTHON_BIN}" - <<PY
import csv
in_path = "${RESULTS_P2}"
rows = list(csv.DictReader(open(in_path, "r", encoding="utf-8")))
kept = [r for r in rows if not (r.get("run_id") == "${RUN_ID}" and r.get("protocol") == "${protocol}" and r.get("baseline") == "audio+eeg" and r.get("phase") == "test")]
with open(in_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(kept)
PY

    eval_test_and_visualize "${protocol}" "${split_parent}" "audio+eeg" "True" "${best_b}" "retry1"
    run_shift_eval_for_p2_b "${protocol}" "${split_parent}" "${best_b}" "retry1"
  done
fi
else
  echo "[4/9] Skip P2 quality gate (no P2 protocol selected)"
fi

echo "[6/9] Generate LaTeX rows"
mkdir -p "${RUN_DIR}/latex"
for spec in "${protocols[@]}"; do
  IFS='|' read -r protocol split_parent <<<"${spec}"
  csv_path="$(result_csv_for_protocol "${protocol}")"
  for baseline in audio-only audio+eeg; do
    "${PYTHON_BIN}" tools/csv_to_latex_row.py \
      --csv "${csv_path}" \
      --protocol "${protocol}" \
      --baseline "${baseline}" \
      > "${RUN_DIR}/latex/${protocol}_${baseline}.tex" || true
  done
done

echo "[7/9] Final sanity snapshots"
if [[ "${has_p2}" -eq 1 ]]; then
  first_p2_split=""
  for spec in "${protocols[@]}"; do
    IFS='|' read -r protocol split_parent <<<"${spec}"
    if [[ "${protocol}" == P2_* ]]; then
      first_p2_split="${split_parent}"
      break
    fi
  done
  if [[ -n "${first_p2_split}" ]]; then
"${PYTHON_BIN}" tools/sanity_check_biotalk.py \
  --split-json "${first_p2_split}/EATalker/test.json" \
  --seed "${SEED}" \
  > "${RUN_DIR}/sanity_p2_loso_lc.json"
  fi
fi

if [[ "${has_p3}" -eq 1 ]]; then
  first_p3_split=""
  for spec in "${protocols[@]}"; do
    IFS='|' read -r protocol split_parent <<<"${spec}"
    if [[ "${protocol}" == P3_* ]]; then
      first_p3_split="${split_parent}"
      break
    fi
  done
  if [[ -n "${first_p3_split}" ]]; then
"${PYTHON_BIN}" tools/sanity_check_biotalk.py \
  --split-json "${first_p3_split}/EATalker/test.json" \
  --seed "${SEED}" \
  > "${RUN_DIR}/sanity_p3_lang.json"
  fi
fi

echo "[8/9] Write run manifest"
"${PYTHON_BIN}" - <<PY > "${RUN_DIR}/run_manifest.json"
import json
payload = {
  "run_id": "${RUN_ID}",
  "seed": ${SEED},
  "source_root": "${SOURCE_ROOT}",
  "split_root": "${SPLIT_ROOT}",
  "results": {
    "P1": "${RESULTS_P1}",
    "P2": "${RESULTS_P2}",
    "P3": "${RESULTS_P3}",
    "shift": "${SHIFT_CSV}",
  },
  "pretrained_ckpt": "${PRETRAINED_CKPT}",
  "quality_gate": {
    "target_mve_improve": ${TARGET_MVE_IMPROVE},
    "target_bsmae_improve": ${TARGET_BSMAE_IMPROVE},
  },
}
print(json.dumps(payload, indent=2, ensure_ascii=False))
PY

echo "[9/9] Completed. Outputs in ${RUN_DIR}"
