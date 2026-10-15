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
RUN_ID="${RUN_ID:-biotalk_p1p2_fast_$(date +%Y%m%d_%H%M%S)}"
CACHE_ROOT="${CACHE_ROOT:-/home/zym/experiments/cache/biotalk_p1p2_fast/${RUN_ID}}"
SEED="${SEED:-42}"
P2_HELDOUT="${P2_HELDOUT:-lws}"
WORKERS="${WORKERS:-8}"
IMG_SIZE="${IMG_SIZE:-256}"

A_EPOCHS="${A_EPOCHS:-40}"
A_GRAD_ACC="${A_GRAD_ACC:-4}"
A_WAV2VEC_LR="${A_WAV2VEC_LR:-0.000005}"
B_STAGE1_EPOCHS="${B_STAGE1_EPOCHS:-18}"
B_STAGE1_GRAD_ACC="${B_STAGE1_GRAD_ACC:-4}"
B_STAGE1_WAV2VEC_LR="${B_STAGE1_WAV2VEC_LR:-0.000005}"
B_STAGE2_EPOCHS="${B_STAGE2_EPOCHS:-24}"
B_STAGE2_GRAD_ACC="${B_STAGE2_GRAD_ACC:-6}"
B_STAGE2_WAV2VEC_LR="${B_STAGE2_WAV2VEC_LR:-0.000004}"
B_RETRY_STAGE2_EPOCHS="${B_RETRY_STAGE2_EPOCHS:-36}"
B_RETRY_EEG_LR_SCALE="${B_RETRY_EEG_LR_SCALE:-8.0}"

HOME_THRESHOLD_GB="${HOME_THRESHOLD_GB:-120}"
MNT_THRESHOLD_GB="${MNT_THRESHOLD_GB:-180}"

mkdir -p "${RUN_ROOT}"
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
mkdir -p "${RUN_DIR}" "${CACHE_ROOT}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "python binary not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "config not found: ${CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -f "${PRETRAINED_CKPT}" ]]; then
  echo "pretrained checkpoint not found: ${PRETRAINED_CKPT}" >&2
  exit 1
fi

if [[ -n "${NPROC_OVERRIDE:-}" ]]; then
  NPROC="${NPROC_OVERRIDE}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _gpu_arr <<<"${CUDA_VISIBLE_DEVICES}"
  NPROC="${#_gpu_arr[@]}"
else
  NPROC="$("${PYTHON_BIN}" - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 1)
PY
)"
fi
if [[ "${NPROC}" -lt 1 ]]; then
  NPROC=1
fi

export PYTHONPATH="/home/zym"
cd "${PROJECT_ROOT}"

echo "RUN_ID=${RUN_ID}" | tee "${RUN_DIR}/run.stdout.log"
echo "NPROC=${NPROC}" | tee -a "${RUN_DIR}/run.stdout.log"
echo "${SEED}" > "${RUN_DIR}/seed.txt"
echo "${PRETRAINED_CKPT}" > "${RUN_DIR}/pretrained_ckpt.txt"
git -C "${PROJECT_ROOT}" rev-parse HEAD > "${RUN_DIR}/git_commit.txt"
cp "${CONFIG_PATH}" "${RUN_DIR}/resolved_config.yaml"
df -h /home /mnt > "${RUN_DIR}/df_before.txt"

RESULTS_P1="${RUN_DIR}/results_P1.csv"
RESULTS_P2="${RUN_DIR}/results_P2.csv"
SHIFT_CSV="${RUN_DIR}/metrics_shift.csv"

protocols=(
  "P1_lc|${SPLIT_ROOT}/P1/lc"
  "P1_zym|${SPLIT_ROOT}/P1/zym"
  "P1_yrx|${SPLIT_ROOT}/P1/yrx"
  "P1_lws|${SPLIT_ROOT}/P1/lws"
  "P1_pj|${SPLIT_ROOT}/P1/pj"
  "P2_loso_${P2_HELDOUT}|${SPLIT_ROOT}/P2/loso_${P2_HELDOUT}"
)

result_csv_for_protocol() {
  local protocol="$1"
  if [[ "${protocol}" == P1_* ]]; then
    echo "${RESULTS_P1}"
  else
    echo "${RESULTS_P2}"
  fi
}

identity_override_for_protocol() {
  local protocol="$1"
  if [[ "${protocol}" == P1_* ]]; then
    echo "1"
  else
    echo "0"
  fi
}

cache_manifest_for_protocol() {
  local protocol="$1"
  echo "${CACHE_ROOT}/${protocol}/cache_manifest.json"
}

build_cache_for_protocol() {
  local protocol="$1"
  local split_parent="$2"
  local cache_dir="${CACHE_ROOT}/${protocol}"
  mkdir -p "${cache_dir}"
  "${PYTHON_BIN}" tools/cache_features.py \
    --config "${CONFIG_PATH}" \
    --split-json "${split_parent}/EATalker/train.json" "${split_parent}/EATalker/val.json" "${split_parent}/EATalker/test.json" \
    --out-dir "${cache_dir}" \
    --manifest-out "$(cache_manifest_for_protocol "${protocol}")" \
    > "${cache_dir}/cache.log" 2>&1
}

train_one() {
  local protocol="$1"
  local split_parent="$2"
  local baseline="$3"
  local stage="$4"
  local use_eeg="$5"
  local weight_path="$6"
  local epochs="$7"
  local grad_acc="$8"
  local wav2vec_lr="$9"
  local eeg_lr_scale="${10}"
  local save_dir="${11}"
  local identity_override
  identity_override="$(identity_override_for_protocol "${protocol}")"
  local cache_manifest
  cache_manifest="$(cache_manifest_for_protocol "${protocol}")"

  mkdir -p "${save_dir}/logs" "${save_dir}/eval"
  local opts=(
    "dataset" "EATalker"
    "data_root" "${split_parent}"
    "train_json_key" "train.json"
    "save_path" "${save_dir}"
    "weight_path" "${weight_path}"
    "do_validate" "True"
    "do_test" "False"
    "use_audio" "True"
    "use_eeg" "${use_eeg}"
    "epochs" "${epochs}"
    "workers" "${WORKERS}"
    "batch_size" "1"
    "gradient_accumulation_steps" "${grad_acc}"
    "save_every" "2"
    "eval_every" "1"
    "test_every" "1"
    "random_id_prob" "0.0"
    "wav2vec_train_last_n_layers" "12"
    "wav2vec_frozen_to_eval" "True"
    "wav2vec_lr" "${wav2vec_lr}"
    "train_stage" "${stage}"
    "stage2_decoder_last_k" "0"
    "stage2_unfreeze_head" "True"
    "identity_policy" "global"
    "identity_num_override" "${identity_override}"
    "use_cached_features" "True"
    "cache_manifest" "${cache_manifest}"
    "fusion_alpha_mode" "scalar"
    "fusion_alpha_init" "0.0"
    "w_lve" "0.02"
    "w_ufve" "0.02"
    "eeg_fusion_lr_scale" "${eeg_lr_scale}"
  )
  printf '%s\n' "${opts[@]}" > "${save_dir}/opts.txt"

  if [[ "${NPROC}" -gt 1 ]]; then
    PYTHONPATH=/home/zym EAT_LOG_DIR="${save_dir}/logs" \
      "${TORCHRUN_BIN}" --nproc_per_node="${NPROC}" --master_port="$((29650 + RANDOM % 500))" \
      -m EATalker.main.train_eatalker --config "${CONFIG_PATH}" "${opts[@]}" \
      2>&1 | tee "${save_dir}/train_stdout.log"
  else
    PYTHONPATH=/home/zym EAT_LOG_DIR="${save_dir}/logs" \
      "${PYTHON_BIN}" -m EATalker.main.train_eatalker --config "${CONFIG_PATH}" "${opts[@]}" \
      2>&1 | tee "${save_dir}/train_stdout.log"
  fi
}

select_best_ckpt() {
  local protocol="$1"
  local split_parent="$2"
  local baseline="$3"
  local use_eeg="$4"
  local run_dir="$5"
  local val_json="${split_parent}/EATalker/val.json"
  local metric_table="${run_dir}/eval/val_ckpt_metrics.tsv"
  mkdir -p "${run_dir}/eval/val"
  : > "${metric_table}"

  shopt -s nullglob
  local ckpts=("${run_dir}"/eatalker_*.pt)
  shopt -u nullglob
  if [[ "${#ckpts[@]}" -eq 0 ]]; then
    echo "No checkpoints found in ${run_dir}" >&2
    exit 1
  fi

  for ckpt in "${ckpts[@]}"; do
    local stem out_json
    stem="$(basename "${ckpt}" .pt)"
    out_json="${run_dir}/eval/val/${stem}.json"
    local extra=()
    if [[ "${use_eeg}" == "True" ]]; then
      extra+=(--use-eeg)
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
      "${extra[@]}" \
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
    str(float(d["FDD"])),
]))
PY
  done

  "${PYTHON_BIN}" - <<PY > "${run_dir}/best_ckpt.txt"
from tools.ckpt_scoring import pick_best_by_rank
rows = []
with open("${metric_table}", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        ckpt, mve, lve, ufve, fdd = line.split("\\t")
        rows.append({"ckpt": ckpt, "MVE": float(mve), "LVE": float(lve), "UFVE": float(ufve), "FDD": float(fdd)})
best = pick_best_by_rank(
    rows,
    metrics=("MVE", "LVE", "UFVE", "FDD"),
    weights={"MVE":0.20, "LVE":0.15, "UFVE":0.35, "FDD":0.30},
)
print(best["ckpt"])
PY
  cat "${run_dir}/best_ckpt.txt"
}

prune_ckpts_keep() {
  local run_dir="$1"
  local keep_ckpt="$2"
  shopt -s nullglob
  local latest=""
  local latest_mtime=0
  for ckpt in "${run_dir}"/eatalker_*.pt; do
    local mt
    mt="$(stat -c %Y "${ckpt}")"
    if (( mt > latest_mtime )); then
      latest_mtime="${mt}"
      latest="${ckpt}"
    fi
  done
  for ckpt in "${run_dir}"/eatalker_*.pt; do
    if [[ "${ckpt}" == "${keep_ckpt}" || "${ckpt}" == "${latest}" ]]; then
      continue
    fi
    rm -f "${ckpt}"
  done
  shopt -u nullglob
}

eval_test() {
  local protocol="$1"
  local split_parent="$2"
  local baseline="$3"
  local use_eeg="$4"
  local weight_path="$5"
  local run_dir="$6"
  local csv_path
  csv_path="$(result_csv_for_protocol "${protocol}")"
  local extra=()
  if [[ "${use_eeg}" == "True" ]]; then
    extra+=(--use-eeg)
  fi
  "${PYTHON_BIN}" tools/eval_biotalk_protocol.py \
    --config "${CONFIG_PATH}" \
    --weight-path "${weight_path}" \
    --split-json "${split_parent}/EATalker/test.json" \
    --protocol "${protocol}" \
    --baseline "${baseline}" \
    --phase "test" \
    --run-id "${RUN_ID}" \
    --metrics-csv "${csv_path}" \
    --per-sample-csv "${run_dir}/eval/test_per_sample.csv" \
    --out-json "${run_dir}/eval/test_summary.json" \
    --visual-root "${RUN_DIR}/visuals" \
    --render-topk 10 \
    --render-worstk 10 \
    --img-size "${IMG_SIZE}" \
    "${extra[@]}" \
    --opts data_root "${split_parent}" dataset EATalker use_audio True use_eeg "${use_eeg}" workers "${WORKERS}" \
    | tee "${run_dir}/eval/test_eval_stdout.log"
}

extract_metric() {
  local json_path="$1"
  local key="$2"
  "${PYTHON_BIN}" - <<PY
import json
with open("${json_path}", "r", encoding="utf-8") as f:
    d = json.load(f)
print(float(d["${key}"]))
PY
}

cleanup_now() {
  local protocol="$1"
  local cache_manifest
  cache_manifest="$(cache_manifest_for_protocol "${protocol}")"
  "${PYTHON_BIN}" tools/cleanup_storage.py \
    --runs-root "${RUN_ROOT}" \
    --cache-root "${CACHE_ROOT}" \
    --active-manifest "${cache_manifest}" \
    --current-run-dir "${RUN_DIR}" \
    --home-threshold-gb "${HOME_THRESHOLD_GB}" \
    --mnt-threshold-gb "${MNT_THRESHOLD_GB}" \
    > "${RUN_DIR}/cleanup_${protocol}.json"
}

run_protocol() {
  local protocol="$1"
  local split_parent="$2"
  local root_dir="${RUN_DIR}/${protocol}"
  mkdir -p "${root_dir}"

  echo "[cache] ${protocol}"
  build_cache_for_protocol "${protocol}" "${split_parent}"

  echo "[train A] ${protocol}"
  local a_dir="${root_dir}/audio-only"
  train_one "${protocol}" "${split_parent}" "audio-only" "joint" "False" "${PRETRAINED_CKPT}" "${A_EPOCHS}" "${A_GRAD_ACC}" "${A_WAV2VEC_LR}" "10.0" "${a_dir}"
  local best_a
  best_a="$(select_best_ckpt "${protocol}" "${split_parent}" "audio-only" "False" "${a_dir}")"
  prune_ckpts_keep "${a_dir}" "${best_a}"
  eval_test "${protocol}" "${split_parent}" "audio-only" "False" "${best_a}" "${a_dir}"

  echo "[train B stage1] ${protocol}"
  local b1_dir="${root_dir}/audio+eeg_stage1"
  train_one "${protocol}" "${split_parent}" "audio+eeg" "stage1" "True" "${best_a}" "${B_STAGE1_EPOCHS}" "${B_STAGE1_GRAD_ACC}" "${B_STAGE1_WAV2VEC_LR}" "10.0" "${b1_dir}"
  local best_b1
  best_b1="$(select_best_ckpt "${protocol}" "${split_parent}" "audio+eeg" "True" "${b1_dir}")"
  prune_ckpts_keep "${b1_dir}" "${best_b1}"

  echo "[train B stage2] ${protocol}"
  local b2_dir="${root_dir}/audio+eeg"
  train_one "${protocol}" "${split_parent}" "audio+eeg" "stage2" "True" "${best_b1}" "${B_STAGE2_EPOCHS}" "${B_STAGE2_GRAD_ACC}" "${B_STAGE2_WAV2VEC_LR}" "10.0" "${b2_dir}"
  local best_b
  best_b="$(select_best_ckpt "${protocol}" "${split_parent}" "audio+eeg" "True" "${b2_dir}")"
  prune_ckpts_keep "${b2_dir}" "${best_b}"
  eval_test "${protocol}" "${split_parent}" "audio+eeg" "True" "${best_b}" "${b2_dir}"

  # One retry if multimodal still degrades in primary metric.
  local a_mve b_mve
  a_mve="$(extract_metric "${a_dir}/eval/test_summary.json" "MVE")"
  b_mve="$(extract_metric "${b2_dir}/eval/test_summary.json" "MVE")"
  if awk "BEGIN {exit !(${b_mve} >= ${a_mve})}"; then
    echo "[retry stage2] ${protocol}: multimodal MVE did not improve, run one retry"
    local b2r_dir="${root_dir}/audio+eeg_retry1"
    train_one "${protocol}" "${split_parent}" "audio+eeg" "stage2" "True" "${best_b1}" "${B_RETRY_STAGE2_EPOCHS}" "${B_STAGE2_GRAD_ACC}" "${B_STAGE2_WAV2VEC_LR}" "${B_RETRY_EEG_LR_SCALE}" "${b2r_dir}"
    local best_b_retry
    best_b_retry="$(select_best_ckpt "${protocol}" "${split_parent}" "audio+eeg" "True" "${b2r_dir}")"
    prune_ckpts_keep "${b2r_dir}" "${best_b_retry}"
    eval_test "${protocol}" "${split_parent}" "audio+eeg" "True" "${best_b_retry}" "${b2r_dir}"
  fi

  if [[ "${protocol}" == P2_* ]]; then
    for offset in -80 -40 -20 20 40 80; do
      "${PYTHON_BIN}" tools/eval_biotalk_protocol.py \
        --config "${CONFIG_PATH}" \
        --weight-path "${best_b}" \
        --split-json "${split_parent}/EATalker/test.json" \
        --protocol "${protocol}" \
        --baseline "audio+eeg" \
        --phase "shift_test" \
        --run-id "${RUN_ID}" \
        --use-eeg \
        --eeg-shift-ms "${offset}" \
        --metrics-csv "${SHIFT_CSV}" \
        --opts data_root "${split_parent}" dataset EATalker use_audio True use_eeg True workers "${WORKERS}" \
        > "${b2_dir}/eval/shift_${offset}.log"
    done
  fi

  cleanup_now "${protocol}"
}

echo "[1/6] build splits"
"${PYTHON_BIN}" tools/build_biotalk_splits.py \
  --source-root "${SOURCE_ROOT}" \
  --out-root "${SPLIT_ROOT}" \
  --seed "${SEED}"

echo "[2/6] run protocols"
for spec in "${protocols[@]}"; do
  IFS='|' read -r protocol split_parent <<<"${spec}"
  run_protocol "${protocol}" "${split_parent}" | tee -a "${RUN_DIR}/run.stdout.log"
done

echo "[3/6] latex rows"
mkdir -p "${RUN_DIR}/latex"
for spec in "${protocols[@]}"; do
  IFS='|' read -r protocol _split_parent <<<"${spec}"
  csv_path="$(result_csv_for_protocol "${protocol}")"
  for baseline in audio-only audio+eeg; do
    "${PYTHON_BIN}" tools/csv_to_latex_row.py \
      --csv "${csv_path}" \
      --protocol "${protocol}" \
      --baseline "${baseline}" \
      --columns "MVE,LVE,UFVE,FDD" \
      > "${RUN_DIR}/latex/${protocol}_${baseline}.tex" || true
  done
done

echo "[4/6] final cleanup"
"${PYTHON_BIN}" tools/cleanup_storage.py \
  --runs-root "${RUN_ROOT}" \
  --cache-root "${CACHE_ROOT}" \
  --current-run-dir "${RUN_DIR}" \
  --home-threshold-gb "${HOME_THRESHOLD_GB}" \
  --mnt-threshold-gb "${MNT_THRESHOLD_GB}" \
  > "${RUN_DIR}/cleanup_final.json"

echo "[5/6] env snapshot"
df -h /home /mnt > "${RUN_DIR}/df_after.txt"
"${PYTHON_BIN}" - <<PY > "${RUN_DIR}/run_manifest.json"
import json, os, platform, sys
payload = {
  "run_id": "${RUN_ID}",
  "seed": ${SEED},
  "project_root": "${PROJECT_ROOT}",
  "run_dir": "${RUN_DIR}",
  "cache_root": "${CACHE_ROOT}",
  "p2_heldout": "${P2_HELDOUT}",
  "nproc": int("${NPROC}"),
  "python": sys.version,
  "platform": platform.platform(),
  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
  "results": {"P1": "${RESULTS_P1}", "P2": "${RESULTS_P2}", "shift": "${SHIFT_CSV}"},
}
print(json.dumps(payload, indent=2, ensure_ascii=False))
PY

echo "[6/6] completed: ${RUN_DIR}"
