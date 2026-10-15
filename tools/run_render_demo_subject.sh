#!/usr/bin/env bash
set -eo pipefail

SUBJECT=""
IMG_SIZE="${IMG_SIZE:-512}"
DECIMATE="${DECIMATE:-2}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subject)
      SUBJECT="$2"
      shift 2
      ;;
    --img-size)
      IMG_SIZE="$2"
      shift 2
      ;;
    --decimate)
      DECIMATE="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$SUBJECT" ]]; then
  echo "Usage: $0 --subject <name> [--img-size 512] [--decimate 2]" >&2
  exit 2
fi

cd /home/zym/EATalker
eval "$(/home/zym/micromamba/bin/micromamba shell hook -s bash)"
micromamba activate eatalker
export PYTHONPATH=/home/zym

OUT_ROOT="/home/zym/EATalker_data_template_v2/${SUBJECT}/render_demo"
DATA_ROOT="/home/zym/EATalker_data_template_v2/${SUBJECT}/EATalker"
PIPELINE_LOG="$OUT_ROOT/logs/pipeline_${IMG_SIZE}.log"

mkdir -p "$OUT_ROOT"/{npz,train,val,test,logs}

echo "[$(date '+%F %T')] SUBJECT=${SUBJECT} START_IMG_${IMG_SIZE}" | tee -a "$PIPELINE_LOG"
for split in train val test; do
  if [[ ! -d "$DATA_ROOT/$split" ]]; then
    echo "[$(date '+%F %T')] FAIL ${split} missing $DATA_ROOT/$split" | tee -a "$PIPELINE_LOG"
    exit 1
  fi
  echo "[$(date '+%F %T')] START ${split}" | tee -a "$PIPELINE_LOG"
  if python -u -m EATalker.tools.render_testset_gt \
    --test-dir "$DATA_ROOT/$split" \
    --out-npz "$OUT_ROOT/npz/${split}_gt_vertices.npz" \
    --out-videos "$OUT_ROOT/$split" \
    --device cuda \
    --decimate "$DECIMATE" \
    --img-size "$IMG_SIZE" \
    >"$OUT_ROOT/logs/${split}_${IMG_SIZE}.log" 2>&1; then
    echo "[$(date '+%F %T')] DONE ${split}" | tee -a "$PIPELINE_LOG"
  else
    code=$?
    echo "[$(date '+%F %T')] FAIL ${split} code=${code}" | tee -a "$PIPELINE_LOG"
    exit $code
  fi
done

echo "[$(date '+%F %T')] SUBJECT=${SUBJECT} ALL_DONE" | tee -a "$PIPELINE_LOG"
