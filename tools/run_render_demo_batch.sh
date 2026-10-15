#!/usr/bin/env bash
set -eo pipefail

SUBJECTS=""
IMG_SIZE="${IMG_SIZE:-512}"
DECIMATE="${DECIMATE:-2}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subjects)
      SUBJECTS="$2"
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

if [[ -z "$SUBJECTS" ]]; then
  echo "Usage: $0 --subjects zym,yrx,lws,pj [--img-size 512] [--decimate 2]" >&2
  exit 2
fi

IFS=',' read -r -a arr <<< "$SUBJECTS"
for raw in "${arr[@]}"; do
  subject="$(echo "$raw" | xargs)"
  if [[ -z "$subject" ]]; then
    continue
  fi
  echo "==== Render subject: $subject (img_size=$IMG_SIZE, decimate=$DECIMATE) ===="
  /home/zym/EATalker/tools/run_render_demo_subject.sh \
    --subject "$subject" \
    --img-size "$IMG_SIZE" \
    --decimate "$DECIMATE"
done

echo "All subjects finished."
