#!/usr/bin/env bash
set -eo pipefail

IMG_SIZE="${IMG_SIZE:-256}"
DECIMATE="${DECIMATE:-2}"

/home/zym/EATalker/tools/run_render_demo_subject.sh \
  --subject lc \
  --img-size "$IMG_SIZE" \
  --decimate "$DECIMATE"
