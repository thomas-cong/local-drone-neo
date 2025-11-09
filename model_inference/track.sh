#!/bin/bash
export LD_LIBRARY_PATH="/home/jetson/.local/decord:$LD_LIBRARY_PATH"

python3 edgetam_tracker.py \
  --input ./test.mp4 \
  --output ./tracked.mp4 \
  --skip-frames 5 \
  --max-objects 1
