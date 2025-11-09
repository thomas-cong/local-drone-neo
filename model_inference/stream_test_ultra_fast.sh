#!/bin/bash
export LD_LIBRARY_PATH="/home/jetson/.local/decord:$LD_LIBRARY_PATH"

# Ultra-fast configuration - maximize speed over quality
python3 depth_seg_streaming.py \
  --source ./hard_test.mp4 \
  --output ./output/hard_output_fast.mp4 \
  --json-output ./output/hard_depth_fast.json \
  --json-interval 15 \
  --process-every 10 \
  --max-objects 2 \
  --min-coverage 0.1 \
  --max-coverage 0.25 \
  --resolution 160x90 \
  --matrix-resolution 40x22 \
  --depth-model depth-anything/Depth-Anything-V2-Small-hf \
  --reinit-every 0 \
  --no-display
