#!/bin/bash
export LD_LIBRARY_PATH="/home/jetson/.local/decord:$LD_LIBRARY_PATH"

# Test with video file and JSON output
python3 depth_seg_streaming.py \
  --source ./hard_test.mp4 \
  --output ./output/hard_output.mp4 \
  --save-json-file ./output/hard_depth_data.json \
  --json-interval 5 \
  --process-every 5 \
  --max-objects 5 \
  --min-coverage 0.1 \
  --max-coverage 0.25 \
  --resolution 320x180 \
  --depth-model depth-anything/Depth-Anything-V2-Small-hf \
  --reinit-every 0 \
  --enable-object-detection \
  --object-detect-every 5 \
  --object-model-size s \
  --object-threshold 0.5 \
  --object-overlap-threshold 0.6 \
  --no-display \
  --mqtt-disable
