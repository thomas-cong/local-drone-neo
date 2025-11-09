#!/bin/bash
export LD_LIBRARY_PATH="/home/jetson/.local/decord:$LD_LIBRARY_PATH"

# Test with video file (headless mode, optimized for speed)
python3 depth_seg_streaming.py \
  --source ./test.mp4 \
  --output ./output/stream_output.mp4 \
  --process-every 3 \
  --max-objects 1 \
  --resolution 320x180 \
  --depth-model depth-anything/Depth-Anything-V2-Small-hf \
  --reinit-every 0 \
  --no-display
