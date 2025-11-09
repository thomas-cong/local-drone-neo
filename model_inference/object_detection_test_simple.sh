#!/bin/bash

# Simple object detection test using YOLOv8
python3 object_bounder_simple.py \
  --input ./hard_test.mp4 \
  --output ./output/hard_test_detected_simple.mp4 \
  --model n \
  --threshold 0.5 \
  --skip-frames 1
