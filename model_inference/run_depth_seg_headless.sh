#!/bin/bash
# Run depth + segmentation streaming on Jetson (HEADLESS MODE)
# No display window - saves last frames and JSON data on exit

# Camera settings
WIDTH=1280
HEIGHT=720
FPS=30

# Processing settings
PROCESS_RES="320x180"
PROCESS_EVERY=3
MAX_OBJECTS=10
REINIT_EVERY=100

# Output settings
SAVE_FRAMES=10             # Save more frames in headless mode
SAVE_DIR="saved_frames"
JSON_OUTPUT="depth_data.json"
JSON_INTERVAL=5            # Save JSON every 5 frames

echo "Starting HEADLESS depth + segmentation streaming..."
echo "Camera: ${WIDTH}x${HEIGHT} @ ${FPS} FPS"
echo "Processing: $PROCESS_RES (every $PROCESS_EVERY frames)"
echo "Output: $SAVE_FRAMES frames -> $SAVE_DIR/, JSON -> $JSON_OUTPUT"
echo ""
echo "Press Ctrl+C to stop and save data"
echo "----------------------------------------"
echo ""

python3 depth_seg_streaming.py \
    --source 0 \
    --use-csi \
    --cam-width $WIDTH \
    --cam-height $HEIGHT \
    --cam-fps $FPS \
    --resolution $PROCESS_RES \
    --process-every $PROCESS_EVERY \
    --max-objects $MAX_OBJECTS \
    --reinit-every $REINIT_EVERY \
    --save-frames $SAVE_FRAMES \
    --save-frames-dir $SAVE_DIR \
    --json-output $JSON_OUTPUT \
    --json-interval $JSON_INTERVAL \
    --no-display
