#!/bin/bash
# Run depth + segmentation streaming from RTSP source

set -euo pipefail

# Default RTSP stream (override by exporting RTSP_URL before calling this script)
RTSP_URL=${RTSP_URL:-"rtsp://10.103.1.5:8554/webcam"}

# Processing settings
PROCESS_RES=${PROCESS_RES:-"320x180"}
PROCESS_EVERY=${PROCESS_EVERY:-3}
MAX_OBJECTS=${MAX_OBJECTS:-6}
REINIT_EVERY=${REINIT_EVERY:-100}

# Output / logging
SAVE_FRAMES=${SAVE_FRAMES:-0}
SAVE_DIR=${SAVE_DIR:-"saved_frames"}
JSON_OUTPUT=${JSON_OUTPUT:-""}          # Leave empty to disable file logging
JSON_INTERVAL=${JSON_INTERVAL:-5}
DISPLAY_OUTPUT=${DISPLAY_OUTPUT:-1}     # Set to 0 to disable display window

echo "Starting RTSP depth + segmentation streaming..."
echo "RTSP URL: ${RTSP_URL}"
echo "Processing: ${PROCESS_RES} (every ${PROCESS_EVERY} frames)"
if [[ "${DISPLAY_OUTPUT}" == "0" ]]; then
    echo "Display: disabled"
else
    echo "Display: enabled"
fi
if [[ -n "${JSON_OUTPUT}" ]]; then
    echo "JSON output -> ${JSON_OUTPUT} (interval ${JSON_INTERVAL})"
else
    echo "JSON output: disabled"
fi
if (( SAVE_FRAMES > 0 )); then
    echo "Saving last ${SAVE_FRAMES} frames to ${SAVE_DIR}/ on exit"
else
    echo "Frame saving: disabled"
fi
echo "----------------------------------------"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CMD=(
    python3 "${SCRIPT_DIR}/depth_seg_streaming.py"
    --source "${RTSP_URL}"
    --resolution "${PROCESS_RES}"
    --process-every "${PROCESS_EVERY}"
    --max-objects "${MAX_OBJECTS}"
    --reinit-every "${REINIT_EVERY}"
    --object-detect-every 5
    --object-threshold 0.5
    --object-overlap-threshold 0.6
)

if (( SAVE_FRAMES > 0 )); then
    CMD+=(--save-frames "${SAVE_FRAMES}" --save-frames-dir "${SAVE_DIR}")
fi

if [[ -n "${JSON_OUTPUT}" ]]; then
    CMD+=(--save-json-file "${JSON_OUTPUT}" --json-interval "${JSON_INTERVAL}")
fi

if [[ "${DISPLAY_OUTPUT}" == "0" ]]; then
    CMD+=(--no-display)
fi

echo "Executing: ${CMD[*]}"
echo ""

exec "${CMD[@]}"

