#!/usr/bin/env bash
#
# Helper script to launch depth_seg_streaming.py against the shared RTSP webcam.
#
# Usage:
#   ./run_depth_seg_rtsp.sh [extra-args...]
# Example (headless processing, higher resolution matrix):
#   ./run_depth_seg_rtsp.sh --no-display --matrix-resolution 200x112
#
# Any extra arguments are forwarded directly to depth_seg_streaming.py.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RTSP_URL="${RTSP_URL:-rtsp://10.103.1.5:8554/webcam}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/depth_seg_streaming.py" \
  --source "${RTSP_URL}" \
  "$@"

