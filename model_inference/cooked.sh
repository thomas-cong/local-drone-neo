python3 depth_seg_streaming.py \
  --source rtsp://127.0.0.1:8554/cam \
  --camera-fps-limit 15 \
  --no-display \
  --save-frames 6 \
  --json-interval 6 \
  --object-model-size s \
  --max-objects 1 \
  --process-every 6