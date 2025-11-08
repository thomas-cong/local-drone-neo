## RTSP to HLS Bridge

1. Ensure `ffmpeg` is installed and available on the system PATH.
2. Instantiate and start the bridge in your application code:

```python
from middleware.helpers.ffmpeg_bridge import FFmpegHLSBridge

bridge = FFmpegHLSBridge(
    rtsp_url="rtsp://jetson.local:8554/primary",
    output_directory="middleware/static/hls/quest",
)
bridge.start()
```

3. The bridge emits an `index.m3u8` playlist and rolling `.ts` segments inside the output directory. Serve that directory via HTTP so the Quest 3 headset can subscribe to the HLS feed.
4. When finished, call `bridge.stop()` to gracefully terminate the underlying ffmpeg process.

### Default tuning

The default configuration re-encodes video with `libx264` (`veryfast`, `zerolatency`), targets ~3 Mbps, enforces frequent keyframes, and delivers sub-second HLS segments with `delete_segments+append_list+round_durations`. This keeps playback smooth in VR players, trading a small amount of latency for fluid motion. Override any parameter through `FFmpegHLSBridge` or the REST API if you need a different profile.

## FastAPI Control Plane

The middleware exposes endpoints under `http://<host>:8000` that orchestrate the streaming pipeline.

- `POST /streams/quest/start` – Launch the ffmpeg bridge. Provide a body like:

  ```json
  {
    "rtsp_url": "rtsp://jetson.local:8554/primary",
    "segment_seconds": 0.5,
    "video_bitrate": "3M",
    "threads": 2
  }
  ```

  If `rtsp_url` is omitted, the service falls back to the `JETSON_RTSP_URL` environment variable.

- `POST /streams/quest/stop` – Gracefully terminate the bridge and release resources.
- `GET /streams/quest/status` – Inspect whether the bridge is running and retrieve the HLS manifest path.
- `POST /quest/hls/session` – Lightweight handshake for the Quest 3; returns the active HLS manifest URL if the stream is up.

Mounting `middleware.main:app` with Uvicorn (`uvicorn middleware.main:app --reload`) will also serve the generated HLS assets at `/hls/quest/index.m3u8`.
