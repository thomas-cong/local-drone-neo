import base64
import json
import os, time, sys
import cv2
import numpy as np
import torch

# -------- Config --------
WIDTH, HEIGHT, FPS = 1280, 720, 30
BITRATE = 6_000_000             # ~6 Mbps
RTSP_URL = "rtsp://127.0.0.1:8554/cam"  # MediaMTX path you'll view at rtsp://<JETSON_IP>:8554/seg
USE_TCP = True                 # True if your network loses UDP packets
MODEL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRES = 0.35
IOU_THRES = 0.6
MASK_ALPHA = 0.35               # overlay transparency
MQTT_ENABLED = os.environ.get("MQTT_ENABLE", "1") != "0"
MQTT_HOST = os.environ.get("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_TOPIC = os.environ.get("MQTT_TOPIC", "fastsam/masks")
MQTT_CLIENT_ID = os.environ.get("MQTT_CLIENT_ID", "fastsam-publisher")
MQTT_QOS = int(os.environ.get("MQTT_QOS", "0"))
INFERENCE_EVERY = max(1, int(os.environ.get("INFERENCE_EVERY", "5")))

# -------- FastSAM load --------
# If your package exposes a `FastSAM` class; otherwise, adapt based on your installed API.
try:
    from ultralytics import FastSAM
except Exception as e:
    print("FastSAM import failed; install fastsam or switch to ultralytics YOLOv8-seg.", e)
    sys.exit(1)

# Pick a weights file (adjust name/path if different)
WEIGHTS = os.environ.get("FASTSAM_WEIGHTS", "FastSAM-s.pt")

model = FastSAM(WEIGHTS)
model.to(MODEL_DEVICE).eval()

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    if MQTT_ENABLED:
        print("paho-mqtt not installed; MQTT mask publishing disabled.")
    MQTT_ENABLED = False

# -------- GStreamer: camera (CSI) → OpenCV --------
# Use Argus for CSI and NVMM→system copy. If you prefer V4L2 /dev/video0 path, replace the source.
cam_pipe = (
    f"nvarguscamerasrc ! "
    f"video/x-raw(memory:NVMM),width={WIDTH},height={HEIGHT},framerate={FPS}/1 ! "
    f"nvvidconv ! video/x-raw,format=BGRx ! "
    f"videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1"
)
cap = cv2.VideoCapture(cam_pipe, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("Failed to open CSI camera via GStreamer.")
    sys.exit(1)

# -------- GStreamer: OpenCV (BGR) → appsrc → NVENC → RTSP --------
# We feed BGR frames into appsrc; convert to NV12, encode with nvv4l2h264enc, then publish via rtspclientsink.
proto = "tcp" if USE_TCP else "udp"
out_pipe = (
    "appsrc emit-signals=false is-live=true do-timestamp=true format=time "
    f"caps=video/x-raw,format=BGR,width={WIDTH},height={HEIGHT},framerate={FPS}/1 ! "
    "videoconvert ! video/x-raw,format=I420 ! "
    "nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
    # Low-latency NVENC settings:
    f"nvv4l2h264enc control-rate=1 bitrate={BITRATE} preset-level=1 "
    "iframeinterval=15 idrinterval=15 insert-sps-pps=1 ! "
    "h264parse config-interval=1 ! "
    "video/x-h264,stream-format=avc,alignment=au ! "
    f"rtspclientsink location={RTSP_URL} protocols={proto} latency=0 do-rtsp-keep-alive=true"
)
writer = cv2.VideoWriter(out_pipe, cv2.CAP_GSTREAMER, 0, FPS, (WIDTH, HEIGHT))
if not writer.isOpened():
    print("Failed to open RTSP publisher pipeline.")
    sys.exit(1)

def np_to_base64(mask: np.ndarray) -> str:
    """Pack a binary mask to base64 (row-major order)."""
    mask_u8 = mask.astype(np.uint8).flatten(order="C")
    packed = np.packbits(mask_u8)
    return base64.b64encode(packed.tobytes()).decode("ascii")

def serialize_mask(mask: np.ndarray):
    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_payload = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        contour_payload.append(cnt.reshape(-1, 2).astype(int).tolist())
    if not contour_payload:
        return None
    ys, xs = np.where(mask_u8)
    bbox = [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max()),
    ]
    return {
        "area": int(mask_u8.sum()),
        "bbox": bbox,
        "contours": contour_payload,
        "mask": np_to_base64(mask_u8),
    }

def publish_masks(mqtt_client, masks, frame_shape):
    if mqtt_client is None or not masks:
        return
    payload_masks = []
    for mask in masks:
        serialized = serialize_mask(mask)
        if serialized is not None:
            payload_masks.append(serialized)
    payload = {
        "timestamp": time.time(),
        "frame": {
            "width": int(frame_shape[1]),
            "height": int(frame_shape[0]),
        },
        "count": len(payload_masks),
        # "masks": payload_masks,
    }
    try:
        mqtt_client.publish(MQTT_TOPIC, json.dumps(payload), qos=MQTT_QOS)
    except Exception as exc:
        print(f"MQTT publish failed: {exc}")

def draw_masks(frame, masks, color=(0, 255, 0), alpha=MASK_ALPHA):
    """Blend multi-object masks onto the BGR frame. Returns annotated frame and mask list."""
    if masks is None or len(masks) == 0:
        return frame, []
    overlay = frame.copy()
    valid_masks = []
    # masks: list of HxW boolean arrays (or tensor)
    for m in masks:
        if isinstance(m, torch.Tensor):
            m = m.detach().to("cpu").numpy()
        m = m.astype(bool)
        if not np.any(m):
            continue
        overlay[m] = (
            overlay[m] * (1 - alpha)
            + np.array(color, dtype=np.float32) * alpha
        ).astype(np.uint8)
        valid_masks.append(m)
        # Optional: contour outline
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (0, 0, 0), 1)
    return overlay, valid_masks

def fastsam_segment(img_bgr):
    """
    Run FastSAM on BGR image and return a list of boolean masks (HxW).
    Adjust this function to your installed FastSAM API.
    """
    # FastSAM expects RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        # Example API; adapt if your FastSAM exposes a different call signature
        results = model(
            img_rgb,
            device=MODEL_DEVICE,
            conf=CONF_THRES,
            iou=IOU_THRES,
            retina_masks=True
        )
    masks: list[np.ndarray] = []
    frame_h, frame_w = img_bgr.shape[:2]

    # Ultralytics FastSAM returns a list[Results]; grab the first item
    if isinstance(results, (list, tuple)):
        results = results[0]

    ul_masks = getattr(results, "masks", None)
    if ul_masks is None:
        return masks

    data = getattr(ul_masks, "data", None)
    if data is None:
        # some APIs expose masks directly as numpy/list
        data = ul_masks

    if isinstance(data, torch.Tensor):
        data = data.detach().to("cpu").numpy()
    elif isinstance(data, list):
        if len(data) == 0:
            return masks
        data = np.stack([np.asarray(m) for m in data], axis=0)

    if not isinstance(data, np.ndarray):
        return masks

    for m in data:
        mask = m.astype(np.float32)
        # ensure mask is 0/1 float
        if mask.max() > 1.0 + 1e-3:
            mask = mask / 255.0
        mask = (mask > 0.5).astype(np.uint8)
        if mask.shape != (frame_h, frame_w):
            mask = cv2.resize(mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
        masks.append(mask.astype(bool))

    return masks

# -------- Main loop --------
frame_interval = 1.0 / FPS
t0 = time.time()
mqtt_client = None
if MQTT_ENABLED and MQTT_AVAILABLE:
    try:
        mqtt_client = mqtt.Client(client_id=MQTT_CLIENT_ID)
        mqtt_client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
        mqtt_client.loop_start()
        print(f"MQTT publishing masks to {MQTT_HOST}:{MQTT_PORT} topic '{MQTT_TOPIC}' (QoS {MQTT_QOS})")
    except Exception as exc:
        mqtt_client = None
        print(f"Failed to connect to MQTT broker: {exc}")
frame_idx = 0
last_masks: list[np.ndarray] = []
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed")
            break
        should_infer = (frame_idx % INFERENCE_EVERY) == 0

        if should_infer:
            # Segmentation (GPU)
            masks = fastsam_segment(frame)
        else:
            masks = last_masks

        # Overlay
        annotated, valid_masks = draw_masks(frame, masks)

        if should_infer:
            last_masks = valid_masks
        elif valid_masks:
            last_masks = valid_masks

        # Push to RTSP
        writer.write(annotated)

        if should_infer and mqtt_client is not None:
            publish_masks(mqtt_client, valid_masks, frame.shape)

        # Simple pacing to target FPS when CPU/GPU is faster than encode
        dt = time.time() - t0
        if dt < frame_interval:
            time.sleep(frame_interval - dt)
        t0 = time.time()
        frame_idx += 1

except KeyboardInterrupt:
    pass
finally:
    writer.release()
    cap.release()
    if mqtt_client is not None:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
