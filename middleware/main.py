import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from helpers.ffmpeg_bridge import FFmpegHLSBridge

try:
    import paho.mqtt.client as mqtt
except ImportError:  # pragma: no cover
    mqtt = None
logger = logging.getLogger("middleware")
logging.basicConfig(level=logging.INFO)

HLS_ROOT_DIR = Path(__file__).parent / "static" / "hls"
QUEST_HLS_DIR = HLS_ROOT_DIR / "quest"
QUEST_HLS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PLAYLIST_NAME = "index.m3u8"
# DEFAULT_RTSP_URL = os.getenv("JETSON_RTSP_URL")
DEFAULT_RTSP_URL = "rtsp://10.103.1.3:8554/cam"

MQTT_ENABLED = os.environ.get("MQTT_ENABLE", "1") != "0"
MQTT_HOST = os.environ.get("MQTT_HOST", "10.103.1.3")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
# MQTT_TOPIC = os.environ.get("MQTT_TOPIC", "fastsam/masks")
# MQTT_CLIENT_ID = os.environ.get("MQTT_CLIENT_ID", "fastsam-subscriber")
DEFAULT_MQTT_TOPIC = os.environ.get("MQTT_TOPIC", "depth/seg")
DEFAULT_MQTT_CLIENT_ID = os.environ.get("MQTT_CLIENT_ID", "depth-seg-subscriber")
MQTT_QOS = int(os.environ.get("MQTT_QOS", "0"))

app = FastAPI(
    title="Local Drone Middleware",
    version="0.1.0",
    description=(
        "Receives telemetry and perception data streams from the Jetson device "
        "and brokers updates to the Quest 3 headset."
    ),
)

app.mount("/hls", StaticFiles(directory=HLS_ROOT_DIR, html=False), name="hls")

bridge_lock = asyncio.Lock()
bridge: Optional[FFmpegHLSBridge] = None

mqtt_lock = asyncio.Lock()
mqtt_client: Optional["mqtt.Client"] = None  # type: ignore[misc]
mqtt_connection_params: Optional[tuple[str, int]] = None
subscribed_topics: set[str] = set()
mqtt_message_count = 0


class SerialFrame(BaseModel):
    """
    Represents a serial data packet emitted alongside the RTSP video stream.

    The Jetson can POST these payloads while it publishes frames over RTSP.
    """

    frame_id: str
    payload: str
    timestamp: Optional[datetime] = None


class QuestStreamDescriptor(BaseModel):
    """
    High-level description of the Quest 3-bound HLS session.

    The concrete payload fields are placeholders and can evolve as we refine
    the client requirements.
    """

    session_id: str
    hls_manifest: Optional[str]
    negotiated_at: datetime
    running: bool
    notes: Optional[str] = None


class QuestHandshake(BaseModel):
    """
    Minimal handshake payload emitted by the Quest 3 headset.

    The headset can identify itself (serial number, user account, etc.) to
    allow the middleware to tailor the outgoing stream.
    """

    client_id: Optional[str] = None
    requested_channel: Optional[str] = None


class StreamStartRequest(BaseModel):
    """Payload for starting the Jetson → Quest streaming bridge."""

    rtsp_url: Optional[str] = None
    copy_video: bool = False
    segment_seconds: Optional[float] = None
    playlist_length: Optional[int] = None
    video_preset: Optional[str] = None
    video_tune: Optional[str] = None
    video_profile: Optional[str] = None
    video_bitrate: Optional[str] = None
    maxrate: Optional[str] = None
    bufsize: Optional[str] = None
    gop_size: Optional[int] = None
    threads: Optional[int] = None
    rtsp_transport: Optional[str] = None
    audio_codec: Optional[str] = None
    audio_bitrate: Optional[str] = None
    audio_channels: Optional[int] = None
    audio_rate: Optional[int] = None
    input_flags: list[str] = []
    extra_flags: list[str] = []


class StreamStatus(BaseModel):
    """Represents the lifecycle state of the ffmpeg bridge."""

    running: bool
    manifest_url: Optional[str] = None
    playlist_path: Optional[str] = None
    message: Optional[str] = None


class MQTTSubscribeRequest(BaseModel):
    """Configures a subscription to the Jetson-published MQTT topic."""

    host: Optional[str] = None
    port: Optional[int] = None
    topic: Optional[str] = None
    qos: Optional[int] = None
    client_id: Optional[str] = None


async def persist_serial_payload(packet: SerialFrame) -> None:
    """Placeholder for persistence/queueing logic."""
    logger.info(
        "Serial payload stored",
        extra={"frame_id": packet.frame_id, "timestamp": packet.timestamp},
    )


@app.post("/rtsp/serial")
async def receive_serial_payload(
    packet: SerialFrame, background_tasks: BackgroundTasks
) -> dict[str, str]:
    """
    Ingests serial metadata emitted in sync with the Jetson RTSP stream.

    The Jetson should POST JSON packets containing the frame identifier and
    any encoded metadata. The payload can be arbitrary (JSON string, base64,
    etc.) depending on the upstream encoder.
    """
    if packet.timestamp is None:
        packet.timestamp = datetime.utcnow()

    background_tasks.add_task(persist_serial_payload, packet)
    return {"status": "accepted", "frame_id": packet.frame_id}


@app.post("/rtsp/serial/raw")
async def receive_serial_raw(request: Request) -> dict[str, str]:
    """
    Alternate entrypoint for raw binary serial buffers.

    This is useful when the Jetson publishes CBOR/MsgPack or raw bytes alongside
    the RTSP stream. Callers should send the POST request with
    ``Content-Type: application/octet-stream``.
    """
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)

    if not body:
        raise HTTPException(status_code=400, detail="Empty serial payload")

    logger.info("Received %d raw serial bytes", len(body))
    # TODO: Dispatch to downstream consumers (queue, websocket, etc.)
    return {"status": "accepted", "bytes": len(body)}


def _manifest_url(request: Request) -> str:
    base_url = str(request.base_url).rstrip("/")
    return f"{base_url}/hls/quest/{DEFAULT_PLAYLIST_NAME}"


@app.post("/streams/quest/start", response_model=StreamStatus)
async def start_quest_stream(
    payload: StreamStartRequest, request: Request
) -> StreamStatus:
    """
    Launches the ffmpeg bridge that converts the Jetson RTSP feed into HLS segments.

    The RTSP URL can be provided per-request or via the ``JETSON_RTSP_URL``
    environment variable.
    """
    rtsp_url = payload.rtsp_url or DEFAULT_RTSP_URL
    if not rtsp_url:
        raise HTTPException(
            status_code=400,
            detail="Missing RTSP URL. Supply it in the request body or set JETSON_RTSP_URL.",
        )

    async with bridge_lock:
        global bridge
        if bridge is not None and bridge.is_running:
            raise HTTPException(status_code=409, detail="Quest stream already running.")

        bridge_kwargs: dict[str, object] = {
            "rtsp_url": rtsp_url,
            "output_directory": QUEST_HLS_DIR,
            "playlist_name": DEFAULT_PLAYLIST_NAME,
            "copy_video": payload.copy_video,
        }

        if payload.segment_seconds is not None:
            bridge_kwargs["segment_seconds"] = payload.segment_seconds
        if payload.playlist_length is not None:
            bridge_kwargs["playlist_length"] = payload.playlist_length
        if payload.video_preset is not None:
            bridge_kwargs["video_preset"] = payload.video_preset
        if payload.video_tune is not None:
            bridge_kwargs["video_tune"] = payload.video_tune
        if payload.video_profile is not None:
            bridge_kwargs["video_profile"] = payload.video_profile
        if payload.video_bitrate is not None:
            bridge_kwargs["video_bitrate"] = payload.video_bitrate
        if payload.maxrate is not None:
            bridge_kwargs["maxrate"] = payload.maxrate
        if payload.bufsize is not None:
            bridge_kwargs["bufsize"] = payload.bufsize
        if payload.gop_size is not None:
            bridge_kwargs["gop_size"] = payload.gop_size
        if payload.threads is not None:
            bridge_kwargs["threads"] = payload.threads
        if payload.rtsp_transport is not None:
            bridge_kwargs["rtsp_transport"] = payload.rtsp_transport
        if payload.audio_codec is not None:
            bridge_kwargs["audio_codec"] = payload.audio_codec
        if payload.audio_bitrate is not None:
            bridge_kwargs["audio_bitrate"] = payload.audio_bitrate
        if payload.audio_channels is not None:
            bridge_kwargs["audio_channels"] = payload.audio_channels
        if payload.audio_rate is not None:
            bridge_kwargs["audio_rate"] = payload.audio_rate
        if payload.input_flags:
            bridge_kwargs["input_flags"] = payload.input_flags
        if payload.extra_flags:
            bridge_kwargs["extra_flags"] = payload.extra_flags

        bridge = FFmpegHLSBridge(**bridge_kwargs)
        bridge.start()

        current_bridge = bridge

    ready = False
    if current_bridge is not None:
        ready = await asyncio.to_thread(current_bridge.ensure_ready)

    manifest = _manifest_url(request) if current_bridge else None
    playlist_path = str(current_bridge.playlist_path) if current_bridge else None
    message = None
    if not ready:
        message = (
            "HLS manifest not detected yet. The Quest headset should retry shortly."
        )

    return StreamStatus(
        running=current_bridge.is_running if current_bridge else False,
        manifest_url=manifest,
        playlist_path=playlist_path,
        message=message,
    )


@app.post("/streams/quest/stop", response_model=StreamStatus)
async def stop_quest_stream() -> StreamStatus:
    """Stops the ffmpeg bridge if it is running."""
    async with bridge_lock:
        global bridge
        if bridge is None or not bridge.is_running:
            raise HTTPException(status_code=409, detail="Quest stream is not running.")

        current_bridge = bridge
        playlist = str(current_bridge.playlist_path)
        bridge = None

    await asyncio.to_thread(current_bridge.stop)

    return StreamStatus(
        running=False,
        manifest_url=None,
        playlist_path=playlist,
        message="Stream bridge stopped.",
    )


@app.get("/streams/quest/status", response_model=StreamStatus)
async def quest_stream_status(request: Request) -> StreamStatus:
    """Returns the current status of the ffmpeg bridge."""
    async with bridge_lock:
        current_bridge = bridge
        running = current_bridge.is_running if current_bridge else False

    manifest = _manifest_url(request) if running else None
    playlist = (
        str(current_bridge.playlist_path) if current_bridge is not None else None
    )

    return StreamStatus(
        running=running,
        manifest_url=manifest,
        playlist_path=playlist,
    )


def _ensure_mqtt_client(
    host: str, port: int, client_id: str
) -> "mqtt.Client":  # type: ignore[name-defined]
    """Initializes and caches the MQTT client instance."""
    if mqtt is None:
        raise HTTPException(
            status_code=500,
            detail="paho-mqtt is not installed. Install it to enable MQTT subscriptions.",
        )

    global mqtt_client, mqtt_connection_params

    if mqtt_client is None:
        logger.info(
            "Initializing MQTT client",
            extra={"host": host, "port": port, "client_id": client_id},
        )

        client = mqtt.Client(client_id=client_id)

        def _on_connect(client, userdata, flags, rc):  # pragma: no cover - callback
            if rc == 0:
                logger.info("Connected to MQTT broker at %s:%s", host, port)
            else:
                logger.error(
                    "Failed to connect to MQTT broker at %s:%s (code %s)",
                    host,
                    port,
                    rc,
                )

        def _on_message(client, userdata, msg):  # pragma: no cover - callback
            global mqtt_message_count
            mqtt_message_count += 1
            data: Optional[dict] = None
            try:
                payload_text = msg.payload.decode("utf-8")
                parsed = json.loads(payload_text)
                if isinstance(parsed, dict):
                    data = parsed
            except (UnicodeDecodeError, json.JSONDecodeError):
                data = None

            # logger.info("MQTT %s => %s", msg.topic, payload)  # noisy payload logging
            # print(f"MQTT message #{mqtt_message_count} ({msg.topic}) keys: {keys}")  # noqa: ERA001
            if data and "object_detections" in data:
                print(
                    f"MQTT message #{mqtt_message_count} ({msg.topic}) "
                    f"object_detections: {data['object_detections']}"
                )

        client.on_connect = _on_connect
        client.on_message = _on_message

        try:
            client.connect(host, port, keepalive=60)
        except Exception as exc:  # pragma: no cover
            logger.exception("MQTT connection error")
            raise HTTPException(
                status_code=502, detail=f"MQTT connection error: {exc}"
            ) from exc

        client.loop_start()

        mqtt_client = client
        mqtt_connection_params = (host, port)

    else:
        current_host, current_port = mqtt_connection_params or (host, port)
        if (host, port) != (current_host, current_port):
            raise HTTPException(
                status_code=409,
                detail=(
                    "MQTT client already connected to "
                    f"{current_host}:{current_port}. Restart the server to change broker."
                ),
            )

    return mqtt_client


@app.post("/mqtt/subscribe")
async def subscribe_mqtt(
    payload: MQTTSubscribeRequest, background_tasks: BackgroundTasks
) -> dict[str, object]:
    """Subscribes to the configured MQTT topic and logs incoming messages."""
    if not MQTT_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="MQTT subscriptions are disabled. Set MQTT_ENABLE=1 to enable.",
        )

    host = payload.host or MQTT_HOST
    port = payload.port or MQTT_PORT
    topic = payload.topic or MQTT_TOPIC
    qos = payload.qos if payload.qos is not None else MQTT_QOS
    client_id = payload.client_id or MQTT_CLIENT_ID

    async with mqtt_lock:
        client = _ensure_mqtt_client(host, port, client_id)

        if topic in subscribed_topics:
            logger.info("Already subscribed to MQTT topic %s", topic)
            return {"status": "exists", "topic": topic, "host": host, "port": port}

        subscribed_topics.add(topic)

        def _subscribe():
            logger.info("Subscribing to MQTT topic %s (QoS %s)", topic, qos)
            result, _ = client.subscribe(topic, qos=qos)
            if result != mqtt.MQTT_ERR_SUCCESS:  # pragma: no cover - network error
                logger.error("MQTT subscribe failed for %s (code %s)", topic, result)
                subscribed_topics.discard(topic)

        background_tasks.add_task(_subscribe)

    return {"status": "subscribing", "topic": topic, "host": host, "port": port}


@app.post("/quest/hls/session", response_model=QuestStreamDescriptor)
async def establish_quest_hls_session(
    handshake: QuestHandshake, request: Request
) -> QuestStreamDescriptor:
    """
    Negotiates an outbound HLS session with the Quest 3 headset.

    Returns a lightweight JSON descriptor that the headset can use to attach to
    the middleware-controlled HLS stream. The payload is intentionally minimal while
    we finalize the exact metadata contract.
    """
    session_id = handshake.client_id or f"quest-{uuid4().hex}"
    async with bridge_lock:
        running = bridge.is_running if bridge else False

    return QuestStreamDescriptor(
        session_id=session_id,
        hls_manifest=_manifest_url(request) if running else None,
        negotiated_at=datetime.utcnow(),
        running=running,
        notes=(
            "Start the Quest stream via /streams/quest/start before requesting the HLS manifest."
            if not running
            else "Quest stream is active. Subscribe to the provided manifest."
        ),
    )


def main() -> None:
    """
    Entry point for local development with ``python middleware/main.py``.

    For production use, run ``uvicorn middleware.main:app --reload`` instead.
    """
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
