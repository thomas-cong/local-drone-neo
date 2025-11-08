"""
Utility helpers for bridging an RTSP source into an HLS output ladder via ffmpeg.

The Jetson publishes an RTSP stream; this module supervises a local ffmpeg process
that ingests that stream and produces a rolling HLS playlist that the Quest 3
headset (or any other HLS-capable client) can consume.

Example
-------
>>> from middleware.ffmpeg_bridge import FFmpegHLSBridge
>>> bridge = FFmpegHLSBridge(
...     rtsp_url="rtsp://jetson.local:8554/primary",
...     output_directory="middleware/static/hls/quest",
... )
>>> bridge.start()
>>> # ... do work ...
>>> bridge.stop()

The command that is generated is equivalent to:

    ffmpeg -i <rtsp_url> -c:v copy -c:a aac -f hls \\
           -hls_time 2 -hls_list_size 6 -hls_flags delete_segments \\
           <output_directory>/index.m3u8

The default parameters favour low latency while keeping the implementation simple.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import subprocess
import threading
import time
from pathlib import Path
from typing import Iterable, List, Optional

import logging

logger = logging.getLogger(__name__)


@dataclass
class FFmpegHLSBridge:
    """
    Supervises an ffmpeg process that transcodes an RTSP input into an HLS output.

    Parameters
    ----------
    rtsp_url:
        The RTSP URL published by the Jetson.
    output_directory:
        Directory that will contain `index.m3u8` and rolling segment files.
    ffmpeg_path:
        Path to the ffmpeg executable (defaults to `ffmpeg` on PATH).
    playlist_name:
        Name for the playlist manifest. Defaults to `index.m3u8`.
    segment_seconds:
        Target duration of each HLS segment in seconds.
    playlist_length:
        Number of segments to retain in the playlist.
    copy_video:
        If True, video is stream-copied (`-c:v copy`); otherwise re-encoded via `libx264`.
    audio_codec:
        Audio codec to emit. AAC is widely supported for HLS.
    input_flags:
        Flags inserted before the ``-i`` argument (e.g. RTSP transport hints).
    extra_flags:
        Additional ffmpeg CLI arguments appended after the core parameters for
        advanced tuning (authentication, filters, etc.).
    """

    rtsp_url: str
    output_directory: Path | str
    ffmpeg_path: str = "ffmpeg"
    playlist_name: str = "index.m3u8"
    segment_seconds: float = 1.0
    playlist_length: int = 8
    copy_video: bool = False
    audio_codec: str = "aac"
    audio_bitrate: str = "128k"
    audio_channels: int = 2
    audio_rate: int = 48000
    video_preset: str = "veryfast"
    video_tune: str = "zerolatency"
    video_profile: Optional[str] = "high"
    video_bitrate: Optional[str] = "3M"
    maxrate: Optional[str] = "3M"
    bufsize: Optional[str] = "6M"
    gop_size: Optional[int] = 60
    threads: int = 2
    rtsp_transport: Optional[str] = "tcp"
    input_flags: Iterable[str] = field(default_factory=list)
    extra_flags: Iterable[str] = field(default_factory=list)

    _process: Optional[subprocess.Popen] = field(default=None, init=False, repr=False)
    _stdout_thread: Optional[threading.Thread] = field(default=None, init=False, repr=False)
    _stderr_thread: Optional[threading.Thread] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.output_directory = Path(self.output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

        if self.rtsp_transport:
            self.input_flags = list(self.input_flags) + ["-rtsp_transport", self.rtsp_transport]

    # --------------------------------------------------------------------- utils
    def _command(self) -> List[str]:
        input_args = list(self.input_flags)

        cmd: List[str] = [self.ffmpeg_path, "-y", *input_args, "-i", self.rtsp_url]
        if self.copy_video:
            cmd.extend(["-c:v", "copy"])
        else:
            cmd.extend(["-c:v", "libx264", "-preset", self.video_preset, "-tune", self.video_tune])
            if self.video_profile:
                cmd.extend(["-profile:v", self.video_profile])
            if self.gop_size:
                cmd.extend(["-g", str(self.gop_size)])
            cmd.extend(["-keyint_min", str(self.gop_size or 30)])
            cmd.extend(
                [
                    "-force_key_frames",
                    f"expr:gte(t,n_forced*{self.segment_seconds})",
                ]
            )
            if self.video_bitrate:
                cmd.extend(["-b:v", self.video_bitrate])
            if self.maxrate:
                cmd.extend(["-maxrate", self.maxrate])
            if self.bufsize:
                cmd.extend(["-bufsize", self.bufsize])
            if self.threads:
                cmd.extend(["-threads", str(self.threads)])

        cmd.extend(
            [
                "-c:a",
                self.audio_codec,
                "-b:a",
                self.audio_bitrate,
                "-ac",
                str(self.audio_channels),
                "-ar",
                str(self.audio_rate),
                "-f",
                "hls",
                "-hls_time",
                str(self.segment_seconds),
                "-hls_list_size",
                str(self.playlist_length),
                "-hls_flags",
                "delete_segments+append_list+round_durations",
                "-hls_segment_type",
                "mpegts",
                "-hls_init_time",
                str(self.segment_seconds),
                "-hls_allow_cache",
                "0",
                str(self.output_directory / self.playlist_name),
            ]
        )

        if self.extra_flags:
            cmd[1:1] = list(self.extra_flags)  # inject after ffmpeg executable

        return cmd

    def _log_stream(self, pipe, level: int) -> None:
        """Continuously logs stdout/stderr from ffmpeg until the process exits."""
        with pipe:
            for line in iter(pipe.readline, b""):
                logger.log(level, "ffmpeg: %s", line.decode(errors="ignore").strip())

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """
        Launches the ffmpeg process.

        Raises
        ------
        RuntimeError: if the process is already running.
        FileNotFoundError: if the ffmpeg executable cannot be located.
        """
        if self.is_running:
            raise RuntimeError("ffmpeg bridge is already running")

        cmd = self._command()
        logger.info("Starting ffmpeg bridge: %s", " ".join(cmd))

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )

        assert self._process.stdout is not None
        assert self._process.stderr is not None

        self._stdout_thread = threading.Thread(
            target=self._log_stream, args=(self._process.stdout, logging.DEBUG), daemon=True
        )
        self._stderr_thread = threading.Thread(
            target=self._log_stream, args=(self._process.stderr, logging.INFO), daemon=True
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """
        Attempts to gracefully terminate the ffmpeg process.

        Parameters
        ----------
        timeout:
            Seconds to wait for ffmpeg to exit before force-killing it.
        """
        if not self.is_running:
            return

        assert self._process is not None
        logger.info("Stopping ffmpeg bridge (pid=%s)", self._process.pid)
        self._process.terminate()

        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg bridge did not exit in %.1fs; killing", timeout)
            self._process.kill()
            self._process.wait()

        self._cleanup_threads()
        self._process = None

    def _cleanup_threads(self) -> None:
        for thread in (self._stdout_thread, self._stderr_thread):
            if thread and thread.is_alive():
                thread.join(timeout=0.5)
        self._stdout_thread = None
        self._stderr_thread = None

    @property
    def playlist_path(self) -> Path:
        """Returns the resolved path to the HLS playlist manifest."""
        return self.output_directory / self.playlist_name

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def ensure_ready(self, wait_seconds: float = 0.5, retries: int = 20) -> bool:
        """
        Polls for the playlist manifest until it exists or retries are exhausted.

        Returns True if the playlist file is present, False otherwise.
        """
        for _ in range(retries):
            if self.playlist_path.exists():
                return True
            time.sleep(wait_seconds)
        return False


__all__ = ["FFmpegHLSBridge"]

