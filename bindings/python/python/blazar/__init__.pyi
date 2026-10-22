"""Type stubs for the compiled `blazar._blazar` extension."""

from typing import Any, Literal, Sequence

import numpy as np

__all__ = ["Calibrations", "Encoder"]

Target = Literal["low_latency", "balanced", "quality"]
Backend = Literal[
    "nvenc_av1", "nvenc_h264", "nvenc_hevc", "x264", "x265", "svtav1", "vp9", "mock"
]
Codec = Literal["h264", "hevc", "av1", "vp9"]
Layout = Literal["thwc", "bcthw"]
Range = Literal["unit01", "signed_unit", "full_u8"]

class Calibrations:
    """A directory of calibration files, loaded once and shared by streams."""

    @staticmethod
    def load(path: str) -> Calibrations:
        """Read every ``*.toml`` under ``path``.

        One invalid file fails the whole load rather than being skipped.
        """
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

class Encoder:
    """One encoder stream, driven by a worker thread.

    ``submit*`` hands frames over and returns without waiting for the codec.
    Use as a context manager: leaving the block closes the stream.
    """

    def __init__(
        self,
        *,
        calibrations: Calibrations,
        identity: dict[str, str],
        target: Target,
        backend: Backend,
        output: str,
        codec: Codec | None = None,
        request_id: str = "stream",
        video_extradata: bytes | None = None,
        queue_depth: int = ...,
    ) -> None: ...
    def submit(
        self,
        frame: np.ndarray | Any,
        *,
        width: int,
        height: int,
        fps: float,
        layout: Layout,
        range: Range,
        pts_ticks: int | None = None,
        scene_cut: bool = False,
    ) -> int:
        """Queue one host frame; returns the assigned PTS."""
    def submit_batch(
        self,
        frames: Sequence[np.ndarray | Any],
        *,
        width: int,
        height: int,
        fps: float,
        layout: Layout,
        range: Range,
        pts_ticks: int | None = None,
        scene_cut: bool = False,
    ) -> int:
        """Queue many host frames in one call."""
    def submit_device_batch(
        self,
        tensor: Any,
        *,
        fps: float,
        range: Range,
        pts_ticks: int | None = None,
        scene_cut: bool = False,
    ) -> int:
        """Queue a CUDA ``(1, 3, T, H, W)`` float32/bfloat16 contiguous tensor."""
    def poll(self, timeout: float | None = None) -> list[dict[str, Any]]:
        """Encoded units produced so far."""
    def flush(self, timeout: float | None = None) -> None:
        """Encode everything queued."""
    def close(self, timeout: float | None = None) -> None:
        """Flush and finalise the output. Safe to call twice."""
    def stats(self) -> dict[str, Any]: ...
    def error_message(self) -> str | None: ...
    def failed_event_message(self) -> str | None: ...
    def __enter__(self) -> Encoder: ...
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool | None: ...
    def __repr__(self) -> str: ...
