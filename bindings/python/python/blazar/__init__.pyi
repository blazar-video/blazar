"""Type stubs for the compiled `blazar._blazar` extension."""

from typing import Any, Literal, Sequence

import numpy as np

__all__ = ["Calibrations", "Encoder", "backends"]

Target = Literal["low_latency", "balanced", "quality"]
Backend = Literal["nvenc", "videotoolbox", "svtav1", "x264", "x265", "vp9", "mock", "auto"]
BackendPool = Literal["any", "hardware", "software"]
Preprocess = Literal["auto", "cpu", "gpu"]
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
        max_buffered_units: int = ...,
        preprocess: Preprocess = "auto",
        backend_pool: BackendPool = "any",
    ) -> None: ...
    def submit(
        self,
        frames: np.ndarray | Any,
        fps: float,
        width: int | None = None,
        height: int | None = None,
        layout: Layout | None = None,
        range: Range = "unit01",
        pts_ticks: int | None = None,
        scene_cut: bool = False,
    ) -> int:
        """Queue frames; a numpy array takes the host path, a CUDA tensor the device path."""

    def poll(self, timeout: float | None = None) -> list[dict[str, Any]]:
        """Encoded units produced so far."""
    def flush(self, timeout: float | None = None) -> None:
        """Encode everything queued."""
    def close(self, timeout: float | None = None) -> None:
        """Flush and finalise the output. Safe to call twice."""
    @property
    def resolved_backend(self) -> str | None: ...

    @property
    def preprocess(self) -> str | None: ...

    @property
    def residency(self) -> str | None: ...

    @property
    def stats(self) -> dict[str, Any]: ...
    def error_message(self) -> str | None: ...
    def failed_event_message(self) -> str | None: ...
    def __enter__(self) -> Encoder: ...
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool | None: ...
    def __repr__(self) -> str: ...

def backends() -> list[str]:
    """Encoder implementations compiled into this wheel (e.g. ``["nvenc", "x264", "mock"]``)."""
