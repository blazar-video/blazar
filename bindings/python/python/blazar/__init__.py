"""Python bindings for the Blazar video encoding harness.

Two entry points:

``Calibrations.load(path)``
    Parse a directory of calibration TOML once. Reuse it across requests.

``Encoder(...)``
    One encode stream writing a fragmented MP4. Create one per request and
    ``close()`` it (or use it as a context manager).

Frames are float32 arrays in C order. The calendar mapping is explicit: the
runtimes never tag colorspace, and numpy cannot represent bf16, so callers pass
``layout``, ``range`` and the frame dimensions per submit. Blazar assumes BT.709
and records that assumption in the stream.

This module is deliberately small. Audio, device tensors and non-file sinks do
not exist yet.
"""

from ._blazar import Calibrations, Encoder, backends, __version__  # noqa: F401

__all__ = ["Calibrations", "Encoder", "backends", "__version__"]
