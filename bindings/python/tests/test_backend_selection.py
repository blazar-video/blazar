"""`backend="auto"` picks from a pool; an explicit backend is honoured as given.

The pool is a constraint: `hardware` never falls back to software, `any` prefers
hardware and falls back, `software` never returns hardware.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import blazar

WIDTH, HEIGHT, FRAMES = 640, 480, 12
CALIBRATIONS = os.environ.get(
    "BLAZAR_CALIBRATIONS",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "calibrations", "default"),
)


def frames() -> list[np.ndarray]:
    out = []
    for index in range(FRAMES):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[:, :, 0] = (np.arange(WIDTH) + index * 8)[None, :] % 256
        frame[:, :, 1] = (np.arange(HEIGHT)[:, None] + index * 4) % 256
        out.append(frame)
    return out


def encode(backend: str, pool: str = "any", codec: str = "h264"):
    encoder = blazar.Encoder(
        calibrations=blazar.Calibrations.load(CALIBRATIONS),
        identity={"family": "wan", "version": "2.1", "quant": "bf16", "size": "1.3b"},
        target="balanced",
        backend=backend,
        codec=codec,
        output=None,
        request_id="backend",
        backend_pool=pool,
    )
    assert encoder.resolved_backend is None, "the backend is unknown before the first frame"
    units = 0
    try:
        for frame in frames():
            encoder.submit(frame, width=WIDTH, height=HEIGHT, fps=24.0, layout="thwc", range="full_u8")
        encoder.close()
    except RuntimeError:
        encoder.close()
        raise
    units = sum(1 for e in encoder.poll() if e["kind"] == "encoded" and e["data"] is not None)
    return encoder.resolved_backend, units


def test_auto_with_any_pool_prefers_hardware() -> None:
    backend, units = encode("auto", "any")
    assert backend == "nvenc_h264"
    assert units > 0


def test_auto_with_software_pool_never_returns_hardware() -> None:
    backend, units = encode("auto", "software")
    assert backend == "x264"
    assert units > 0


def test_auto_with_hardware_pool_does_not_fall_back_for_a_software_only_codec() -> None:
    # vp9 has no hardware backend in this build: the pool constraint must win
    # over "something can serve this request".
    with pytest.raises(RuntimeError, match="no backend"):
        encode("auto", "hardware", codec="vp9")


def test_an_explicit_backend_is_honoured_as_named() -> None:
    backend, units = encode("x264")
    assert backend == "x264"
    assert units > 0


def test_unknown_pool_is_rejected_before_any_frame() -> None:
    with pytest.raises(ValueError, match="backend_pool must be"):
        blazar.Encoder(
            calibrations=blazar.Calibrations.load(CALIBRATIONS),
            identity={"family": "wan", "version": "2.1", "quant": "bf16", "size": "1.3b"},
            target="balanced",
            backend="auto",
            codec="h264",
            output=None,
            backend_pool="Hardware",
        )


def test_backend_names_the_implementation_not_the_codec() -> None:
    backend, units = encode("x264")
    assert backend == "x264" and units > 0
    backend, units = encode("nvenc", codec="h264")
    assert backend == "nvenc_h264" and units > 0


def test_an_implementation_that_cannot_encode_the_codec_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not encode"):
        encode("x264", codec="hevc")


def test_a_pair_shaped_backend_is_rejected_with_the_fix() -> None:
    with pytest.raises(ValueError, match="restates the codec"):
        encode("nvenc_h264")
