"""Placement of RGB->NV12 conversion is explicit, reported, and does not change
the bitstream class.

`preprocess="auto"` resolves per backend and per frame location; `"cpu"` and
`"gpu"` are honoured or rejected, never silently substituted. These tests need a
CUDA device with NVENC, so they skip when Blazar has no hardware backend.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import blazar

WIDTH, HEIGHT, FRAMES = 640, 480, 24
CALIBRATIONS = os.environ.get(
    "BLAZAR_CALIBRATIONS",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "calibrations", "default"),
)


def frames() -> list[np.ndarray]:
    """Moving content, so the encoder codes something rather than flat frames."""
    out = []
    for index in range(FRAMES):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[:, :, 0] = (np.arange(WIDTH) + index * 8)[None, :] % 256
        frame[:, :, 1] = (np.arange(HEIGHT)[:, None] + index * 4) % 256
        frame[:, :, 2] = 64 + index
        out.append(frame)
    return out


def encode(preprocess: str, backend: str = "nvenc", codec: str = "h264"):
    encoder = blazar.Encoder(
        calibrations=blazar.Calibrations.load(CALIBRATIONS),
        identity={"family": "wan", "version": "2.1", "quant": "bf16", "size": "1.3b"},
        target="balanced",
        backend=backend,
        codec=codec,
        output=None,
        request_id=f"preprocess-{preprocess}-{backend}",
        preprocess=preprocess,
    )
    for frame in frames():
        encoder.submit(frame, width=WIDTH, height=HEIGHT, fps=24.0, layout="thwc", range="full_u8")
    encoder.close()
    units = [e for e in encoder.poll() if e["kind"] == "encoded" and e["data"] is not None]
    return encoder.preprocess, len(units), sum(len(u["data"]) for u in units)


def needs_nvenc() -> None:
    try:
        blazar.Encoder(
            calibrations=blazar.Calibrations.load(CALIBRATIONS),
            identity={"family": "wan", "version": "2.1", "quant": "bf16", "size": "1.3b"},
            target="balanced",
            backend="nvenc",
            codec="h264",
            output=None,
        )
    except Exception as error:  # noqa: BLE001 - any failure means no hardware here
        pytest.skip(f"NVENC unavailable: {error}")


def test_auto_resolves_to_the_device_for_a_host_frame_on_nvenc() -> None:
    needs_nvenc()
    site, units, _ = encode("auto")
    assert site == "gpu", "NVENC encodes from the surface the kernel wrote, so the upload is the only copy"
    assert units == FRAMES


def test_cpu_placement_is_honoured_and_produces_the_same_stream_class() -> None:
    needs_nvenc()
    auto_site, auto_units, auto_bytes = encode("auto")
    cpu_site, cpu_units, cpu_bytes = encode("cpu")
    assert auto_site == "gpu" and cpu_site == "cpu"
    assert cpu_units == auto_units == FRAMES
    # Placement changes cost, not correctness: the two conversions agree within
    # one level, so the streams land in the same size class.
    assert abs(cpu_bytes - auto_bytes) / auto_bytes < 0.05


def test_gpu_placement_is_honoured() -> None:
    needs_nvenc()
    site, units, _ = encode("gpu")
    assert site == "gpu"
    assert units == FRAMES


def test_cpu_codec_converts_host_frames_on_the_cpu_by_default() -> None:
    needs_nvenc()
    site, units, _ = encode("auto", backend="x264", codec="h264")
    assert site == "cpu", "a device conversion for a CPU codec would only add a download"
    assert units == FRAMES


def test_unknown_placement_is_rejected_before_any_frame() -> None:
    with pytest.raises(ValueError, match="preprocess must be one of"):
        blazar.Encoder(
            calibrations=blazar.Calibrations.load(CALIBRATIONS),
            identity={"family": "wan", "version": "2.1", "quant": "bf16", "size": "1.3b"},
            target="balanced",
            backend="x264",
            codec="h264",
            output=None,
            preprocess="CPU",
        )


def test_device_tensor_with_cpu_placement_is_an_error_not_a_silent_device_convert() -> None:
    torch = pytest.importorskip("torch")
    needs_nvenc()
    encoder = blazar.Encoder(
        calibrations=blazar.Calibrations.load(CALIBRATIONS),
        identity={"family": "wan", "version": "2.1", "quant": "bf16", "size": "1.3b"},
        target="balanced",
        backend="nvenc",
        codec="h264",
        output=None,
        preprocess="cpu",
    )
    tensor = torch.zeros((1, 3, 4, HEIGHT, WIDTH), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="device tensor"):
        encoder.submit(tensor, fps=24.0, range="signed_unit")
