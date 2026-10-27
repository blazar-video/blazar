"""One `submit`: host arrays and CUDA tensors take the path their type implies,
and residency and layout are fixed by the first frame.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import blazar

WIDTH, HEIGHT, FRAMES = 320, 240, 6
CALIBRATIONS = os.environ.get(
    "BLAZAR_CALIBRATIONS",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "calibrations", "default"),
)


def encoder(**kwargs):
    return blazar.Encoder(
        calibrations=blazar.Calibrations.load(CALIBRATIONS),
        identity={"family": "wan", "version": "2.1", "quant": "bf16", "size": "1.3b"},
        target="balanced",
        backend="nvenc",
        codec="h264",
        output=None,
        request_id="dispatch",
        **kwargs,
    )


def host_frame(index: int) -> np.ndarray:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:, :, 0] = (np.arange(WIDTH) + index * 8)[None, :] % 256
    frame[:, :, 1] = (np.arange(HEIGHT)[:, None] + index * 4) % 256
    return frame


def test_residency_is_reported_and_fixed_by_the_first_frame() -> None:
    enc = encoder()
    assert enc.residency is None
    enc.submit(host_frame(0), width=WIDTH, height=HEIGHT, fps=24.0, layout="thwc", range="full_u8")
    assert enc.residency == "host"
    enc.close()


def test_a_cuda_tensor_takes_the_device_path_through_the_same_call() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    enc = encoder()
    tensor = torch.zeros((1, 3, FRAMES, HEIGHT, WIDTH), dtype=torch.bfloat16, device="cuda")
    enc.submit(tensor, fps=24.0, range="signed_unit")
    assert enc.residency == "device"
    enc.close()
    units = [e for e in enc.poll() if e["kind"] == "encoded" and e["data"] is not None]
    assert len(units) == FRAMES


def test_mixing_residencies_on_one_stream_is_refused() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    enc = encoder()
    enc.submit(host_frame(0), width=WIDTH, height=HEIGHT, fps=24.0, layout="thwc", range="full_u8")
    tensor = torch.zeros((1, 3, 2, HEIGHT, WIDTH), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="opened for host frames"):
        enc.submit(tensor, fps=24.0, range="signed_unit")
    enc.close()


def test_layout_is_fixed_at_open() -> None:
    enc = encoder(layout="thwc")
    planar = np.zeros((3, HEIGHT, WIDTH), dtype=np.uint8)
    with pytest.raises(ValueError, match="is fixed for this stream"):
        enc.submit(planar, width=WIDTH, height=HEIGHT, fps=24.0, layout="bcthw", range="full_u8")
    enc.close()


def test_unknown_layout_is_rejected_at_open() -> None:
    with pytest.raises(ValueError, match="layout"):
        encoder(layout="twhc")
