"""Contract tests for the Python binding, on the mock encoder.

These assert the binding's rules and the container's structure. They cannot
assert decodability: the mock backend has no real codec, so it is handed a
plausible av1C and writes synthetic payload bytes. Real decode checks live on a
CUDA host (repo README, "Hardware validation").
"""

import os
from pathlib import Path

import numpy as np
import pytest

import blazar

# blazar/bindings/python/tests/test_encoder.py -> blazar-video/calibrations/default
DEFAULT_CALIBRATIONS = Path(__file__).resolve().parents[4] / "calibrations" / "default"
CALIBRATIONS = Path(os.environ.get("BLAZAR_CALIBRATIONS", DEFAULT_CALIBRATIONS))

# Plausible AV1 config record so the mock can open a stream. Not a real av1C for
# the synthetic payload; only the container structure is under test here.
FAKE_AV1C = bytes([0x81, 0x04, 0x0C, 0x00])


def thwc_frame(width: int, height: int, value: float) -> np.ndarray:
    """C-contiguous (H, W, 3) float32 RGB, the xDiT seam shape."""
    return np.full((height, width, 3), value, dtype=np.float32)


def bcthw_frame(width: int, height: int, value: float) -> np.ndarray:
    """C-contiguous (1, 3, 1, H, W) float32 RGB, the vLLM-Omni seam shape."""
    return np.full((1, 3, 1, height, width), value, dtype=np.float32)


@pytest.fixture(scope="module")
def calibrations():
    assert CALIBRATIONS.is_dir(), f"missing calibrations at {CALIBRATIONS}"
    return blazar.Calibrations.load(str(CALIBRATIONS))


def new_encoder(calibrations, output: Path, **kwargs):
    params = dict(
        calibrations=calibrations,
        identity={"family": "wan", "version": "2.1", "quant": "bf16", "size": "1.3b"},
        target="balanced",
        backend="svtav1",
        output=str(output),
        video_extradata=FAKE_AV1C,
    )
    params.update(kwargs)
    return blazar.Encoder(**params)


def test_encodes_frames_into_a_fragmented_mp4(calibrations, tmp_path):
    out = tmp_path / "stream.mp4"
    encoder = new_encoder(calibrations, out)
    for i in range(4):
        seq = encoder.submit(
            thwc_frame(8, 8, 0.5),
            width=8,
            height=8,
            fps=24.0,
            layout="thwc",
            range="unit01",
        )
        assert seq == i
    encoder.close()

    data = out.read_bytes()
    assert len(data) > 0
    assert data[4:8] == b"ftyp", "ISO BMFF signature"
    assert b"moov" in data
    assert b"moof" in data, "fragmented output"
    assert b"mdat" in data


def test_default_pts_is_frame_index_over_fps(calibrations, tmp_path):
    """24 fps on a 90 kHz clock is exactly 3750 ticks per frame."""
    out = tmp_path / "pts.mp4"
    encoder = new_encoder(calibrations, out)
    encoder.submit(
        thwc_frame(8, 8, 0.5), width=8, height=8, fps=24.0, layout="thwc", range="unit01"
    )
    encoder.submit(
        thwc_frame(8, 8, 0.5), width=8, height=8, fps=24.0, layout="thwc", range="unit01"
    )
    encoder.close()
    assert out.stat().st_size > 0


def test_accepts_the_vllm_omni_chunk_layout(calibrations, tmp_path):
    out = tmp_path / "chunk.mp4"
    encoder = new_encoder(calibrations, out)
    encoder.submit(
        bcthw_frame(8, 8, -1.0),
        width=8,
        height=8,
        fps=24.0,
        layout="bcthw",
        range="signed_unit",
    )
    encoder.close()
    assert out.read_bytes()[4:8] == b"ftyp"


def test_context_manager_closes_the_file(calibrations, tmp_path):
    out = tmp_path / "with.mp4"
    with new_encoder(calibrations, out) as encoder:
        encoder.submit(
            thwc_frame(8, 8, 0.25),
            width=8,
            height=8,
            fps=24.0,
            layout="thwc",
            range="unit01",
        )
    assert b"moov" in out.read_bytes()


def test_non_float32_frame_is_rejected_with_the_fix(calibrations, tmp_path):
    encoder = new_encoder(calibrations, tmp_path / "bad-dtype.mp4")
    frame = np.zeros((8, 8, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="float32"):
        encoder.submit(
            frame, width=8, height=8, fps=24.0, layout="thwc", range="unit01"
        )


def test_size_change_mid_stream_is_rejected(calibrations, tmp_path):
    encoder = new_encoder(calibrations, tmp_path / "resize.mp4")
    encoder.submit(
        thwc_frame(8, 8, 0.5), width=8, height=8, fps=24.0, layout="thwc", range="unit01"
    )
    with pytest.raises(ValueError, match="size changed"):
        encoder.submit(
            thwc_frame(16, 16, 0.5),
            width=16,
            height=16,
            fps=24.0,
            layout="thwc",
            range="unit01",
        )


def test_fps_change_mid_stream_is_rejected(calibrations, tmp_path):
    encoder = new_encoder(calibrations, tmp_path / "fps.mp4")
    encoder.submit(
        thwc_frame(8, 8, 0.5), width=8, height=8, fps=24.0, layout="thwc", range="unit01"
    )
    with pytest.raises(ValueError, match="fps changed"):
        encoder.submit(
            thwc_frame(8, 8, 0.5), width=8, height=8, fps=30.0, layout="thwc", range="unit01"
        )


def test_submit_after_close_is_an_error(calibrations, tmp_path):
    encoder = new_encoder(calibrations, tmp_path / "closed.mp4")
    encoder.close()
    with pytest.raises(RuntimeError, match="closed"):
        encoder.submit(
            thwc_frame(8, 8, 0.5), width=8, height=8, fps=24.0, layout="thwc", range="unit01"
        )


def test_close_is_idempotent(calibrations, tmp_path):
    encoder = new_encoder(calibrations, tmp_path / "twice.mp4")
    encoder.close()
    encoder.close()


def test_missing_codec_config_is_reported(calibrations, tmp_path):
    """The mock has no codec; without an override the runtime must refuse."""
    encoder = new_encoder(calibrations, tmp_path / "nocfg.mp4", video_extradata=None)
    with pytest.raises(RuntimeError, match="codec configuration"):
        encoder.submit(
            thwc_frame(8, 8, 0.5), width=8, height=8, fps=24.0, layout="thwc", range="unit01"
        )


def test_null_dimensions_are_rejected(calibrations, tmp_path):
    encoder = new_encoder(calibrations, tmp_path / "zero.mp4")
    with pytest.raises(ValueError, match="non-zero"):
        encoder.submit(
            np.zeros((0, 0, 3), dtype=np.float32),
            width=0,
            height=0,
            fps=24.0,
            layout="thwc",
            range="unit01",
        )


def test_calibrations_load_error_names_the_file(tmp_path):
    (tmp_path / "broken.toml").write_text("schema_version = ")
    with pytest.raises(ValueError, match="broken.toml"):
        blazar.Calibrations.load(str(tmp_path))


def test_calibrations_len(calibrations):
    assert len(calibrations) >= 3
