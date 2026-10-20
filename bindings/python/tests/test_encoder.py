"""Contract tests for the Python binding, on the mock encoder (`rav1e`).

`svtav1` is a real encoder once `enc-svtav1` is compiled in, so the mock
contract is tested through the backend that is always a mock.

These assert the binding's rules, the worker/queue behaviour, and the container's
structure. They cannot assert decodability: the mock backend has no real codec,
so it is handed a plausible av1C and writes synthetic payload bytes. Real decode
checks live on a CUDA host (repo README, "Hardware validation").
"""

import os
import threading
import time
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

# Enough frames to see ordering and a forced keyframe. The shipped
# svtav1_balanced default has keyint=120, so cadence alone yields no second
# keyframe here; the scene cut does.
FRAMES = 8


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
        backend="rav1e",
        output=str(output),
        video_extradata=FAKE_AV1C,
    )
    params.update(kwargs)
    return blazar.Encoder(**params)


def submit_default(encoder, width=8, height=8, fps=24.0, scene_cut=False):
    return encoder.submit(
        thwc_frame(width, height, 0.5),
        width=width,
        height=height,
        fps=fps,
        layout="thwc",
        range="unit01",
        scene_cut=scene_cut,
    )


def drain(encoder, expected: int, timeout: float = 5.0) -> list:
    events = []
    deadline = time.monotonic() + timeout
    while len([e for e in events if e["kind"] == "encoded"]) < expected:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError(f"timed out waiting for {expected} events, got {events}")
        events.extend(encoder.poll(min(remaining, 0.25)))
    return events


def test_encodes_frames_into_a_fragmented_mp4(calibrations, tmp_path):
    out = tmp_path / "stream.mp4"
    encoder = new_encoder(calibrations, out)
    for i in range(FRAMES):
        assert submit_default(encoder) == i
    encoder.close()

    data = out.read_bytes()
    assert len(data) > 0
    assert data[4:8] == b"ftyp", "ISO BMFF signature"
    assert b"moov" in data
    assert b"moof" in data, "fragmented output"
    assert b"mdat" in data


def test_events_arrive_in_submission_order_with_synthesized_pts(calibrations, tmp_path):
    encoder = new_encoder(calibrations, tmp_path / "events.mp4")
    for index in range(FRAMES):
        submit_default(encoder, scene_cut=(index == 5))

    events = [e for e in drain(encoder, FRAMES) if e["kind"] == "encoded"]
    assert [e["seq"] for e in events] == list(range(FRAMES))
    assert [e["pts"] for e in events] == [i * 3750 for i in range(FRAMES)]
    # Frame 0 is the initial keyframe; frame 5 was submitted as a scene cut.
    assert [e["keyframe"] for e in events] == [i in (0, 5) for i in range(FRAMES)]
    assert all(e["bytes"] > 0 for e in events)
    assert all(e["encode_us"] >= 0 for e in events)
    encoder.close()


def test_flush_is_a_barrier(calibrations, tmp_path):
    encoder = new_encoder(calibrations, tmp_path / "barrier.mp4")
    for _ in range(FRAMES):
        submit_default(encoder)
    encoder.flush(timeout=5.0)
    stats = encoder.stats
    assert stats["encoded"] == stats["submitted"] == FRAMES
    assert stats["bytes"] > 0
    encoder.close()


def test_stats_start_at_zero_and_track_submissions(calibrations, tmp_path):
    encoder = new_encoder(calibrations, tmp_path / "stats.mp4")
    assert encoder.stats["submitted"] == 0
    assert encoder.stats["encoded"] == 0
    submit_default(encoder)
    assert encoder.stats["submitted"] == 1
    encoder.close()


def test_submit_returns_before_the_encode_finishes(calibrations, tmp_path):
    """The caller (a decode loop) must not wait for the encoder."""
    encoder = new_encoder(calibrations, tmp_path / "async.mp4", queue_depth=8)
    started = time.perf_counter()
    for _ in range(4):
        submit_default(encoder, width=320, height=240)
    submit_seconds = time.perf_counter() - started
    encoder.flush(timeout=10.0)
    stats = encoder.stats
    encoder.close()

    # With queue_depth=8 the four enqueues must not wait for any encode. An
    # absolute bound is used rather than comparing against encode time: under
    # load the loop itself can be slow without the encoder blocking it.
    assert submit_seconds < 0.05, (
        f"submit loop took {submit_seconds:.3f}s - encode is blocking the caller"
    )
    assert stats["encoded"] == 4


def test_python_threads_run_while_the_queue_is_full(calibrations, tmp_path, monkeypatch):
    """A full queue blocks submit, but with the GIL released."""
    # The mock has no real encode cost; ask it to simulate one so the queue can
    # actually fill.
    monkeypatch.setenv("BLAZAR_MOCK_DELAY_MS", "10")
    encoder = new_encoder(calibrations, tmp_path / "gil.mp4", queue_depth=1)
    ticks = 0
    stop = False

    def spin():
        nonlocal ticks
        while not stop:
            ticks += 1

    spinner = threading.Thread(target=spin)
    spinner.start()
    started = time.perf_counter()
    for _ in range(8):
        submit_default(encoder, width=320, height=240)
    elapsed = time.perf_counter() - started
    stop = True
    spinner.join()
    encoder.close()

    assert elapsed > 0.05, "expected the bounded queue to block the caller"
    assert ticks > 1000, f"other Python thread made no progress ({ticks} ticks): GIL held"


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
        submit_default(encoder)
    assert b"moov" in out.read_bytes()


def test_non_float32_frame_is_rejected_with_the_fix(calibrations, tmp_path):
    encoder = new_encoder(calibrations, tmp_path / "bad-dtype.mp4")
    frame = np.zeros((8, 8, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="float32"):
        encoder.submit(
            frame, width=8, height=8, fps=24.0, layout="thwc", range="unit01"
        )
    encoder.close()


def test_uint8_frames_are_accepted_and_match_f32(calibrations, tmp_path):
    # The same solid colours must encode identically whether they arrive as
    # uint8 or as the equivalent f32 values: solid frames make the conversion
    # exact, so any difference would be a real bug rather than rounding.
    colours = [(0, 0, 0), (255, 255, 255), (255, 0, 0), (17, 200, 90)]
    width = height = 8

    def encode(cast):
        out = tmp_path / f"{cast.__name__}.mp4"
        encoder = new_encoder(calibrations, out)
        for colour in colours:
            plane = np.empty((height, width, 3), dtype=cast)
            plane[:, :] = colour
            encoder.submit(
                plane, width=width, height=height, fps=24.0,
                layout="thwc", range="unit01",
            )
        encoder.close()
        return out.read_bytes()

    def as_uint8(colour):
        return np.array(colour, dtype=np.uint8)

    def as_float32(colour):
        return np.array(colour, dtype=np.float32) / np.float32(255.0)

    assert encode(as_uint8) == encode(as_float32)


def test_uint8_frames_work_in_the_model_layout(calibrations, tmp_path):
    out = tmp_path / "uint8-bcthw.mp4"
    encoder = new_encoder(calibrations, out)
    for value in (0, 64, 128, 255):
        frame = np.full((1, 3, 1, 8, 8), value, dtype=np.uint8)
        encoder.submit(
            frame, width=8, height=8, fps=24.0, layout="bcthw", range="unit01"
        )
    encoder.close()
    data = out.read_bytes()
    assert data[4:8] == b"ftyp"


def test_size_change_mid_stream_is_rejected(calibrations, tmp_path):
    encoder = new_encoder(calibrations, tmp_path / "resize.mp4")
    submit_default(encoder)
    with pytest.raises(ValueError, match="size changed"):
        submit_default(encoder, width=16, height=16)
    encoder.close()


def test_fps_change_mid_stream_is_rejected(calibrations, tmp_path):
    encoder = new_encoder(calibrations, tmp_path / "fps.mp4")
    submit_default(encoder)
    with pytest.raises(ValueError, match="fps changed"):
        submit_default(encoder, fps=30.0)
    encoder.close()


def test_submit_after_close_is_an_error(calibrations, tmp_path):
    encoder = new_encoder(calibrations, tmp_path / "closed.mp4")
    encoder.close()
    with pytest.raises(RuntimeError, match="closed"):
        submit_default(encoder)


def test_close_is_idempotent(calibrations, tmp_path):
    encoder = new_encoder(calibrations, tmp_path / "twice.mp4")
    encoder.close()
    encoder.close()


def test_missing_codec_config_fails_on_close_and_reports_an_event(calibrations, tmp_path):
    """The mock has no codec; without an override the worker must refuse."""
    encoder = new_encoder(calibrations, tmp_path / "nocfg.mp4", video_extradata=None)
    submit_default(encoder)  # accepted by the queue; the worker rejects it
    with pytest.raises(RuntimeError, match="codec configuration"):
        encoder.close()
    kinds = [event["kind"] for event in encoder.poll()]
    assert "failed" in kinds
    with pytest.raises(RuntimeError, match="codec configuration"):
        submit_default(encoder)


def test_queue_depth_must_be_positive(calibrations, tmp_path):
    with pytest.raises(ValueError, match="queue_depth"):
        new_encoder(calibrations, tmp_path / "zero.mp4", queue_depth=0)


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
    encoder.close()


def test_calibrations_load_error_names_the_file(tmp_path):
    (tmp_path / "broken.toml").write_text("schema_version = ")
    with pytest.raises(ValueError, match="broken.toml"):
        blazar.Calibrations.load(str(tmp_path))


def test_calibrations_len(calibrations):
    assert len(calibrations) >= 3


def test_submit_batch_encodes_every_frame_in_one_submission(calibrations, tmp_path):
    """A chunk goes to the worker as one job and one event."""
    out = tmp_path / "batch.mp4"
    encoder = new_encoder(calibrations, out)
    frames = [bcthw_frame(160, 120, 0.2 + 0.05 * index) for index in range(4)]

    seq = encoder.submit_batch(
        frames,
        width=160,
        height=120,
        fps=24.0,
        layout="bcthw",
        range="signed_unit",
        scene_cut=True,
    )
    encoder.flush(timeout=10.0)
    stats = encoder.stats
    events = encoder.poll()
    encoder.close()

    assert isinstance(seq, int)
    assert stats["encoded"] == 4, f"expected 4 frames, got {stats['encoded']}"
    encoded_events = [e for e in events if e.get("kind") == "encoded"]
    assert len(encoded_events) == 1, f"expected one batch event, got {encoded_events}"
    assert encoded_events[0]["units"] == 4, (
        f"batch event reported {encoded_events[0]['units']} units"
    )


def test_submit_batch_rejects_mixed_geometry(calibrations, tmp_path):
    encoder = new_encoder(calibrations, tmp_path / "mixed.mp4")
    good = bcthw_frame(160, 120, 0.5)
    encoder.submit_batch(
        [good], width=160, height=120, fps=24.0, layout="bcthw", range="signed_unit"
    )
    with pytest.raises(ValueError):
        encoder.submit_batch(
            [good],
            width=320,
            height=240,
            fps=24.0,
            layout="bcthw",
            range="signed_unit",
        )
    encoder.close()
