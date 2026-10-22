"""Encoded units can be collected in memory instead of written to a file."""

import numpy as np
import pytest

import blazar

CALIBRATIONS = "../../../calibrations/default"


def _calibrations():
    try:
        return blazar.Calibrations.load(CALIBRATIONS)
    except Exception:  # pragma: no cover - running outside the repo
        pytest.skip("default calibrations not available")


def _encoder(**kwargs):
    cals = _calibrations()
    try:
        return blazar.Encoder(
            calibrations=cals,
            identity={"family": "wan", "version": "2.1", "quant": "bf16", "size": "1.3b"},
            target="balanced",
            backend="x264",
            codec="h264",
            request_id="memory",
            **kwargs,
        )
    except RuntimeError as error:  # built without enc-x264
        pytest.skip(f"x264 backend unavailable: {error}")


def test_units_are_returned_instead_of_written(tmp_path):
    encoder = _encoder(output=None)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    for _ in range(3):
        encoder.submit(frame, width=64, height=64, fps=24.0, layout="thwc",
                       range="full_u8")
    encoder.close()

    events = encoder.poll(timeout=2.0)
    payloads = [e for e in events if e["kind"] == "encoded" and e["data"] is not None]

    assert len(payloads) == 3
    assert sum(len(e["data"]) for e in payloads) == encoder.stats["bytes"]
    assert all(isinstance(e["data"], bytes) for e in payloads)
    # decode order, not presentation order: B-frames reorder the middle units
    assert [e["pts"] for e in payloads][0] == 0
    assert payloads[0]["keyframe"] is True


def test_a_file_is_not_created(tmp_path):
    encoder = _encoder(output=None)
    encoder.submit(np.zeros((64, 64, 3), dtype=np.uint8), width=64, height=64,
                   fps=24.0, layout="thwc", range="full_u8")
    encoder.close()
    assert list(tmp_path.iterdir()) == []
