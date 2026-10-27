"""Hardware test: Python -> NVENC -> fMP4 -> decode.

Skipped unless BLAZAR_NVENC=1 (and the binding was built with
`maturin develop --features nvenc`). This is the only place the Python path is
proven end to end, so it asserts on the decoded media, not the container.

Run on a CUDA host:

    BLAZAR_NVENC=1 pytest bindings/python/tests/test_hardware.py -v
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import blazar

pytestmark = pytest.mark.skipif(
    os.environ.get("BLAZAR_NVENC") != "1",
    reason="hardware test: set BLAZAR_NVENC=1 on a CUDA host",
)

CALIBRATIONS = Path(
    os.environ.get(
        "BLAZAR_CALIBRATIONS",
        Path(__file__).resolve().parents[4] / "calibrations" / "default",
    )
)


def frame(width: int, height: int, index: int) -> np.ndarray:
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    shift = (index % 8) / 8.0
    out = np.empty((height, width, 3), dtype=np.float32)
    out[..., 0] = (xs / max(width - 1, 1) + shift) % 1.0
    out[..., 1] = (ys / max(height - 1, 1) + shift) % 1.0
    out[..., 2] = 0.5
    return out


def ffprobe_json(path: Path, *args: str) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-of", "json", *args, str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_nvenc_output_decodes(tmp_path):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not installed")

    out = tmp_path / "nvenc.mp4"
    width, height, fps, count = 320, 240, 24.0, 24
    calibrations = blazar.Calibrations.load(str(CALIBRATIONS))

    with blazar.Encoder(
        calibrations=calibrations,
        identity={"family": "wan", "version": "2.1", "quant": "bf16", "size": "1.3b"},
        target="low_latency",
        backend="nvenc",
        codec="av1",
        output=str(out),
        video_extradata=None,  # the real encoder derives av1C from its keyframe
        request_id="nvenc-pytest",
    ) as stream:
        for index in range(count):
            stream.submit(
                frame(width, height, index),
                width=width,
                height=height,
                fps=fps,
                layout="thwc",
                range="unit01",
                scene_cut=(index == 0),
            )

    # Full decode: fails on corrupt bitstream or bad framing.
    subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-i", str(out), "-f", "null", "-"],
        check=True,
        capture_output=True,
    )

    counted = ffprobe_json(
        out, "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames",
    )
    stream = counted["streams"][0]
    assert int(stream["nb_read_frames"]) == count

    meta = ffprobe_json(
        out,
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate",
        "-show_entries",
        "format=duration",
    )
    video = meta["streams"][0]
    assert video["codec_name"] == "av1"
    assert (video["width"], video["height"]) == (width, height)
    assert video["avg_frame_rate"] == "24/1"
    assert abs(float(meta["format"]["duration"]) - count / fps) < 0.01
