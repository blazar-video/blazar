"""Submitting a host frame must not cost interpreter time.

The zero-copy path releases the GIL by construction: `submit` is handed a pointer
and a Rust worker does everything else. The host path is the interesting one,
because it has to copy the caller's numpy buffer, and copying that under the GIL
serialises the caller against every other Python thread in the process.

Measured before the fix at 1088p: 35.6 fps alone, 11.3 fps with one other Python
thread running. The assertion here is deliberately loose — this is a timing test
and the ratio it guards is 3x, not 5%.
"""

from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest

import blazar

WIDTH, HEIGHT, FRAMES = 640, 480, 24
CALIBRATIONS = os.environ.get(
    "BLAZAR_CALIBRATIONS",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "calibrations", "default"),
)
SECONDS = 2.0


def python_loop(out: list[int], stop: threading.Event) -> None:
    """Pure-Python work: no C call that could release the interpreter."""
    n = 0
    while not stop.is_set():
        for _ in range(10_000):
            n += 1
    out.append(n)


def measure(encode: bool) -> float:
    """Iterations of the loop per second, with or without encoding alongside."""
    frames = [
        np.full((HEIGHT, WIDTH, 3), i % 256, dtype=np.uint8) for i in range(FRAMES)
    ]
    encoder = None
    if encode:
        encoder = blazar.Encoder(
            calibrations=blazar.Calibrations.load(CALIBRATIONS),
            identity={"family": "wan", "version": "2.1", "quant": "u8", "size": "480p"},
            target="balanced",
            backend="nvenc",
            codec="h264",
            output=None,
            request_id="gil-test",
            preprocess="host",
        )
        # Warm the session so opening it is not part of the measurement.
        encoder.submit(frames[0], width=WIDTH, height=HEIGHT, fps=24.0, layout="thwc", range="full_u8")
        encoder.poll()

    out: list[int] = []
    stop = threading.Event()
    loop = threading.Thread(target=python_loop, args=(out, stop))
    loop.start()
    deadline = time.perf_counter() + SECONDS
    try:
        while time.perf_counter() < deadline:
            if encoder is not None:
                for frame in frames:
                    encoder.submit(
                        frame, width=WIDTH, height=HEIGHT, fps=24.0, layout="thwc", range="full_u8"
                    )
            else:
                time.sleep(0.001)
    finally:
        stop.set()
        loop.join()
        if encoder is not None:
            encoder.close()
    return out[0] / SECONDS


def test_host_frames_do_not_starve_other_python_threads() -> None:
    """The copy must happen with the interpreter released.

    A ratio rather than an absolute rate: the machine this runs on is not
    specified anywhere, and the claim is about interference, not speed.
    """
    baseline = measure(encode=False)
    with_encoding = measure(encode=True)
    ratio = with_encoding / baseline
    assert ratio > 0.6, (
        f"another Python thread kept only {ratio:.0%} of its throughput while host "
        f"frames were being submitted ({baseline:.0f} -> {with_encoding:.0f} iters/s); "
        "the frame copy is holding the GIL"
    )
