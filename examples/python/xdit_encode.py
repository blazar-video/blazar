#!/usr/bin/env python3
"""Encode xDiT output with Blazar.

xDiT's public seam is host numpy: ``DiffusionOutput.videos`` is a list of
``(H, W, 3)`` float32 arrays in ``[0, 1]`` (THWC, RGB). The whole clip exists
before encoding starts, so this is an offline encode, not a low-latency stream:
the point is real AV1, exact 90 kHz timestamps and calibration-driven settings.

Two frame sources:

  --fake             (default) a moving gradient with the same dtype/shape/range
                     as the xDiT seam. Runs anywhere, including a Mac.
  --from-npy PATH    frames saved from a real xDiT run:
                     ``np.save(path, np.stack(output.videos))`` for a
                     ``(F, H, W, 3)`` float32 array in ``[0, 1]``.

Examples:
    python examples/python/xdit_encode.py --fake
    python examples/python/xdit_encode.py --from-npy /tmp/xdit.npy \\
        --backend nvenc_av1 --output /tmp/xdit.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import blazar

# Mock backend has no codec, so it needs a plausible AV1 config to open a stream.
# The payload is synthetic, so the result is not decodable; use nvenc_av1 for a
# real file.
FAKE_AV1C = bytes([0x81, 0x04, 0x0C, 0x00])


def repo_calibrations() -> Path:
    # examples/python/xdit_encode.py -> blazar/ -> blazar-video/calibrations
    return Path(__file__).resolve().parents[2].parent / "calibrations" / "default"


def fake_frames(count: int, width: int, height: int) -> np.ndarray:
    """(F, H, W, 3) float32 in [0, 1]: a gradient that moves per frame."""
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    frames = np.empty((count, height, width, 3), dtype=np.float32)
    for i in range(count):
        shift = i / max(count, 1)
        frames[i, ..., 0] = (xs / max(width - 1, 1) + shift) % 1.0
        frames[i, ..., 1] = (ys / max(height - 1, 1) + shift) % 1.0
        frames[i, ..., 2] = 0.5
    return frames


def load_frames(path: Path) -> np.ndarray:
    frames = np.load(path)
    if frames.dtype != np.float32:
        raise SystemExit(f"{path}: expected float32, got {frames.dtype}")
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise SystemExit(f"{path}: expected (F, H, W, 3), got {frames.shape}")
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fake", action="store_true", help="synthetic frames (default)")
    parser.add_argument("--from-npy", type=Path, help="frames from a real xDiT run")
    parser.add_argument("--frames", type=int, default=24, help="fake frames (default 24)")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--output", type=Path, default=Path("/tmp/xdit_blazar.mp4"))
    parser.add_argument("--backend", default="svtav1", choices=["svtav1", "nvenc_av1", "rav1e"])
    parser.add_argument("--target", default="balanced", choices=["low_latency", "balanced", "quality"])
    parser.add_argument("--calibrations", type=Path, default=repo_calibrations())
    args = parser.parse_args()
    if args.fake and args.from_npy:
        parser.error("choose either --fake or --from-npy")

    if args.from_npy:
        frames = load_frames(args.from_npy)
    else:
        frames = fake_frames(args.frames, args.width, args.height)

    count, height, width, _ = frames.shape
    if args.from_npy is None and (width, height) != (args.width, args.height):
        raise SystemExit("internal: fake frame geometry mismatch")

    # The mock encoder cannot produce a codec configuration; supply a placeholder
    # so the container structure is exercised. nvenc derives the real av1C.
    # Only the mock (rav1e) needs a canned av1C. SVT-AV1 and NVENC derive a real
    # one from the first keyframe, and overriding it would put a lie in the
    # container that a decoder may or may not ignore.
    extradata = FAKE_AV1C if args.backend == "rav1e" else None
    if extradata is not None:
        print("note: mock backend - output is NOT decodable, structure only")

    calibrations = blazar.Calibrations.load(str(args.calibrations))
    started = time.perf_counter()
    with blazar.Encoder(
        calibrations=calibrations,
        identity={"family": "wan", "version": "2.1", "quant": "bf16", "size": "1.3b"},
        target=args.target,
        backend=args.backend,
        output=str(args.output),
        video_extradata=extradata,
        request_id="xdit-example",
    ) as stream:
        for index in range(count):
            stream.submit(
                np.ascontiguousarray(frames[index]),
                width=width,
                height=height,
                fps=args.fps,
                layout="thwc",
                range="unit01",
                scene_cut=(index == 0),
            )
    elapsed = time.perf_counter() - started

    size = args.output.stat().st_size
    print(f"frames    : {count} ({width}x{height} @ {args.fps:g} fps)")
    print(f"duration  : {count / args.fps:.3f} s")
    print(f"encode    : {elapsed * 1000:.1f} ms total, {elapsed / count * 1000:.2f} ms/frame")
    print(f"output    : {args.output} ({size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
