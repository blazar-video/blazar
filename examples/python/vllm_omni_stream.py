#!/usr/bin/env python3
"""Stream vLLM-Omni decode chunks into Blazar.

vLLM-Omni's streaming decoder emits ``(1, 3, T, H, W)`` tensors in ``[-1, 1]``
per autoregressive block, on the device. Phase 1 of Blazar takes host float32,
so the boundary adaptation is ``chunk.cpu().float().numpy()`` -- one copy per
chunk, before the per-frame submit. Device zero-copy (DLPack) is Phase 2.

This example treats each decoded chunk as it arrives and submits its frames
immediately, which is the point: time-to-first-encoded-frame tracks the model,
not the clip length. It prints that number.

Frame sources:

  --fake             (default) chunks emitted on a timer, same dtype/shape/range
                     as the real seam, so the timing story is visible anywhere.
  --from-npy PATH    a decoded ``(F, H, W, 3)`` float32 array in ``[-1, 1]``;
                     chunks are sliced from it.

Wiring the real hook (documented, Phase 2 work): intercept the tensor at the
producer side, before ``MultimodalPayload._to_cpu`` -- i.e. in
``OmniRequestState.add_multimodal_tensor`` (output_processor.py) or the
``post_decode`` callback of ``experimental/ar_diffusion/streaming_decode.py``.
Do not monkeypatch ``from_raw``; the chunk boundary is the natural submit
boundary, and it is available before the unconditional host copy.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import blazar

FAKE_AV1C = bytes([0x81, 0x04, 0x0C, 0x00])


def repo_calibrations() -> Path:
    return Path(__file__).resolve().parents[2].parent / "calibrations" / "default"


def fake_chunk(index: int, frames: int, width: int, height: int) -> np.ndarray:
    """(1, 3, T, H, W) float32 in [-1, 1], the streaming decoder's layout."""
    chunk = np.empty((1, 3, frames, height, width), dtype=np.float32)
    for t in range(frames):
        level = -1.0 + 2.0 * ((index * frames + t) % 24) / 23.0
        chunk[0, :, t, :, :] = level
    return chunk


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fake", action="store_true", help="synthetic chunks (default)")
    parser.add_argument("--from-npy", type=Path, help="decoded (F, H, W, 3) in [-1, 1]")
    parser.add_argument("--chunks", type=int, default=8, help="fake chunks (default 8)")
    parser.add_argument("--frames-per-chunk", type=int, default=3)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--chunk-delay-ms", type=float, default=40.0,
                        help="fake decode latency between chunks (default 40 ms)")
    parser.add_argument("--output", type=Path, default=Path("/tmp/vllm_omni_blazar.mp4"))
    parser.add_argument("--backend", default="svtav1", choices=["svtav1", "nvenc_av1", "rav1e"])
    parser.add_argument("--target", default="low_latency", choices=["low_latency", "balanced", "quality"])
    parser.add_argument("--calibrations", type=Path, default=repo_calibrations())
    args = parser.parse_args()
    if args.fake and args.from_npy:
        parser.error("choose either --fake or --from-npy")

    frames: np.ndarray | None = None
    if args.from_npy:
        frames = np.load(args.from_npy)
        if frames.dtype != np.float32 or frames.ndim != 4 or frames.shape[-1] != 3:
            raise SystemExit(f"{args.from_npy}: expected (F, H, W, 3) float32")
        height, width = frames.shape[1], frames.shape[2]
    else:
        height, width = args.height, args.width

    extradata = None if args.backend == "nvenc_av1" else FAKE_AV1C
    if extradata is not None:
        print("note: mock backend - output is NOT decodable, structure only")

    calibrations = blazar.Calibrations.load(str(args.calibrations))
    submitted = 0
    first_frame_at: float | None = None
    chunk_times: list[float] = []
    started = time.perf_counter()

    with blazar.Encoder(
        calibrations=calibrations,
        identity={"family": "wan", "version": "2.1", "quant": "bf16", "size": "1.3b"},
        target=args.target,
        backend=args.backend,
        output=str(args.output),
        video_extradata=extradata,
        request_id="vllm-omni-example",
    ) as stream:
        for chunk_index in range(args.chunks):
            chunk_started = time.perf_counter()

            if frames is not None:
                lo = chunk_index * args.frames_per_chunk
                hi = min(lo + args.frames_per_chunk, len(frames))
                if lo >= hi:
                    break
                chunk = frames[lo:hi]
            else:
                time.sleep(args.chunk_delay_ms / 1000.0)
                chunk = fake_chunk(
                    chunk_index, args.frames_per_chunk, width, height
                )[0]  # drop the batch dim: (3, T, H, W)

            for t in range(chunk.shape[1] if frames is None else len(chunk)):
                # Host adaptation at the documented boundary.
                if frames is None:
                    frame = np.ascontiguousarray(chunk[:, t, :, :].transpose(1, 2, 0))
                else:
                    frame = np.ascontiguousarray(chunk[t])
                stream.submit(
                    frame,
                    width=width,
                    height=height,
                    fps=args.fps,
                    layout="thwc",
                    range="signed_unit",
                    scene_cut=(submitted == 0),
                )
                submitted += 1
                if first_frame_at is None:
                    first_frame_at = time.perf_counter() - started

            chunk_times.append((time.perf_counter() - chunk_started) * 1000.0)

    elapsed = time.perf_counter() - started
    print(f"chunks    : {len(chunk_times)} x {args.frames_per_chunk} frames")
    print(f"frames    : {submitted} ({width}x{height} @ {args.fps:g} fps)")
    if first_frame_at is not None:
        print(f"first frame encoded {first_frame_at * 1000:.1f} ms after start")
    if chunk_times:
        print(f"per chunk : {min(chunk_times):.2f}..{max(chunk_times):.2f} ms")
    print(f"total     : {elapsed * 1000:.1f} ms")
    print(f"output    : {args.output} ({args.output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
