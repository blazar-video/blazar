"""Encode frames and collect the encoded units in memory instead of a file.

Useful when the units are forwarded somewhere else (a socket, a test harness, a
sink of your own) rather than muxed into an MP4.

    python examples/python/memory_sink.py --calibrations ../../calibrations/default

The stream is opened with ``output=None``, so nothing is written to disk. Each
encoded access unit comes back from ``poll()`` as an event carrying ``data``.

Units arrive in *decode* order, which is why the PTS below are not monotonic once
the encoder reorders frames with B-frames; ``dts`` is the monotonic one. Callers
that need presentation order sort on ``pts``, or ask for a low-latency calibration
that disables B-frames.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import blazar


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--calibrations", default="calibrations/default")
    parser.add_argument("--backend", default="x264")
    parser.add_argument("--codec", default="h264")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--frames", type=int, default=24)
    args = parser.parse_args(argv)

    calibrations = blazar.Calibrations.load(args.calibrations)
    encoder = blazar.Encoder(
        calibrations=calibrations,
        identity={"family": "wan", "version": "2.1", "quant": "bf16", "size": "1.3b"},
        target="balanced",
        backend=args.backend,
        codec=args.codec,
        # No path: the units stay in memory.
        output=None,
        request_id="memory-sink-example",
    )

    # A frame that moves, so the encoder has something to code.
    for n in range(args.frames):
        frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        frame[:, (n * 8) % args.width :] = 200
        encoder.submit(
            frame,
            width=args.width,
            height=args.height,
            fps=24.0,
            layout="thwc",
            range="full_u8",
        )

    # Encoders with lookahead or B-frames emit nothing at submit time, so the
    # last units appear during close().
    encoder.close()

    payloads = [
        event
        for event in encoder.poll()
        if event["kind"] == "encoded" and event["data"] is not None
    ]
    if not payloads:
        print("no encoded units came back", file=sys.stderr)
        return 1

    total = sum(len(event["data"]) for event in payloads)
    print(f"{args.backend}/{args.codec}: {len(payloads)} units, {total} bytes")
    print(f"first unit: pts={payloads[0]['pts']} keyframe={payloads[0]['keyframe']}")
    for event in payloads[:5]:
        print(f"  pts={event['pts']:<8} dts={event['dts']!s:<8} "
              f"keyframe={event['keyframe']!s:<5} {len(event['data']):>6} bytes")

    stats = encoder.stats
    if total != stats["bytes"]:
        print(
            f"payload total {total} does not match stats bytes {stats['bytes']}",
            file=sys.stderr,
        )
        return 1
    print(f"matches stats: {stats['bytes']} bytes over {stats['encoded']} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
