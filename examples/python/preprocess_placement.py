"""Show where RGB→NV12 conversion runs, and that placement changes cost not output.

`preprocess="auto"` follows the backend: NVENC converts on the device because
the kernel writes into the surface NVENC encodes from, while a CPU codec
converts on the host because a device conversion would only add a download.

    python examples/python/preprocess_placement.py --calibrations ../../calibrations/default

Each placement encodes the same 24 frames. The unit counts must match; the byte
totals may differ slightly because the two conversions round differently.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import blazar

WIDTH, HEIGHT, FRAMES = 640, 480, 24


def frames() -> list[np.ndarray]:
    """Moving content, so the encoder has something to code."""
    out = []
    for index in range(FRAMES):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[:, :, 0] = (np.arange(WIDTH) + index * 8)[None, :] % 256
        frame[:, :, 1] = (np.arange(HEIGHT)[:, None] + index * 4) % 256
        frame[:, :, 2] = 64 + index
        out.append(frame)
    return out


def encode(calibrations, placement: str, backend: str) -> tuple[str, int, int]:
    encoder = blazar.Encoder(
        calibrations=calibrations,
        identity={"family": "wan", "version": "2.1", "quant": "bf16", "size": "1.3b"},
        target="balanced",
        backend=backend,
        codec="h264",
        output=None,
        request_id=f"placement-{placement}",
        preprocess=placement,
    )
    for frame in frames():
        encoder.submit(
            frame, width=WIDTH, height=HEIGHT, fps=24.0, layout="thwc", range="full_u8"
        )
    encoder.close()
    units = [e for e in encoder.poll() if e["kind"] == "encoded" and e["data"] is not None]
    return encoder.preprocess, len(units), sum(len(u["data"]) for u in units)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--calibrations", default="calibrations/default")
    parser.add_argument("--backend", default="nvenc")
    args = parser.parse_args(argv)

    calibrations = blazar.Calibrations.load(args.calibrations)
    results = {}
    for placement in ("auto", "host", "device"):
        try:
            site, units, total = encode(calibrations, placement, args.backend)
        except ValueError as error:
            print(f"{placement:<5} unavailable: {error}")
            continue
        results[placement] = units
        print(f"{placement:<5} -> {site:<3} {units:>3} units  {total:>8} bytes")

    if len(set(results.values())) > 1:
        print(f"placements disagree on unit count: {results}", file=sys.stderr)
        return 1
    print(f"all placements encoded {FRAMES} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
