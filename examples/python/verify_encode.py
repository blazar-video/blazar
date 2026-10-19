#!/usr/bin/env python3
"""Prove that Blazar encoded a file, from evidence outside Blazar's own claims.

The check is a cross-reference, not a log claim:

* Blazar's Rust worker reports `frames` and `bytes` (sum of the encoded units it
  wrote) in its `encoder session finished` log line.
* ffprobe reports the packets that are actually in the container and their sizes.

If the counts match and the per-packet sizes sum to the worker's byte count, the
payload in the file is exactly what Blazar's encoder produced. A log line alone
would not show that; a file alone would not show who produced it.

    python examples/python/verify_encode.py out.mp4 --log encode.log

Exit code 0 on success, 1 on any mismatch. Works without ffmpeg by reporting
that the container could not be inspected (exit 2), so "verified" is never
claimed without evidence.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

FRAMES = re.compile(r"frames=(\d+)")
BYTES = re.compile(r"bytes=(\d+)")
MAX_US = re.compile(r"max_encode_us=(\d+)")
BACKEND = re.compile(r"backend=(\S+)")
CODEC = re.compile(r"codec=(\S+)")


def probe(path: Path) -> dict:
    entries = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    packets = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "packet=size", "-of", "csv=p=0", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    sizes = [int(line) for line in packets.stdout.split() if line.strip().isdigit()]
    stream = json.loads(entries.stdout)["streams"][0]
    return {"stream": stream, "packet_sizes": sizes}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("--log", type=Path, required=True, help="stderr captured from the run")
    parser.add_argument("--expect-backend", default="nvenc_av1")
    args = parser.parse_args()

    if not args.video.is_file():
        raise SystemExit(f"{args.video}: not found")
    if shutil.which("ffprobe") is None:
        print("ffprobe not installed: cannot inspect the container, refusing to claim verified", file=sys.stderr)
        return 2

    ansi = re.compile(r"\x1b\[[0-9;]*m")
    log = [ansi.sub("", line) for line in args.log.read_text(errors="replace").splitlines()]
    started = [line for line in log if "encoder worker started" in line]
    finished = [line for line in log if "encoder session finished" in line]
    if not started:
        raise SystemExit("no 'encoder worker started' line in the log: Blazar's worker never ran")
    if not finished:
        raise SystemExit("no 'encoder session finished' line in the log: the session did not complete")

    backend = BACKEND.search(started[-1])
    codec = CODEC.search(started[-1])
    frames = FRAMES.search(finished[-1])
    byte_count = BYTES.search(finished[-1])
    max_us = MAX_US.search(finished[-1])
    if not all((frames, byte_count, max_us)):
        raise SystemExit(f"could not parse the summary line: {finished[-1]}")

    probed = probe(args.video)
    packet_sizes = probed["packet_sizes"]
    stream = probed["stream"]
    reported_frames = int(frames.group(1))
    reported_bytes = int(byte_count.group(1))
    file_bytes = sum(packet_sizes)

    print(f"blazar log : backend={backend.group(1) if backend else '?'} "
          f"codec={codec.group(1) if codec else '?'} frames={reported_frames} bytes={reported_bytes} "
          f"max_encode_us={max_us.group(1)}")
    print(f"container  : codec={stream['codec_name']} {stream['width']}x{stream['height']} "
          f"packets={len(packet_sizes)} payload_bytes={file_bytes}")

    failures = []
    if backend is None or backend.group(1) != args.expect_backend:
        failures.append(f"log backend is {backend.group(1) if backend else '?'}, expected {args.expect_backend}")
    if stream["codec_name"] != "av1":
        failures.append(f"container codec is {stream['codec_name']}, expected av1")
    if len(packet_sizes) != reported_frames:
        failures.append(f"container has {len(packet_sizes)} packets, worker reported {reported_frames} frames")
    if file_bytes != reported_bytes:
        failures.append(f"container payload is {file_bytes} bytes, worker reported {reported_bytes}")
    if int(max_us.group(1)) == 0:
        failures.append("worker reported a zero encode time, which cannot be a real encode")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("PASS: the container's payload is exactly the encoder's output")
    return 0


if __name__ == "__main__":
    sys.exit(main())
