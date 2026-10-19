#!/usr/bin/env bash
# Validate the NVENC path on a CUDA host (mosk). Run from the blazar/ repo root.
#
# The `nvenc` feature is off by default so the workspace builds on machines
# without CUDA. These tests do real encodes and are the only place the hardware
# path is exercised. Decode validation needs ffmpeg (static build is fine).
set -euo pipefail

export CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"
export CUDA_PATH="${CUDA_PATH:-$CUDA_ROOT}"
export NVIDIA_VIDEO_CODEC_SDK_PATH="${NVIDIA_VIDEO_CODEC_SDK_PATH:-/usr/lib/wsl/lib}"
export LD_LIBRARY_PATH="${NVIDIA_VIDEO_CODEC_SDK_PATH}:${CUDA_ROOT}/lib64:${LD_LIBRARY_PATH:-}"

echo "== workspace tests (no hardware) =="
cargo test --workspace

echo "== NVENC encoder hardware tests =="
cargo test -p blazar-encoder --features nvenc --test nvenc_hardware -- --test-threads=1

echo "== NVENC end-to-end: host frame -> fMP4 =="
cargo test -p blazar-runtime --features nvenc --test nvenc_e2e -- --nocapture

SMOKE="/tmp/blazar-nvenc-smoke.mp4"
UNALIGNED="/tmp/blazar-nvenc-unaligned.mp4"

if ! command -v ffprobe >/dev/null 2>&1; then
  echo "ffprobe/ffmpeg not installed: container structure verified, NOT decoded" >&2
  echo "static build: https://johnvansickle.com/ffmpeg/" >&2
  exit 0
fi

echo "== smoke file: container =="
ffprobe -v error \
  -show_entries stream=codec_name,profile,width,height,pix_fmt,color_range,color_space,time_base,avg_frame_rate \
  -show_entries format=format_name,duration,size \
  -of default=noprint_wrappers=1 "$SMOKE"

echo "== smoke file: timing (drift check) =="
ffprobe -v error -select_streams v:0 -show_entries packet=pts,dts,duration -of csv=p=0 "$SMOKE"
echo "unique durations above; a constant per-sample duration in the stream timebase is expected"

echo "== smoke file: full decode (fails on corrupt AV1) =="
ffmpeg -v error -xerror -i "$SMOKE" -f null -

echo "== smoke file: decoded frame count =="
ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=nb_read_frames -of default=noprint_wrappers=1:nokey=1 "$SMOKE"

echo "== unaligned width (834x480): row alignment =="
# The driver allocates an 896-byte stride at this width. If the backend pads
# rows to the width instead of the stride, every decoded row mixes neighbours.
ffmpeg -v error -xerror -i "$UNALIGNED" -frames:v 1 -f rawvideo -pix_fmt gray -y /tmp/unaligned.gray
python3 - <<'PY'
W, H = 834, 480
data = open("/tmp/unaligned.gray", "rb").read()
assert len(data) == W * H, f"decoded {len(data)} bytes, expected {W * H}"
non_uniform = [row for row in range(H) if len(set(data[row * W:(row + 1) * W])) != 1]
print(f"non_uniform_rows={len(non_uniform)}/{H}")
if non_uniform:
    raise SystemExit(f"row skew detected, first rows: {non_uniform[:5]}")
PY

echo "== NVENC validation passed =="
