#!/usr/bin/env bash
# Validate the NVENC path on a CUDA host (mosk). Run from the blazar/ repo root.
#
# The `nvenc` feature is off by default so the workspace builds on machines
# without CUDA. These tests do real encodes and are the only place the hardware
# path is exercised.
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
if ! command -v ffprobe >/dev/null 2>&1; then
  echo "ffprobe/ffmpeg not installed: container structure verified, decode NOT verified" >&2
  echo "install with: sudo apt-get install -y ffmpeg" >&2
  exit 0
fi

echo "== container probe =="
ffprobe -v error -show_entries stream=codec_name,width,height,pix_fmt,avg_frame_rate \
  -show_entries format=format_name,duration -of default=noprint_wrappers=1 "$SMOKE"

echo "== full decode (fails on corrupt AV1) =="
# -xerror turns any decode warning into a non-zero exit.
ffmpeg -v error -xerror -i "$SMOKE" -f null -

echo "== counted frames =="
ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=nb_read_frames -of default=noprint_wrappers=1:nokey=1 "$SMOKE"
