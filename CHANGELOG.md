# Changelog

All notable changes to this project. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to the versioning policy in [RELEASING.md](RELEASING.md):
while `0.x`, a minor bump may break the API, and a change to the bitstream at
identical settings is always a minor bump.

## [Unreleased]

## [0.1.0] - TBD

First release. The encoder pipeline, the calibration system and the Python
binding are in place; the API is not yet stable.

### Added

- **Encoder runtime** with a dedicated worker thread per stream, so `submit`
  returns without waiting for the codec, and ordered emission guarded by DTS.
- **Backends**: NVENC (H.264, HEVC, AV1) with zero-copy device ingest, x264,
  x265, SVT-AV1, VP9, and VideoToolbox (H.264, HEVC) on macOS via raw framework
  FFI with no extra crates. `mock` for tests.
- **Resolution ingest**: host arrays or a CUDA tensor shaped `(1, 3, T, H, W)`,
  in `u8`, `f32` or `bf16`, in either layout.
- **Fused SIMD RGB→YUV420** for all six layout/dtype combinations on NEON and
  AVX2, 2.1–3.7× the scalar reference at 1080p.
- **Calibrations**: a versioned, measurable configuration per
  (model × scene × backend × target), with 27 files across nine codecs and a
  bitrate-budget target selector (0.35 / 0.60 / 1.0).
- **Non-file output**: `output=None` keeps encoded units in memory and returns
  them from `poll()`, bounded so a full buffer raises rather than dropping a
  frame.
- **`preprocess="auto" | "cpu" | "gpu"`**: RGB→NV12 placement is the caller's
  choice, reported back as `encoder.preprocess`.
- **`backend="auto"` with `backend_pool="any" | "hardware" | "software"`**:
  hardware preferred, software the fallback, and `hardware` never silently
  becomes software. `encoder.resolved_backend` reports what answered.
- **One `submit`**: a numpy array takes the host path, a CUDA tensor the device
  path. Residency and layout are fixed at open, and a mixed sequence is refused.
- **SGLang interop**: a registry plugin seam, a frame-callback hook, and a
  `sitecustomize` patch that encodes decoded Wan frames on the device inside
  SGLang's spawned scheduler.

### Changed

- `codec` and `backend` are separate dimensions: `backend` names the
  implementation (`nvenc`), `codec` the wire codec (`h264`).

### Removed

- `rav1e`, which was never an encoder here. Its stub had answered to a real codec
  name.

### Fixed

- Dropping an NVENC session with frames still buffered in the B-frame reorder
  queue segfaulted inside the driver; the teardown no longer emits EOS.
- VideoToolbox units carried `duration: 0`, and `420v` (video range) was used for
  full-range content. Both produced output that looked plausible and decoded
  wrong.
- `reporting` and per-frame summaries omitted `dts`, which the docs claimed.
- Host `u8` frames were rejected by NVENC's preprocess stage; the convert kernel
  gained an 8-bit path (17.77 → 6.56 ms/frame at 1088p).

### Known gaps

- **Host `bf16` ingest is rejected**: the binding infers dtype from the buffer
  format and numpy has no bf16 dtype. Needs an explicit `dtype=` argument.
- **`bf16` device ingest is slower than `f32`** (4.48 vs 3.20 ms/frame, 1080p) for
  reasons not yet established. Do not size a deployment on it.
- **Latent export is not implemented**: a corpus regenerates clips to change the
  decoder instead of re-decoding saved latents.
- **Hardware suites are manual**: the self-hosted NVENC and VideoToolbox runners
  do not exist yet, so a release in this state is validated on the CPU path only.

### Performance

1080p, RTX 4080 SUPER, NVENC H.264 ConstQP 30, frames starting on the GPU:

| path | ms/frame |
|---|---|
| Blazar device `bf16` (zero copy) | 3.88 |
| Blazar host `u8` | 6.35 |
| PyAV | 9.37 |
| ffmpeg CLI | 19.81 |

Concurrency (720p, 200 frames/thread): Blazar scales 4.0× across 4 Python
threads; PyAV 1.9× and flat at two, because its per-frame work runs under the GIL.
Method and caveats: `.llm/state/.../benchmark-nvenc-2026-10-25.md`.
