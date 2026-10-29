# Changelog

All notable changes to this project. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to the versioning policy in [RELEASING.md](RELEASING.md):
while `0.x`, a minor bump may break the API, and a change to the bitstream at
identical settings is always a minor bump.

## [Unreleased]

## [0.1.2] - 2026-10-29

Packaging only: `blazar-video` itself is unchanged from 0.1.1. This tag exists so
`blazar-video-gpl` can be published, because publishing only ever happens from a
tag and 0.1.1 had already gone out.

### Fixed

- **The GPL wheel is built, not rewritten.** 0.1.1's archive rewriter produced a
  wheel PyPI refused with "ZIP archive not accepted: Trailing data". The wheel is
  now produced by maturin from a generated project (`scripts/build_gpl.py`) that
  contains the crate and the Python package together, so the metadata — Name,
  `License-Expression: GPL-3.0-or-later`, the GPL classifier, `NOTICE.md` as a
  licence file — is written by the build, not patched onto the result.

## [0.1.1] - 2026-10-29

Adds the GPL distribution, and fixes what the first release shipped wrong.

### Added

- **`blazar-video-gpl`**: the same library built with x264 and x265, published
  under `GPL-3.0-or-later`. Apache-2.0 code can be combined into a GPLv3 work, so
  this is legal where shipping those encoders in the Apache wheel would not be.
  The wheel bundles both libraries and every wheel carries a `NOTICE.md` naming
  them, their licences and where their source lives.

### Fixed

- **`blazar.__version__` reported `0.0.0`** while the wheel metadata said `0.1.0`.
  The module hardcoded it; it now re-exports the extension's, which is built from
  the crate, so there is one source instead of a fifth that drifts.
- **`__pycache__` was packaged** from the build machine's working tree, putting
  another interpreter's bytecode inside the wheel.
- **A wheel with NVENC on a machine with no driver panicked** inside the encoder
  worker instead of failing the session: `cudarc`'s loader panics rather than
  returning an error. Availability is a runtime probe now, so `preprocess="auto"`
  selects the host path and a session opened for NVENC fails with a message that
  names the driver.
- **`preprocess="gpu"` reported "needs the gpu-preprocess feature"** on a machine
  where the feature was compiled in and the driver was missing, which points the
  reader at the wrong fix.

### Changed

- **`preprocess` takes `"auto"`, `"host"` or `"device"`.** `"gpu"` and `"cpu"` are
  rejected, not aliased. VideoToolbox is hardware encoding that converts on the
  host, so the old names were wrong for it in both directions; `host`/`device`
  match what `Encoder.residency` already reported.

### Known gaps

- Wheels are not abi3: `cp311` on macOS and Windows, `cp312` on the Linux and GPL
  wheels, depending on the interpreter each job builds against. A wheel will not
  install on another Python minor version. `abi3-py39` would collapse them to one
  per platform.
- NVENC on Windows is build-verified only. There is no GPU runner, so the wheel
  proves it imports without a driver and refuses with a driver message.
- No AAC. Opus is the supported audio codec.

## [0.1.0] - 2026-10-28

First release. The encoder pipeline, the calibration system and the Python
binding are in place; the API is not yet stable.

### Added

- **Opus audio**: `Encoder(audio="opus")` plus `submit_audio(samples, sample_rate=48000)`.
  The dOps pre-skip comes from the encoder's lookahead, so a decoder discards
  exactly the samples the encoder delayed, and audio/video stay aligned.

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
- **`preprocess="auto" | "host" | "device"`**: RGB→NV12 placement is the caller's
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

- **AAC is not implemented.** The usable encoders are non-free (`libfdk-aac`) or
  drag in libavcodec, so 0.1 ships Opus only.
- **Pre-skip lives in dOps, not the timeline.** oxideav rejects negative packet
  timestamps, so audio starts at zero; the residual offset is the encoder
  lookahead (312 samples, ~6.5 ms).

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
