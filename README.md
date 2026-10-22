# blazar — video encoding for generative models

Encodes frames from generative video runtimes (SGLang Diffusion, vLLM-Omni,
diffusers) as fast as the model produces them. Frames stay on the GPU when the
caller has device tensors; a worker thread owns the encoder so the sampling loop
never blocks on codec setup.

Verified end to end against SGLang Diffusion (Wan 2.1 1.3B) and the diffusers
Wan pipelines. Calibration data comes from the companion `calibrator` repo.

## Workspace layout

| Crate | Role |
|-------|------|
| `blazar-core` | Shared contract types: `FrameRef`/`FrameDesc`, `EncodedUnit`, `StreamParams`, `ModelIdentity`, `Target`, `Pts`/`Timebase`. No I/O. |
| `blazar-calibration` | TOML calibration schema, validating loader, specificity index for model/target lookup. |
| `blazar-preprocess` | Preprocess trait plus CPU implementation (RGB→YUV, range normalise, 4:2:0 subsample, 8-bit quantise) and an NVRTC-compiled CUDA kernel for the same conversion. |
| `blazar-encoder` | `VideoEncoder` trait, calibration→config translation, and the codec backends: NVENC (AV1/H.264/HEVC), x264, x265, SVT-AV1, VP9, plus a mock encoder for contract tests. |
| `blazar-sink` | `Sink` trait, `OrderGuard`, `DebugFileSink`, and `Fmp4FileSink` on a pinned `oxideav-mp4` fork (AV1/H.264/HEVC + Opus/AAC). |
| `blazar-runtime` | `Encoder::new` → `open_stream` → `submit_video`/`flush`/`close`, the encoder worker, session limiting, and GPU preprocessing glue. |
| `blazar-*-sys` | FFI bindings: `svtav1`, `vpx`, `x265` (x265 resolves the encoder symbol per library build, so Debian/macOS builds both work). |

## What works

**Ingest.** Host frames as `thwc`/`bcthw`, `u8` or `f32`; device tensors as
`bcthw`/`bchw` with a `torch` buffer, including non-contiguous views. `u8`
ingest skips the float conversion and is byte-identical to `f32` output.

**Codecs.** NVENC AV1/H.264/HEVC (zero-copy from `cuMemAllocPitch` allocations),
x264, x265, SVT-AV1, VP9. Every backend writes BT.709 full-range VUI; CPU codec
packet timestamps come from the encoder's output picture, not the submitted
frame, so B-frame reordering keeps DTS valid.

**Sinks.** Fragmented MP4 via the `oxideav-mp4` fork; `DebugFileSink` for
inspection. The muxer is opened after the first keyframe, when `avcC`/`hvcC`/`av1C`
is known.

**Batching.** `encode_batch`/`submit_batch` stage several frames through one
kernel launch and split them across the encoder pool.

**Multi-stream.** `BLAZAR_MAX_NVENC_SESSIONS` (default 8) queues callers instead
of letting the driver refuse a session. The GIL is released while waiting.

**Interop.** `bindings/python/python/blazar/sglang_interop.py` plus a
`sitecustomize` shim let SGLang use blazar for its output encode path. The shim
hooks the decode stage SGLang already has.

**Python.** `bindings/python` (PyO3 + maturin) exposes `blazar.Encoder` and
`blazar.Calibrations`:

```python
import blazar

cals = blazar.Calibrations.load("calibrations/default")
with blazar.Encoder(calibrations=cals, identity={"family": "wan", "version": "2.1",
                                                 "quant": "bf16", "size": "1.3b"},
                    target="balanced", backend="nvenc_h264", codec="h264",
                    output="out.mp4") as stream:
    stream.submit(frame, width=W, height=H, fps=24.0, layout="thwc", range="unit01")
    # or, with device tensors:
    stream.submit_device_batch(tensor, fps=24.0, range="unit01")
```

## Measured performance

1080p, 240 frames, RTX 4080 SUPER, unless noted. Wall time per frame includes
preprocess and encode; bitrate and PSNR are for the same source.

| Configuration | ms/frame | kbps | PSNR (dB) |
|---|---|---|---|
| ffmpeg libx264, CRF 30, medium (fair comparison: inputs materialised first) | 4.84 | 295 | 43.29 |
| blazar x264, `u8` host frames | 4.89 | 302 | 42.51 |
| blazar x264, `f32` host frames | 6.78 | 302 | 42.51 |
| blazar x264, bf16 device tensor | 4.51 | 304 | 42.61 |
| blazar NVENC H.264, device tensor | **2.18** | 197 | 39.76 |

CPU encode against `ffmpeg` at equal settings is a wash; the win is the device
path, and NVENC keeps it.

**Device tensors beat host frames** (1080p, 120 frames):

| Backend | device | host | speedup |
|---|---|---|---|
| NVENC H.264 | 4.09 ms/frame | 20.27 ms/frame | 4.96× |
| x264 | 5.51 ms/frame | 15.26 ms/frame | 2.77× |

**Batch kernel** (NVIDIA Nsight Compute A/B): 2.28–2.72× over per-frame launches.

**Concurrent NVENC streams** (16 frames per stream, 832×480): 1 stream 13.9
ms/frame, 4 streams 4.59, 8 streams **3.51**. An RTX 4080 SUPER accepts 9
sessions; the 10th fails, which is why the session limiter exists.

**Ingest width matters more than arithmetic.** Converting `f32` host frames costs
6.78 ms/frame; `u8` frames cost 4.90 for the same output. An AVX2 implementation
of the conversion was written, measured slower (16.1 vs 11.2 ms/frame, because the routine reads
every 25 MB plane twice), and removed.

## Limits worth knowing

- NVENC has no lossless mode. Lossless calibration requires the CPU codecs
  (x264 CRF 0, x265 `lossless=1`, SVT-AV1 `--lossless`, VP9 `lossless=1`).
- Hardware encoders reject frames below one CTU (x265, 64×64 by default).
- Consumer GPUs cap concurrent NVENC sessions (9 measured on an RTX 4080 SUPER).
  Datacenter parts do not.
- Software H.264/HEVC (x264/x265) are GPL. The features are off by default.
- AV1 B-frames are not wired up. H.264/HEVC support them.
- NVENC flush cannot drain the driver's lookahead; segment boundaries force a
  keyframe instead.
- The SDK bindings are built against the 12.1 headers. Newer drivers accept
  them; an older driver gets a named `EncodeError`, not a panic.

## Build and test

```sh
cargo test --workspace                                  # no codec libraries needed
cargo test -p blazar-encoder --features nvenc           # needs CUDA + driver
cargo test -p blazar-encoder --features "nvenc enc-x264 enc-x265 enc-svtav1 enc-vp9"
```

Codec features are per-library because they pull system dependencies:

| Feature | Needs |
|---|---|
| `nvenc` | CUDA toolkit, `libnvidia-encode` |
| `enc-x264` | `libx264` (GPL) |
| `enc-x265` | `libx265` (GPL) |
| `enc-svtav1` | `libSvtAv1Enc` |
| `enc-vp9` | `libvpx` |
| `gpu-preprocess` | `nvenc`; NVRTC-compiles the NV12 kernel at runtime |

Hardware tests are ignored by default and run on a machine with a GPU:

```sh
cargo test -p blazar-encoder --features nvenc --test nvenc_hardware -- --test-threads=1
```

## What's next

**Calibrated ladders shipping with the runtime.** The `calibrator` repo generates
lossless masters, sweeps each codec, scores with VMAF (CUDA, 26× faster than the
CPU CLI) and emits calibration files. Plan is a global ladder per target
(`low latency`, `balanced`, `high quality`) that any model can use, plus
per-model ladders that override it when measured.

**An 8-bit generation corpus.** Masters generated with distilled, fp8-quantised
video models (SANA-Video 5B, Wan 2.2 5B, Fast H3) so the calibration reflects
what is actually served. Every clip is FFV1 in RGB, and the model revision,
quantisation and sampler settings are part of the clip key.

**Upstream integration instead of shims.** The SGLang work is a pluggable video
encoder plus a frame callback, so a deployment selects an encoder by name rather
than monkeypatching. The diffusers equivalent is an `encode_video` backend
protocol.

**More backends.** AV1 B-frames on NVENC once the driver path is verified, plus a
second hardware vendor when there is a machine to test on.
