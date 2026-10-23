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
| `blazar-preprocess` | Preprocess trait, the CPU implementation (RGB→YUV, range normalise, 4:2:0 subsample, 8-bit quantise) with NEON/AVX2 kernels for every layout and dtype, and an NVRTC-compiled CUDA kernel for the same conversion. |
| `blazar-encoder` | `VideoEncoder` trait, calibration→config translation, and the codec backends: NVENC (AV1/H.264/HEVC), VideoToolbox (H.264/HEVC, macOS), x264, x265, SVT-AV1, VP9, plus a mock encoder for contract tests. |
| `blazar-sink` | `Sink` trait, `OrderGuard`, `DebugFileSink`, `MemorySink` (units handed back instead of written), and `Fmp4FileSink` on a pinned `oxideav-mp4` fork (AV1/H.264/HEVC + Opus/AAC). |
| `blazar-runtime` | `Encoder::new` → `open_stream` → `submit_video`/`flush`/`close`, the encoder worker, session limiting, and GPU preprocessing glue. |
| `blazar-*-sys` | FFI bindings: `svtav1`, `vpx`, `x265` (x265 resolves the encoder symbol per library build, so Debian/macOS builds both work). |

## What works

**Ingest.** Host frames as `thwc`/`bcthw`, `u8` or `f32`; device tensors as
`bcthw`/`bchw` in `float32` or `bfloat16`, including non-contiguous views. `u8`
ingest skips the float conversion and is byte-identical to `f32` output.

**Codecs.** NVENC AV1/H.264/HEVC (zero-copy from `cuMemAllocPitch` allocations),
VideoToolbox H.264/HEVC on macOS, x264, x265, SVT-AV1, VP9. Every backend writes
BT.709 full-range VUI; CPU codec packet timestamps come from the encoder's output
picture, not the submitted frame, so B-frame reordering keeps DTS valid.

**Output.** `output="file.mp4"` muxes fragmented MP4; `output=None` keeps the
encoded units in memory and hands them back from `poll()`, so a caller can forward
them (WebRTC, a test harness) without touching the filesystem.

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
`blazar.Calibrations`. Every argument, method, layout and error is documented in
[`docs/python-api.md`](docs/python-api.md); runnable examples are in
[`examples/python/`](examples/python/):

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

240 real 1080p frames from one clip, encoded to a file. The clock covers the submit
loop through `close()`; encoder construction happens before it and the frames are
already in the dtype and layout each row names. Best of two runs, RTX 4080 SUPER
with an i7-14700KF host, WSL2.

| Backend | Frames arrive as | u8 | f32 | bf16 |
|---|---|---|---|---|
| NVENC H.264 | host arrays, CPU rgb2yuv | not supported | 8.46 | not supported |
| NVENC H.264 | device tensor, GPU rgb2yuv | not supported | **3.20** | 4.48 |
| x264 | host arrays, CPU rgb2yuv | **4.65** | 6.74 | not supported |
| x264 | device tensor, GPU rgb2yuv | not supported | 6.68 | 7.06 |

What the table says:

- **The fastest path depends on the codec.** A hardware encoder that can read a
device buffer is 2.6x faster than staging to the host (3.20 against 8.46). A
software encoder is the opposite: x264 on host `u8` arrays (4.65) beats the same
x264 on a device tensor (6.68), because the frames have to come back for the encoder
to read them and nothing else needs the GPU.
- **Ingest dtype is a transfer cost, not an arithmetic one.** An 1080p frame is 6 MB
as `u8` and 25 MB as `f32`, and the conversion kernel costs the same either way
(0.44-0.62 ms measured). The 2.1 ms gap between x264's `u8` and `f32` rows is the
copy.
- **`bf16` device ingest is not a win yet.** It moves half the bytes of `f32` and
measured *slower* (4.48 against 3.20) for reasons not yet established. Do not size
a deployment on it until that is explained.

The four "not supported" cells are real gaps, not omissions:

- NVENC host frames cannot be `u8`: the GPU preprocess stage runs even for host
  input and has no `u8` branch, so `u8` frames must be widened to `f32` first.
- Device tensors must be `float32` or `bfloat16`; `uint8` is rejected, because a
  VAE does not emit it.
- Host `bf16` is rejected because the binding infers dtype from the buffer format
  and bfloat16 has no numpy dtype to infer from. An explicit dtype argument would
  close this.

For comparison, measured on the same box in an earlier pass and not re-run since:
SGLang's `CudaVideoEncoder` (ffmpeg/libx264) 9.13 ms/frame, vLLM-Omni's PyAV path
48.07. Both stage through host memory and use library default settings; blazar uses
a calibration target, so treat this as "what each stack costs as it ships", not a
quality-matched comparison.

**Batch kernel** (Nsight Compute A/B): 2.28-2.72x over per-frame launches, device
path only. `submit_batch` on host frames is slower than a per-frame loop, because the
frames have to cross into the runtime either way.

**Concurrent NVENC streams** (16 frames per stream, 832x480): 1 stream 13.9 ms/frame,
4 streams 4.59, 8 streams **3.51**. An RTX 4080 SUPER accepts 9 sessions and rejects
the 10th, which is why the session limiter exists.

**The colour conversion is not the bottleneck.** RGB→YUV420 has vector kernels
(NEON and AVX2) for all six combinations of layout and dtype, 2.1-3.7x faster than
the scalar reference at 1080p:

| Source | macOS (NEON) | Windows (AVX2) | WSL2 (AVX2) | Scalar |
|---|---|---|---|---|
| `thwc` u8 | 0.47 | 0.47 | 0.88 | 1.21-2.24 |
| `bcthw` u8 | 0.43 | 0.44 | 0.57 | 1.27-2.21 |
| `thwc` f32 | 0.46 | 0.52 | 0.76 | 1.59-2.62 |
| `bcthw` f32 | 0.46 | 0.44 | 0.66 | 1.61-2.62 |
| `thwc` bf16 | 0.47 | 0.62 | 1.16 | 1.28-2.33 |
| `bcthw` bf16 | 0.45 | 0.55 | 0.87 | 1.37-2.33 |

Numbers on this machine move by 10-15% between runs, so treat differences under that
as noise.

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
- VideoToolbox has no CRF. `RateControl::Crf` maps onto its quality knob
  (`1 - crf/51`), so the same setting is not the same bitrate as x264/x265.
  `request_keyframe` is a no-op: forcing one needs a frame property that is not
  wired up.

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
| `enc-videotoolbox` | macOS only; links the system frameworks, nothing to install |

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

## License

Apache-2.0. See `LICENSE`.

Note that the GPL codec backends are opt-in build features and are off by
default: enabling `enc-x264` or `enc-x265` links GPL libraries into the binary
and makes the resulting artifact GPL, not Apache-2.0.
