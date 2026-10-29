# Blazar Video Encoder

Blazar is an encoder for generative video with optimized paths for both batched and streamed model outputs. Blazar offers Python, C and Rust bindings. 

## Features

- [x] Hardware encode: NVENC (H.264/AVC, H.265/HEVC, AV1), VideoToolbox (H.265/AVC, H.265/HEVC)
- [x] Software encode: libx264 (H.264/AVC), libx265 (H.265/HEVC), libvpx (VP9), SVT-AV1 (AV1)
- [x] File output: fMP4
- [x] Buffer output
- [x] GPU-Accelerated YUV Conversion
- [x] SIMD-Accelerated YUV Conversion
- [x] Runtime selectable configuration files

## Links

- [Python API Docs](docs/python-api.md)
- [Calibrations](https://github.com/blazar-video/calibrations)

## How it works

The general goal of Blazar is to plugin to existing generative video frameworks and offer an encoder with native compatibility
with model outputs.

```python
import blazar

cals = blazar.Calibrations.load("calibrations/default")
with blazar.Encoder(calibrations=cals, identity={"family": "wan", "version": "2.1",
                                                 "quant": "bf16", "size": "1.3b"},
                    target="balanced", backend="nvenc", codec="h264",
                    output="out.mp4") as stream:
    stream.submit(frame, width=W, height=H, fps=24.0, layout="thwc", range="unit01")
    # or, with device tensors:
    stream.submit(tensor, fps=24.0, range="unit01")
```

Create the encoder with a pointer to the [calibrations](https://github.com/blazar-video/calibrations) directory you wish to use. These calibrations can be for a mixture of models, scene types, codecs, and codec backends.

## Performance

mosk (WSL2), RTX 4080 SUPER, 2026-10-29. 121 frames of real LTX-2.5 content at
24 fps. Encoder settings mirrored everywhere: `h264_nvenc`, `preset p5`,
`ConstQP 30`, GOP 120, 2 B-frames, yuv420p (Blazar maps `crf` to `ConstQP`).

**Two rules every table follows**, because breaking them produced numbers that
looked comparable and were not:

- **The source is a `(1, 3, T, H, W)` tensor already resident on the device**, in
  the shape SGLang's decode stage produces, built before the clock starts.
  Nothing in any clock uploads source frames — this reproduces the model's output,
  so the pixels are already there.
- **Session open is excluded from every row.** It is a fixed 200–400 ms and it
  amortises differently per path: a faster path completes its frames sooner and
  therefore pays proportionally more of it, which turns a constant into an
  apparent throughput difference. Measuring it inclusive inflated the device row
  from 2.20 ms to 4.15 ms and shrank its margin over PyAV from 3.9× to 2.3×.

### 1. Steady state, per frame

Each row starts from the device tensor and pays only what its own path needs,
including its own device→host copy where it needs one. Median of three sequential
runs.

| Path | What it does from the device tensor | 720p | 1088p |
|---|---|---|---|
| blazar, device tensor | kernel writes NV12 into the surface NVENC reads | **1.10 ms** (909 fps) | **2.20 ms** (455 fps) |
| blazar, host path | quantise → copy to host → CPU convert → upload NV12 | 3.54 ms (282 fps) | 5.37 ms (186 fps) |
| PyAV | quantise → copy to host → swscale → encode | 2.73 ms (367 fps) | 8.60 ms (116 fps) |
| ffmpeg CLI | quantise → copy to host → pipe raw → swscale → encode | not measured | 8.24 ms (121 fps) |

- **The device path is 2.5× PyAV at 720p and 3.9× at 1088p**, and the margin grows
  with frame size because the copy and the conversion are what it does not pay.
- **The host path beats PyAV at 1088p (1.6×) and loses at 720p (0.77×).** It pays
  the same copy PyAV does, so what is left is conversion: ours amortises better on
  a larger frame and worse on a small one. The ordering is not
  resolution-independent and is reported as measured.
- **The CLI's 8.24 ms is a marginal cost**, `(t(121) − t(81)) / 40`, because a
  separate process has no session open we can time directly; process start and
  codec init cancel in the difference. It is not a per-frame number in the same
  sense as the others and is shown for scale only.

**QP is not a quality match between libraries.** At the same nominal `ConstQP 30`
Blazar spends 45–49% more bits per pixel than PyAV, so the fps figures above are
not quality-matched. Bitrate-matched (PyAV at QP 27 = 218.3 bits/kpx against
Blazar's 224.0) the two are level inside noise, and that noise is large enough that
PyAV measured *faster* at QP 28 than at QP 27. Treat the margin between the two as
a throughput comparison at unequal quality, not as a like-for-like win.

**Caveat on PyAV's 1088p row.** Across this session the identical PyAV pipeline
measured 5.07, 6.06, 8.24 and 8.60 ms/frame with the same code, content and
settings. That band is wider than the device-vs-PyAV margin, so treat the ratio as
somewhere between 2.3× and 3.9× until the interleaved re-measurement lands. The
device path is stable across every session at 2.20–2.49 ms.

### 2. The GIL

1920×1088, 6 s window, sessions warmed. A second thread runs a pure-Python loop
(no C calls that could release the interpreter); the share of its throughput that
survives says who is holding the GIL. Loop baseline 69.6 M iters/s. **Retention above 100% is possible** — the same
loop beside an active GPU process measured 106–108%, so an idle interpreter is not
at its boost clock. Read the comparison between arms, not the percentage.

**"Pre-staged" means the numpy arrays exist in host memory before the clock**, so
these rows time `submit()` and the encode only. They exclude the per-frame work a
real pipeline does to produce those arrays — quantise, device→host copy, reshape —
which is deliberate: it isolates the library's own interpreter behaviour from the
caller's torch work. It also means these frame rates are **not** comparable with
table 1, which includes that work. The rates here are lower despite doing less,
because a competing Python thread is saturating the interpreter throughout.

| Workload | fps beside a Python thread | Python loop keeps |
|---|---|---|
| blazar, device tensor | **359.3** | 106% (noise) |
| blazar, pre-staged host arrays | 83.3 | 91% (89–92%) |
| PyAV, pre-staged host arrays | 33.3 | 92% (90–93%) |

- **Device tensors cost no interpreter time**: `submit` hands a pointer to a Rust
  worker. Nothing else here can say that, because everything else moves pixels
  through Python first.
- **On identical host input Blazar does 2.5× PyAV's frame rate for the same
  interpreter cost** (91% vs 92%, 0.1 points apart over three runs each). The cost
  is the same because the work is the same; what differs is what each gets done.
- **If your loop does the device→host copy per frame in Python**, that becomes the
  bottleneck rather than the library: 186.7 fps alone against 14.7 fps under
  contention. Hand over the tensor instead.

Scaling with encoder threads, same 1920×1088, one Python thread throughout,
baseline 75.8 M iters/s:

| Streams | blazar device | python | blazar host | python | PyAV | python |
|---|---|---|---|---|---|---|
| 1 | 420.1 fps | 91% | 14.6 fps | 93% | 11.4 fps | 85% |
| 2 | 822.9 fps | 92% | 21.1 fps | 90% | 15.6 fps | 79% |
| 4 | 885.7 fps | **86%** | 41.1 fps | 86% | 28.2 fps | **64%** |

- **Like-for-like (host columns, same per-frame Python work as PyAV), the gap
  grows with concurrency**: 1.28× at one stream, 1.46× at four.
- **PyAV's Python thread degrades as streams are added** (85% → 64%) because each
  thread adds interpreter work to one GIL. Blazar's added threads cost the
  interpreter nothing (91% → 86%, one thread's worth of variance).
- **The device row is a capability PyAV does not have**, 21× the host path at four
  streams. It is not a GIL measurement and should not be read as one.

`poll()` after every submit costs nothing measurable (557.2 vs 563.3 fps, inside
variance). An earlier table claimed ~7%; it also ran a busy Python thread inside
the timed region, which starved every arm and was published as PyAV being 13×
slower than it is.

**Where it stops.** Eight streams reach 626–641 fps at 1088p and 1359.7 fps at
720p, then flatten, with `nvidia-smi` reporting the NVENC engine at 69% median /
100% peak and SM at 41%. Past four streams the hardware encoder is the limit.

### 3. Where the time goes

`stats["encode_us"]` covers the whole submit path, so it cannot say whether the
encoder is the bottleneck. The worker counts time inside `nvEncEncodePicture`
itself, exposed as `blazar._blazar._nvenc_encode_micros()` for benchmarks. 1088p
device tensors:

| Streams | aggregate fps | wall per frame | driver call | driver share |
|---|---|---|---|---|
| 1 | 389.0 | 2.57 ms | 0.422 ms | 17% |
| 2 | 794.3 | 1.26 ms | 0.275 ms | 23% |
| 4 | 883.2 | 1.13 ms | 0.243 ms | 23% |

- **The encode call is not where the time is**: 17% of the wall clock at one
  stream. Of the remaining 2.15 ms, a kernel-only measurement of the RGB→NV12
  conversion came to ~1.37 ms, which puts most of the frame in the conversion
  rather than the encode. If single-stream throughput is the goal, the kernel is
  the target — that number is from a separate kernel benchmark, not from a table
  here, because a pipeline cannot isolate it.
- **The call shortens per call as streams are added** (0.42 → 0.24 ms) while
  growing as a share: with one stream each call waits behind that stream's
  conversion, with four there is always work in flight. That is an interpretation;
  the measurement behind it is the share rising while the call shortens.
- These are CPU times inside the call — `nvEncEncodePicture` queues work, so this
  is not GPU encode time.

### Retired numbers

Kept here so nobody re-derives them from an old copy of this file:

- **Host-staged Blazar beating PyAV by 1.5×**: it compared a device-side conversion
  against a software one under the same label. On the canonical pipeline it is
  1.6× at 1088p and 0.77× at 720p.
- **PyAV at 33.7 fps single-threaded**: that harness ran a Python spin thread
  inside the timed region. Without it, 313–367 fps depending on the arm.
- **`ms/frame` figures that included session open** (1.7–3.4 ms of every number):
  they penalised the fastest path most, as described at the top of this section.

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

