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

Measured on mosk (WSL2), RTX 4080 SUPER, 2026-10-29. Source: 121 frames of real
LTX-2.5 content at 24 fps, re-encoded to the target size.

Every path starts from the same `(1, 3, 121, H, W)` CUDA tensor, which is what a
model hands over, and **every row pays its own device→host copy inside the
clock**. Encoder settings are mirrored: `h264_nvenc`, `preset p5`, `ConstQP 30`,
GOP 120, 2 B-frames, yuv420p (Blazar maps `crf` to `ConstQP`).

Two numbers per row, because one of them is not throughput: **steady** is the
median per-frame cost over three runs once the session is open, and **open** is
what a fresh NVENC session costs. 121 frames is short enough that session open
dominates a naive average — it accounts for 1.7–3.4 ms of every "ms/frame" the
previous version of this table published.

| Path | Conversion | 720p steady | 720p open | 1088p steady | 1088p open |
|---|---|---|---|---|---|
| blazar, device tensor | device kernel, no copy | **1.38 ms** (726 fps) | 233 ms | **2.46 ms** (407 fps) | 217 ms |
| blazar, host arrays | device kernel | 2.59 ms (386 fps) | 223 ms | 5.21 ms (192 fps) | 231 ms |
| PyAV, host arrays | swscale | 2.26 ms (442 fps) | 209 ms | 6.06 ms (165 fps) | 420 ms |
| ffmpeg CLI, raw pipe | swscale | 5.47 ms (183 fps) | — | 10.42 ms (96 fps) | — |

What this says:

- **Zero-copy is the win, and it grows with resolution**: 1.6× over PyAV at 720p,
  2.5× at 1088p. It is the only path that does not move the frame.
- **Host-staged Blazar is not faster than PyAV.** At 720p it is slightly slower
  (2.59 vs 2.26 ms); at 1088p slightly faster (5.21 vs 6.06 ms). Both pay the same
  device→host copy, so what is left is a conversion difference of a few tenths of
  a millisecond in either direction, inside run-to-run variance. An earlier
  version of this table claimed 1.5×; that comparison did not pay the copy on both
  sides and does not survive a clean measurement.
- **The CLI is 2.4× slower than PyAV at the same settings**, from the raw pipe and
  a separate process, not from the encode.

### Concurrency and the Python GIL

1920×1088, 100 frames per thread, one encoder per thread, host `u8` arrays through
`preprocess="auto"` (so each thread pays its own upload and device conversion —
this is the path a numpy frame takes, not the zero-copy one). NVENC settings
mirrored. Every session is warmed before the clock, so this measures throughput and
not session open. Best of three.

| Arm | 1 thread | 2 threads | 4 threads | scaling @ 4 | per-thread @ 4 |
|---|---|---|---|---|---|
| blazar | 210.9 fps | 350.9 fps | **563.3 fps** | 2.7× | 140.8 |
| blazar_poll | 207.8 fps | 341.7 fps | 557.2 fps | 2.7× | 139.3 |
| pyav | 163.5 fps | 200.6 fps | 266.0 fps | **1.6×** | 66.5 |

- **Blazar scales to 2.7× on four threads, PyAV to 1.6×**, and PyAV's per-thread
  rate drops by 2.5× (163.5 → 66.5 fps) as threads are added: `from_ndarray`,
  `reformat` and `mux` all run in the interpreter, so its threads take turns.
  Blazar's `submit` enqueues onto that encoder's Rust worker and returns.
- **Aggregate gap is 2.1× at four threads** (563.3 vs 266.0), against 1.3× on a
  single stream at the same size and path.
- **Polling after every submit costs nothing measurable** (557.2 vs 563.3, inside
  variance). An earlier version of this table reported ~7%; it also ran a busy
  Python thread for the whole timed region to count background iterations, which
  starved the interpreter for every arm and was reported as PyAV being 13× slower
  than it is. That thread is gone, and so is the claim.

**Where the scaling stops, and why.** 2.7× on four threads is not a software
limit. At eight threads the aggregate reaches 626–641 fps (3.0×) and flattens,
with `nvidia-smi` reporting the **NVENC engine at 69% median / 100% peak** while SM
utilisation peaks at 41%. At that point the hardware encoder is the bottleneck, not
Blazar, the GIL, or the conversion. The same shape appeared at 720p (2.8× at four
threads, saturating near 1360 fps with the encoder at 98% peak): one engine does
roughly 1.5 ms of work per 1088p frame, so the aggregate saturates around 650 fps
at this size.

Content differs from the table above — synthetic gradients at a fixed QP encode
slower than real footage — so the absolute rates are not comparable across the two
tables; the scaling is.

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

