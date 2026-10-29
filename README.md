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

mosk (WSL2), RTX 4080 SUPER, 2026-10-29. Source: 121 frames of real LTX-2.5
content at 24 fps. Encoder settings mirrored everywhere: `h264_nvenc`, `preset p5`,
`ConstQP 30`, GOP 120, 2 B-frames, yuv420p (Blazar maps `crf` to `ConstQP`).

Every table states its basis, and the numbers are not comparable across them:

1. **The canonical pipeline** — a device-resident tensor in SGLang's shape,
   nothing uploaded inside the clock, each path paying only its own copy.
2. **Conversion in isolation** — the copy excluded, for comparing kernels. Not a
   pipeline, and labelled as such.
3. **The GIL** — throughput and interpreter interference with other threads.
4. **Where the time goes** — instrumented inside the encoder call.

Run-to-run variance on this box reaches 1.8× on the host paths, so treat these as
orders of magnitude; every table says how many runs it is.

### 1. The canonical pipeline

The invariant, and the only thing that makes these rows comparable:

- the source is a `(1, 3, T, H, W)` tensor **already resident on the device**, in
  the shape SGLang's decode stage produces, built before the clock starts;
- **nothing in the clock uploads source frames.** There is nothing to upload: this
  reproduces the model's output, so the pixels are already there;
- each row pays only what its own path needs to get from that tensor to encoded
  bytes — including its own device→host copy, if it needs one at all.

121 frames of real LTX-2.5 content, one stream, median of three, session open
included (a fresh NVENC session is 200–400 ms, i.e. 1.7–3.3 ms per frame at this
clip length; it is a fixed cost, so read the ordering rather than the offset).

| Path | What it does from the device tensor | 720p | 1088p |
|---|---|---|---|
| blazar, device tensor | kernel writes NV12 into the surface NVENC reads | **3.09 ms** (324 fps) | **4.15 ms** (241 fps) |
| blazar, host path | quantise → copy to host → CPU convert → upload NV12 | 5.39 ms (186 fps) | 7.06 ms (142 fps) |
| PyAV | quantise → copy to host → swscale → encode | 3.78 ms (265 fps) | 9.65 ms (104 fps) |
| ffmpeg CLI | quantise → copy to host → pipe raw → swscale → encode | 6.86 ms (146 fps) | 13.46 ms (74 fps) |

- **The device path is 1.2× PyAV at 720p and 2.3× at 1088p.** The margin grows with
  frame size because it is the copy and the conversion that it does not pay.
- **The host path wins at 1088p (1.4× over PyAV) and loses at 720p (0.70×).** It
  pays the same copy PyAV does, so what is left is conversion: ours is faster than
  swscale once the frame is big enough to amortise it, and slower when it is not.
  Reported as measured; there is no version of this where the ordering is
  resolution-independent.
- **PyAV's own cost is nearly all copy**: 3.78 → 9.65 ms from 720p to 1088p (2.5×
  for 2.27× the pixels), which is what a host-side pipeline looks like.

### 2. Conversion in isolation (not a pipeline)

Two questions the table above cannot answer, because it includes the copy: whose
conversion is faster, and how much of the per-frame cost is conversion at all.
These rows hold host arrays in memory and start the clock with the copy excluded —
useful for comparing kernels, **not** representative of a pipeline, which is why
they are not in the main table.

| Conversion, host arrays, copy excluded | 720p | 1088p |
|---|---|---|
| Blazar `preprocess="device"` (upload + kernel) | 284.9 fps (3.51 ms) | 204.5 fps (4.89 ms) |
| PyAV (swscale) | 280.4 fps (3.57 ms) | 164.0 fps (6.10 ms) |
| Blazar `preprocess="host"` (CPU SIMD) | 184.8 fps (5.41 ms) | 140.6 fps (7.11 ms) |

- **On the CPU, swscale beats our kernel**: 1.52× at 720p, 1.17× at 1088p. A
  twenty-year-old hand-tuned path wins on the host.
- **On the device there is no competitor**, because PyAV cannot convert there.
- **The QP number is not a quality match between libraries.** At the same nominal
  `ConstQP 30`, Blazar spends 45–49% more bits per pixel. Matched on bitrate (PyAV
  at QP 27 = 218.3 bits/kpx against Blazar's 224.0) PyAV runs 257.6 fps against
  284.9 — parity inside a noise band that also produced PyAV running *faster* at
  QP 28 than QP 27.

### 3. The GIL

A second thread runs a pure-Python loop (no C calls that could release the
interpreter); the share of its throughput that survives says who is holding the
GIL. Loop baseline: 69.6 M iters/s. **Retention above 100% is possible** — the same
loop beside an active GPU process measured 106–108%, so an idle interpreter is not
at its boost clock. Read the comparison between arms, not the absolute percentage.

| Workload | fps beside a Python thread | Python loop keeps |
|---|---|---|
| blazar, device tensor | **359.3** | 106% (noise) |
| blazar, pre-staged host arrays | 83.3 | 91% (89–92%) |
| PyAV, pre-staged host arrays | 33.3 | 92% (90–93%) |

- **Device tensors cost no interpreter time**: `submit` hands a pointer to a Rust
  worker. Nothing else here can say that, because everything else moves pixels
  through Python first.
- **On identical host input Blazar does 2.5× PyAV's frame rate for the same
  interpreter cost** (91% vs 92%, 0.1 points apart across three runs each). The
  cost is the same because the work is the same — per-frame Python. What differs is
  what each gets done for it.
- **If your loop does the device→host copy per frame in Python**, that becomes the
  bottleneck rather than the library: 186.7 fps alone against 14.7 fps under
  contention. Hand over the tensor instead.

Scaling with encoder threads, same one Python thread throughout, baseline 75.8 M
iters/s:

| Streams | blazar device | python | blazar host | python | PyAV | python |
|---|---|---|---|---|---|---|
| 1 | 420.1 fps | 91% | 14.6 fps | 93% | 11.4 fps | 85% |
| 2 | 822.9 fps | 92% | 21.1 fps | 90% | 15.6 fps | 79% |
| 4 | 885.7 fps | **86%** | 41.1 fps | 86% | 28.2 fps | **64%** |

- **Like-for-like (the host columns, same per-frame Python work as PyAV), the gap
  grows with concurrency**: 1.28× at one stream, 1.46× at four.
- **PyAV's Python thread degrades as streams are added** (85% → 64%) because each
  thread adds interpreter work to one GIL. Blazar's added threads cost the
  interpreter nothing (91% → 86%, which is one thread's worth of variance).
- **The device row is a capability PyAV does not have**, 21× the host path at four
  streams. It is not a GIL measurement and should not be read as one.

`poll()` after every submit costs nothing measurable (557.2 vs 563.3 fps, inside
variance). An earlier table claimed ~7%; it also ran a busy Python thread inside
the timed region, which starved every arm and was published as PyAV being 13×
slower than it is.

**Where it stops.** Eight streams reach 626–641 fps at 1088p and 1359.7 fps at
720p, then flatten, with `nvidia-smi` reporting the NVENC engine at 69% median /
100% peak and SM at 41%. Past four streams the hardware encoder is the limit, not
Blazar or the GIL.

### 4. Where the time goes

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
  stream. The RGB→NV12 kernel measures ~1.37 ms of the remaining 2.15 ms, which
  makes the conversion the largest component and the encode the smallest. If
  single-stream throughput is the goal, the kernel is the target, not the encoder
  settings.
- **The call shortens per call as streams are added** (0.42 → 0.24 ms) while
  growing as a share: with one stream each call waits behind that stream's
  conversion, with four there is always work in flight. That is an interpretation;
  the measurement behind it is the share rising while the call shortens.
- These are CPU times inside the call — `nvEncEncodePicture` queues work, so this
  is not GPU encode time.

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

