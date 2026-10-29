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
| blazar, host u8 → device kernel | upload + GPU kernel | 2.59 ms (386 fps) | 223 ms | 5.21 ms (192 fps) | 231 ms |
| PyAV, host u8 → swscale | CPU, then upload NV12 | 2.26 ms (442 fps) | 209 ms | 6.06 ms (165 fps) | 420 ms |
| ffmpeg CLI, raw pipe | swscale | 5.47 ms (183 fps) | — | 10.42 ms (96 fps) | — |

What this says:

- **Zero-copy is the win, and it grows with resolution**: 1.6× over PyAV at 720p,
  2.5× at 1088p. It is the only path that does not move the frame.
- **These are not the same work.** PyAV has no device-side conversion: `sws_scale`
  runs on the CPU and NVENC takes the resulting NV12. Blazar's row uploads the RGB
  frame and converts it in a kernel. The like-for-like pair is Blazar
  `preprocess="host"` against PyAV, measured in the next section — and there PyAV
  is faster.
- An earlier version of this table claimed host-staged Blazar beat PyAV by 1.5×.
  It compared a device-side conversion against a software one and called it
  like-for-like. It does not survive measurement.
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
- **These rows use device-side conversion, not the software path.** `auto` on
  NVENC resolves to the device, so each thread uploads its frame and the kernel
  writes NV12 into the surface NVENC reads. At 1088p on one thread the same clip
  gives 202.8 fps that way against **140.6 fps** with `preprocess="host"`, which
  converts on the CPU and then uploads NV12 — 1.44× slower, converting the same
  bytes (`184.8 bits/kpx` in both). The software path exists because CPU codecs
  need it; it is not the faster path on NVENC.
- **Polling after every submit costs nothing measurable** (557.2 vs 563.3, inside
  variance). An earlier version of this table reported ~7%; it also ran a busy
  Python thread for the whole timed region to count background iterations, which
  starved the interpreter for every arm and was reported as PyAV being 13× slower
  than it is. That thread is gone, and so is the claim.

### What the GIL actually costs

Throughput scaling is a proxy — it cannot separate "the encoder releases the
interpreter" from "the encoder is simply faster". This measures it directly: a
second thread runs a pure-Python loop (no C calls that might release the
interpreter), and the share of its throughput that survives tells you who is
holding the GIL. 1920×1088, 6 s window, sessions warmed.

"Python loop keeps" is that share: iterations/second of the loop while an encoder
runs beside it, divided by the same loop's iterations/second with nothing else
running. It is measured against a baseline taken by the same script, because the
loop body differs between scripts and the rates are not comparable across them.
Baseline here: 69.6 M iters/s. Retention below 100% can also come from clocks
rather than contention — running the same loop beside an active GPU process
measured 106–108% on this box, so an idle interpreter is evidently not at its
boost clock. The metric to read is the *comparison between arms*, on the same
baseline.

| Workload | fps while a Python thread runs | Python loop keeps |
|---|---|---|
| blazar, device tensor (zero copy) | **359.3** | 106% (noise) |
| blazar, pre-staged host arrays | 83.3 | 91% |
| pyav, pre-staged host arrays | 33.3 | 92% |

- **Device tensors cost no interpreter time at all.** `submit` hands a pointer to
  a Rust worker; the calling thread is free before the frame is touched. Nothing
  else in this table can say that, because everything else has to move the pixels
  through Python first.
- **Host arrays keep 91% of the Python thread's throughput — and 2.5× PyAV's fps**
  (83.3 vs 33.3) on identical input. The frame copy runs with the GIL released.
- **The host path costs the interpreter the same as PyAV's, and that is the
  point.** Three runs each against a 39.2 M iters/s baseline: Blazar 91%
  (89–92%), PyAV 92% (90–93%) — 0.1 percentage points apart, spreads overlapping.
  Both do per-frame Python work, so both pay for it. What differs is what each gets
  done for that price: 83.3 fps against 33.3 fps.

**A caveat about this box.** Repeating the host-array arm across this session gave
83.3–150 fps for the same code and input, and the device arm 333–364 fps. Run-to-
run variance here is up to 1.8× on the host path, so treat any single number as an
order of magnitude, not a measurement. The `tests/test_gil.py` guard asserts a
ratio (>60% retention), not a rate, for that reason.

**If your pipeline does the device→host copy per frame in Python** — quantise,
`.cpu()`, reshape, submit — the GIL cost is then your own loop, not the encoder's,
and it dominates: measured 14.7 fps under contention against 186.7 fps alone. The
fix for that is to hand over the tensor and let the encoder fetch it.

### Scaling the number of encoder threads

A serving process is several streams plus the work around them, so the useful
question is what happens when encoder threads are added while one Python thread
keeps working. 1920×1088, 6 s, one Python loop thread throughout, loop baseline
75.8 M iters/s.

Two blazar rows, because only one of them is comparable to PyAV:

| Encoder threads | blazar **device** fps | python kept | blazar **host** fps | python kept | pyav fps | python kept |
|---|---|---|---|---|---|---|
| 1 | 420.1 | 91% | 14.6 | 93% | 11.4 | 85% |
| 2 | 822.9 | 92% | 21.1 | 90% | 15.6 | 79% |
| 4 | 885.7 | 86% | 41.1 | 86% | 28.2 | 64% |

- **Device vs host is not a like-for-like comparison with PyAV.** The device row
  hands over a tensor: no per-frame Python work at all. The host row quantises,
  copies to host and submits an array — the same per-frame work PyAV does — so it
  is the row to compare against PyAV, and it is labelled as such.
- **Like-for-like, the gap grows with concurrency**: 1.28× at one thread, 1.35× at
  two, 1.46× at four (41.1 vs 28.2 fps). Same frames, same conversion site, same
  per-frame Python.
- **The Python thread degrades with thread count for PyAV and not for Blazar**:
  93% → 86% against 85% → 64%. Each PyAV encoder thread adds interpreter-bound
  work to a single GIL; Blazar's added threads cost the interpreter nothing.
- **Blazar's device path is the capability PyAV does not have**: 885.7 fps at four
  threads, 21× the host path, because the frames never leave the device.

PyAV's ratio scaling (2.47× at four threads on the host row) looks better than
Blazar's (2.82×) in ratio terms while being an order of magnitude slower in
absolute terms — ratios reward a low starting point, and the absolute numbers are
what a serving path runs on.

**Where the scaling stops, and why.** 2.7× on four threads is not a software
limit. At eight threads the aggregate reaches 626–641 fps (3.0×) and flattens,
with `nvidia-smi` reporting the **NVENC engine at 69% median / 100% peak** while SM
utilisation peaks at 41%. At that point the hardware encoder is the bottleneck, not
Blazar, the GIL, or the conversion. The same shape appeared at 720p (2.8× at four
threads, saturating near 1360 fps with the encoder at 98% peak): one engine does
roughly 1.5 ms of work per 1088p frame, so the aggregate saturates around 650 fps
at this size.

### Same clip, same frame count, two sizes

The tables above use different content per test, so this one holds everything
constant except resolution: one real clip resized to each size, 100 frames per
thread, warmed sessions, host `u8` arrays (no device→host copy inside the clock,
which is what makes this a pure encode+convert comparison), best of three.

| Arm | Size | t | fps | Gpx/s | Mbit/s | bits/kpx |
|---|---|---|---|---|---|---|
| blazar | 1280×720 | 1 | 284.9 | 0.26 | 58.8 | 224.0 |
| blazar | 1280×720 | 4 | 798.5 | 0.74 | 164.8 | 224.0 |
| pyav | 1280×720 | 1 | 280.4 | 0.26 | 38.8 | 150.0 |
| pyav | 1280×720 | 4 | 427.3 | 0.39 | 59.1 | 150.0 |
| blazar | 1920×1088 | 1 | 204.5 | 0.43 | 79.0 | 184.8 |
| blazar | 1920×1088 | 4 | 550.9 | 1.15 | 212.7 | 184.8 |
| pyav | 1920×1088 | 1 | 164.0 | 0.34 | 43.6 | 127.2 |
| pyav | 1920×1088 | 4 | 247.4 | 0.52 | 65.7 | 127.2 |

- **Pixel throughput is not resolution-independent, and larger is better**:
  1088p delivers 1.56× the pixels/s of 720p at four threads (1.63× at one). The
  per-frame costs — session work, API calls, mux — do not scale with picture size,
  so they are amortised better across 2.27× the pixels.
- **Scaling is resolution-independent**: 2.80× at 720p and 2.69× at 1088p for
  Blazar; 1.52× and 1.51× for PyAV. Same GIL story at both sizes.
- **The QP number is not a quality match between libraries.** At the same nominal
  `ConstQP 30`, Blazar spends 49% more bits per pixel at 720p and 45% more at
  1088p. Matching bitrate means giving PyAV a lower QP: at 720p, PyAV QP 27 lands
  at 218.3 bits/kpx against Blazar's 224.0, and runs at 257.6 fps against 284.9 —
  **parity inside the ~10% run-to-run noise** (PyAV at QP 28 measured *faster*
  than at QP 27, which is how large the noise is at this frame count).

**Software conversion against software conversion** (same clip, no device→host
copy, one thread, warmed):

| conversion | 720p | 1088p |
|---|---|---|
| Blazar `preprocess="host"` (CPU SIMD) | 184.8 fps (5.41 ms) | 140.6 fps (7.11 ms) |
| PyAV (CPU swscale) | 280.4 fps (3.57 ms) | 164.0 fps (6.10 ms) |
| Blazar `preprocess="device"` (upload + kernel) | 284.9 fps (3.51 ms) | 204.5 fps (4.89 ms) |

- **PyAV's software conversion beats ours**: 1.52× at 720p and 1.17× at 1088p for
  identical bytes. libswscale is a decades-old hand-tuned path and our kernel is
  not beating it on the CPU; the reason to use Blazar is not that it converts
  faster on the host.
- **Blazar's advantage is the conversion PyAV cannot do**: on the device. That puts
  it level with PyAV's software path at 720p (284.9 vs 280.4) and 1.25× ahead at
  1088p, without touching the CPU.

So the honest summary is: on host conversion PyAV is faster, on device conversion
Blazar is at parity to 1.25×, and the large multiple the old table claimed on the
host path was a device-vs-software comparison wearing the same label.

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

