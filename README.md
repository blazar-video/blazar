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
    stream.submit_device_batch(tensor, fps=24.0, range="unit01")
```

Create the encoder with a pointer to the [calibrations](https://github.com/blazar-video/calibrations) directory you wish to use. These calibrations can be for a mixture of models, scene types, codecs, and codec backends.

## Performance

This comparison table was done on a Core i7 14700KF with a 4080 Super in WSL. What we are measuring here is a BF16 (or F32 in the case of the blazar F32 row) (1, 3, 121, 1088, 1920) tensor being passed into the encoder pipelines for these scenarios:
1. Blazar device bf16 (BF16 input tensor, GPU YUV conversion, NVENC)
2. Blazar device u8 -> CPU YUV Conversion -> NVENC
3. PyAV device u8 -> CPU YUV Conversion -> NVENC
4. FFmpeg device u8 -> IPC -> CPU YUV Conversion -> NVENC

| Path | Conversion | 1088p ms/frame | 1088p fps | 720p ms/frame |
|---|---|---|---|---|
| blazar, device tensor | device kernel, no copy | **3.88** | 258 | **2.78** |
| blazar, host arrays | device kernel | 6.35 | 158 | 3.70 |
| PyAV, host arrays | swscale | 9.37 | 107 | 4.29 |
| ffmpeg CLI, raw pipe | swscale | 19.81 | 51 | 8.94 |

### Concurrency and the Python GIL

This test is done on the same machine as above.

| Arm | 1 thread | 2 threads | 4 threads | scaling | per-thread at 4 |
|---|---|---|---|---|---|
| blazar | 155.9 fps | 383.0 | **620.8** | **4.0x** | 155.2 |
| PyAV | 33.7 | 61.9 | 63.0 | **1.9x** | 15.7 |

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

