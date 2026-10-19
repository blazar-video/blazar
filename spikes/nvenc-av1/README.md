# NVENC-AV1 hardware spike

A standalone capability probe, **not part of the workspace** (excluded in the
root `Cargo.toml`). It links the NVIDIA Video Codec SDK + CUDA, so it only builds
and runs on a machine with an NVENC-capable NVIDIA GPU and the CUDA toolkit —
validated on **mosk** (RTX 4080 SUPER, WSL2 Ubuntu, CUDA 13.1).

## What it proves

- `nvidia-video-codec-sdk` (git) + `cudarc 0.19.10` build against CUDA 13.1.
- A CUDA context + NVENC-AV1 session opens on the GPU.
- Synthetic ARGB frames encode to a real AV1 bitstream.

Result on the 4080: 30× 832×480 frames → ~201 KB AV1, 1 keyframe.

## The dependency trap

The **published** `nvidia-video-codec-sdk 0.4.0` pins `cudarc 0.16.6`, which
panics with *"Unsupported cuda toolkit version 13.1"*. The fix (see `Cargo.toml`):
use the crate from **git main** (it depends on cudarc 0.19) and pin cudarc with the
`cuda-13010` + `dynamic-loading` features so no compile-time CUDA gate applies.

## Build & run (on mosk / an NVENC host)

```sh
export CUDA_ROOT=/usr/local/cuda
export PATH=$CUDA_ROOT/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$CUDA_ROOT/lib64:$LD_LIBRARY_PATH
# NVENC build.rs looks in the CUDA dir by default; on WSL the libs are here:
export NVIDIA_VIDEO_CODEC_SDK_PATH=/usr/lib/wsl/lib
cargo run
```

## Lessons folded into the real backend design (Phase 2)

- NVENC AV1 input formats on the 4080: NV12, YV12, IYUV, YUV420_10BIT, ARGB, ABGR,
  AYUV, ARGB10, ABGR10. The device-path preprocess should target **NV12**.
- With `enable_picture_type_decision()`, NVENC runs asynchronously: `encode_picture`
  returns `NeedMoreInput` while buffering, and output for a frame emerges on a later
  call. A correct drain needs a **pool of output bitstreams** (one per in-flight
  frame) plus an `end_of_stream()` flush. This is the concrete shape of the
  architecture §4 lookahead, confirmed against hardware.
- The backend takes an `Arc<CudaContext>`, which it can share with the device-ingest
  path for the in-process zero-copy topology (architecture §7).
