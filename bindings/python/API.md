# blazar Python API

`import blazar` after `maturin develop`. Every class lives in the compiled module
`blazar._blazar` and is re-exported from `blazar`.

## `blazar.Calibrations`

A directory of calibration files, loaded once and shared.

```python
cals = blazar.Calibrations.load("calibrations/default")
```

| Member | Meaning |
|---|---|
| `Calibrations.load(path)` | Read every `*.toml` under `path`. Raises if a file is invalid; one bad file fails the load rather than being skipped. |
| `len(cals)` | Number of calibrations loaded. |

## `blazar.Encoder`

One encoder stream. Owns a worker thread; `submit*` hands frames over and returns
without waiting for the codec.

```python
with blazar.Encoder(calibrations=cals, identity={...}, target="balanced",
                    backend="nvenc_h264", codec="h264", output="out.mp4") as stream:
    stream.submit_device_batch(tensor, fps=24.0, range="unit01")
```

### Constructor

| Argument | Type | Meaning |
|---|---|---|
| `calibrations` | `Calibrations` | Loaded calibration set. |
| `identity` | `dict` | Model identity: `{"family", "version", "quant", "size"}`. Used to pick the calibration. |
| `target` | `str` | `"low_latency"`, `"balanced"` or `"quality"`. |
| `backend` | `str` | `"nvenc_av1"`, `"nvenc_h264"`, `"nvenc_hevc"`, `"x264"`, `"x265"`, `"svtav1"`, `"vp9"`, or `"mock"` (tests only, compresses nothing). |
| `output` | `str` | Output path. |
| `codec` | `str` | Wire codec: `"h264"`, `"hevc"`, `"av1"`, `"vp9"`. Required for CPU backends, because `backend` and `codec` are separate choices. |
| `request_id` | `str` | Tag used in logs and events. Defaults to `"stream"`. |
| `video_extradata` | `bytes \| None` | Codec configuration override. Leave `None` for real encoders. |
| `queue_depth` | `int` | Frames the worker may hold ahead of the encoder. |

### Methods

| Method | Returns | Meaning |
|---|---|---|
| `submit(frame, width, height, fps, layout, range, pts_ticks=None, scene_cut=False)` | `int` | Queue one host frame. `layout` is `"thwc"` (`H, W, 3`) or `"bcthw"` (`3, H, W`), optionally with a leading batch axis. `range` is `"unit01"`, `"signed_unit"` or `"full_u8"`. Returns the assigned PTS. |
| `submit_batch(frames, width, height, fps, layout, range, pts_ticks=None, scene_cut=False)` | `int` | Queue many host frames in one call. |
| `submit_device_batch(tensor, fps, range, pts_ticks=None, scene_cut=False)` | `int` | Queue a CUDA tensor shaped `(1, 3, T, H, W)`, `torch.float32` or `torch.bfloat16`, contiguous. This is the fast path: no host copy and, for CPU codecs, colour conversion on the GPU. |
| `poll(timeout=None)` | `list[dict]` | Encoded units produced so far, each with `track`, `pts`, `dts`, `keyframe`, `data`, `path`. |
| `flush(timeout=None)` | `None` | Encode everything queued. |
| `close(timeout=None)` | `None` | Flush and finalise the output. Safe to call twice. |
| `stats()` | `dict` | Counters for the stream. |
| `error_message()` | `str \| None` | Why the stream failed, if it did. |
| `failed_event_message()` | `str \| None` | Error recorded by the worker, including backend-specific text. |

The object is a context manager: leaving the `with` block calls `close()`.

### Errors

Encoder construction and submission raise `ValueError` for bad arguments
(unknown target, wrong tensor dtype, non-contiguous tensor) and `RuntimeError`
for backend failures. A backend whose build feature is off raises rather than
silently substituting a stub. When a consumer GPU's driver refuses a new NVENC
session the message says so and names `BLAZAR_MAX_NVENC_SESSIONS`.

## Environment

| Variable | Meaning |
|---|---|
| `BLAZAR_MAX_NVENC_SESSIONS` | Concurrent NVENC sessions this process will open before queueing. Default 8. |

## Types and layouts

| Name | Shape | Notes |
|---|---|---|
| `thwc` | `(H, W, 3)` interleaved | What `numpy` video decoders give you. |
| `bcthw` | `(3, H, W)` planar | What a VAE emits for a single frame. |
| device tensor | `(1, 3, T, H, W)` | `bfloat16` is fastest; `float32` also accepted. |

`range="unit01"` means values in `[0, 1]`, `"signed_unit"` means `[-1, 1]`, and
`"full_u8"` means `0..255`. Passing
the wrong range is a silent colour error, not an exception.

## Interop

`blazar.sglang_interop.install(on_frames)` wires the runtime into SGLang's decode
stage for a given SGLang version, and `device_output(...)` returns a context
manager that yields the device tensor. See `sglang_interop.py`.
