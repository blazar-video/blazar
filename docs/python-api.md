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
                    backend="nvenc", codec="h264", output="out.mp4") as stream:
    stream.submit(tensor, fps=24.0, range="unit01")
```

### Constructor

| Argument | Type | Meaning |
|---|---|---|
| `calibrations` | `Calibrations` | Loaded calibration set. |
| `identity` | `dict` | Model identity: `{"family", "version", "quant", "size"}`. Used to pick the calibration. |
| `target` | `str` | `"low_latency"`, `"balanced"` or `"quality"`. |
| `backend` | `str` | Implementation: `"nvenc"`, `"videotoolbox"` (macOS), `"svtav1"`, `"x264"`, `"x265"`, `"vp9"`, `"mock"` (tests only, compresses nothing), or `"auto"`. The codec comes from `codec`, not from the name; see [Backend selection](#backend-selection). |
| `backend_pool` | `str` | What `backend="auto"` may choose: `"any"` (default, hardware preferred then software), `"hardware"` (never falls back), `"software"`. Ignored when `backend` is named. |
| `output` | `str \| None` | Output path, or `None` to keep encoded units in memory and hand them back from `poll()`. |
| `max_buffered_units` | `int` | When `output` is `None`: how many encoded units may sit in the buffer before a write fails. Defaults to 240. |
| `layout` | `str \| None` | `"thwc"` or `"bcthw"`, fixed for the stream: a container written with two interpretations of one shape is unplayable. `None` infers it from the first frame and locks. |
| `codec` | `str` | Wire codec: `"h264"`, `"hevc"`, `"av1"`, `"vp9"`. Pairing it with an implementation that cannot encode it is an error at construction. |
| `request_id` | `str` | Tag used in logs and events. Defaults to `"stream"`. |
| `video_extradata` | `bytes \| None` | Codec configuration override. Leave `None` for real encoders. |
| `queue_depth` | `int` | Frames the worker may hold ahead of the encoder. |

### Methods

| Method | Returns | Meaning |
|---|---|---|
| `submit(frames, fps, width=None, height=None, layout=None, range="unit01", pts_ticks=None, scene_cut=False)` | `int` | Queue frames. A numpy array takes the host path and is copied once; a CUDA tensor takes the device path and is read in place. The type decides, so there is one call rather than one per residency; `encoder.residency` reports which path the stream took. A sequence of host frames is accepted and encoded in order; a sequence mixing host and device is an error. `width`/`height` are required for host frames and carried by the tensor's shape for device frames. `layout` defaults to the one fixed at `Encoder(...)`. `range` is `"unit01"`, `"signed_unit"` or `"full_u8"`, and is validated against the dtype. Returns the last assigned PTS. |
| `poll(timeout=None)` | `list[dict]` | Encoded events, all the same shape: `kind` (`"encoded"`, `"failed"` or `"closed"`), `seq`, `pts`, `dts`, `keyframe`, `units`, `bytes`, `encode_us`, `queue_depth`, `data`. A per-frame event summarises one submitted frame and reports how many access units it produced; with `output=None` each unit also arrives as its own event with `units` 1 and `data` holding the encoded bytes. With an output file, `data` is `None`. `pts` is presentation order and is deliberately not monotonic once B-frames reorder; `dts` is the monotonic one. |
| `flush(timeout=None)` | `None` | Encode everything queued. |
| `close(timeout=None)` | `None` | Flush and finalise the output. Safe to call twice. |
| `resolved_backend` | `str \| None` | Which backend actually answered. `None` until the first frame when `backend="auto"`, because the choice depends on what opens a session here. A property. |
| `preprocess` | `str` | Where host frames are converted to NV12: `"host"` or `"device"`. A property; see [Preprocess placement](#preprocess-placement). |
| `residency` | `str \| None` | `"host"` or `"device"`, fixed by the first submit. A property. |
| `stats` | `dict` | Counters for the stream: `encoded`, `bytes`, `failed`, `max_encode_us`. A property, not a method. |
| `error_message()` | `str \| None` | Why the stream failed, if it did. |
| `failed_event_message()` | `str \| None` | Error recorded by the worker, including backend-specific text. |

The object is a context manager: leaving the `with` block calls `close()`.

### Collecting encoded frames instead of writing a file

```python
enc = blazar.Encoder(calibrations=cals, identity=identity, target="balanced",
                     backend="x264", codec="h264", output=None)
for frame in frames:
    enc.submit(frame, width=W, height=H, fps=24.0, layout="thwc", range="full_u8")
enc.close()
for event in enc.poll():
    if event["kind"] == "encoded" and event["data"] is not None:
        send(event["data"])     # one encoded access unit
```

Units arrive in decode order, so `pts` is not monotonic when the encoder reorders
with B-frames. The buffer is bounded: if the caller stops draining it, a write fails
with an error naming the limit rather than dropping a frame silently. Encoders with
lookahead or B-frames emit nothing at submit time, so most units surface during
`close()`, which is why the loop above polls after closing.

### Errors

Encoder construction and submission raise `ValueError` for bad arguments
(unknown target, wrong tensor dtype, non-contiguous tensor) and `RuntimeError`
for backend failures. A backend whose build feature is off raises rather than
silently substituting a stub. When a consumer GPU's driver refuses a new NVENC
session the message says so and names `BLAZAR_MAX_NVENC_SESSIONS`.

## Backend selection

`backend` names an **implementation**; `codec` names the wire codec. They are not
redundant: `nvenc` encodes h264, hevc or av1, and `codec` says which.

| `backend` | codecs |
|---|---|
| `nvenc` | h264, hevc, av1 |
| `videotoolbox` | h264, hevc |
| `svtav1` | av1 |
| `x264` / `x265` | h264 / hevc |
| `vp9` | vp9 |
| `mock` | any (tests only, produces no data) |
| `auto` | picks from `backend_pool` |

`backend_pool` constrains `auto` and is a constraint, not a preference:

| pool | behaviour |
|---|---|
| `"any"` (default) | hardware first, software as the fallback |
| `"hardware"` | hardware only; **errors rather than falling back** |
| `"software"` | software only |

```python
enc = blazar.Encoder(..., backend="auto", backend_pool="hardware", codec="h264")
enc.resolved_backend    # "nvenc_h264" once the stream has opened, else None
```

Availability cannot be queried cheaply — an NVENC session costs ~50-100 ms and one
of the 9 consumer session slots — so `auto` tries candidates in a fixed order and
takes the first that opens. Consequences worth knowing:

- The resolved backend is logged and reported, so a fallback is never invisible.
- The calibration is resolved **after** selection and keyed on what answered, so a
  fallback that changes quality is recorded rather than hidden.
- A stream's backend can differ between runs when hardware is transiently
  unavailable (a busy box hitting the session cap is the realistic case). Name the
  backend explicitly when you need byte-stability — corpus generation and
  calibration sweeps both do.

A pair-shaped spelling (`backend="nvenc_h264"`) is rejected: it restates `codec`,
and the two could disagree. Name the implementation and pass `codec=`.

## Preprocess placement

RGB→NV12 conversion can run on the host or on the device. `Encoder(..., preprocess=...)`
chooses, and `encoder.preprocess` reports what a host frame gets:

| value | meaning |
|---|---|
| `"auto"` (default) | Follow the backend: NVENC converts on the device (the kernel writes straight into the surface NVENC encodes from, so the upload is the only copy); CPU codecs convert on the host, because a device conversion would have to be downloaded again. |
| `"host"` | Always convert on the host. |
| `"device"` | Always convert on the device. Requires a wheel built with `gpu-preprocess` *and* a loadable CUDA driver; otherwise constructing the encoder raises `ValueError` naming whichever is missing. VideoToolbox is hardware encoding but converts on the host, so `"device"` does not apply to it. |

Two behaviours worth knowing:

- **Placement is a cost, not a contract.** The runtime performs whatever copy a
  placement implies. You are never required to move a tensor to host before
  submitting it.
- **Device tensors convert on the device.** `preprocess="host"` with
  `submit` with a device tensor raises rather than silently converting on the device:
  honouring it would mean downloading the tensor, which is not implemented yet.

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
