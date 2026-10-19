# Python bindings

PyO3 bindings over `blazar-runtime`, built with maturin.

## Build

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install maturin numpy pytest

# mock encoder (no CUDA): enough to exercise the binding and the muxer
maturin develop

# hardware AV1 on a CUDA host
maturin develop --features nvenc
```

`bindings/python` is its own cargo workspace on purpose: `extension-module` does
not link libpython, which breaks `cargo test` linking, so the portable
`cargo test --workspace` must not see it.

## Test

```sh
pytest bindings/python/tests -v
```

The suite uses the mock encoder and asserts container structure, not decodability:
the mock has no real codec, so it is given a plausible `video_extradata` and
writes synthetic payload bytes. Decode validation stays on hardware (see the
repo README, "Hardware validation").

## Use

```python
import numpy as np
import blazar

calibrations = blazar.Calibrations.load("calibrations/default")

with blazar.Encoder(
    calibrations=calibrations,
    identity={"family": "wan", "version": "2.1", "quant": "bf16", "size": "1.3b"},
    target="balanced",
    backend="nvenc_av1",           # "svtav1" (mock here) on machines without CUDA
    output="out.mp4",
) as stream:
    for i, frame in enumerate(frames):        # frame: (H, W, 3) float32 in [0, 1]
        stream.submit(
            frame, width=W, height=H, fps=24.0,
            layout="thwc", range="unit01",
            scene_cut=(i == 0),
        )
```

Defaults synthesise PTS as `frame_index * 90000 / fps`; pass `pts_ticks=` to
place frames explicitly (vLLM-Omni chunks carry a cumulative frame index).

## Boundaries

- Video only. `submit_audio` does not exist on the Python object.
- The sink is a file. There is no WebSocket/streaming destination yet, so this
  measures the encode path rather than delivering a live stream.
- Device tensors are Phase 2: the frame is copied to host and then into Rust.
