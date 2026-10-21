# blazar — the Rust runtime

Model-aware, low-latency video encoding runtime for generative video runtimes
(xDiT, vLLM-Omni). The design docs (`architecture.md`, `calibration-format.md`)
live outside this repo for now.

## Workspace layout

| Crate | Role | Docs ref |
|-------|------|----------|
| `blazar-core` | Shared contract types: `FrameRef`/`FrameDesc`, `EncodedUnit`/`TrackId`, `StreamParams`, `ModelIdentity`, `Target`, `Pts`/`Timebase`. No I/O. | §3, §3.1, §6, §7.1 |
| `blazar-calibration` | TOML calibration: schema, validating loader (per-file safe failure), specificity index with source tie-break + logging. | calibration-format.md; §5.1 |
| `blazar-preprocess` | Preprocess stage trait + CPU impl (RGB→YUV, range-normalize, 4:2:0 subsample, 8-bit quantize). Parameterized by `FrameDesc` (two ingest conventions). | §7 |
| `blazar-encoder` | `VideoEncoder` trait + `MockVideoEncoder` (full contract, no real codec yet); calibration→`EncoderConfig` translation; AV1 `av1C` derivation from a keyframe. Real NVENC backends for AV1, H.264 and HEVC behind the `nvenc` feature. | §3.1, §6 |
| `blazar-sink` | `Sink` trait + `Feedback` channel + `OrderGuard`; `DebugFileSink` (working, inspectable) and `Fmp4FileSink` backed by the forked `oxideav-mp4` muxer (AV1/H.264/HEVC + Opus/AAC). | §6 |
| `blazar-runtime` | Operator API (`Encoder::new` → `open_stream` → `submit_video`/`flush`/`close`) and the Phase-1 synchronous pipeline spine. | §3, §4, §9 |

## Phase 1 scope (what works today)

Host-frame ingest → CPU preprocess → mock encode → DebugFileSink, driven by the
default calibration. Video-only, but the types carry audio + PTS from day one so
audio is additive (§3.1). End-to-end in `blazar-runtime/tests/pipeline.rs`.

## Known Phase-1 boundaries (honest stubs, not fakes)

- `Fmp4FileSink` uses the pinned `oxideav-mp4` fork revision. It requires codec
  configuration extradata and supports AV1/H.264/HEVC video plus Opus/AAC audio.
  `Sink::flush()` emits an explicit fMP4 fragment; `close()` writes the final trailer.
- `MockVideoEncoder` implements the encoder *contract* (keyframe cadence, ordering,
  flush/finish) but does not compress. It stands in for SVT-AV1/rav1e until those
  backends land.
- The NVENC backends are real (`--features nvenc`) and hardware-tested on mosk:
AV1, H.264 and HEVC. H.264/HEVC input is Annex B from the driver; `nal.rs`
turns it into length-prefixed samples plus `avcC`/`hvcC`. For host ingest the
RGB→NV12 conversion runs as an NVRTC-compiled kernel on the GPU, inside the
encoder worker; `CpuPreprocess` remains for direct callers and other backends.
  It derives `av1C` from the first keyframe, forces IDRs itself (driver GOP
  boundaries produce intra-only frames, which are not sync samples), and does not
  support B-frames yet. `VideoEncoder::flush` cannot drain NVENC lookahead without
  end-of-stream; segment boundaries are handled by forcing a keyframe instead.
- Codec configuration is resolved *after* the first encode: `VideoEncoder::codec_config()`
  returns `av1C`/`avcC`/`hvcC` once the encoder has emitted a keyframe, and the sink is
  opened at that point. `Output::video_extradata` is an explicit override for backends
  without a real codec (the mock); a real encoder ignores nothing silently, so leave it
  `None`.
- Device (GPU/DLPack) ingest: the `FrameBuffer::Device` variant exists but the CPU
  preprocess rejects it. Device path is Phase 2 (and is where vLLM-Omni's zero-copy
  seam pays off, §7).

## Python bindings

`bindings/python` (PyO3 + maturin) exposes the runtime to Python:

```python
import blazar
cals = blazar.Calibrations.load("calibrations/default")
with blazar.Encoder(calibrations=cals, identity={...}, target="balanced",
                    backend="nvenc_av1", output="out.mp4") as stream:
    stream.submit(frame, width=W, height=H, fps=24.0,
                  layout="thwc", range="unit01")
```

Frames are C-contiguous float32 RGB; `layout` (`thwc` for a whole-clip seam,
`bcthw` for vLLM-Omni), `range` (`unit01`, `signed_unit`) and the geometry are
explicit per submit. Pull requests to "just figure it out" will be declined; the
seams do not tag any of it.

The Rust worker logs session start (backend, codec, target, calibration count),
periodic progress and a final summary (frames, payload bytes, max encode time) to
stderr. Set `BLAZAR_LOG=debug` for per-frame lines, `off` to silence it.
`examples/python/verify_encode.py out.mp4 --log run.log` cross-checks that
summary against the container: packet count must equal the worker's frame count
and the packets' payload must sum to the worker's byte count, so "Blazar encoded
this" is evidence rather than a claim.

Build and test instructions: `bindings/python/README.md`. Examples:

```sh
python examples/python/sglang_encode.py --fake      # swizzle seam, stand-in stage
python examples/python/vllm_omni_stream.py --fake   # chunked decode, timing output
```

Both run without the models (`--fake`) and take `--from-npy` for real frames.
`sglang_encode.py` exercises the device-output swizzle: it installs
`blazar.sglang_interop`'s hook on a stand-in stage with the same method shape as
SGLang's `DiffusersExecutionStage._postprocess_output`, so the install +
on-device postprocess + submit path runs without SGLang or a GPU (it falls back
to the host submit off CUDA, labelled, so the result is a real decodable file).
`--real` patches the actual SGLang class; the native Wan pipeline runs its
scheduler in a spawned child, so use the `BLAZAR_SWIZZLE=1` sidecar in
`bindings/python/sglang_patch` for that path instead.
`vllm_omni_stream.py --real` drives vLLM-Omni's own `WanStreamingDecoder` with
real Wan VAE weights on the GPU (needs `torch`, `diffusers`, and a vLLM-Omni
checkout; latents are noise, since the AR transformer is not part of that path).

`vllm_omni_generate.py` goes the other way: it generates with the real
vLLM-Omni engine (Wan2.1 on a 16 GiB card), takes the pipeline's frames, and
feeds them straight to Blazar. No intermediate video file, no second encode.
Neither is a live stream yet: the only sink is a file, so these measure the
encode path rather than deliver playback.


## Build & test

```
cargo build
cargo test        # 107 tests
cargo clippy --all-targets
```

### Hardware validation (manual)

The `nvenc` tests encode on a real GPU, so they never run in CI and no script in
this repo wraps them: the environment below is specific to the host. On a CUDA
machine with NVENC (bare metal or WSL2):

```
export CUDA_ROOT=/usr/local/cuda                       # your toolkit
export NVIDIA_VIDEO_CODEC_SDK_PATH=<dir with libnvidia-encode.so.1>
                                                       # WSL2: /usr/lib/wsl/lib
export LD_LIBRARY_PATH=$NVIDIA_VIDEO_CODEC_SDK_PATH:$CUDA_ROOT/lib64:$LD_LIBRARY_PATH

cargo test --workspace
cargo test -p blazar-encoder --features nvenc --test nvenc_hardware -- --test-threads=1
cargo test -p blazar-runtime --features nvenc --test nvenc_e2e -- --nocapture
```

That writes real fMP4s to `/tmp/blazar-nvenc-smoke.mp4` (320x240) and
`/tmp/blazar-nvenc-unaligned.mp4` (834x480, driver stride 896). Then verify the
media, not just the container: `ffprobe` the stream metadata and duration, decode
with `ffmpeg -xerror -i <file> -f null -`, and for the unaligned file decode one
frame to raw gray and check no row mixes neighbours. A container that parses is
not a stream that decodes.

Toolchain: the arm64 toolchain must be the rustup default on Apple Silicon
(`rustup default <version>-aarch64-apple-darwin`); the x86_64 one fails with
"Bad CPU type".
