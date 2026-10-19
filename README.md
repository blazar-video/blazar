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
| `blazar-encoder` | `VideoEncoder` trait + `MockVideoEncoder` (full contract, no real codec yet); calibration→`EncoderConfig` translation; AV1 `av1C` derivation from a keyframe. Real NVENC-AV1 behind the `nvenc` feature. | §3.1, §6 |
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
- The NVENC-AV1 backend is real (`--features nvenc`) and hardware-tested on mosk.
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
