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
| `blazar-encoder` | `VideoEncoder` trait + `MockVideoEncoder` (full contract, no real codec yet); calibration→`EncoderConfig` translation. Real SVT-AV1 behind the `svtav1` feature (not yet implemented). | §3.1, §6 |
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
  flush/finish) but does not compress. The real encoder backends must satisfy the same
  contract; NVENC is the next backend milestone.
- Device (GPU/DLPack) ingest: the `FrameBuffer::Device` variant exists but the CPU
  preprocess rejects it. Device path is Phase 2 (and is where vLLM-Omni's zero-copy
  seam pays off, §7).

## Build & test

```
cargo build
cargo test        # 59 tests
cargo clippy --all-targets
```

Toolchain: the arm64 toolchain must be the rustup default on Apple Silicon
(`rustup default <version>-aarch64-apple-darwin`); the x86_64 one fails with
"Bad CPU type".
