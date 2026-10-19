//! NVENC AV1 backend.
//!
//! Split into a testable core and a thin hardware shell:
//!
//! * [`config`] — calibration → NVENC parameter policy (no SDK types).
//! * [`keyframe`] — IDR cadence (driver GOPs are intra-only, not sync samples).
//! * [`nv12`] — planar YUV 4:2:0 → NV12 with explicit row pitch.
//! * [`pool`] — output-bitstream slot state machine for the async pipeline.
//! * `worker` — the SDK-facing session, only compiled with the `nvenc` feature.

pub mod config;
pub mod keyframe;
pub mod nv12;
pub mod pool;

#[cfg(feature = "nvenc")]
mod worker;

#[cfg(feature = "nvenc")]
pub use worker::NvencEncoder;
