//! `blazar-core` — the shared contract types for the Blazar runtime.
//!
//! No I/O, no CUDA, no encoders. Just the value objects every other crate
//! agrees on: frame handoff ([`frame`]), timing ([`timebase`]), encoded output
//! ([`unit`]), model identity ([`identity`]), operator target ([`target`]), and
//! sink/stream parameters ([`stream_params`]).
//!
//! These types are deliberately audio-aware from day one (architecture §3.1):
//! Phase 1 is video-only, but [`unit::TrackId`], [`stream_params::AudioTrackParams`],
//! and the shared [`timebase`] all carry audio so it is additive, not a retrofit.

pub mod error;
pub mod frame;
pub mod identity;
pub mod stream_params;
pub mod target;
pub mod timebase;
pub mod unit;

pub use error::{CoreError, Result};
pub use frame::{
    BufferOwner, ColorRange, ColorSpace, CudaEvent, CudaStream, Dtype, FrameBuffer, FrameDesc,
    FrameRef, IpcHandle, Layout,
};
pub use identity::ModelIdentity;
pub use stream_params::{
    AudioCodec, AudioTrackParams, Container, StreamParams, VideoCodec, VideoTrackParams,
};
pub use target::Target;
pub use timebase::{Pts, Timebase};
pub use unit::{EncodedUnit, TrackId};
