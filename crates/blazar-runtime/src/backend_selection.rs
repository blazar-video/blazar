// SPDX-License-Identifier: Apache-2.0
//! Which backend serves a request, and what may be considered.
//!
//! `backend="auto"` picks from a pool the caller constrains: `hardware`,
//! `software`, or `any`. The distinction matters in both directions — a caller
//! who asks for hardware must never silently get software, and a caller who does
//! not care should get hardware when it is there.
//!
//! Selection is a fixed order over candidates that pass four filters, never a
//! runtime benchmark: the same request on the same machine must always resolve
//! to the same backend, or a corpus stops describing itself.

use blazar_core::{VideoCodec, VideoEncoderBackend};

/// Which implementations a request may be served by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendPool {
    /// Hardware first, software as the fallback.
    Any,
    /// Only hardware. Never falls back: the caller asked for a constraint.
    Hardware,
    /// Only software.
    Software,
}

impl core::str::FromStr for BackendPool {
    type Err = SelectionError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "any" => Ok(Self::Any),
            "hardware" => Ok(Self::Hardware),
            "software" => Ok(Self::Software),
            other => Err(SelectionError::UnknownPool(other.to_owned())),
        }
    }
}

impl BackendPool {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Any => "any",
            Self::Hardware => "hardware",
            Self::Software => "software",
        }
    }
}

/// Why a request could not be served.
#[derive(Debug, PartialEq, Eq)]
pub enum SelectionError {
    /// The pool allows nothing that supports this codec.
    NoCandidate {
        codec: VideoCodec,
        pool: BackendPool,
        rejected: Vec<Rejection>,
    },
    /// Anything that did support the codec was not built or not usable here.
    NoneAvailable {
        codec: VideoCodec,
        pool: BackendPool,
        rejected: Vec<Rejection>,
    },
    UnknownPool(String),
}

/// Why one backend was not used. Reported so a failure says what to fix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rejection {
    pub backend: VideoEncoderBackend,
    pub reason: RejectionReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectionReason {
    /// The backend exists but does not encode this codec.
    WrongCodec,
    /// Excluded by the pool (`software` asked for, hardware found, ...).
    OutsidePool,
    /// Not compiled into this wheel.
    NotBuilt,
    /// Compiled, but no device or no session could be opened.
    Unavailable,
}

impl core::fmt::Display for SelectionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        fn explain(rejected: &[Rejection]) -> String {
            rejected
                .iter()
                .map(|r| format!("{}: {:?}", backend_name(r.backend), r.reason))
                .collect::<Vec<_>>()
                .join(", ")
        }
        match self {
            Self::NoCandidate {
                codec,
                pool,
                rejected,
            } => write!(
                f,
                "no backend in pool {:?} encodes {codec:?} (considered: {})",
                pool.as_str(),
                explain(rejected)
            ),
            Self::NoneAvailable {
                codec,
                pool,
                rejected,
            } => write!(
                f,
                "every backend in pool {:?} that encodes {codec:?} was unusable (considered: {})",
                pool.as_str(),
                explain(rejected)
            ),
            Self::UnknownPool(value) => write!(
                f,
                "backend_pool must be \"any\", \"hardware\" or \"software\" (got {value:?})"
            ),
        }
    }
}

impl std::error::Error for SelectionError {}

/// A backend considered for this request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Candidate {
    pub backend: VideoEncoderBackend,
    /// Compiled into this wheel.
    pub built: bool,
    /// A session opens here (only meaningful when `built`).
    pub usable: bool,
}

impl Candidate {
    #[must_use]
    pub fn new(backend: VideoEncoderBackend, built: bool, usable: bool) -> Self {
        Self {
            backend,
            built,
            usable,
        }
    }
}

#[must_use]
pub fn backend_name(backend: VideoEncoderBackend) -> &'static str {
    match backend {
        VideoEncoderBackend::Mock => "mock",
        VideoEncoderBackend::SvtAv1 => "svtav1",
        VideoEncoderBackend::NvencAv1 => "nvenc_av1",
        VideoEncoderBackend::NvencH264 => "nvenc_h264",
        VideoEncoderBackend::NvencHevc => "nvenc_hevc",
        VideoEncoderBackend::X264 => "x264",
        VideoEncoderBackend::X265 => "x265",
        VideoEncoderBackend::Vp9 => "vp9",
        VideoEncoderBackend::VideoToolboxH264 => "videotoolbox_h264",
        VideoEncoderBackend::VideoToolboxHevc => "videotoolbox_hevc",
    }
}

/// Whether a backend runs on an accelerator.
#[must_use]
pub fn is_hardware(backend: VideoEncoderBackend) -> bool {
    matches!(
        backend,
        VideoEncoderBackend::NvencAv1
            | VideoEncoderBackend::NvencH264
            | VideoEncoderBackend::NvencHevc
            | VideoEncoderBackend::VideoToolboxH264
            | VideoEncoderBackend::VideoToolboxHevc
    )
}

/// Whether a backend can serve `codec`.
///
/// `mock` is special: it emits no compressed data and carries whatever codec the
/// caller names, so tests do not depend on real hardware or a real codec build.
#[must_use]
pub fn serves(backend: VideoEncoderBackend, codec: VideoCodec) -> bool {
    matches!(backend, VideoEncoderBackend::Mock) || codec_of(backend) == codec
}

/// The codec a backend encodes.
#[must_use]
pub fn codec_of(backend: VideoEncoderBackend) -> VideoCodec {
    match backend {
        VideoEncoderBackend::NvencAv1 | VideoEncoderBackend::SvtAv1 => VideoCodec::Av1,
        VideoEncoderBackend::NvencH264
        | VideoEncoderBackend::X264
        | VideoEncoderBackend::VideoToolboxH264 => VideoCodec::H264,
        VideoEncoderBackend::NvencHevc
        | VideoEncoderBackend::X265
        | VideoEncoderBackend::VideoToolboxHevc => VideoCodec::Hevc,
        VideoEncoderBackend::Vp9 => VideoCodec::Vp9,
        // The mock produces no compressed data; it serves whichever codec the
        // caller names, so tests do not depend on real hardware.
        VideoEncoderBackend::Mock => VideoCodec::H264,
    }
}

/// Fixed preference order. Hardware first: an accelerator leaves the CPU free
/// for the rest of the pipeline. Within a tier the order is arbitrary but fixed.
const ORDER: &[VideoEncoderBackend] = &[
    VideoEncoderBackend::NvencH264,
    VideoEncoderBackend::NvencHevc,
    VideoEncoderBackend::NvencAv1,
    VideoEncoderBackend::VideoToolboxH264,
    VideoEncoderBackend::VideoToolboxHevc,
    VideoEncoderBackend::SvtAv1,
    VideoEncoderBackend::X264,
    VideoEncoderBackend::X265,
    VideoEncoderBackend::Vp9,
];

/// Backends that could serve `codec` under `pool`, in preference order.
///
/// Availability is not knowable without opening a session, so the caller tries
/// these in turn and reports the ones that failed. `built` comes from the cargo
/// features, which are static.
#[must_use]
pub fn candidates_for(
    codec: VideoCodec,
    pool: BackendPool,
    built: bool,
) -> Vec<VideoEncoderBackend> {
    ORDER
        .iter()
        .copied()
        .filter(|b| serves(*b, codec))
        .filter(|b| match pool {
            BackendPool::Any => true,
            BackendPool::Hardware => is_hardware(*b),
            BackendPool::Software => !is_hardware(*b),
        })
        .filter(|b| built || !is_built(*b))
        .collect()
}

/// Whether this build has the backend compiled in.
#[must_use]
pub fn is_built(backend: VideoEncoderBackend) -> bool {
    match backend {
        VideoEncoderBackend::Mock => true,
        VideoEncoderBackend::NvencAv1
        | VideoEncoderBackend::NvencH264
        | VideoEncoderBackend::NvencHevc => cfg!(feature = "nvenc"),
        VideoEncoderBackend::SvtAv1 => cfg!(feature = "enc-svtav1"),
        VideoEncoderBackend::X264 => cfg!(feature = "enc-x264"),
        VideoEncoderBackend::X265 => cfg!(feature = "enc-x265"),
        VideoEncoderBackend::Vp9 => cfg!(feature = "enc-vp9"),
        VideoEncoderBackend::VideoToolboxH264 | VideoEncoderBackend::VideoToolboxHevc => {
            cfg!(all(feature = "enc-videotoolbox", target_os = "macos"))
        }
    }
}

/// Pick a backend for `codec` from `pool`.
///
/// Filters, in order: codec support, pool, built, usable. The first survivor of
/// the fixed [`ORDER`] wins. A `Hardware` pool that finds no usable hardware
/// returns an error rather than dropping to software: that is the constraint the
/// caller asked for.
pub fn select(
    codec: VideoCodec,
    pool: BackendPool,
    candidates: &[Candidate],
) -> Result<VideoEncoderBackend, SelectionError> {
    let mut rejected = Vec::new();
    let mut supported_any_pool = false;

    for backend in ORDER.iter().copied() {
        let candidate = candidates
            .iter()
            .find(|c| c.backend == backend)
            .copied()
            .unwrap_or(Candidate::new(backend, false, false));

        if !serves(backend, codec) {
            continue;
        }
        supported_any_pool = true;
        let hardware = is_hardware(backend);
        if matches!(pool, BackendPool::Hardware) && !hardware
            || matches!(pool, BackendPool::Software) && hardware
        {
            rejected.push(Rejection {
                backend,
                reason: RejectionReason::OutsidePool,
            });
            continue;
        }
        if !candidate.built {
            rejected.push(Rejection {
                backend,
                reason: RejectionReason::NotBuilt,
            });
            continue;
        }
        if !candidate.usable {
            rejected.push(Rejection {
                backend,
                reason: RejectionReason::Unavailable,
            });
            continue;
        }
        return Ok(backend);
    }

    if supported_any_pool {
        Err(SelectionError::NoneAvailable {
            codec,
            pool,
            rejected,
        })
    } else {
        Err(SelectionError::NoCandidate {
            codec,
            pool,
            rejected,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all(built: bool, usable: bool) -> Vec<Candidate> {
        ORDER
            .iter()
            .map(|b| Candidate::new(*b, built, usable))
            .collect()
    }

    fn only(backend: VideoEncoderBackend, built: bool, usable: bool) -> Vec<Candidate> {
        ORDER
            .iter()
            .map(|b| Candidate::new(*b, *b == backend && built, *b == backend && usable))
            .collect()
    }

    #[test]
    fn every_backend_encodes_exactly_one_codec() {
        // The pair matrix: a backend that claimed two codecs would make this
        // whole API ambiguous.
        for backend in ORDER.iter().copied() {
            assert!(!backend_name(backend).is_empty());
        }
        assert_eq!(codec_of(VideoEncoderBackend::X264), VideoCodec::H264);
        assert_eq!(codec_of(VideoEncoderBackend::X265), VideoCodec::Hevc);
        assert_eq!(codec_of(VideoEncoderBackend::SvtAv1), VideoCodec::Av1);
        assert_eq!(codec_of(VideoEncoderBackend::Vp9), VideoCodec::Vp9);
        assert_eq!(codec_of(VideoEncoderBackend::NvencHevc), VideoCodec::Hevc);
    }

    #[test]
    fn any_prefers_hardware_and_falls_back_to_software() {
        assert_eq!(
            select(VideoCodec::H264, BackendPool::Any, &all(true, true)),
            Ok(VideoEncoderBackend::NvencH264)
        );
        // No usable hardware: software serves the request.
        let no_hw: Vec<Candidate> = ORDER
            .iter()
            .map(|b| Candidate::new(*b, true, !is_hardware(*b)))
            .collect();
        assert_eq!(
            select(VideoCodec::H264, BackendPool::Any, &no_hw),
            Ok(VideoEncoderBackend::X264)
        );
    }

    #[test]
    fn hardware_pool_never_falls_back_to_software() {
        let no_hw: Vec<Candidate> = ORDER
            .iter()
            .map(|b| Candidate::new(*b, true, !is_hardware(*b)))
            .collect();
        let error = select(VideoCodec::H264, BackendPool::Hardware, &no_hw).unwrap_err();
        match error {
            SelectionError::NoneAvailable { rejected, .. } => assert!(
                rejected
                    .iter()
                    .any(|r| r.reason == RejectionReason::Unavailable),
                "the error must name the hardware that was not usable: {rejected:?}"
            ),
            other => panic!("expected NoneAvailable, got {other:?}"),
        }
    }

    #[test]
    fn software_pool_never_returns_hardware() {
        let picked = select(VideoCodec::Hevc, BackendPool::Software, &all(true, true)).unwrap();
        assert_eq!(picked, VideoEncoderBackend::X265);
        assert!(!is_hardware(picked));
    }

    #[test]
    fn a_codec_only_outside_the_pool_is_reported_as_unavailable() {
        // vp9 has exactly one backend and it is software, so asking for hardware
        // is a pool problem, not a missing-codec problem.
        let error = select(VideoCodec::Vp9, BackendPool::Hardware, &all(true, true)).unwrap_err();
        match error {
            SelectionError::NoneAvailable { rejected, .. } => assert!(
                rejected
                    .iter()
                    .any(|r| r.reason == RejectionReason::OutsidePool),
                "{rejected:?}"
            ),
            other => panic!("expected NoneAvailable, got {other:?}"),
        }
    }

    #[test]
    fn a_backend_that_is_not_built_is_skipped_with_a_reason() {
        let built_but_not_usable = only(VideoEncoderBackend::NvencH264, true, false);
        match select(VideoCodec::H264, BackendPool::Any, &built_but_not_usable).unwrap_err() {
            SelectionError::NoneAvailable { rejected, .. } => assert!(
                rejected
                    .iter()
                    .any(|r| r.backend == VideoEncoderBackend::NvencH264
                        && r.reason == RejectionReason::Unavailable),
                "the backend that was built but unusable must be named: {rejected:?}"
            ),
            other => panic!("expected NoneAvailable, got {other:?}"),
        }

        let not_built = only(VideoEncoderBackend::NvencH264, false, false);
        let error = select(VideoCodec::H264, BackendPool::Hardware, &not_built).unwrap_err();
        match error {
            SelectionError::NoneAvailable { rejected, .. } => {
                assert_eq!(rejected[0].reason, RejectionReason::NotBuilt)
            }
            other => panic!("expected NoneAvailable, got {other:?}"),
        }
    }

    #[test]
    fn selection_is_deterministic_across_repeated_calls() {
        let candidates = all(true, true);
        let first = select(VideoCodec::H264, BackendPool::Any, &candidates).unwrap();
        for _ in 0..5 {
            assert_eq!(
                select(VideoCodec::H264, BackendPool::Any, &candidates).unwrap(),
                first,
                "the same request must resolve to the same backend"
            );
        }
    }

    #[test]
    fn pools_parse_and_reject_typos() {
        assert_eq!("any".parse(), Ok(BackendPool::Any));
        assert_eq!("hardware".parse(), Ok(BackendPool::Hardware));
        assert_eq!("software".parse(), Ok(BackendPool::Software));
        assert_eq!(
            "Hardware".parse::<BackendPool>(),
            Err(SelectionError::UnknownPool("Hardware".to_owned()))
        );
    }
}
