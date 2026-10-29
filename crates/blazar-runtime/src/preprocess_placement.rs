// SPDX-License-Identifier: Apache-2.0
//! Where RGB→NV12 happens, and what that implies for the caller.
//!
//! Placement is chosen by the caller (`preprocess="auto" | "host" | "device"`) but
//! the *transfers* are never the caller's problem: whichever site is chosen,
//! the runtime performs whatever copy that site requires. So a placement is a
//! cost, not a contract about where the tensor lives.
//!
//! [`resolve`] is pure — no CUDA, no encoder — because the resolution rule is
//! the part that is easy to get subtly wrong and the part callers rely on when
//! they assert `encoder.preprocess`.

use std::str::FromStr;

/// What the caller asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreprocessPlacement {
    /// Pick from where the frame already is. Never moves data needlessly.
    Auto,
    /// Always convert on the host, downloading a device tensor first.
    Host,
    /// Always convert on the device, uploading a host frame first.
    Device,
}

/// Where the conversion actually runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreprocessSite {
    Host,
    Device,
}

/// Where the frame already is when `submit` is called.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceLocation {
    Host,
    Device,
}

/// Why a placement could not be honoured. Never a silent substitution: the two
/// cases here are a caller error and an unavailable build, both of which the
/// caller can act on.
#[derive(Debug, PartialEq, Eq)]
pub enum PlacementError {
    /// `gpu` was requested but this build or this machine cannot do it.
    ///
    /// The two causes need different answers from the caller — rebuild with the
    /// feature, or use a machine with a driver — so they are distinguished
    /// rather than reported as "unavailable".
    DeviceUnavailable {
        /// True when the feature is compiled in and the driver is what is missing.
        driver_missing: bool,
    },
    /// Not one of `auto`, `cpu`, `gpu`.
    Unknown(String),
}

impl core::fmt::Display for PlacementError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PlacementError::DeviceUnavailable {
                driver_missing: false,
            } => write!(
                f,
                "preprocess=\"gpu\" needs the gpu-preprocess feature; build with \
                 --features gpu-preprocess or pass preprocess=\"cpu\""
            ),
            PlacementError::DeviceUnavailable {
                driver_missing: true,
            } => write!(
                f,
                "preprocess=\"gpu\" needs a CUDA driver, which is not loadable on this \
                 machine; pass preprocess=\"cpu\" or \"auto\""
            ),
            PlacementError::Unknown(value) => write!(
                f,
                "preprocess must be one of \"auto\", \"cpu\", \"gpu\" (got {value:?})"
            ),
        }
    }
}

impl std::error::Error for PlacementError {}

impl FromStr for PreprocessPlacement {
    type Err = PlacementError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "auto" => Ok(Self::Auto),
            "host" => Ok(Self::Host),
            "device" => Ok(Self::Device),
            other => Err(PlacementError::Unknown(other.to_owned())),
        }
    }
}

impl PreprocessPlacement {
    /// The string a caller passed, so `encoder.preprocess` round-trips.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Host => "host",
            Self::Device => "device",
        }
    }
}

impl PreprocessSite {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Host => "host",
            Self::Device => "device",
        }
    }
}

/// Decide where the conversion runs.
///
/// `Auto` follows the frame where the backend can use the device: a device
/// tensor converts on the device, and so does a host frame when the backend
/// converts into memory it already has registered (`NVENC` writes the kernel
/// output straight into the pitched surface it encodes from, so the upload is
/// the only transfer). A CPU codec has no such surface, so converting on the
/// device would add a download — measured 6.78 ms/frame on the CPU against
/// 7.98 with the round trip, and `auto` picks the cheaper one.
///
/// When GPU preprocessing is unavailable, a device tensor still falls back to
/// `Host`: that path needs a download, not a device kernel.
pub fn resolve(
    requested: PreprocessPlacement,
    source: SourceLocation,
    backend_converts_on_device: bool,
    gpu_available: bool,
) -> Result<PreprocessSite, PlacementError> {
    match requested {
        PreprocessPlacement::Host => Ok(PreprocessSite::Host),
        PreprocessPlacement::Device if !gpu_available => Err(PlacementError::DeviceUnavailable {
            // `resolve` is told whether the feature exists separately from
            // whether the driver loads; the caller passes the conjunction, so
            // ask the build which half failed.
            driver_missing: cfg!(feature = "gpu-preprocess"),
        }),
        PreprocessPlacement::Device => Ok(PreprocessSite::Device),
        PreprocessPlacement::Auto => Ok(match source {
            _ if !gpu_available => PreprocessSite::Host,
            SourceLocation::Device => PreprocessSite::Device,
            SourceLocation::Host if backend_converts_on_device => PreprocessSite::Device,
            SourceLocation::Host => PreprocessSite::Host,
        }),
    }
}

/// Whether honouring this placement implies a transfer the caller did not make
/// themselves. Logged once per stream so a placement cost is visible rather
/// than inferred from timings.
#[must_use]
pub fn implies_copy(site: PreprocessSite, source: SourceLocation) -> bool {
    matches!(
        (site, source),
        (PreprocessSite::Device, SourceLocation::Host)
            | (PreprocessSite::Host, SourceLocation::Device)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const GPU: bool = true;
    const NO_GPU: bool = false;
    /// NVENC: the kernel writes into the surface being encoded, so a host frame
    /// needs one upload and no download.
    const DEVICE_CONVERT: bool = true;
    /// x264/x265/SVT-AV1: no registered surface, so a device conversion has to
    /// be downloaded before the CPU encoder sees it.
    const HOST_EXPECTED: bool = false;

    #[test]
    fn auto_sends_a_device_tensor_to_the_device() {
        for backend_converts_on_device in [DEVICE_CONVERT, HOST_EXPECTED] {
            assert_eq!(
                resolve(
                    PreprocessPlacement::Auto,
                    SourceLocation::Device,
                    backend_converts_on_device,
                    GPU
                ),
                Ok(PreprocessSite::Device)
            );
        }
    }

    #[test]
    fn auto_converts_a_host_frame_on_the_device_when_the_backend_can_use_it() {
        assert_eq!(
            resolve(
                PreprocessPlacement::Auto,
                SourceLocation::Host,
                DEVICE_CONVERT,
                GPU
            ),
            Ok(PreprocessSite::Device),
            "NVENC encodes from the surface the kernel wrote, so there is no download to avoid"
        );
    }

    #[test]
    fn auto_converts_a_host_frame_on_the_cpu_for_a_cpu_codec() {
        assert_eq!(
            resolve(
                PreprocessPlacement::Auto,
                SourceLocation::Host,
                HOST_EXPECTED,
                GPU
            ),
            Ok(PreprocessSite::Host),
            "a device conversion for a CPU codec buys a download it cannot use"
        );
    }

    #[test]
    fn auto_falls_back_to_the_cpu_without_gpu_preprocessing() {
        for source in [SourceLocation::Host, SourceLocation::Device] {
            for backend_converts_on_device in [DEVICE_CONVERT, HOST_EXPECTED] {
                assert_eq!(
                    resolve(
                        PreprocessPlacement::Auto,
                        source,
                        backend_converts_on_device,
                        NO_GPU
                    ),
                    Ok(PreprocessSite::Host)
                );
            }
        }
    }

    #[test]
    fn cpu_is_honoured_for_either_location() {
        for source in [SourceLocation::Host, SourceLocation::Device] {
            for gpu_available in [GPU, NO_GPU] {
                assert_eq!(
                    resolve(
                        PreprocessPlacement::Host,
                        source,
                        DEVICE_CONVERT,
                        gpu_available
                    ),
                    Ok(PreprocessSite::Host),
                    "cpu placement downloads a device tensor rather than refusing"
                );
            }
        }
    }

    #[test]
    fn gpu_is_honoured_for_either_location() {
        for source in [SourceLocation::Host, SourceLocation::Device] {
            assert_eq!(
                resolve(PreprocessPlacement::Device, source, DEVICE_CONVERT, GPU),
                Ok(PreprocessSite::Device)
            );
        }
    }

    #[test]
    fn gpu_without_the_feature_is_an_error_not_a_fallback() {
        for source in [SourceLocation::Host, SourceLocation::Device] {
            assert_eq!(
                resolve(PreprocessPlacement::Device, source, DEVICE_CONVERT, NO_GPU),
                Err(PlacementError::DeviceUnavailable {
                    driver_missing: cfg!(feature = "gpu-preprocess"),
                })
            );
        }
    }

    #[test]
    fn the_gpu_unavailable_message_names_the_thing_that_is_missing() {
        let compiled = PlacementError::DeviceUnavailable {
            driver_missing: false,
        }
        .to_string();
        assert!(compiled.contains("feature"), "{compiled}");
        let no_driver = PlacementError::DeviceUnavailable {
            driver_missing: true,
        }
        .to_string();
        assert!(no_driver.contains("CUDA driver"), "{no_driver}");
    }

    #[test]
    fn transfers_are_reported_only_when_placement_moves_data() {
        assert!(!implies_copy(PreprocessSite::Host, SourceLocation::Host));
        assert!(!implies_copy(
            PreprocessSite::Device,
            SourceLocation::Device
        ));
        assert!(implies_copy(PreprocessSite::Host, SourceLocation::Device));
        assert!(implies_copy(PreprocessSite::Device, SourceLocation::Host));
    }

    #[test]
    fn placement_strings_round_trip_and_reject_typos() {
        for (text, placement) in [
            ("auto", PreprocessPlacement::Auto),
            ("host", PreprocessPlacement::Host),
            ("device", PreprocessPlacement::Device),
        ] {
            assert_eq!(text.parse(), Ok(placement));
            assert_eq!(placement.as_str(), text);
        }
        assert_eq!(
            "CPU".parse::<PreprocessPlacement>(),
            Err(PlacementError::Unknown("CPU".to_owned())),
            "case matters: a silent accept hides a typo'd deployment"
        );
        assert_eq!(
            "cuda".parse::<PreprocessPlacement>(),
            Err(PlacementError::Unknown("cuda".to_owned()))
        );
    }
}
