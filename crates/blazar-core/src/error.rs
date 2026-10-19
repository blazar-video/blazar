//! Shared error type for the core contracts.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("unsupported frame layout for this operation: {0:?}")]
    UnsupportedLayout(crate::frame::Layout),

    #[error("unsupported dtype for this operation: {0:?}")]
    UnsupportedDtype(crate::frame::Dtype),

    #[error("frame buffer residency mismatch: {0}")]
    ResidencyMismatch(String),

    #[error("frame bytes ({got}) do not match descriptor ({expected}) for {w}x{h}")]
    SizeMismatch {
        got: usize,
        expected: usize,
        w: u32,
        h: u32,
    },
}

pub type Result<T> = core::result::Result<T, CoreError>;
