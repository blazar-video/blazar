// SPDX-License-Identifier: Apache-2.0
//! Calibration resolution index (calibration-format §3, §4.1; architecture §5.1).
//!
//! Resolution rules, as finalized:
//! 1. A candidate is eligible only if its codec+target match exactly and every
//!    wildcard-capable axis matches the concrete query.
//! 2. Rank eligible candidates by **specificity** (count of non-wildcard axes).
//!    Specificity is the ONLY ranking axis.
//! 3. At **equal** specificity, `source` breaks the tie: User > Calibrator >
//!    Default. The tie-break is always LOGGED, never silent.
//! 4. A terminal default (full-wildcard) for each (codec, target) always
//!    resolves, so lookup never fails.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use blazar_core::{ModelIdentity, Target};
use thiserror::Error;

use crate::loader::{load_str, Calibration, LoadError};

/// A concrete lookup query built from the running stream.
#[derive(Debug, Clone)]
pub struct Query {
    pub family: String,
    pub version: String,
    pub quant: String,
    pub size: String,
    pub scene: String,
    pub codec: String,
    pub target: String,
}

impl Query {
    pub fn new(identity: &ModelIdentity, scene: &str, codec: &str, target: Target) -> Self {
        Self {
            family: identity.family.clone(),
            version: identity.version.clone(),
            quant: identity.quant.clone(),
            size: identity.size.clone(),
            scene: scene.to_string(),
            codec: codec.to_string(),
            target: target.as_key().to_string(),
        }
    }
}

/// Why a directory of calibrations could not be loaded as a whole.
#[derive(Debug, Error)]
pub enum LoadDirError {
    #[error("calibrations directory I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("no *.toml calibrations found in {path}")]
    Empty { path: PathBuf },

    #[error("invalid calibration file {path}: {source}")]
    InvalidFile {
        path: PathBuf,
        #[source]
        source: LoadError,
    },
}

/// An in-memory index of loaded calibrations for one (codec, target) space.
/// The runtime typically holds one index per active (codec, target) deployment,
/// but the index itself filters on codec+target so a shared index also works.
#[derive(Debug, Clone, Default)]
pub struct CalibrationIndex {
    entries: Vec<Calibration>,
}

impl CalibrationIndex {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn insert(&mut self, c: Calibration) {
        self.entries.push(c);
    }

    /// Load every `*.toml` file in `dir` into a new index.
    ///
    /// Strict on purpose: a malformed or invalid calibration fails the whole
    /// load and names the file. Embedders run this once at startup, where a typo
    /// should be loud rather than silently degrade a stream to the default. The
    /// runtime's own per-file "log and skip" policy (calibration-format §4.1) is
    /// a separate policy; a caller that wants it can load files individually.
    ///
    /// Files are read in sorted path order so the index is deterministic.
    /// Duplicate keys are allowed: resolution ranks by specificity, then source,
    /// and logs equal-specificity ties.
    pub fn load_dir(dir: impl AsRef<Path>) -> Result<Self, LoadDirError> {
        let dir = dir.as_ref();
        let read_dir = fs::read_dir(dir).map_err(|source| LoadDirError::Io {
            path: dir.to_path_buf(),
            source,
        })?;

        let mut paths: Vec<PathBuf> = Vec::new();
        for entry in read_dir {
            let entry = entry.map_err(|source| LoadDirError::Io {
                path: dir.to_path_buf(),
                source,
            })?;
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) == Some("toml") {
                paths.push(path);
            }
        }
        paths.sort();

        if paths.is_empty() {
            return Err(LoadDirError::Empty {
                path: dir.to_path_buf(),
            });
        }

        let mut index = Self::new();
        for path in paths {
            let text = fs::read_to_string(&path).map_err(|source| LoadDirError::Io {
                path: path.clone(),
                source,
            })?;
            let calibration = load_str(&text, path.display().to_string()).map_err(|source| {
                LoadDirError::InvalidFile {
                    path: path.clone(),
                    source,
                }
            })?;
            index.insert(calibration);
        }
        Ok(index)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn eligible(&self, q: &Query) -> Vec<&Calibration> {
        self.entries
            .iter()
            .filter(|c| {
                c.key.codec == q.codec
                    && c.key.target == q.target
                    && c.key.family.matches(&q.family)
                    && c.key.version.matches(&q.version)
                    && c.key.quant.matches(&q.quant)
                    && c.key.size.matches(&q.size)
                    && c.key.scene.matches(&q.scene)
            })
            .collect()
    }

    /// Resolve the best calibration for `q`, or `None` if nothing (not even a
    /// default) is eligible. A well-provisioned index always contains the
    /// terminal default, so `None` signals a provisioning bug, not a normal
    /// runtime state.
    pub fn resolve(&self, q: &Query) -> Option<&Calibration> {
        let eligible = self.eligible(q);
        let max_spec = eligible.iter().map(|c| c.key.specificity()).max()?;
        let mut top: Vec<&Calibration> = eligible
            .into_iter()
            .filter(|c| c.key.specificity() == max_spec)
            .collect();

        if top.len() == 1 {
            return Some(top[0]);
        }

        // Tie at equal specificity → source breaks it, and we LOG it.
        top.sort_by_key(|c| std::cmp::Reverse(c.source.rank()));
        let winner = *top.first()?;
        self.log_conflict(q, &top, winner);
        Some(winner)
    }

    fn log_conflict(&self, q: &Query, tied: &[&Calibration], winner: &Calibration) {
        // Human-readable key for the log line.
        let key = format!(
            "({}/{}/{}/{}, {}, {}, {})",
            q.family, q.version, q.quant, q.size, q.scene, q.codec, q.target
        );
        let mut lines = String::new();
        for c in tied {
            let mark = if std::ptr::eq(*c, winner) {
                "  [WINS]"
            } else {
                ""
            };
            lines.push_str(&format!("\n     {:?} {}{}", c.source, c.origin, mark));
        }
        tracing::warn!("calibration conflict at key {key}:{lines}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::load_str;

    fn cal(source: &str, scene: &str, origin: &str) -> Calibration {
        let txt = format!(
            r#"
schema_version = 1
source = "{source}"
[key]
model = {{ family = "wan", version = "2.1", quant = "*", size = "*" }}
scene = "{scene}"
codec = "svtav1"
target = "balanced"
[video.common]
rate_control = "crf"
crf = 30
keyint = 120
bframes = 3
pix_fmt = "yuv420p"
"#
        );
        load_str(&txt, origin).unwrap()
    }

    fn default_cal() -> Calibration {
        let txt = r#"
schema_version = 1
source = "default"
[key]
model = { family = "*", version = "*", quant = "*", size = "*" }
scene = "*"
codec = "svtav1"
target = "balanced"
[video.common]
rate_control = "crf"
crf = 30
keyint = 120
bframes = 3
pix_fmt = "yuv420p"
"#;
        load_str(txt, "default/svtav1_balanced.toml").unwrap()
    }

    fn query(scene: &str) -> Query {
        let id = ModelIdentity::new("wan", "2.1", "bf16", "1.3b");
        Query::new(&id, scene, "svtav1", Target::Balanced)
    }

    #[test]
    fn default_always_resolves() {
        let mut idx = CalibrationIndex::new();
        idx.insert(default_cal());
        // Query for a scene no specific calibration covers.
        let got = idx.resolve(&query("never_seen_scene")).unwrap();
        assert_eq!(got.source, crate::schema::Source::Default);
    }

    #[test]
    fn more_specific_beats_default() {
        let mut idx = CalibrationIndex::new();
        idx.insert(default_cal());
        idx.insert(cal("calibrator", "cinematic", "gen/cine.toml"));
        let got = idx.resolve(&query("cinematic")).unwrap();
        assert_eq!(
            got.key.scene,
            crate::schema::MatchAxis::Exact("cinematic".into())
        );
        assert_eq!(got.source, crate::schema::Source::Calibrator);
    }

    #[test]
    fn source_does_not_change_specificity() {
        // A user calibration at LOWER specificity must NOT beat a more-specific
        // generated one. This is the "bounded blast radius" invariant.
        let mut idx = CalibrationIndex::new();
        idx.insert(default_cal());
        // user, scene-agnostic (specificity: family+version = 2)
        idx.insert(cal("user", "*", "users/wan21_any.toml"));
        // calibrator, cinematic (specificity: family+version+scene = 3)
        idx.insert(cal("calibrator", "cinematic", "gen/cine.toml"));

        let got = idx.resolve(&query("cinematic")).unwrap();
        // The more-specific calibrator entry wins despite user being present.
        assert_eq!(got.source, crate::schema::Source::Calibrator);
        assert_eq!(
            got.key.scene,
            crate::schema::MatchAxis::Exact("cinematic".into())
        );
    }

    #[test]
    fn user_wins_tie_at_equal_specificity() {
        // Same specificity (both family+version+scene=3) → user wins the tie.
        let mut idx = CalibrationIndex::new();
        idx.insert(cal("calibrator", "cinematic", "gen/cine.toml"));
        idx.insert(cal("user", "cinematic", "users/cine.toml"));
        let got = idx.resolve(&query("cinematic")).unwrap();
        assert_eq!(got.source, crate::schema::Source::User);
        assert_eq!(got.origin, "users/cine.toml");
    }

    #[test]
    fn empty_index_resolves_to_none() {
        let idx = CalibrationIndex::new();
        assert!(idx.resolve(&query("cinematic")).is_none());
    }

    #[test]
    fn wrong_codec_is_not_eligible() {
        let mut idx = CalibrationIndex::new();
        idx.insert(default_cal()); // svtav1
        let id = ModelIdentity::new("wan", "2.1", "bf16", "1.3b");
        let q = Query::new(&id, "cinematic", "x265", Target::Balanced);
        assert!(idx.resolve(&q).is_none());
    }
}
