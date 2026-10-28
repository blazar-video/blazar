// SPDX-License-Identifier: Apache-2.0
//! Keyframe cadence for backends whose driver-scheduled GOP is not enough.
//!
//! NVENC's `gopLength` boundary produces an AV1 INTRA_ONLY frame on this driver,
//! and INTRA_ONLY is not a random-access point (not an ISOBMFF sync sample).
//! Real IDRs have to be requested explicitly. This schedule decides when, so the
//! policy is unit-tested instead of buried in the hardware loop.
//!
//! Cadence is "keyframe on frame 0, then every `keyint` frames", with a caller
//! request (scene cut / PLI) resetting the count.

/// Tracks when the next frame must be forced to an IDR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyframeSchedule {
    keyint: u32,
    frames_since_key: u32,
}

impl KeyframeSchedule {
    /// Start a schedule. The first submitted frame is always a keyframe so the
    /// container gets its codec configuration immediately.
    pub fn new(keyint: u32) -> Self {
        let keyint = keyint.max(1);
        Self {
            keyint,
            // Due immediately.
            frames_since_key: keyint,
        }
    }

    /// Decide whether the next frame is a keyframe, advancing the schedule.
    ///
    /// `caller_requested` is a scene cut or a transport PLI. It wins, and
    /// restarts the cadence so a forced keyframe is not followed immediately by
    /// a scheduled one.
    pub fn next_is_keyframe(&mut self, caller_requested: bool) -> bool {
        let scheduled = self.frames_since_key >= self.keyint;
        let keyframe = caller_requested || scheduled;
        if keyframe {
            // This frame is the first of the new GOP, so the next frame is one
            // frame away from the keyframe, not zero.
            self.frames_since_key = 1;
        } else {
            self.frames_since_key += 1;
        }
        keyframe
    }

    pub fn keyint(&self) -> u32 {
        self.keyint
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(schedule: &mut KeyframeSchedule, frames: u32) -> Vec<bool> {
        (0..frames)
            .map(|_| schedule.next_is_keyframe(false))
            .collect()
    }

    #[test]
    fn first_frame_is_always_a_keyframe() {
        let mut schedule = KeyframeSchedule::new(12);
        assert!(schedule.next_is_keyframe(false));
    }

    #[test]
    fn cadence_matches_keyint() {
        let mut schedule = KeyframeSchedule::new(4);
        assert_eq!(
            run(&mut schedule, 10),
            vec![true, false, false, false, true, false, false, false, true, false]
        );
    }

    #[test]
    fn caller_request_forces_a_keyframe_and_restarts_the_cadence() {
        let mut schedule = KeyframeSchedule::new(4);
        assert!(schedule.next_is_keyframe(false)); // frame 0
        assert!(!schedule.next_is_keyframe(false)); // 1
        assert!(schedule.next_is_keyframe(true)); // forced at 2
        assert!(!schedule.next_is_keyframe(false)); // 3
        assert!(!schedule.next_is_keyframe(false)); // 4
        assert!(!schedule.next_is_keyframe(false)); // 5
        assert!(schedule.next_is_keyframe(false)); // 6 = 2 + keyint
    }

    #[test]
    fn keyint_zero_is_treated_as_every_frame() {
        let mut schedule = KeyframeSchedule::new(0);
        assert_eq!(schedule.keyint(), 1);
        assert_eq!(run(&mut schedule, 4), vec![true, true, true, true]);
    }
}
