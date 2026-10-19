//! Output-bitstream slot bookkeeping for the asynchronous NVENC pipeline.
//!
//! NVENC's output is not necessarily the slot you just submitted: with
//! lookahead it buffers frames, returns `NeedMoreInput`, and later completes
//! them in submission order. Locking a slot that has not been filled returns
//! **stale data from its previous use**, not an error, so the backend must know
//! which slots are in flight and what timestamp each one is expected to carry.
//!
//! This type is deliberately free of SDK types so the state machine can be
//! tested without a GPU. The worker owns one `SlotPool` alongside its vector of
//! `Bitstream`s; slot index `i` refers to `bitstreams[i]`.

use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotState {
    Free,
    /// Claimed for a slot whose frame has not been submitted to the encoder
    /// yet. A batched conversion reserves every destination before the kernel
    /// runs, but only submission order matters for draining.
    Reserved,
    InFlight { expected_pts: u64 },
}

/// A fixed set of output slots plus the order in which they were submitted.
#[derive(Debug)]
pub struct SlotPool {
    slots: Vec<SlotState>,
    /// Submission order: the order outputs are expected to complete in.
    order: VecDeque<usize>,
}

impl SlotPool {
    /// Create a pool of at least one slot. One slot is the minimum that can
    /// encode a frame at all; callers that need lookahead choose more.
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            slots: vec![SlotState::Free; capacity],
            order: VecDeque::with_capacity(capacity),
        }
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Claim a free slot for a frame with this presentation timestamp.
    /// Returns `None` when every slot is in flight, which means the caller must
    /// drain before submitting more.
    pub fn acquire(&mut self, expected_pts: u64) -> Option<usize> {
        let slot = self.reserve()?;
        self.submit(slot, expected_pts);
        Some(slot)
    }

    /// Claim a free slot without submitting anything.
    ///
    /// A batch reserves all of its destinations before the kernel launch, but
    /// `drain_ready` must not touch them until the frames are actually handed
    /// to the encoder.
    pub fn reserve(&mut self) -> Option<usize> {
        let slot = self
            .slots
            .iter()
            .position(|state| *state == SlotState::Free)?;
        self.slots[slot] = SlotState::Reserved;
        Some(slot)
    }

    /// Move a reserved slot into the submission order.
    pub fn submit(&mut self, slot: usize, expected_pts: u64) {
        if let Some(state) = self.slots.get_mut(slot) {
            *state = SlotState::InFlight { expected_pts };
            self.order.push_back(slot);
        }
    }

    /// Give a reserved slot back without submitting it.
    pub fn unreserve(&mut self, slot: usize) {
        if let Some(state) = self.slots.get_mut(slot) {
            if *state == SlotState::Reserved {
                *state = SlotState::Free;
            }
        }
    }

    /// Mark a slot as delivered. Safe to call on a free slot (no-op).
    pub fn release(&mut self, slot: usize) {
        if let Some(state) = self.slots.get_mut(slot) {
            if *state != SlotState::Free {
                *state = SlotState::Free;
                self.order.retain(|candidate| *candidate != slot);
            }
        }
    }

    /// In-flight slots in submission order, paired with the PTS each one must
    /// report when it is finally lockable.
    pub fn in_flight(&self) -> Vec<(usize, u64)> {
        self.order
            .iter()
            .filter_map(|slot| match self.slots[*slot] {
                SlotState::InFlight { expected_pts } => Some((*slot, expected_pts)),
                SlotState::Free | SlotState::Reserved => None,
            })
            .collect()
    }

    pub fn in_flight_count(&self) -> usize {
        self.order.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquire_hands_out_distinct_slots_in_submission_order() {
        let mut pool = SlotPool::new(3);
        assert_eq!(pool.acquire(100), Some(0));
        assert_eq!(pool.acquire(200), Some(1));
        assert_eq!(pool.acquire(300), Some(2));
        assert_eq!(pool.in_flight(), vec![(0, 100), (1, 200), (2, 300)]);
        assert_eq!(pool.in_flight_count(), 3);
    }

    #[test]
    fn exhausted_pool_reports_none_instead_of_reusing_a_slot() {
        let mut pool = SlotPool::new(2);
        assert_eq!(pool.acquire(0), Some(0));
        assert_eq!(pool.acquire(1), Some(1));
        assert_eq!(pool.acquire(2), None);
    }

    #[test]
    fn released_slots_are_reused_and_drop_out_of_the_order() {
        let mut pool = SlotPool::new(2);
        let a = pool.acquire(10).unwrap();
        let b = pool.acquire(20).unwrap();
        pool.release(a);
        assert_eq!(pool.in_flight(), vec![(b, 20)]);
        // The freed slot is reused before growing the order.
        assert_eq!(pool.acquire(30), Some(a));
        assert_eq!(pool.in_flight(), vec![(b, 20), (a, 30)]);
    }

    #[test]
    fn releasing_a_free_slot_is_a_no_op() {
        let mut pool = SlotPool::new(2);
        pool.release(0);
        pool.release(99);
        assert_eq!(pool.in_flight_count(), 0);
        assert_eq!(pool.acquire(1), Some(0));
    }

    #[test]
    fn a_pool_never_has_zero_slots() {
        let mut pool = SlotPool::new(0);
        assert_eq!(pool.capacity(), 1);
        assert_eq!(pool.acquire(0), Some(0));
        assert_eq!(pool.acquire(1), None);
    }
}
