use std::collections::HashMap;
use std::hash::Hash;

use crate::join_semilattice::BoundedJoinSemilattice;
use crate::join_semilattice::JoinSemilattice;

/// Internal lattice state of a grow-only counter.
///
/// This is the classic **GCounter** lattice: a map from replica IDs
/// to non-decreasing counts. The join is pointwise `max` on the
/// per-replica components.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GCounterState<Id>
where
    Id: Eq + Hash,
{
    counts: HashMap<Id, u64>,
}

impl<Id> GCounterState<Id>
where
    Id: Eq + Hash + Clone,
{
    // Create an empty state (no replicas, logically all zero).
    pub fn new() -> Self {
        Self {
            counts: HashMap::new(),
        }
    }

    // Reads raw per-replica counts.
    pub fn counts(&self) -> &HashMap<Id, u64> {
        &self.counts
    }

    /// Total value of the counter, interpreted as: the sum of all
    /// per-replica increment counts observed so far.
    pub fn value(&self) -> u64 {
        self.counts.values().copied().sum()
    }

    /// Local, monotone increment for a given replica ID.
    ///
    /// This does **not** enforce any replica discipline; that is the
    /// job of the higher-level [`GCounter`]. Here, we only ensure the
    /// component is monotonically increasing.
    pub fn inc_for(&mut self, id: &Id, delta: u64) {
        let entry = self.counts.entry(id.clone()).or_insert(0);
        *entry = entry.saturating_add(delta);
    }
}

impl<Id> JoinSemilattice for GCounterState<Id>
where
    Id: Eq + Hash + Clone,
{
    fn join(&self, other: &Self) -> Self {
        let mut out = self.counts.clone();
        for (id, &n_other) in &other.counts {
            out.entry(id.clone())
                .and_modify(|n_here| {
                    if *n_here < n_other {
                        *n_here = n_other;
                    }
                })
                .or_insert(n_other);
        }
        GCounterState { counts: out }
    }
}

impl<Id> BoundedJoinSemilattice for GCounterState<Id>
where
    Id: Eq + Hash + Clone,
{
    // Bottom = empty map (all components implicitly 0).
    fn bottom() -> Self {
        Self::new()
    }
}

/// A classic **grow-only counter CRDT (GCounter)**.
///
/// Each replica has:
/// - a unique `Id` (e.g. string, UUID, u64),
/// - a local counter component in the lattice state,
/// - a `join`-based merge with other replicas' states.
///
/// The observable value is the **sum of all per-replica components**.
/// Updates are monotone and merges are
/// associative/commutative/idempotent, so replicas converge under
/// arbitrary message reordering and duplication.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GCounter<Id>
where
    Id: Eq + Hash + Clone,
{
    id: Id,
    state: GCounterState<Id>,
}

impl<Id> GCounter<Id>
where
    Id: Eq + Hash + Clone,
{
    // Create a new GCounter for this replica ID.
    pub fn new(id: Id) -> Self {
        Self {
            id,
            state: GCounterState::bottom(),
        }
    }

    // Replica ID for this counter.
    pub fn id(&self) -> &Id {
        &self.id
    }

    // Current value (sum of all components).
    pub fn value(&self) -> u64 {
        self.state.value()
    }

    // Read the underlying lattice (for replication).
    pub fn state(&self) -> &GCounterState<Id> {
        &self.state
    }

    /// Monotone local increment on **this replica**.
    pub fn inc(&mut self, delta: u64) {
        self.state.inc_for(&self.id, delta);
    }

    /// Merge a remote state into this replica using lattice `join`.
    pub fn merge(&mut self, remote: &GCounterState<Id>) {
        self.state = self.state.join(remote);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gcounter_local_inc() {
        let mut c = GCounter::new("A");
        assert_eq!(c.value(), 0);

        c.inc(3);
        c.inc(2);
        assert_eq!(c.value(), 5);
    }

    #[test]
    fn gcounter_merge_converges() {
        // Two replicas A and B.
        let mut a = GCounter::new("A");
        let mut b = GCounter::new("B");

        // Local updates.
        a.inc(3); // A: 3
        b.inc(5); // B: 5

        // Exchange states (in any order).
        let a_state = a.state().clone();
        let b_state = b.state().clone();

        a.merge(&b_state);
        b.merge(&a_state);

        assert_eq!(a.value(), 8);
        assert_eq!(b.value(), 8);

        // More local updates after merge.
        a.inc(2); // A: 5, B: 5
        let a_state2 = a.state().clone();

        b.merge(&a_state2);
        assert_eq!(a.value(), 10);
        assert_eq!(b.value(), 10);
    }

    #[test]
    fn gcounter_merge_is_idempotent_and_commutative() {
        let mut a = GCounter::new(1u32);
        let mut b = GCounter::new(2u32);

        a.inc(1);
        b.inc(2);

        let s1 = a.state().clone();
        let s2 = b.state().clone();

        // A merge B
        let mut a1 = a.clone();
        a1.merge(&s2);

        // B merge A
        let mut b1 = b.clone();
        b1.merge(&s1);

        // A merge B twice
        let mut a2 = a.clone();
        a2.merge(&s2);
        a2.merge(&s2);

        assert_eq!(a1.value(), b1.value());
        assert_eq!(a1.value(), a2.value());
    }
}
