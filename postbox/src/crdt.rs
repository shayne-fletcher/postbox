//! Basic **state-based CRDTs** built on lattice primitives.
//!
//! This module provides a small collection of classic convergent
//! replicated data types implemented in terms of [`JoinSemilattice`]
//! / [`BoundedJoinSemilattice`] from [`crate::join_semilattice`]. All
//! of these are *state-based* CRDTs: replicas exchange full lattice
//! states and merge them with `join`, guaranteeing convergence under
//! arbitrary message reordering and duplication.
//!
//! Included types:
//!
//! - [`GCounterState`] / [`GCounter`]:
//!   grow-only counter where each replica maintains a per-replica
//!   component; merge is pointwise `max`, and the logical value is
//!   the sum of all components.
//!
//! - [`PNCounterState`] / [`PNCounter`]:
//!   a **PN-Counter** built from two GCounters (`p` for increments,
//!   `n` for decrements), with logical value `sum(p) - sum(n)`.
//!
//! - [`GSet`]:
//!   grow-only set where updates only **add** elements and merges use
//!   set union.
use std::collections::HashMap;
use std::collections::HashSet;
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
    /// Lattice join for GCounter state: pointwise `max` over
    /// per-replica counts.
    ///
    /// Each replica ID maps to the **largest** count we have ever
    /// seen for that replica, from either side. Concretely, for every
    /// `(id → n)` in `other`, we update `out[id]` to `max(out[id],
    /// n)` (or insert `n` if `id` is new). This makes `join`
    /// associative, commutative, and idempotent, which is exactly
    /// what we need for a state-based CRDT merge.
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

/// Internal lattice state of a **PN-Counter**.
///
/// A PN-Counter is represented as two grow-only counters:
/// - `p`: counts increments
/// - `n`: counts decrements
///
/// The lattice order is componentwise, and the join is just joining
/// each component (`p` and `n`) using their lattice join.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PNCounterState<Id>
where
    Id: Eq + Hash,
{
    pub(crate) p: GCounterState<Id>,
    pub(crate) n: GCounterState<Id>,
}

impl<Id> PNCounterState<Id>
where
    Id: Eq + Hash + Clone,
{
    /// Create a zero-initialized PN state (p = 0, n = 0).
    pub fn new() -> Self {
        Self {
            p: GCounterState::bottom(),
            n: GCounterState::bottom(),
        }
    }

    /// Logical value = sum(p) - sum(n).
    pub fn value(&self) -> i64 {
        self.p.value() as i64 - self.n.value() as i64
    }
}

impl<Id> JoinSemilattice for PNCounterState<Id>
where
    Id: Eq + Hash + Clone,
{
    /// Lattice join: componentwise join on `p` and `n`.
    fn join(&self, other: &Self) -> Self {
        Self {
            p: self.p.join(&other.p),
            n: self.n.join(&other.n),
        }
    }
}

impl<Id> BoundedJoinSemilattice for PNCounterState<Id>
where
    Id: Eq + Hash + Clone,
{
    /// Bottom = both components at bottom (all zeros).
    fn bottom() -> Self {
        Self::new()
    }
}

/// A **PN-Counter** CRDT: supports increments *and* decrements.
///
/// It is implemented as:
/// - `p`: a grow-only counter for positive increments
/// - `n`: a grow-only counter for negative increments (decrements)
///
/// The observable value is `p_total - n_total`. Merges use lattice
/// join on the underlying [`PNCounterState`], so replicas converge
/// under arbitrary message reordering and duplication.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PNCounter<Id>
where
    Id: Eq + Hash,
{
    id: Id,
    state: PNCounterState<Id>,
}

impl<Id> PNCounter<Id>
where
    Id: Eq + Hash + Clone,
{
    /// Create a new PN-Counter for this replica ID.
    pub fn new(id: Id) -> Self {
        Self {
            id,
            state: PNCounterState::new(),
        }
    }

    /// Replica ID for this counter.
    pub fn id(&self) -> &Id {
        &self.id
    }

    /// Logical value (may be negative).
    pub fn value(&self) -> i64 {
        self.state.value()
    }

    /// Access the underlying lattice state (for replication).
    pub fn state(&self) -> &PNCounterState<Id> {
        &self.state
    }

    /// Replace the underlying state (mainly for tests /
    /// reconstruction).
    pub fn set_state(&mut self, state: PNCounterState<Id>) {
        self.state = state;
    }

    /// Monotone local increment on this replica.
    pub fn inc(&mut self, delta: u64) {
        self.state.p.inc_for(&self.id, delta);
    }

    /// Monotone local decrement on this replica.
    ///
    /// Implemented as incrementing the `n` component; the logical
    /// value is `p_total - n_total`.
    pub fn dec(&mut self, delta: u64) {
        self.state.n.inc_for(&self.id, delta);
    }

    /// Merge a remote state using lattice join.
    pub fn merge(&mut self, remote: &PNCounterState<Id>) {
        self.state = self.state.join(remote);
    }
}

/// A **grow-only set (G-Set)** CRDT.
///
/// State is a set of elements; updates only ever **add** elements,
/// and merging replicas uses set union. This is the set analogue of
/// `GCounter`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GSet<T>
where
    T: Eq + Hash,
{
    elems: HashSet<T>,
}

impl<T> GSet<T>
where
    T: Eq + Hash + Clone,
{
    /// Create an empty grow-only set.
    pub fn new() -> Self {
        Self {
            elems: HashSet::new(),
        }
    }

    /// Insert an element (monotone: once present, it never
    /// disappears).
    pub fn insert(&mut self, x: T) {
        self.elems.insert(x);
    }

    /// Current elements.
    pub fn elements(&self) -> &HashSet<T> {
        &self.elems
    }

    /// Does the set contain this element?
    pub fn contains(&self, x: &T) -> bool {
        self.elems.contains(x)
    }

    /// Merge a remote state into this one using lattice join (union).
    pub fn merge(&mut self, other: &GSet<T>) {
        self.elems = self.elems.join(&other.elems);
    }
}

impl<T> JoinSemilattice for GSet<T>
where
    T: Eq + Hash + Clone,
{
    /// Lattice join: union of sets.
    fn join(&self, other: &Self) -> Self {
        let mut out = self.elems.clone();
        out.extend(other.elems.iter().cloned());
        GSet { elems: out }
    }
}

impl<T> BoundedJoinSemilattice for GSet<T>
where
    T: Eq + Hash + Clone,
{
    /// Bottom = empty set.
    fn bottom() -> Self {
        Self::new()
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

    #[test]
    fn gset_local_add() {
        let mut s = GSet::new();
        assert!(!s.contains(&"a"));

        s.insert("a");
        s.insert("b");
        s.insert("b"); // idempotent

        assert!(s.contains(&"a"));
        assert!(s.contains(&"b"));
        assert_eq!(s.elements().len(), 2);
    }

    #[test]
    fn gset_merge_converges() {
        let mut a = GSet::new();
        let mut b = GSet::new();

        a.insert("a");
        a.insert("b");
        b.insert("b");
        b.insert("c");

        let a_state = a.clone();
        let b_state = b.clone();

        a.merge(&b_state);
        b.merge(&a_state);

        assert!(a.contains(&"a"));
        assert!(a.contains(&"b"));
        assert!(a.contains(&"c"));

        assert_eq!(a, b);
    }

    #[test]
    fn pncounter_local_inc_and_dec() {
        let mut c = PNCounter::new("A");
        assert_eq!(c.value(), 0);

        c.inc(5); // +5
        assert_eq!(c.value(), 5);

        c.dec(2); // -2
        assert_eq!(c.value(), 3);

        c.dec(10); // can go negative
        assert_eq!(c.value(), -7);
    }

    // This test simulates two PN-Counter replicas (A and B) that
    // update locally, then exchange states and converge.
    //
    // Internally, a PN-Counter is two GCounters:
    //   - p: grow-only counts for increments
    //   - n: grow-only counts for decrements
    // Each replica only bumps its own entries in p/n; merging uses
    // lattice join (componentwise max on p and n). The logical value
    // is sum(p) - sum(n).
    //
    // Here:
    //   A does  +10, -3   → locally thinks 7
    //   B does  +4,  -1   → locally thinks 3
    // After exchanging and joining their states, both see:
    //   p: {A:10, B:4}, n: {A:3, B:1} → (10+4) - (3+1) = 10
    // The assertions check that both replicas converge to the same
    // value (10), regardless of the order of merges.
    #[test]
    fn pncounter_merge_converges() {
        // Two replicas: A and B.
        let mut a = PNCounter::new("A");
        let mut b = PNCounter::new("B");

        // Local updates:
        a.inc(10); // +10
        a.dec(3); // -3   => A thinks: 7
        b.inc(4); // +4
        b.dec(1); // -1   => B thinks: 3

        let a_state = a.state().clone();
        let b_state = b.state().clone();

        // Exchange states (in any order).
        a.merge(&b_state);
        b.merge(&a_state);

        // Total: (10+4) - (3+1) = 14 - 4 = 10
        assert_eq!(a.value(), 10);
        assert_eq!(b.value(), 10);
    }

    #[test]
    fn pncounter_merge_is_idempotent_and_commutative() {
        let mut a = PNCounter::new(1u32);
        let mut b = PNCounter::new(2u32);

        a.inc(3);
        a.dec(1); // net +2
        b.inc(2);
        b.dec(5); // net -3

        let s1 = a.state().clone();
        let s2 = b.state().clone();

        // A merge B
        let mut a1 = a.clone();
        a1.merge(&s2);

        // B merge A
        let mut b1 = b.clone();
        b1.merge(&s1);

        // A merge B twice (idempotence)
        let mut a2 = a.clone();
        a2.merge(&s2);
        a2.merge(&s2);

        assert_eq!(a1.value(), b1.value());
        assert_eq!(a1.value(), a2.value());
    }
}
