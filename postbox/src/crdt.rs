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
//! - [`crate::crdt::GCounterState`] / [`crate::crdt::GCounter`]:
//!   grow-only counter where each replica maintains a per-replica
//!   component; merge is pointwise `max`, and the logical value is
//!   the sum of all components.
//!
//! - [`crate::crdt::PNCounterState`] / [`crate::crdt::PNCounter`]:
//!   a **PN-Counter** built from two GCounters (`p` for increments,
//!   `n` for decrements), with logical value `sum(p) - sum(n)`.
//!
//! - [`crate::crdt::GSet`]:
//!   grow-only set where updates only **add** elements and merges use
//!   set union.
//!
//! - [`crate::crdt::TwoPSetState`] / [`crate::crdt::TwoPSet`]:
//!   a **Two-Phase Set** that supports both adds and removes, but
//!   once an element is removed, it cannot be re-added (simpler than
//!   OR-Set).
//!
//! - [`crate::crdt::ORSetState`] / [`crate::crdt::ORSet`]:
//!   an add-wins **Observed-Remove Set** that supports both inserts
//!   and removes by tagging adds and recording tombstones for
//!   observed tags; merges are just lattice joins on the underlying
//!   add/remove sets.
//!
//! - [`crate::crdt::LWW`]:
//!   a **Last-Writer-Wins register** lattice storing `(value, ts)`
//!   and resolving conflicts by picking the value with the larger
//!   timestamp.
//!
//! - [`crate::crdt::MVRegister`]:
//!   a **Multi-Value Register** that keeps all values written at the
//!   maximum timestamp, handling concurrent writes by keeping all of them.
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
    /// Create an empty state (no replicas, logically all zero).
    pub fn new() -> Self {
        Self {
            counts: HashMap::new(),
        }
    }

    /// Reads raw per-replica counts.
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
    /// Create a new GCounter for this replica ID.
    pub fn new(id: Id) -> Self {
        Self {
            id,
            state: GCounterState::bottom(),
        }
    }

    /// Replica ID for this counter.
    pub fn id(&self) -> &Id {
        &self.id
    }

    /// Current value (sum of all components).
    pub fn value(&self) -> u64 {
        self.state.value()
    }

    /// Read the underlying lattice state (for replication).
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

/// Internal lattice state of a **Two-Phase Set (2P-Set)**.
///
/// A 2P-Set has two grow-only sets:
/// - `adds`: elements that` have been added
/// - `removes`: elements that have been removed
///
/// An element is in the logical set iff it's in `adds` and NOT in
/// `removes`. Once an element appears in `removes`, it can never be
/// in the logical set again (the "two-phase" property).
///
/// Lattice order is componentwise (both sets grow); join is union of
/// both components.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TwoPSetState<T>
where
    T: Eq + Hash,
{
    adds: HashSet<T>,
    removes: HashSet<T>,
}

impl<T> TwoPSetState<T>
where
    T: Eq + Hash + Clone,
{
    /// Empty 2P-Set: no adds, no removes.
    pub fn new() -> Self {
        Self {
            adds: HashSet::new(),
            removes: HashSet::new(),
        }
    }

    /// Elements that have been added.
    pub fn adds(&self) -> &HashSet<T> {
        &self.adds
    }

    /// Underlying remove-set (tombstones).
    pub fn removes(&self) -> &HashSet<T> {
        &self.removes
    }

    /// Logical elements: in `adds` but not in `removes`.
    pub fn elements(&self) -> HashSet<T> {
        self.adds.difference(&self.removes).cloned().collect()
    }

    /// Check if an element is in the logical set.
    pub fn contains(&self, x: &T) -> bool {
        self.adds.contains(x) && !self.removes.contains(x)
    }
}

impl<T> JoinSemilattice for TwoPSetState<T>
where
    T: Eq + Hash + Clone,
{
    /// Lattice join: union on both `adds` and `removes`.
    fn join(&self, other: &Self) -> Self {
        let mut adds = self.adds.clone();
        adds.extend(other.adds.iter().cloned());

        let mut removes = self.removes.clone();
        removes.extend(other.removes.iter().cloned());

        Self { adds, removes }
    }
}

impl<T> BoundedJoinSemilattice for TwoPSetState<T>
where
    T: Eq + Hash + Clone,
{
    /// Bottom = empty adds and removes.
    fn bottom() -> Self {
        Self::new()
    }
}

/// A **Two-Phase Set (2P-Set)** CRDT.
///
/// Supports adding and removing elements, but once removed, an
/// element cannot be re-added (unlike OR-Set). This makes it simpler
/// but less flexible than OR-Set.
///
/// The 2P-Set maintains two sets:
/// - additions: elements that have been added
/// - removals: tombstones for removed elements
///
/// An element is in the set iff it's been added but not removed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TwoPSet<T>
where
    T: Eq + Hash + Clone,
{
    state: TwoPSetState<T>,
}

impl<T> TwoPSet<T>
where
    T: Eq + Hash + Clone,
{
    /// Create a new empty 2P-Set.
    pub fn new() -> Self {
        Self {
            state: TwoPSetState::bottom(),
        }
    }

    /// Current logical elements.
    pub fn elements(&self) -> HashSet<T> {
        self.state.elements()
    }

    /// Check if the set contains an element.
    pub fn contains(&self, x: &T) -> bool {
        self.state.contains(x)
    }

    /// Access underlying lattice state (for replication).
    pub fn state(&self) -> &TwoPSetState<T> {
        &self.state
    }

    /// Replace underlying state.
    pub fn set_state(&mut self, state: TwoPSetState<T>) {
        self.state = state;
    }

    /// Add an element to the set.
    /// If the element was previously removed, this has no effect
    /// (2P-Set property).
    pub fn add(&mut self, x: T) {
        self.state.adds.insert(x);
    }

    /// Remove an element from the set.
    /// Once removed, the element cannot be re-added.
    pub fn remove(&mut self, x: T) {
        self.state.removes.insert(x);
    }

    /// Merge a remote state using lattice join.
    pub fn merge(&mut self, remote: &TwoPSetState<T>) {
        self.state = self.state.join(remote);
    }
}

/// Internal lattice state of an **add-wins observed-remove set
/// (OR-Set)**.
///
/// State is a pair of grow-only sets:
/// - `adds`:    tagged insertions `(elem, tag)`
/// - `removes`: tagged removals  `(elem, tag)`
///
/// The logical value of the set is:
///   `{ x | ∃ tag. (x, tag) ∈ adds ∧ (x, tag) ∉ removes }`.
///
/// Lattice order is componentwise (`adds` and `removes` are both
/// grow-only sets); join is their union. This makes it a classic
/// state-based OR-Set lattice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ORSetState<T, Tag>
where
    T: Eq + Hash,
    Tag: Eq + Hash,
{
    adds: HashSet<(T, Tag)>,
    removes: HashSet<(T, Tag)>,
}

impl<T, Tag> ORSetState<T, Tag>
where
    T: Eq + Hash + Clone,
    Tag: Eq + Hash + Clone,
{
    /// Empty OR-Set state: no adds, no removes.
    pub fn new() -> Self {
        Self {
            adds: HashSet::new(),
            removes: HashSet::new(),
        }
    }

    /// Underlying add-tag set.
    pub fn adds(&self) -> &HashSet<(T, Tag)> {
        &self.adds
    }

    /// Underlying remove-tag set.
    pub fn removes(&self) -> &HashSet<(T, Tag)> {
        &self.removes
    }

    /// Logical elements present in this state.
    pub fn elements(&self) -> HashSet<T> {
        let mut out = HashSet::new();
        for (x, t) in &self.adds {
            if !self.removes.contains(&(x.clone(), t.clone())) {
                out.insert(x.clone());
            }
        }
        out
    }
}

impl<T, Tag> JoinSemilattice for ORSetState<T, Tag>
where
    T: Eq + Hash + Clone,
    Tag: Eq + Hash + Clone,
{
    /// Lattice join: union on adds and removes.
    ///
    /// This is just the product lattice of two G-Sets:
    ///   (adds1, removes1) ⊔ (adds2, removes2)
    ///   = (adds1 ∪ adds2, removes1 ∪ removes2)
    fn join(&self, other: &Self) -> Self {
        let mut adds = self.adds.clone();
        adds.extend(other.adds.iter().cloned());

        let mut removes = self.removes.clone();
        removes.extend(other.removes.iter().cloned());

        Self { adds, removes }
    }
}

impl<T, Tag> BoundedJoinSemilattice for ORSetState<T, Tag>
where
    T: Eq + Hash + Clone,
    Tag: Eq + Hash + Clone,
{
    /// Bottom = empty adds/removes.
    fn bottom() -> Self {
        Self::new()
    }
}

/// An **add-wins OR-Set** CRDT.
///
/// Each `add(x)` is tagged with a unique, per-replica identifier. A
/// `remove(x)` records tombstones for all tags of `x` currently
/// visible at that replica. The logical element set is:
///
///   `{ x | ∃ tag. (x, tag) ∈ adds ∧ (x, tag) ∉ removes }`.
///
/// This implementation uses tags of the form `(Id, u64)` where `Id`
/// is the replica ID and `u64` is a local counter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ORSet<Id, T>
where
    Id: Eq + Hash + Clone,
    T: Eq + Hash + Clone,
{
    id: Id,
    clock: u64,
    state: ORSetState<T, (Id, u64)>,
}

impl<Id, T> ORSet<Id, T>
where
    Id: Eq + Hash + Clone,
    T: Eq + Hash + Clone,
{
    /// Create a new OR-Set for this replica ID.
    pub fn new(id: Id) -> Self {
        Self {
            id,
            clock: 0,
            state: ORSetState::bottom(),
        }
    }

    /// Replica ID.
    pub fn id(&self) -> &Id {
        &self.id
    }

    /// Current logical elements.
    pub fn elements(&self) -> HashSet<T> {
        self.state.elements()
    }

    /// Access underlying lattice state (for replication).
    pub fn state(&self) -> &ORSetState<T, (Id, u64)> {
        &self.state
    }

    /// Replace underlying state (mainly for tests / reconstruction).
    pub fn set_state(&mut self, state: ORSetState<T, (Id, u64)>) {
        self.state = state;
    }

    /// Add an element. This is implemented as:
    ///   - bump the local clock
    ///   - insert (elem, (id, clock)) into the adds set
    pub fn add(&mut self, x: T) {
        self.clock = self.clock.wrapping_add(1);
        let tag = (self.id.clone(), self.clock);
        self.state.adds.insert((x, tag));
    }

    /// Remove an element. Implemented as adding tombstones in the
    /// removes set for all known tags of `x`.
    pub fn remove(&mut self, x: &T) {
        // Collect tags for x that we currently know about.
        let tags: Vec<_> = self
            .state
            .adds
            .iter()
            .filter_map(|(e, tag)| if e == x { Some(tag.clone()) } else { None })
            .collect();

        for tag in tags {
            self.state.removes.insert((x.clone(), tag));
        }
    }

    /// Merge a remote state using lattice join.
    pub fn merge(&mut self, remote: &ORSetState<T, (Id, u64)>) {
        self.state = self.state.join(remote);
    }

    /// Does the logical set contain this element?
    pub fn contains(&self, x: &T) -> bool {
        self.elements().contains(x)
    }
}

/// A **Last-Writer-Wins register** lattice.
///
/// The state is a pair `(value, ts)` where `ts` is a logical
/// timestamp (e.g. Lamport clock, HLC, or monotone counter). Lattice
/// ordering is by timestamp: on join, the value with the **larger
/// timestamp** wins, and ties are broken deterministically in favor
/// of the left-hand operand.
///
/// This makes `LWW<T>` a simple register-style lattice that can be
/// used as the payload in higher-level CRDTs or in `Option<LWW<T>>`
/// when a true bottom element is needed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LWW<T> {
    /// The current value of the register.
    pub value: T,
    /// The logical timestamp associated with this value.
    pub ts: u64,
}

impl<T: Clone> JoinSemilattice for LWW<T> {
    fn join(&self, other: &Self) -> Self {
        if other.ts > self.ts {
            other.clone()
        } else {
            self.clone()
        }
    }
}

/// A **Multi-Value Register** lattice.
///
/// Unlike LWW which picks a single winner, MVRegister keeps all
/// values that were written at the maximum timestamp. This handles
/// concurrent writes by keeping all of them.
///
/// Structure: a set of `(value, timestamp, replica_id)` entries.
/// - Join: union the sets, then keep only entries with max timestamp
/// - Read: all values at the max timestamp (may be multiple if
///   concurrent)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MVRegister<T, Ts, Id>
where
    T: Eq + Hash + Clone,
    Ts: Eq + Hash + Ord + Clone,
    Id: Eq + Hash + Clone,
{
    entries: HashSet<(T, Ts, Id)>,
}

impl<T, Ts, Id> MVRegister<T, Ts, Id>
where
    T: Eq + Hash + Clone,
    Ts: Eq + Hash + Ord + Clone,
    Id: Eq + Hash + Clone,
{
    /// Create an empty MVRegister.
    pub fn new() -> Self {
        Self {
            entries: HashSet::new(),
        }
    }

    /// Write a value with the given timestamp and replica ID.
    pub fn write(&mut self, value: T, timestamp: Ts, replica_id: Id) {
        self.entries.insert((value, timestamp, replica_id));
        self.prune_dominated();
    }

    /// Get all current values (may be multiple if there were concurrent writes).
    pub fn values(&self) -> Vec<T> {
        self.entries.iter().map(|(v, _, _)| v.clone()).collect()
    }

    /// Get the maximum timestamp.
    pub fn max_timestamp(&self) -> Option<&Ts> {
        self.entries.iter().map(|(_, ts, _)| ts).max()
    }

    /// Remove entries that are dominated (have timestamps less than
    /// max).
    fn prune_dominated(&mut self) {
        if let Some(max_ts) = self.max_timestamp().cloned() {
            self.entries.retain(|(_, ts, _)| ts == &max_ts);
        }
    }

    /// Number of concurrent values.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Is the register empty?
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl<T, Ts, Id> JoinSemilattice for MVRegister<T, Ts, Id>
where
    T: Eq + Hash + Clone,
    Ts: Eq + Hash + Ord + Clone,
    Id: Eq + Hash + Clone,
{
    /// Lattice join: union entries, then keep only those with max
    /// timestamp.
    fn join(&self, other: &Self) -> Self {
        let mut entries = self.entries.clone();
        entries.extend(other.entries.iter().cloned());

        let mut result = Self { entries };
        result.prune_dominated();
        result
    }
}

impl<T, Ts, Id> BoundedJoinSemilattice for MVRegister<T, Ts, Id>
where
    T: Eq + Hash + Clone,
    Ts: Eq + Hash + Ord + Clone,
    Id: Eq + Hash + Clone,
{
    /// Bottom = no entries.
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
    fn twopset_local_add_and_remove() {
        let mut s = TwoPSet::new();

        s.add("a");
        s.add("b");
        assert!(s.contains(&"a"));
        assert!(s.contains(&"b"));

        s.remove("a");
        assert!(!s.contains(&"a"));
        assert!(s.contains(&"b"));
    }

    #[test]
    fn twopset_cannot_readd_after_remove() {
        let mut s = TwoPSet::new();

        s.add("x");
        assert!(s.contains(&"x"));

        s.remove("x");
        assert!(!s.contains(&"x"));

        // This add has no effect - x is permanently removed
        s.add("x");
        assert!(!s.contains(&"x"));
    }

    #[test]
    fn twopset_merge_converges() {
        let mut a = TwoPSet::new();
        let mut b = TwoPSet::new();

        // A adds "x" and "y"
        a.add("x");
        a.add("y");

        // B adds "y" and "z", then removes "y"
        b.add("y");
        b.add("z");
        b.remove("y");

        let a_state = a.state().clone();
        let b_state = b.state().clone();

        // Merge in both directions
        a.merge(&b_state);
        b.merge(&a_state);

        // Both should converge to: {x, z}
        // (y was removed by B, so it's gone even though A added it)
        assert!(a.contains(&"x"));
        assert!(!a.contains(&"y"));
        assert!(a.contains(&"z"));

        assert_eq!(a.elements(), b.elements());
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

    #[test]
    fn orset_local_add_and_remove() {
        let mut s = ORSet::new("A");

        s.add("a");
        s.add("b");
        assert!(s.contains(&"a"));
        assert!(s.contains(&"b"));

        s.remove(&"a");
        assert!(!s.contains(&"a"));
        assert!(s.contains(&"b"));
    }

    #[test]
    fn orset_merge_converges() {
        // Two replicas: A and B.
        let mut a = ORSet::new("A");
        let mut b = ORSet::new("B");

        // Both add "x" independently.
        a.add("x");
        b.add("x");

        // A removes "x" based on what it has seen so far.
        a.remove(&"x");

        // Capture states and merge in both directions.
        let a_state = a.state().clone();
        let b_state = b.state().clone();

        a.merge(&b_state);
        b.merge(&a_state);

        // Whatever the winning policy for this pattern, both replicas
        // must converge to the same logical view.
        assert_eq!(a.elements(), b.elements());
    }

    #[test]
    fn lww_join_picks_newer_timestamp() {
        let old = LWW {
            value: "old",
            ts: 5,
        };
        let new = LWW {
            value: "new",
            ts: 10,
        };

        let joined = old.join(&new);

        assert_eq!(joined.value, "new");
        assert_eq!(joined.ts, 10);
    }

    #[test]
    fn lww_join_is_commutative_and_idempotent() {
        let a = LWW { value: "a", ts: 7 };
        let b = LWW { value: "b", ts: 3 };

        // Commutative: a ⊔ b == b ⊔ a
        let ab = a.join(&b);
        let ba = b.join(&a);
        assert_eq!(ab, ba);

        // Idempotent: a ⊔ a == a
        let aa = a.join(&a);
        assert_eq!(aa, a);
    }

    #[test]
    fn mvregister_keeps_concurrent_writes() {
        let mut a = MVRegister::new();
        let mut b = MVRegister::new();

        // Both replicas write at timestamp 10 (concurrent)
        a.write("value_a", 10, "A");
        b.write("value_b", 10, "B");

        let merged = a.join(&b);

        // Should keep both values since they're concurrent
        assert_eq!(merged.len(), 2);
        assert!(merged.values().contains(&"value_a"));
        assert!(merged.values().contains(&"value_b"));
    }

    #[test]
    fn mvregister_overwrites_older_values() {
        let mut r = MVRegister::new();

        r.write("old", 5, "A");
        r.write("new", 10, "B");

        // Only the newer value should remain
        assert_eq!(r.len(), 1);
        assert_eq!(r.values(), vec!["new"]);
    }

    #[test]
    fn mvregister_ignores_stale_writes() {
        let mut r = MVRegister::new();

        r.write("current", 10, "A");
        r.write("stale", 5, "B"); // This gets pruned immediately

        // Only the value at max timestamp remains
        assert_eq!(r.len(), 1);
        assert_eq!(r.values(), vec!["current"]);
    }

    #[test]
    fn gcounter_accessor_methods() {
        let mut c = GCounter::new("A");
        assert_eq!(c.id(), &"A");
        assert_eq!(c.value(), 0);

        c.inc(5);
        assert_eq!(c.value(), 5);
        assert_eq!(c.state().counts().get(&"A"), Some(&5));
    }

    #[test]
    fn pncounter_accessor_methods() {
        let mut c = PNCounter::new("A");
        assert_eq!(c.id(), &"A");
        assert_eq!(c.value(), 0);

        c.inc(10);
        c.dec(3);
        assert_eq!(c.value(), 7);

        let s = c.state().clone();
        let mut c2 = PNCounter::new("B");
        c2.set_state(s);
        assert_eq!(c2.value(), 7);
    }

    #[test]
    fn gset_accessor_methods() {
        let mut s = GSet::new();
        assert!(s.elements().is_empty());
        assert!(!s.contains(&1));

        s.insert(1);
        s.insert(2);
        assert_eq!(s.elements().len(), 2);
        assert!(s.contains(&1));
        assert!(!s.contains(&3));
    }

    #[test]
    fn twopset_accessor_methods() {
        let mut s = TwoPSet::new();
        assert!(s.elements().is_empty());
        assert!(!s.contains(&1));

        s.add(1);
        s.add(2);
        assert_eq!(s.elements().len(), 2);
        assert!(s.contains(&1));

        s.remove(1);
        assert!(!s.contains(&1));
        assert_eq!(s.elements().len(), 1);

        let state = s.state().clone();
        assert_eq!(state.adds().len(), 2);
        assert_eq!(state.removes().len(), 1);

        let mut s2 = TwoPSet::new();
        s2.set_state(state);
        assert_eq!(s2.elements().len(), 1);
    }

    #[test]
    fn orset_accessor_methods() {
        let mut s = ORSet::new("A");
        assert_eq!(s.id(), &"A");
        assert!(s.elements().is_empty());
        assert!(!s.contains(&1));

        s.add(1);
        s.add(2);
        assert_eq!(s.elements().len(), 2);
        assert!(s.contains(&1));

        let state = s.state().clone();
        assert_eq!(state.adds().len(), 2);
        assert_eq!(state.removes().len(), 0);

        let mut s2 = ORSet::new("B");
        s2.set_state(state);
        assert_eq!(s2.elements().len(), 2);
    }

    #[test]
    fn lww_value_and_timestamp() {
        let r1 = LWW { value: 42, ts: 10 };
        assert_eq!(r1.value, 42);
        assert_eq!(r1.ts, 10);

        let r2 = LWW { value: 99, ts: 5 };
        let j = r1.join(&r2);
        assert_eq!(j.value, 42); // Newer timestamp wins
        assert_eq!(j.ts, 10);
    }
}
