//! Core **join-semilattice** traits and building blocks.
//!
//! This module re-exports [`JoinSemilattice`] and
//! [`BoundedJoinSemilattice`] from [`algebra_core`], plus provides a
//! small toolkit of concrete lattices and combinators that are
//! convenient for CRDTs and other monotone data structures.
//!
//! # Overview
//!
//! - [`JoinSemilattice`]: types with an associative, commutative,
//!   idempotent `join` (∨).
//! - [`BoundedJoinSemilattice`]: lattices with an explicit bottom
//!   element (⊥).
//!
//! # Provided lattices
//!
//! - [`crate::join_semilattice::Max`] / [`crate::join_semilattice::Min`]:
//!   wrapper types where `join` is `max` / `min` on the inner value.
//! - [`crate::join_semilattice::Any`] / [`crate::join_semilattice::All`]:
//!   boolean lattices where `join` is `||` (OR) / `&&` (AND).
//! - [`crate::join_semilattice::BitOr`] / [`crate::join_semilattice::BitAnd`]:
//!   bitwise lattices for bitflags and integer masks (OR / AND).
//! - [`std::collections::HashSet`], [`std::collections::BTreeSet`]:
//!   sets with `join = union` and bottom = empty set.
//! - [`crate::join_semilattice::LatticeMap`]:
//!   pointwise map lattice over `HashMap<K, V>` where `V` is a
//!   lattice; `join` unions keys and joins overlapping values.
//! - Tuples up to arity 4:
//!   product lattices with componentwise `join` and bottom.
//! - [`Option`]:
//!   lifted lattice with `None` as bottom and `Some(a) ⊔ Some(b)`
//!   delegating to `L`.
//! - [`crate::join_semilattice::JoinOf`] / [`crate::join_semilattice::NonEmptyJoinOf`]:
//!   helpers for collecting iterators of lattice values by joining
//!   them (treating empty as ⊥ or returning `None`).
//!
//! # Example
//!
//! ```rust
//! use postbox::join_semilattice::{JoinSemilattice, Max};
//!
//! let x: Max<_> = 1.into();
//! let y: Max<_> = 3.into();
//!
//! // join = max
//! let z = x.join(&y);
//! assert_eq!(z.0, 3);
//! ```
//!
//! These primitives are intentionally small and generic. Higher-level
//! CRDTs in this crate build on them to express their replica states
//! as lattices with well-defined, convergent merge operations.

// Re-export traits from algebra-core
pub use algebra_core::{BoundedJoinSemilattice, JoinSemilattice};

use std::collections::HashMap;
use std::hash::Hash;
use std::ops::Deref;

// join = max

/// Newtype wrapper turning an `Ord` type into a **max-semilattice**.
///
/// `Max<T>` interprets `join` as taking the maximum of two values:
///
/// - `Max(a).join(&Max(b)) == Max(max(a, b))`
///
/// This is handy when you want to treat a plain ordered type (like
/// `u64` or `Instant`) as a lattice element, e.g. for timestamps,
/// counters, or high-water marks.
///
/// When `T: Bounded`, the bottom element is `Max(T::min_value())`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Max<T>(pub T);

impl<T: Ord + Clone> JoinSemilattice for Max<T> {
    fn join(&self, other: &Self) -> Self {
        if self.0 >= other.0 {
            self.clone()
        } else {
            other.clone()
        }
    }
}

impl<T: Ord + Clone + Default + num_traits::Bounded> BoundedJoinSemilattice for Max<T> {
    fn bottom() -> Self {
        Max(num_traits::Bounded::min_value())
    }
}

impl<T> From<T> for Max<T> {
    fn from(value: T) -> Self {
        Max(value)
    }
}

// algebra-core trait implementations for Max

impl<T: Ord + Clone> algebra_core::Semigroup for Max<T> {
    fn combine(&self, other: &Self) -> Self {
        self.join(other)
    }

    fn combine_assign(&mut self, other: &Self) {
        self.join_assign(other);
    }
}

impl<T: Ord + Clone + Default + num_traits::Bounded> algebra_core::Monoid for Max<T> {
    fn empty() -> Self {
        Self::bottom()
    }

    fn concat<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Self>,
    {
        Self::join_all_from_bottom(iter)
    }
}

impl<T: Ord + Clone + Default + num_traits::Bounded> algebra_core::CommutativeMonoid for Max<T> {}

// join = min

/// Newtype wrapper turning an `Ord` type into a **min-semilattice**.
///
/// `Min<T>` interprets `join` as taking the minimum of two values:
///
/// - `Min(a).join(&Min(b)) == Min(min(a, b))`
///
/// This is useful when you want the lattice order to represent "no
/// larger than" (e.g. deadlines, lower-bounds, or minima over a set).
///
/// When `T: Bounded`, the bottom element is `Min(T::max_value())`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Min<T>(pub T);

impl<T: Ord + Clone> JoinSemilattice for Min<T> {
    fn join(&self, other: &Self) -> Self {
        if self.0 <= other.0 {
            self.clone()
        } else {
            other.clone()
        }
    }
}

impl<T: Ord + Clone + Default + num_traits::Bounded> BoundedJoinSemilattice for Min<T> {
    fn bottom() -> Self {
        Min(num_traits::Bounded::max_value())
    }
}

impl<T> From<T> for Min<T> {
    fn from(value: T) -> Self {
        Min(value)
    }
}

// algebra-core trait implementations for Min

impl<T: Ord + Clone> algebra_core::Semigroup for Min<T> {
    fn combine(&self, other: &Self) -> Self {
        self.join(other)
    }

    fn combine_assign(&mut self, other: &Self) {
        self.join_assign(other);
    }
}

impl<T: Ord + Clone + Default + num_traits::Bounded> algebra_core::Monoid for Min<T> {
    fn empty() -> Self {
        Self::bottom()
    }

    fn concat<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Self>,
    {
        Self::join_all_from_bottom(iter)
    }
}

impl<T: Ord + Clone + Default + num_traits::Bounded> algebra_core::CommutativeMonoid for Min<T> {}

// join = OR (disjunction)

/// Newtype wrapper for `bool` where `join` is logical OR (disjunction).
///
/// `Any(bool)` forms a join-semilattice under the natural boolean order
/// (false < true):
///
/// - `Any(a).join(&Any(b)) == Any(a || b)`
/// - Bottom element is `Any(false)`
///
/// This is useful for combining boolean flags where "any true means true",
/// such as error flags, presence checks, or monitoring conditions.
///
/// # Example
///
/// ```rust
/// use postbox::join_semilattice::{JoinSemilattice, BoundedJoinSemilattice, Any};
///
/// let x: Any = false.into();
/// let y: Any = true.into();
/// assert_eq!(x.join(&y), true.into());
/// assert_eq!(Any::bottom(), false.into());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Any(pub bool);

impl JoinSemilattice for Any {
    fn join(&self, other: &Self) -> Self {
        Any(self.0 || other.0)
    }
}

impl BoundedJoinSemilattice for Any {
    fn bottom() -> Self {
        Any(false)
    }
}

impl From<bool> for Any {
    fn from(value: bool) -> Self {
        Any(value)
    }
}

// algebra-core trait implementations for Any

impl algebra_core::Semigroup for Any {
    fn combine(&self, other: &Self) -> Self {
        self.join(other)
    }

    fn combine_assign(&mut self, other: &Self) {
        self.join_assign(other);
    }
}

impl algebra_core::Monoid for Any {
    fn empty() -> Self {
        Self::bottom()
    }

    fn concat<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Self>,
    {
        Self::join_all_from_bottom(iter)
    }
}

impl algebra_core::CommutativeMonoid for Any {}

// join = AND (conjunction, dual order)

/// Newtype wrapper for `bool` where `join` is logical AND (conjunction).
///
/// `All(bool)` forms a join-semilattice under the dual boolean order
/// (true < false):
///
/// - `All(a).join(&All(b)) == All(a && b)`
/// - Bottom element is `All(true)`
///
/// This is useful for combining boolean conditions where "all must be
/// true", such as validation checks, preconditions, or invariants.
///
/// Note: The dual order may seem counterintuitive, but it makes `AND`
/// the join operation (least upper bound). This mirrors how [`Min`]
/// uses the dual order to make `min` the join.
///
/// # Example
///
/// ```rust
/// use postbox::join_semilattice::{JoinSemilattice, BoundedJoinSemilattice, All};
///
/// let x: All = true.into();
/// let y: All = false.into();
/// assert_eq!(x.join(&y), false.into());
/// assert_eq!(All::bottom(), true.into());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct All(pub bool);

impl JoinSemilattice for All {
    fn join(&self, other: &Self) -> Self {
        All(self.0 && other.0)
    }
}

impl BoundedJoinSemilattice for All {
    fn bottom() -> Self {
        All(true)
    }
}

impl From<bool> for All {
    fn from(value: bool) -> Self {
        All(value)
    }
}

// algebra-core trait implementations for All

impl algebra_core::Semigroup for All {
    fn combine(&self, other: &Self) -> Self {
        self.join(other)
    }

    fn combine_assign(&mut self, other: &Self) {
        self.join_assign(other);
    }
}

impl algebra_core::Monoid for All {
    fn empty() -> Self {
        Self::bottom()
    }

    fn concat<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Self>,
    {
        Self::join_all_from_bottom(iter)
    }
}

impl algebra_core::CommutativeMonoid for All {}

// join = bitwise OR

/// Lattice wrapper for bitflags and integers with join = bitwise OR.
///
/// `BitOr<T>` wraps any type implementing [`std::ops::BitOr`] and
/// treats bitwise OR as the join operation. This is useful for:
///
/// - **Bitflags**: types generated by the `bitflags!` macro, where
///   bits represent a set of flags and OR corresponds to set union.
/// - **Integer masks**: `u8`, `u16`, `u32`, etc., used as bit sets.
///
/// The induced partial order is subset-of-bits: `x ≤ y` iff `x | y ==
/// y`.
///
/// - `join = |` (bitwise OR)
/// - Bottom element is `T::default()` (the "all-zero / empty" value)
///
/// Note: `Any(bool)` is essentially the specialized boolean case of
/// `BitOr<bool>`.
///
/// # Examples
///
/// ## With integers
///
/// ```rust
/// use postbox::join_semilattice::{JoinSemilattice, BoundedJoinSemilattice, BitOr};
///
/// let x = BitOr(0b0011u8);
/// let y = BitOr(0b0101u8);
/// assert_eq!(x.join(&y), BitOr(0b0111u8));
/// assert_eq!(BitOr::<u8>::bottom(), BitOr(0));
/// ```
///
/// ## With bitflags
///
/// ```rust
/// # #[cfg(feature = "bitflags")] {
/// use bitflags::bitflags;
/// use postbox::join_semilattice::{JoinSemilattice, BitOr};
///
/// bitflags! {
///     #[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
///     struct Flags: u32 {
///         const A = 0b001;
///         const B = 0b010;
///         const C = 0b100;
///     }
/// }
///
/// let x = BitOr(Flags::A);
/// let y = BitOr(Flags::B | Flags::C);
/// let joined = x.join(&y);
/// assert_eq!(joined.0, Flags::A | Flags::B | Flags::C);
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BitOr<T>(pub T);

impl<T> BitOr<T> {
    /// Extract the underlying value.
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T> JoinSemilattice for BitOr<T>
where
    T: Copy + std::ops::BitOr<Output = T>,
{
    fn join(&self, other: &Self) -> Self {
        BitOr(self.0 | other.0)
    }
}

impl<T> BoundedJoinSemilattice for BitOr<T>
where
    T: Copy + std::ops::BitOr<Output = T> + Default,
{
    fn bottom() -> Self {
        // For bitflags and integers, `Default` is the empty set (all
        // zeros).
        BitOr(T::default())
    }
}

impl<T> From<T> for BitOr<T> {
    fn from(value: T) -> Self {
        BitOr(value)
    }
}

// algebra-core trait implementations for BitOr

impl<T> algebra_core::Semigroup for BitOr<T>
where
    T: Copy + std::ops::BitOr<Output = T>,
{
    fn combine(&self, other: &Self) -> Self {
        self.join(other)
    }

    fn combine_assign(&mut self, other: &Self) {
        self.join_assign(other);
    }
}

impl<T> algebra_core::Monoid for BitOr<T>
where
    T: Copy + std::ops::BitOr<Output = T> + Default,
{
    fn empty() -> Self {
        Self::bottom()
    }

    fn concat<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Self>,
    {
        Self::join_all_from_bottom(iter)
    }
}

impl<T> algebra_core::CommutativeMonoid for BitOr<T> where
    T: Copy + std::ops::BitOr<Output = T> + Default
{
}

// join = bitwise AND

/// Lattice wrapper for bitflags and integers with join = bitwise AND.
///
/// `BitAnd<T>` wraps any type implementing [`std::ops::BitAnd`] and
/// treats bitwise AND as the join operation. This represents the dual
/// lattice to [`BitOr`]:
///
/// - **Integers**: tracks which bits remain set across all values
/// - **Bitflags**: tracks which flags are present in all instances
///
/// The induced partial order is superset-of-bits (dual to [`BitOr`]):
/// `x ≤ y` iff `x & y == x`.
///
/// - `join = &` (bitwise AND)
/// - Bottom element is all bits set (the "universal" value)
///
/// This is useful for tracking invariants that must hold across all
/// replicas, or computing the intersection of flag sets.
///
/// # Examples
///
/// ## With integers
///
/// ```rust
/// use postbox::join_semilattice::{JoinSemilattice, BoundedJoinSemilattice, BitAnd};
///
/// let x = BitAnd(0b1111u8);
/// let y = BitAnd(0b1010u8);
/// assert_eq!(x.join(&y), BitAnd(0b1010u8));
/// assert_eq!(BitAnd::<u8>::bottom(), BitAnd(0xFF));
/// ```
///
/// ## With bitflags
///
/// ```rust
/// # #[cfg(feature = "bitflags")] {
/// use bitflags::bitflags;
/// use postbox::join_semilattice::{JoinSemilattice, BitAnd};
///
/// bitflags! {
///     #[derive(Clone, Copy, Debug, PartialEq, Eq)]
///     struct Flags: u32 {
///         const A = 0b001;
///         const B = 0b010;
///         const C = 0b100;
///     }
/// }
///
/// impl Flags {
///     const fn all_flags() -> Self {
///         Self::from_bits_truncate(0b111)
///     }
/// }
///
/// let x = BitAnd(Flags::A | Flags::B | Flags::C);
/// let y = BitAnd(Flags::A | Flags::B);
/// let joined = x.join(&y);
/// assert_eq!(joined.0, Flags::A | Flags::B);
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BitAnd<T>(pub T);

impl<T> BitAnd<T> {
    /// Extract the underlying value.
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T> JoinSemilattice for BitAnd<T>
where
    T: Copy + std::ops::BitAnd<Output = T>,
{
    fn join(&self, other: &Self) -> Self {
        BitAnd(self.0 & other.0)
    }
}

impl<T> BoundedJoinSemilattice for BitAnd<T>
where
    T: Copy + std::ops::BitAnd<Output = T> + num_traits::Bounded,
{
    fn bottom() -> Self {
        // For bitwise AND, the identity is all bits set (maximum value).
        // This represents the "universal" set in the dual order.
        BitAnd(num_traits::Bounded::max_value())
    }
}

impl<T> From<T> for BitAnd<T> {
    fn from(value: T) -> Self {
        BitAnd(value)
    }
}

// algebra-core trait implementations for BitAnd

impl<T> algebra_core::Semigroup for BitAnd<T>
where
    T: Copy + std::ops::BitAnd<Output = T>,
{
    fn combine(&self, other: &Self) -> Self {
        self.join(other)
    }

    fn combine_assign(&mut self, other: &Self) {
        self.join_assign(other);
    }
}

impl<T> algebra_core::Monoid for BitAnd<T>
where
    T: Copy + std::ops::BitAnd<Output = T> + num_traits::Bounded,
{
    fn empty() -> Self {
        Self::bottom()
    }

    fn concat<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Self>,
    {
        Self::join_all_from_bottom(iter)
    }
}

impl<T> algebra_core::CommutativeMonoid for BitAnd<T> where
    T: Copy + std::ops::BitAnd<Output = T> + num_traits::Bounded
{
}

/// Pointwise map lattice over `HashMap`.
///
/// Keys are optional; values form a join-semilattice. The induced
/// lattice order is:
///
///   m1 ≤ m2  iff  for all k, m1\[k\] ≤ m2\[k\]
///
/// Operationally, `join` is:
/// - keys: union of the key sets
/// - values: pointwise `join` on overlapping keys
///
/// Bottom is the empty map.
///
/// This is a reusable building block for CRDT states that look like
/// "map from IDs to lattice values".
#[derive(Clone, Debug)]
pub struct LatticeMap<K, V> {
    inner: HashMap<K, V>,
}

impl<K, V> LatticeMap<K, V>
where
    K: Eq + Hash,
{
    /// Create an empty map lattice.
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    /// Insert or replace a value for a key.
    pub fn insert(&mut self, k: K, v: V) {
        self.inner.insert(k, v);
    }

    /// Get a reference to the value for this key, if present.
    pub fn get(&self, k: &K) -> Option<&V> {
        self.inner.get(k)
    }

    /// Iterate over `(key, value)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.inner.iter()
    }

    /// Access the underlying `HashMap`.
    pub fn as_inner(&self) -> &HashMap<K, V> {
        &self.inner
    }

    /// Consume the wrapper and return the underlying `HashMap`.
    pub fn into_inner(self) -> HashMap<K, V> {
        self.inner
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Is the map empty?
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl<K, V> JoinSemilattice for LatticeMap<K, V>
where
    K: Eq + Hash + Clone,
    V: JoinSemilattice + Clone,
{
    fn join(&self, other: &Self) -> Self {
        let mut out = self.inner.clone();

        for (k, v_other) in &other.inner {
            out.entry(k.clone())
                .and_modify(|v_here| {
                    *v_here = v_here.join(v_other);
                })
                .or_insert_with(|| v_other.clone());
        }

        LatticeMap { inner: out }
    }
}

impl<K, V> BoundedJoinSemilattice for LatticeMap<K, V>
where
    K: Eq + Hash + Clone,
    V: BoundedJoinSemilattice + Clone,
{
    fn bottom() -> Self {
        LatticeMap {
            inner: HashMap::new(),
        }
    }
}

// JoinOf<L> (bounded)

/// A wrapper type for collecting values using their lattice `join`.
///
/// `JoinOf<L>` turns any iterator of `L` into a single value by
/// repeatedly applying `join`. It is especially useful when working
/// with CRDT state that naturally accumulates via joins.
///
/// This type implements [`FromIterator`] for both owned `L` and
/// references `&L`, allowing you to write:
///
/// ```
/// use postbox::join_semilattice::{JoinOf, JoinSemilattice};
/// use std::collections::HashSet;
///
/// let a: HashSet<_> = [1, 2].into_iter().collect();
/// let b: HashSet<_> = [2, 3].into_iter().collect();
///
/// // Join = union, so collecting produces the union.
/// let JoinOf(u) = [a, b].into_iter().collect::<JoinOf<_>>();
/// assert_eq!(u, [1, 2, 3].into_iter().collect());
/// ```
///
/// # Empty iterators
///
/// If the iterator is empty, the result is the **bottom** element of
/// the lattice (`L::bottom()`), as required by
/// [`BoundedJoinSemilattice`]. This makes `JoinOf` appropriate for
/// "total" reductions where an empty input still yields a
/// well-defined lattice value.
///
/// See also [`NonEmptyJoinOf`], which provides a fallible variant
/// that returns `None` on empty iterators.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JoinOf<L>(pub L);

impl<L> JoinOf<L> {
    /// Unwrap the inner value.
    pub fn into_inner(self) -> L {
        self.0
    }
}

impl<L> Deref for JoinOf<L> {
    type Target = L;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Collect with join, treating an empty iterator as ⊥.
impl<L> std::iter::FromIterator<L> for JoinOf<L>
where
    L: BoundedJoinSemilattice,
{
    fn from_iter<T: IntoIterator<Item = L>>(iter: T) -> Self {
        let acc = iter.into_iter().fold(L::bottom(), |acc, x| acc.join(&x));
        JoinOf(acc)
    }
}

/// Collect with join from references, cloning items. Still treats
/// empty iterator as ⊥.
impl<'a, L> std::iter::FromIterator<&'a L> for JoinOf<L>
where
    L: BoundedJoinSemilattice + Clone,
{
    fn from_iter<T: IntoIterator<Item = &'a L>>(iter: T) -> Self {
        let acc = iter.into_iter().fold(L::bottom(), |acc, x| acc.join(x));
        JoinOf(acc)
    }
}

// NonEmptyJoinOf<L>

/// A wrapper for non-empty joins over a `JoinSemilattice`.
///
/// `NonEmptyJoinOf<L>` represents the join of a **non-empty**
/// collection of lattice values. Unlike [`JoinOf`], it does *not*
/// assume the existence of a bottom element and therefore does not
/// define behavior for empty iterators by itself.
///
/// Instead, it provides the helper constructors
/// [`NonEmptyJoinOf::from_iter_nonempty`] and
/// [`NonEmptyJoinOf::from_iter_nonempty_ref`], which return `None`
/// when given an empty iterator:
///
/// ```
/// use postbox::join_semilattice::{NonEmptyJoinOf, JoinSemilattice};
/// use std::collections::HashSet;
///
/// let a: HashSet<_> = [1, 2].into_iter().collect();
/// let b: HashSet<_> = [2, 3].into_iter().collect();
///
/// let ne = NonEmptyJoinOf::from_iter_nonempty([a, b]).unwrap();
/// // `ne` holds the union of both sets.
/// ```
///
/// This is useful when you want the type system (via `Option`) to
/// distinguish between "no values" and "the join of at least one
/// value", without requiring a [`BoundedJoinSemilattice`] bottom
/// element.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonEmptyJoinOf<L>(pub L);

impl<L> NonEmptyJoinOf<L> {
    /// Unwrap the inner value.
    pub fn into_inner(self) -> L {
        self.0
    }

    /// Fallible collect for non-empty join. Returns `None` if the
    /// iterator is empty.
    pub fn from_iter_nonempty<I>(iter: I) -> Option<Self>
    where
        I: IntoIterator<Item = L>,
        L: JoinSemilattice,
    {
        let mut it = iter.into_iter();
        let first = it.next()?;
        let acc = it.fold(first, |acc, x| acc.join(&x));
        Some(NonEmptyJoinOf(acc))
    }

    /// Reference-based variant (clones items).
    pub fn from_iter_nonempty_ref<'a, I>(iter: I) -> Option<Self>
    where
        I: IntoIterator<Item = &'a L>,
        L: JoinSemilattice + Clone + 'a,
    {
        let mut it = iter.into_iter();
        let first = it.next()?.clone();
        let acc = it.fold(first, |acc, x| acc.join(x));
        Some(NonEmptyJoinOf(acc))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    // Helper to build a set quickly.
    fn set<T>(xs: &[T]) -> HashSet<T>
    where
        T: Eq + std::hash::Hash + Clone,
    {
        xs.iter().cloned().collect()
    }

    #[test]
    fn joinof_collect_moves() {
        let a: HashSet<_> = [1, 2].into_iter().collect();
        let b: HashSet<_> = [2, 3].into_iter().collect();
        let c: HashSet<_> = [3, 4].into_iter().collect();

        // Total collect (empty → bottom):
        let JoinOf(union_all) = [a.clone(), b.clone(), c.clone()]
            .into_iter()
            .collect::<JoinOf<_>>();

        assert_eq!(union_all, set(&[1, 2, 3, 4]));
    }

    #[test]
    fn joinof_collect_refs() {
        let a: HashSet<_> = [1, 2].into_iter().collect();
        let b: HashSet<_> = [2, 3].into_iter().collect();
        let c: HashSet<_> = [3, 4].into_iter().collect();

        // From references (no moves):
        let JoinOf(union_ref) = [&a, &b, &c].into_iter().collect::<JoinOf<_>>();

        assert_eq!(union_ref, set(&[1, 2, 3, 4]));
    }

    #[test]
    fn nonempty_join_collect() {
        let a: HashSet<_> = [1, 2].into_iter().collect();
        let b: HashSet<_> = [2, 3].into_iter().collect();
        let c: HashSet<_> = [3, 4].into_iter().collect();

        // Non-empty collect:
        let ne = NonEmptyJoinOf::from_iter_nonempty([a, b, c]).unwrap();
        assert_eq!(ne.0, set(&[1, 2, 3, 4]));
    }

    #[test]
    fn nonempty_join_empty_is_none() {
        // Empty case for non-empty variant:
        let none: Option<NonEmptyJoinOf<HashSet<i32>>> =
            NonEmptyJoinOf::from_iter_nonempty(std::iter::empty());
        assert!(none.is_none());
    }

    #[test]
    fn joinof_empty_is_bottom() {
        // For the bounded collector, empty → bottom (∅ for sets).
        let JoinOf(u) = std::iter::empty::<HashSet<i32>>().collect::<JoinOf<_>>();
        assert!(u.is_empty());
    }

    #[test]
    fn product_join_is_componentwise() {
        // A = Max<u32>, join = max
        let a = (Max(1u32), Max(10u32));
        let b = (Max(3u32), Max(7u32));

        let j = a.join(&b);

        // Componentwise max: (max(1,3), max(10,7)) = (3, 10)
        assert_eq!(j, (Max(3u32), Max(10u32)));
    }

    #[test]
    fn product_bottom_is_pair_of_bottoms() {
        // Use a simple bounded lattice: HashSet join = union, bottom = ∅
        type L = (HashSet<i32>, HashSet<i32>);

        let b: L = <L as BoundedJoinSemilattice>::bottom();

        // Both components should be the bottom of HashSet, i.e., empty.
        assert!(b.0.is_empty());
        assert!(b.1.is_empty());
    }

    #[cfg(all(test, feature = "derive"))]
    mod derive_tests {
        use super::*;

        use crate::join_semilattice::BoundedJoinSemilattice;
        use crate::join_semilattice::JoinSemilattice;

        use postbox_derive::BoundedJoinSemilattice;
        use postbox_derive::JoinSemilattice;

        #[derive(Debug, Clone, PartialEq, Eq, JoinSemilattice, BoundedJoinSemilattice)]
        struct Foo {
            // Both of these already have JoinSemilattice /
            // BoundedJoinSemilattice impls.
            a: Max<i32>,     // join = max, bottom = min_value()
            b: HashSet<i32>, // join = union, bottom = ∅
        }

        #[test]
        fn derived_join_is_fieldwise() {
            let x = Foo {
                a: Max(1),
                b: set(&[1, 2]), // helper from outer tests module
            };
            let y = Foo {
                a: Max(3),
                b: set(&[2, 3]),
            };

            let z = x.join(&y);

            // Check that each field is joined individually.
            assert_eq!(z.a, x.a.join(&y.a));
            assert_eq!(z.b, x.b.join(&y.b));

            // And that the whole Foo behaves as expected.
            assert_eq!(z.a, Max(3));
            assert_eq!(z.b, set(&[1, 2, 3]));
        }

        #[test]
        fn derived_bottom_is_struct_of_bottoms() {
            let b = Foo::bottom();

            // Max<i32> bottom is min_value()
            assert_eq!(b.a, Max(num_traits::Bounded::min_value()));

            // HashSet bottom is empty set
            assert!(b.b.is_empty());
        }
    }

    #[test]
    fn lattice_map_join_is_pointwise() {
        // Values are Max<i32>, so join = max on each entry.
        let mut m1: LatticeMap<&str, Max<i32>> = LatticeMap::new();
        m1.insert("a", Max(1));
        m1.insert("b", Max(10));

        let mut m2: LatticeMap<&str, Max<i32>> = LatticeMap::new();
        m2.insert("b", Max(7));
        m2.insert("c", Max(3));

        let j = m1.join(&m2);

        // Keys: union of {a,b} and {b,c} = {a,b,c}. Values: pointwise
        // max.
        assert_eq!(j.get(&"a"), Some(&Max(1)));
        assert_eq!(j.get(&"b"), Some(&Max(10))); // max(10, 7)
        assert_eq!(j.get(&"c"), Some(&Max(3)));
    }

    #[test]
    fn any_join_is_or() {
        // join = OR
        assert_eq!(Any(false).join(&Any(false)), Any(false));
        assert_eq!(Any(false).join(&Any(true)), Any(true));
        assert_eq!(Any(true).join(&Any(false)), Any(true));
        assert_eq!(Any(true).join(&Any(true)), Any(true));
    }

    #[test]
    fn any_bottom_is_false() {
        assert_eq!(Any::bottom(), Any(false));
        // false is the identity for OR
        assert_eq!(Any(false).join(&Any(true)), Any(true));
        assert_eq!(Any(true).join(&Any(false)), Any(true));
    }

    #[test]
    fn any_is_idempotent() {
        assert_eq!(Any(false).join(&Any(false)), Any(false));
        assert_eq!(Any(true).join(&Any(true)), Any(true));
    }

    #[test]
    fn all_join_is_and() {
        // join = AND
        assert_eq!(All(false).join(&All(false)), All(false));
        assert_eq!(All(false).join(&All(true)), All(false));
        assert_eq!(All(true).join(&All(false)), All(false));
        assert_eq!(All(true).join(&All(true)), All(true));
    }

    #[test]
    fn all_bottom_is_true() {
        assert_eq!(All::bottom(), All(true));
        // true is the identity for AND
        assert_eq!(All(true).join(&All(false)), All(false));
        assert_eq!(All(false).join(&All(true)), All(false));
    }

    #[test]
    fn all_is_idempotent() {
        assert_eq!(All(false).join(&All(false)), All(false));
        assert_eq!(All(true).join(&All(true)), All(true));
    }

    #[test]
    fn bitor_join_is_bitwise_or() {
        // u8 example
        let x = BitOr(0b0011u8);
        let y = BitOr(0b0101u8);
        assert_eq!(x.join(&y), BitOr(0b0111u8));

        // u16 example
        let a = BitOr(0x00FFu16);
        let b = BitOr(0xFF00u16);
        assert_eq!(a.join(&b), BitOr(0xFFFFu16));
    }

    #[test]
    fn bitor_bottom_is_zero() {
        assert_eq!(BitOr::<u8>::bottom(), BitOr(0));
        assert_eq!(BitOr::<u16>::bottom(), BitOr(0));
        assert_eq!(BitOr::<u32>::bottom(), BitOr(0));
        assert_eq!(BitOr::<u64>::bottom(), BitOr(0));
    }

    #[test]
    fn bitor_is_idempotent() {
        let x = BitOr(0b1010u8);
        assert_eq!(x.join(&x), x);
    }

    #[test]
    fn bitor_is_commutative() {
        let x = BitOr(0b0011u8);
        let y = BitOr(0b0101u8);
        assert_eq!(x.join(&y), y.join(&x));
    }

    #[test]
    fn bitor_is_associative() {
        let x = BitOr(0b0001u8);
        let y = BitOr(0b0010u8);
        let z = BitOr(0b0100u8);
        assert_eq!(x.join(&y).join(&z), x.join(&y.join(&z)));
    }

    #[test]
    fn bitor_into_inner() {
        let x = BitOr(0b1010u8);
        assert_eq!(x.into_inner(), 0b1010u8);
    }

    #[test]
    fn bitand_join_is_bitwise_and() {
        // u8 example
        let x = BitAnd(0b1111u8);
        let y = BitAnd(0b1010u8);
        assert_eq!(x.join(&y), BitAnd(0b1010u8));

        // u16 example
        let a = BitAnd(0xFF00u16);
        let b = BitAnd(0xF0F0u16);
        assert_eq!(a.join(&b), BitAnd(0xF000u16));
    }

    #[test]
    fn bitand_bottom_is_all_bits_set() {
        assert_eq!(BitAnd::<u8>::bottom(), BitAnd(0xFF));
        assert_eq!(BitAnd::<u16>::bottom(), BitAnd(0xFFFF));
        assert_eq!(BitAnd::<u32>::bottom(), BitAnd(0xFFFFFFFF));
        assert_eq!(BitAnd::<u64>::bottom(), BitAnd(0xFFFFFFFFFFFFFFFF));
    }

    #[test]
    fn bitand_is_idempotent() {
        let x = BitAnd(0b1010u8);
        assert_eq!(x.join(&x), x);
    }

    #[test]
    fn bitand_is_commutative() {
        let x = BitAnd(0b1100u8);
        let y = BitAnd(0b1010u8);
        assert_eq!(x.join(&y), y.join(&x));
    }

    #[test]
    fn bitand_is_associative() {
        let x = BitAnd(0b1111u8);
        let y = BitAnd(0b1100u8);
        let z = BitAnd(0b1010u8);
        assert_eq!(x.join(&y).join(&z), x.join(&y.join(&z)));
    }

    #[test]
    fn bitand_into_inner() {
        let x = BitAnd(0b1010u8);
        assert_eq!(x.into_inner(), 0b1010u8);
    }

    #[test]
    fn bitand_bottom_is_identity() {
        let x = BitAnd(0b1010u8);
        let bottom = BitAnd::<u8>::bottom();
        assert_eq!(bottom.join(&x), x);
        assert_eq!(x.join(&bottom), x);
    }

    #[cfg(all(test, feature = "bitflags"))]
    mod bitflags_tests {
        use super::*;
        use bitflags::bitflags;

        #[test]
        fn bitor_works_with_bitflags() {
            bitflags! {
                #[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
                struct TestFlags: u32 {
                    const FLAG_A = 0b0001;
                    const FLAG_B = 0b0010;
                    const FLAG_C = 0b0100;
                    const FLAG_D = 0b1000;
                }
            }

            let x = BitOr(TestFlags::FLAG_A | TestFlags::FLAG_B);
            let y = BitOr(TestFlags::FLAG_C);
            let z = x.join(&y);

            assert_eq!(
                z.0,
                TestFlags::FLAG_A | TestFlags::FLAG_B | TestFlags::FLAG_C
            );
            assert!(z.0.contains(TestFlags::FLAG_A));
            assert!(z.0.contains(TestFlags::FLAG_B));
            assert!(z.0.contains(TestFlags::FLAG_C));
            assert!(!z.0.contains(TestFlags::FLAG_D));

            // Test bottom
            let bottom = BitOr::<TestFlags>::bottom();
            assert_eq!(bottom.0, TestFlags::empty());

            // Bottom is identity
            assert_eq!(bottom.join(&x), x);
            assert_eq!(x.join(&bottom), x);
        }

        #[test]
        fn bitand_works_with_bitflags() {
            bitflags! {
                #[derive(Clone, Copy, Debug, PartialEq, Eq)]
                struct TestFlags: u32 {
                    const FLAG_A = 0b0001;
                    const FLAG_B = 0b0010;
                    const FLAG_C = 0b0100;
                    const FLAG_D = 0b1000;
                }
            }

            let x = BitAnd(TestFlags::FLAG_A | TestFlags::FLAG_B | TestFlags::FLAG_C);
            let y = BitAnd(TestFlags::FLAG_A | TestFlags::FLAG_B);
            let z = x.join(&y);

            // Intersection: only flags present in both
            assert_eq!(z.0, TestFlags::FLAG_A | TestFlags::FLAG_B);
            assert!(z.0.contains(TestFlags::FLAG_A));
            assert!(z.0.contains(TestFlags::FLAG_B));
            assert!(!z.0.contains(TestFlags::FLAG_C));
            assert!(!z.0.contains(TestFlags::FLAG_D));

            // Joining with all flags gives the intersection
            let all_flags = BitAnd(TestFlags::all());
            assert_eq!(all_flags.join(&x), x);
            assert_eq!(x.join(&all_flags), x);
        }
    }

    // Tests for algebra-core integration

    #[test]
    fn lattice_is_semigroup() {
        use algebra_core::Semigroup;

        let x: Max<i32> = 3.into();
        let y: Max<i32> = 5.into();

        // combine() works via blanket impl
        let z = x.combine(&y);
        assert_eq!(z, Max(5));

        // Associativity
        let a: Max<i32> = 1.into();
        let b: Max<i32> = 2.into();
        let c: Max<i32> = 3.into();
        assert_eq!(a.combine(&b).combine(&c), a.combine(&b.combine(&c)));
    }

    #[test]
    fn bounded_lattice_is_monoid() {
        use algebra_core::{Monoid, Semigroup};

        // empty() works via blanket impl
        let e = Max::<i32>::empty();
        assert_eq!(e, Max(i32::MIN));

        // Identity laws
        let x: Max<i32> = 42.into();
        assert_eq!(Max::empty().combine(&x), x);
        assert_eq!(x.combine(&Max::empty()), x);

        // concat() works via blanket impl
        let values = vec![Max(1), Max(3), Max(2)];
        assert_eq!(Max::concat(values), Max(3));

        // concat of empty is identity
        let empty: Vec<Max<i32>> = vec![];
        assert_eq!(Max::concat(empty), Max::empty());
    }

    #[test]
    fn bounded_lattice_is_commutative_monoid() {
        use algebra_core::{CommutativeMonoid, Semigroup};

        // Check that we can use the trait bound
        fn use_commutative_monoid<T: CommutativeMonoid>(a: &T, b: &T) -> T {
            a.combine(b)
        }

        let x: Max<i32> = 3.into();
        let y: Max<i32> = 5.into();

        // Commutativity
        assert_eq!(x.combine(&y), y.combine(&x));
        assert_eq!(use_commutative_monoid(&x, &y), Max(5));
    }
}
