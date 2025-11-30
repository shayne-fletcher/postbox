//! Core **join-semilattice** traits and building blocks.
//!
//! This module defines [`JoinSemilattice`] and
//! [`BoundedJoinSemilattice`], plus a small toolkit of concrete
//! lattices and combinators that are convenient for CRDTs and other
//! monotone data structures.
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
//! - [`crate::join_semilattice::BitOr`]:
//!   bitwise OR lattice for bitflags and integer masks.
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
//! let x = Max(1);
//! let y = Max(3);
//!
//! // join = max
//! let z = x.join(&y);
//! assert_eq!(z.0, 3);
//! ```
//!
//! These primitives are intentionally small and generic. Higher-level
//! CRDTs in this crate build on them to express their replica states
//! as lattices with well-defined, convergent merge operations.
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::Hash;
use std::ops::Deref;

/// A type whose values form a **join-semilattice**.
///
/// Provides a binary `join` (∨) that must be **associative**,
/// **commutative**, and **idempotent** (not enforced by the type
/// system):
///   - Commutative: a.join(b) == b.join(a)
///   - Associative: a.join(b).join(c) == a.join(b.join(c))
///   - Idempotent: a.join(a) == a
///
/// The induced partial order is: `x ≤ y` iff `x.join(&y) == y`. When
/// a bottom element exists, also implement `BoundedJoinSemilattice`
/// with `bottom()`.
pub trait JoinSemilattice: Sized {
    /// The join (least upper bound).
    fn join(&self, other: &Self) -> Self;

    /// In-place variant.
    fn join_assign(&mut self, other: &Self) {
        let x = self.join(other);
        *self = x;
    }

    /// Derived partial order: x <= y iff x ∨ y = y.
    fn leq(&self, other: &Self) -> bool
    where
        Self: PartialEq,
    {
        self.join(other) == *other
    }

    /// Join a finite iterator of values. Returns `None` for empty
    /// iterators.
    fn join_all<I>(it: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self>,
        Self: Sized,
    {
        it.into_iter().reduce(|acc, x| acc.join(&x))
    }

    /// A by-reference variant to avoid moving items.
    fn join_all_ref<'a, I>(it: I) -> Option<Self>
    where
        I: IntoIterator<Item = &'a Self>,
        Self: Sized + 'a + Clone,
    {
        it.into_iter().cloned().reduce(|acc, x| acc.join(&x))
    }
}

/// A join-semilattice with a bottom element (⊥).
pub trait BoundedJoinSemilattice: JoinSemilattice {
    /// The bottom element of the lattice (⊥).
    ///
    /// This is the least element w.r.t. the induced partial order:
    /// for all `x`, `bottom().join(&x) == x`.
    fn bottom() -> Self;

    /// Join a finite iterator of values, starting from ⊥.
    ///
    /// This is equivalent to:
    ///
    /// ```ignore
    /// it.into_iter().fold(Self::bottom(), |acc, x| acc.join(&x))
    /// ```
    ///
    /// It never returns `None`: an empty iterator produces
    /// `bottom()`.
    fn join_all_from_bottom<I>(it: I) -> Self
    where
        I: IntoIterator<Item = Self>,
    {
        it.into_iter().fold(Self::bottom(), |acc, x| acc.join(&x))
    }
}

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
/// let x = Any(false);
/// let y = Any(true);
/// assert_eq!(x.join(&y), Any(true));
/// assert_eq!(Any::bottom(), Any(false));
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
/// let x = All(true);
/// let y = All(false);
/// assert_eq!(x.join(&y), All(false));
/// assert_eq!(All::bottom(), All(true));
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

// HashSet: join = union

impl<T: Eq + Hash + Clone> JoinSemilattice for HashSet<T> {
    fn join(&self, other: &Self) -> Self {
        if self.len() >= other.len() {
            let mut out = self.clone();
            out.extend(other.iter().cloned());
            out
        } else {
            let mut out = other.clone();
            out.extend(self.iter().cloned());
            out
        }
    }
}

impl<T: Eq + Hash + Clone> BoundedJoinSemilattice for HashSet<T> {
    fn bottom() -> Self {
        HashSet::new()
    }
}

// BTreeSet: join = union.

impl<T: Ord + Clone> JoinSemilattice for BTreeSet<T> {
    fn join(&self, other: &Self) -> Self {
        // BTreeSet::union returns an iterator.
        self.union(other).cloned().collect()
    }
}

impl<T: Ord + Clone> BoundedJoinSemilattice for BTreeSet<T> {
    fn bottom() -> Self {
        BTreeSet::new()
    }
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

// Option: None as bottom; Some joins inner

impl<L: JoinSemilattice + Clone> JoinSemilattice for Option<L> {
    fn join(&self, other: &Self) -> Self {
        match (self, other) {
            (None, x) | (x, None) => x.clone(),
            (Some(a), Some(b)) => Some(a.join(b)),
        }
    }
}

impl<L: JoinSemilattice + Clone> BoundedJoinSemilattice for Option<L> {
    fn bottom() -> Self {
        None
    }
}

// Product lattice: componentwise join

// impl<A, B> JoinSemilattice for (A, B)
// where
//     A: JoinSemilattice + Clone,
//     B: JoinSemilattice + Clone,
// {
//     fn join(&self, other: &Self) -> Self {
//         (self.0.join(&other.0), self.1.join(&other.1))
//     }
// }
//
// impl<A, B> BoundedJoinSemilattice for (A, B)
// where
//     A: BoundedJoinSemilattice + Clone,
//     B: BoundedJoinSemilattice + Clone,
// {
//     fn bottom() -> Self {
//         (A::bottom(), B::bottom())
//     }
// }

macro_rules! impl_product_lattice {
    ( $( $T:ident : $idx:tt ),+ ) => {
        impl<$( $T ),+> JoinSemilattice for ( $( $T, )+ )
        where
            $( $T: JoinSemilattice ),+
        {
            fn join(&self, other: &Self) -> Self {
                (
                    $( self.$idx.join(&other.$idx) ),+
                )
            }
        }

        impl<$( $T ),+> BoundedJoinSemilattice for ( $( $T, )+ )
        where
            $( $T: BoundedJoinSemilattice ),+
        {
            fn bottom() -> Self {
                (
                    $( $T::bottom() ),+
                )
            }
        }
    }
}

impl_product_lattice!(A:0, B:1);
impl_product_lattice!(A:0, B:1, C:2);
impl_product_lattice!(A:0, B:1, C:2, D:3);

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
    fn set<T: Eq + std::hash::Hash>(xs: &[T]) -> HashSet<T>
    where
        T: Clone,
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
    }
}
