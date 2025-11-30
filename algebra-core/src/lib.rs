#![deny(missing_docs)]
//! # algebra-core — Core algebraic abstractions
//!
//! This crate provides fundamental algebraic structures as Rust
//! traits:
//!
//! - [`Semigroup`]: associative binary operation
//! - [`Monoid`]: semigroup with identity element
//! - [`CommutativeMonoid`]: monoid with commutative operation
//! - [`JoinSemilattice`]: associative, commutative, idempotent operation
//! - [`BoundedJoinSemilattice`]: join-semilattice with bottom element
//! - [`Group`]: monoid with inverse elements
//! - [`AbelianGroup`]: commutative group
//!
//! ## Quick start
//!
//! ```rust
//! use algebra_core::{Semigroup, Monoid, CommutativeMonoid};
//!
//! // Integers under addition form a commutative monoid
//! #[derive(Clone, Copy, Debug, PartialEq, Eq)]
//! struct Sum(i32);
//!
//! impl Semigroup for Sum {
//!     fn combine(&self, other: &Self) -> Self {
//!         Sum(self.0 + other.0)
//!     }
//! }
//!
//! impl Monoid for Sum {
//!     fn empty() -> Self {
//!         Sum(0)
//!     }
//! }
//!
//! // Use it
//! let x = Sum(3);
//! let y = Sum(5);
//! assert_eq!(x.combine(&y), Sum(8));
//! assert_eq!(Sum::empty().combine(&x), x);
//! ```
//!
//! ## Standard library implementations
//!
//! This crate provides implementations for common standard library types:
//!
//! ### Sets (union as join/combine)
//!
//! - **[`HashSet<T>`](std::collections::HashSet)**: `JoinSemilattice`,
//!   `BoundedJoinSemilattice`, `Semigroup`, `Monoid`, `CommutativeMonoid`
//! - **[`BTreeSet<T>`](std::collections::BTreeSet)**: `JoinSemilattice`,
//!   `BoundedJoinSemilattice`, `Semigroup`, `Monoid`, `CommutativeMonoid`
//!
//! ### Optional values (lifted monoid/lattice)
//!
//! - **[`Option<M>`](Option)** (where `M: Semigroup + Clone`): `Semigroup`
//! - **[`Option<M>`](Option)** (where `M: Monoid + Clone`): `Monoid`, `Semigroup`
//! - **[`Option<M>`](Option)** (where `M: CommutativeMonoid + Clone`): `CommutativeMonoid`
//! - **[`Option<L>`](Option)** (where `L: JoinSemilattice + Clone`):
//!   `JoinSemilattice`, `BoundedJoinSemilattice`
//!   - `None` is bottom/empty, `Some(a) ⊔ Some(b) = Some(a ⊔ b)`
//!
//! ### Tuples (product algebras)
//!
//! - **`()`**: All traits (trivial implementations)
//! - **`(A,)`, `(A, B)`, `(A, B, C)`, `(A, B, C, D)`**:
//!   - `Semigroup`, `Monoid`, `CommutativeMonoid` (componentwise)
//!   - `JoinSemilattice`, `BoundedJoinSemilattice` (componentwise)
//!
//! All tuple implementations require component types to implement the
//! corresponding trait, and operations are applied componentwise.

// Make the current crate visible as `algebra_core` for consistency
extern crate self as algebra_core;

/// A **semigroup**: a type with an associative binary operation.
///
/// Laws (not enforced by type system):
///
/// - **Associative**:
///   `a.combine(b).combine(c) == a.combine(b.combine(c))`
///
/// # Example
///
/// ```rust
/// use algebra_core::Semigroup;
///
/// #[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// struct Max(i32);
///
/// impl Semigroup for Max {
///     fn combine(&self, other: &Self) -> Self {
///         Max(self.0.max(other.0))
///     }
/// }
///
/// let x = Max(3);
/// let y = Max(5);
/// let z = Max(2);
/// assert_eq!(x.combine(&y).combine(&z), x.combine(&y.combine(&z)));
/// ```
pub trait Semigroup: Sized {
    /// Combine two elements associatively.
    fn combine(&self, other: &Self) -> Self;

    /// In-place combine.
    fn combine_assign(&mut self, other: &Self) {
        *self = self.combine(other);
    }
}

/// A **monoid**: a semigroup with an identity element.
///
/// Laws (not enforced by type system):
///
/// - **Associative**:
///   `a.combine(b).combine(c) == a.combine(b.combine(c))`
/// - **Left identity**: `empty().combine(a) == a`
/// - **Right identity**: `a.combine(empty()) == a`
///
/// # Example
///
/// ```rust
/// use algebra_core::{Semigroup, Monoid};
///
/// #[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// struct Product(i32);
///
/// impl Semigroup for Product {
///     fn combine(&self, other: &Self) -> Self {
///         Product(self.0 * other.0)
///     }
/// }
///
/// impl Monoid for Product {
///     fn empty() -> Self {
///         Product(1)
///     }
/// }
///
/// let x = Product(3);
/// let y = Product(5);
/// assert_eq!(x.combine(&y), Product(15));
/// assert_eq!(Product::empty().combine(&x), x);
/// assert_eq!(x.combine(&Product::empty()), x);
/// ```
pub trait Monoid: Semigroup {
    /// The identity element.
    fn empty() -> Self;

    /// Fold an iterator using combine, starting from empty.
    fn concat<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Self>,
    {
        iter.into_iter()
            .fold(Self::empty(), |acc, x| acc.combine(&x))
    }
}

/// A **commutative monoid**: a monoid where combine is commutative.
///
/// Laws (not enforced by type system):
///
/// - **Associative**:
///   `a.combine(b).combine(c) == a.combine(b.combine(c))`
/// - **Commutative**: `a.combine(b) == b.combine(a)`
/// - **Identity**: `a.combine(empty()) == a == empty().combine(a)`
///
/// # Example
///
/// ```rust
/// use algebra_core::{Semigroup, Monoid, CommutativeMonoid};
///
/// #[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// struct Sum(i32);
///
/// impl Semigroup for Sum {
///     fn combine(&self, other: &Self) -> Self {
///         Sum(self.0 + other.0)
///     }
/// }
///
/// impl Monoid for Sum {
///     fn empty() -> Self {
///         Sum(0)
///     }
/// }
///
/// impl CommutativeMonoid for Sum {}
///
/// let x = Sum(3);
/// let y = Sum(5);
/// assert_eq!(x.combine(&y), y.combine(&x)); // commutative
/// ```
pub trait CommutativeMonoid: Monoid {
    // Marker trait - laws are documented above
}

/// A **group**: a monoid where every element has an inverse.
///
/// Laws (not enforced by type system):
///
/// - **Associative**: `a.combine(b).combine(c) == a.combine(b.combine(c))`
/// - **Identity**: `a.combine(empty()) == a == empty().combine(a)`
/// - **Inverse**: `a.combine(a.inverse()) == empty() == a.inverse().combine(a)`
///
/// # Example
///
/// ```rust
/// use algebra_core::{Semigroup, Monoid, Group};
///
/// #[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// struct AddInt(i32);
///
/// impl Semigroup for AddInt {
///     fn combine(&self, other: &Self) -> Self {
///         AddInt(self.0 + other.0)
///     }
/// }
///
/// impl Monoid for AddInt {
///     fn empty() -> Self {
///         AddInt(0)
///     }
/// }
///
/// impl Group for AddInt {
///     fn inverse(&self) -> Self {
///         AddInt(-self.0)
///     }
/// }
///
/// let x = AddInt(5);
/// let inv = x.inverse();
/// assert_eq!(x.combine(&inv), AddInt::empty());
/// ```
pub trait Group: Monoid {
    /// Return the inverse of this element.
    fn inverse(&self) -> Self;
}

/// An **abelian group** (commutative group): a group where combine is
/// commutative.
///
/// Laws (not enforced by type system):
///
/// - **Associative**: `a.combine(b).combine(c) == a.combine(b.combine(c))`
/// - **Commutative**: `a.combine(b) == b.combine(a)`
/// - **Identity**: `a.combine(empty()) == a == empty().combine(a)`
/// - **Inverse**: `a.combine(a.inverse()) == empty() == a.inverse().combine(a)`
///
/// Named after mathematician Niels Henrik Abel.
pub trait AbelianGroup: Group + CommutativeMonoid {
    // Marker trait - combines Group and CommutativeMonoid
}

/// A **join-semilattice**: a type with an associative, commutative,
/// idempotent binary operation.
///
/// Laws (not enforced by type system):
///
/// - **Associative**: `a.join(b).join(c) == a.join(b.join(c))`
/// - **Commutative**: `a.join(b) == b.join(a)`
/// - **Idempotent**: `a.join(a) == a`
///
/// The `join` operation computes the least upper bound (supremum) in the
/// induced partial order: `x ≤ y` iff `x.join(y) == y`.
///
/// # Example
///
/// ```rust
/// use algebra_core::JoinSemilattice;
/// use std::collections::HashSet;
///
/// let a: HashSet<_> = [1, 2].into_iter().collect();
/// let b: HashSet<_> = [2, 3].into_iter().collect();
///
/// // join = union
/// let c = a.join(&b);
/// assert_eq!(c, [1, 2, 3].into_iter().collect());
///
/// // Idempotent
/// assert_eq!(a.join(&a), a);
/// ```
pub trait JoinSemilattice: Sized {
    /// The join (least upper bound).
    fn join(&self, other: &Self) -> Self;

    /// In-place variant.
    fn join_assign(&mut self, other: &Self) {
        *self = self.join(other);
    }

    /// Derived partial order: x ≤ y iff x ∨ y = y.
    fn leq(&self, other: &Self) -> bool
    where
        Self: PartialEq,
    {
        self.join(other) == *other
    }

    /// Join a finite iterator of values. Returns `None` for empty iterators.
    fn join_all<I>(it: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self>,
    {
        it.into_iter().reduce(|acc, x| acc.join(&x))
    }
}

/// A **bounded join-semilattice**: a join-semilattice with a bottom element.
///
/// Laws (not enforced by type system):
///
/// - **Associative**: `a.join(b).join(c) == a.join(b.join(c))`
/// - **Commutative**: `a.join(b) == b.join(a)`
/// - **Idempotent**: `a.join(a) == a`
/// - **Identity**: `bottom().join(a) == a == a.join(bottom())`
///
/// The bottom element (⊥) is the least element in the partial order.
///
/// # Example
///
/// ```rust
/// use algebra_core::{BoundedJoinSemilattice, JoinSemilattice};
/// use std::collections::HashSet;
///
/// let a: HashSet<_> = [1, 2].into_iter().collect();
///
/// // bottom = empty set
/// let bottom = HashSet::<i32>::bottom();
/// assert!(bottom.is_empty());
///
/// // Identity law
/// assert_eq!(bottom.join(&a), a);
/// assert_eq!(a.join(&bottom), a);
/// ```
pub trait BoundedJoinSemilattice: JoinSemilattice {
    /// The bottom element of the lattice (⊥).
    ///
    /// This is the least element w.r.t. the induced partial order:
    /// for all `x`, `bottom().join(x) == x`.
    fn bottom() -> Self;

    /// Join a finite iterator of values, starting from ⊥.
    ///
    /// Never returns `None`: an empty iterator produces `bottom()`.
    fn join_all_from_bottom<I>(it: I) -> Self
    where
        I: IntoIterator<Item = Self>,
    {
        it.into_iter().fold(Self::bottom(), |acc, x| acc.join(&x))
    }
}

// Implementations for standard library types

use std::collections::{BTreeSet, HashSet};
use std::hash::Hash;

// HashSet: join = union

impl<T: Eq + Hash + Clone> JoinSemilattice for HashSet<T> {
    fn join(&self, other: &Self) -> Self {
        self.union(other).cloned().collect()
    }
}

impl<T: Eq + Hash + Clone> BoundedJoinSemilattice for HashSet<T> {
    fn bottom() -> Self {
        HashSet::new()
    }
}

impl<T: Eq + Hash + Clone> Semigroup for HashSet<T> {
    fn combine(&self, other: &Self) -> Self {
        self.join(other)
    }
}

impl<T: Eq + Hash + Clone> Monoid for HashSet<T> {
    fn empty() -> Self {
        Self::bottom()
    }
}

impl<T: Eq + Hash + Clone> CommutativeMonoid for HashSet<T> {}

// BTreeSet: join = union

impl<T: Ord + Clone> JoinSemilattice for BTreeSet<T> {
    fn join(&self, other: &Self) -> Self {
        self.union(other).cloned().collect()
    }
}

impl<T: Ord + Clone> BoundedJoinSemilattice for BTreeSet<T> {
    fn bottom() -> Self {
        BTreeSet::new()
    }
}

impl<T: Ord + Clone> Semigroup for BTreeSet<T> {
    fn combine(&self, other: &Self) -> Self {
        self.join(other)
    }
}

impl<T: Ord + Clone> Monoid for BTreeSet<T> {
    fn empty() -> Self {
        Self::bottom()
    }
}

impl<T: Ord + Clone> CommutativeMonoid for BTreeSet<T> {}

// Option: lifted lattice

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

impl<M: Semigroup + Clone> Semigroup for Option<M> {
    fn combine(&self, other: &Self) -> Self {
        match (self, other) {
            (None, x) | (x, None) => x.clone(),
            (Some(a), Some(b)) => Some(a.combine(b)),
        }
    }
}

impl<M: Monoid + Clone> Monoid for Option<M> {
    fn empty() -> Self {
        None
    }
}

impl<M: CommutativeMonoid + Clone> CommutativeMonoid for Option<M> {}

// Unit type

impl JoinSemilattice for () {
    fn join(&self, _other: &Self) -> Self {}
}

impl BoundedJoinSemilattice for () {
    fn bottom() -> Self {}
}

impl Semigroup for () {
    fn combine(&self, _other: &Self) -> Self {}
}

impl Monoid for () {
    fn empty() -> Self {}
}

impl CommutativeMonoid for () {}

impl Group for () {
    fn inverse(&self) -> Self {}
}

impl AbelianGroup for () {}

// Tuples: product lattices

macro_rules! impl_product_lattice {
    ( $( $T:ident : $idx:tt ),+ ) => {
        impl<$( $T ),+> JoinSemilattice for ( $( $T, )+ )
        where
            $( $T: JoinSemilattice ),+
        {
            fn join(&self, other: &Self) -> Self {
                (
                    $( self.$idx.join(&other.$idx), )+
                )
            }
        }

        impl<$( $T ),+> BoundedJoinSemilattice for ( $( $T, )+ )
        where
            $( $T: BoundedJoinSemilattice ),+
        {
            fn bottom() -> Self {
                (
                    $( $T::bottom(), )+
                )
            }
        }
    }
}

impl_product_lattice!(A:0);
impl_product_lattice!(A:0, B:1);
impl_product_lattice!(A:0, B:1, C:2);
impl_product_lattice!(A:0, B:1, C:2, D:3);

// Tuples: product monoids

macro_rules! impl_product_monoid {
    ( $( $T:ident : $idx:tt ),+ ) => {
        impl<$( $T ),+> Semigroup for ( $( $T, )+ )
        where
            $( $T: Semigroup ),+
        {
            fn combine(&self, other: &Self) -> Self {
                (
                    $( self.$idx.combine(&other.$idx), )+
                )
            }
        }

        impl<$( $T ),+> Monoid for ( $( $T, )+ )
        where
            $( $T: Monoid ),+
        {
            fn empty() -> Self {
                (
                    $( $T::empty(), )+
                )
            }
        }

        impl<$( $T ),+> CommutativeMonoid for ( $( $T, )+ )
        where
            $( $T: CommutativeMonoid ),+
        {
        }
    }
}

impl_product_monoid!(A:0);
impl_product_monoid!(A:0, B:1);
impl_product_monoid!(A:0, B:1, C:2);
impl_product_monoid!(A:0, B:1, C:2, D:3);

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Sum(i32);

    impl Semigroup for Sum {
        fn combine(&self, other: &Self) -> Self {
            Sum(self.0 + other.0)
        }
    }

    impl Monoid for Sum {
        fn empty() -> Self {
            Sum(0)
        }
    }

    impl CommutativeMonoid for Sum {}

    #[test]
    fn semigroup_combine_works() {
        let x = Sum(3);
        let y = Sum(5);
        assert_eq!(x.combine(&y), Sum(8));
    }

    #[test]
    fn semigroup_is_associative() {
        let x = Sum(1);
        let y = Sum(2);
        let z = Sum(3);
        assert_eq!(x.combine(&y).combine(&z), x.combine(&y.combine(&z)));
    }

    #[test]
    fn monoid_has_identity() {
        let x = Sum(5);
        assert_eq!(Sum::empty().combine(&x), x);
        assert_eq!(x.combine(&Sum::empty()), x);
    }

    #[test]
    fn monoid_concat_works() {
        let values = vec![Sum(1), Sum(2), Sum(3)];
        assert_eq!(Sum::concat(values), Sum(6));
    }

    #[test]
    fn monoid_concat_empty_is_identity() {
        let empty: Vec<Sum> = vec![];
        assert_eq!(Sum::concat(empty), Sum::empty());
    }

    #[test]
    fn commutative_monoid_is_commutative() {
        let x = Sum(3);
        let y = Sum(5);
        assert_eq!(x.combine(&y), y.combine(&x));
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct AddInt(i32);

    impl Semigroup for AddInt {
        fn combine(&self, other: &Self) -> Self {
            AddInt(self.0 + other.0)
        }
    }

    impl Monoid for AddInt {
        fn empty() -> Self {
            AddInt(0)
        }
    }

    impl CommutativeMonoid for AddInt {}

    impl Group for AddInt {
        fn inverse(&self) -> Self {
            AddInt(-self.0)
        }
    }

    impl AbelianGroup for AddInt {}

    #[test]
    fn group_has_inverse() {
        let x = AddInt(5);
        assert_eq!(x.combine(&x.inverse()), AddInt::empty());
        assert_eq!(x.inverse().combine(&x), AddInt::empty());
    }

    #[test]
    fn abelian_group_is_commutative() {
        let x = AddInt(3);
        let y = AddInt(-7);
        assert_eq!(x.combine(&y), y.combine(&x));
    }
}
