#![deny(missing_docs)]
//! # algebra-core — Core algebraic abstractions
//!
//! **Part of the [postbox workspace](../index.html)**
//!
//! This crate provides fundamental algebraic structures as Rust
//! traits:
//!
//! - [`Semigroup`]: associative binary operation
//! - [`Monoid`]: semigroup with identity element
//! - [`CommutativeMonoid`]: monoid with commutative operation
//! - [`Group`]: monoid with inverse elements
//! - [`AbelianGroup`]: commutative group
//! - [`SemigroupHom`]: structure-preserving map between semigroups
//! - [`MonoidHom`]: structure-preserving map between monoids
//! - [`JoinSemilattice`]: associative, commutative, idempotent operation
//! - [`BoundedJoinSemilattice`]: join-semilattice with bottom element
//!
//! ## Quick start
//!
//! ```rust
//! use algebra_core::{Semigroup, Monoid, CommutativeMonoid, Sum};
//!
//! // Sum<T> is provided by the library for addition monoids
//! let a = Sum(5);
//! let b = Sum(3);
//! let c = a.combine(&b);
//! assert_eq!(c, Sum(8));
//! assert_eq!(Sum::<i32>::empty(), Sum(0));
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
//! ### Strings (concatenation)
//!
//! - **[`String`](String)**: `Semigroup`, `Monoid`
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

/// A **semigroup homomorphism**: a structure-preserving map between
/// semigroups.
///
/// A homomorphism `f: S → T` preserves the semigroup operation:
///
/// Laws (not enforced by type system):
///
/// - **Preserve combine**: `f(x.combine(y)) == f(x).combine(f(y))`
///
/// # Example
///
/// ```rust
/// use algebra_core::{Semigroup, Monoid, SemigroupHom};
///
/// // Wrapper for usize with addition
/// #[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// struct Sum(usize);
///
/// impl Semigroup for Sum {
///     fn combine(&self, other: &Self) -> Self {
///         Sum(self.0 + other.0)
///     }
/// }
///
/// impl Monoid for Sum {
///     fn empty() -> Self { Sum(0) }
/// }
///
/// // String length is a homomorphism from (String, concat) to (Sum, +)
/// struct Length;
///
/// impl SemigroupHom for Length {
///     type Source = String;
///     type Target = Sum;
///
///     fn apply(&self, s: &String) -> Sum {
///         Sum(s.len())
///     }
/// }
///
/// // Verify: len(s1 + s2) = len(s1) + len(s2)
/// let len = Length;
/// let s1 = String::from("hello");
/// let s2 = String::from("world");
/// assert_eq!(
///     len.apply(&s1.clone().combine(&s2)),
///     len.apply(&s1).combine(&len.apply(&s2))
/// );
/// ```
pub trait SemigroupHom {
    /// The source semigroup
    type Source: Semigroup;

    /// The target semigroup
    type Target: Semigroup;

    /// Apply the homomorphism
    fn apply(&self, x: &Self::Source) -> Self::Target;
}

/// Helper trait for explicitly specifying source and target types.
///
/// This is a blanket-implemented alias that allows writing
/// `T: SemigroupHomFromTo<S, T>` instead of
/// `T: SemigroupHom<Source = S, Target = T>`.
pub trait SemigroupHomFromTo<S: Semigroup, T: Semigroup>:
    SemigroupHom<Source = S, Target = T>
{
}

impl<H, S, T> SemigroupHomFromTo<S, T> for H
where
    H: SemigroupHom<Source = S, Target = T>,
    S: Semigroup,
    T: Semigroup,
{
}

/// A **monoid homomorphism**: a structure-preserving map between
/// monoids.
///
/// A homomorphism `f: M → N` preserves both the monoid operation and
/// identity:
///
/// Laws (not enforced by type system):
///
/// - **Preserve combine**: `f(x.combine(y)) == f(x).combine(f(y))`
/// - **Preserve identity**: `f(M::empty()) == N::empty()`
///
/// # Example
///
/// ```rust
/// use algebra_core::{Monoid, Semigroup, MonoidHom, SemigroupHom};
/// use std::collections::HashSet;
///
/// // Wrapper for usize with addition
/// #[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// struct Sum(usize);
///
/// impl Semigroup for Sum {
///     fn combine(&self, other: &Self) -> Self {
///         Sum(self.0 + other.0)
///     }
/// }
///
/// impl Monoid for Sum {
///     fn empty() -> Self { Sum(0) }
/// }
///
/// // Set cardinality: a monoid homomorphism from (HashSet, ∪) to
/// // (Sum, +)
/// // Note: Exact homomorphism property holds for disjoint unions
/// struct Cardinality;
///
/// impl SemigroupHom for Cardinality {
///     type Source = HashSet<i32>;
///     type Target = Sum;
///
///     fn apply(&self, s: &HashSet<i32>) -> Sum {
///         Sum(s.len())
///     }
/// }
///
/// impl MonoidHom for Cardinality {}
///
/// // Verify identity preservation: |∅| = 0
/// let card = Cardinality;
/// assert_eq!(card.apply(&HashSet::empty()), Sum::empty());
/// ```
pub trait MonoidHom: SemigroupHom {
    // Source and Target are already constrained to be Semigroups. We
    // further require them to be Monoids (but Rust doesn't let us
    // re-constrain associated types, so this is documented in the
    // laws)
}

/// Helper trait for explicitly specifying source and target monoids.
///
/// This is a blanket-implemented alias that allows writing
/// `T: MonoidHomFromTo<M, N>` instead of
/// `T: MonoidHom<Source = M, Target = N>`.
pub trait MonoidHomFromTo<M: Monoid, N: Monoid>: MonoidHom<Source = M, Target = N> {}

impl<H, M, N> MonoidHomFromTo<M, N> for H
where
    H: MonoidHom<Source = M, Target = N>,
    M: Monoid,
    N: Monoid,
{
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

// String: combine = concatenation

impl Semigroup for String {
    fn combine(&self, other: &Self) -> Self {
        let mut result = self.clone();
        result.push_str(other);
        result
    }
}

impl Monoid for String {
    fn empty() -> Self {
        String::new()
    }
}

// ============================================================
// Numeric wrappers: Sum and Product
// ============================================================

/// Wrapper type for the addition monoid.
///
/// `Sum<T>` forms a monoid under addition (`+`), making it useful for
/// accumulating numeric values, counting, or gradient accumulation in
/// automatic differentiation.
///
/// # Examples
///
/// ```
/// use algebra_core::{Monoid, Semigroup, Sum};
///
/// let a = Sum(5);
/// let b = Sum(3);
/// let c = a.combine(&b);
/// assert_eq!(c, Sum(8));
///
/// assert_eq!(Sum::<i32>::empty(), Sum(0));
/// ```
///
/// # Use in Autodiff
///
/// Gradient accumulation in reverse-mode automatic differentiation
/// is fundamentally `Sum<f64>`:
///
/// ```
/// use algebra_core::{Semigroup, Sum};
///
/// let grad1 = Sum(0.5);
/// let grad2 = Sum(0.3);
/// let total_grad = grad1.combine(&grad2);
/// assert_eq!(total_grad.0, 0.8);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Sum<T>(pub T);

impl<T: std::ops::Add<Output = T> + Clone> Semigroup for Sum<T> {
    fn combine(&self, other: &Self) -> Self {
        Sum(self.0.clone() + other.0.clone())
    }
}

impl<T: std::ops::Add<Output = T> + Clone + num_traits::Zero> Monoid for Sum<T> {
    fn empty() -> Self {
        Sum(T::zero())
    }
}

impl<T: std::ops::Add<Output = T> + Clone + num_traits::Zero> CommutativeMonoid for Sum<T> {}

/// Wrapper type for the multiplication monoid.
///
/// `Product<T>` forms a monoid under multiplication (`*`), useful for
/// computing products, scaling factors, or combining probabilities.
///
/// # Examples
///
/// ```
/// use algebra_core::{Monoid, Semigroup, Product};
///
/// let a = Product(5);
/// let b = Product(3);
/// let c = a.combine(&b);
/// assert_eq!(c, Product(15));
///
/// assert_eq!(Product::<i32>::empty(), Product(1));
/// ```
///
/// # Combining Probabilities
///
/// ```
/// use algebra_core::{Semigroup, Product};
///
/// let prob1 = Product(0.5);
/// let prob2 = Product(0.5);
/// let joint_prob = prob1.combine(&prob2);
/// assert_eq!(joint_prob.0, 0.25);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Product<T>(pub T);

impl<T: std::ops::Mul<Output = T> + Clone> Semigroup for Product<T> {
    fn combine(&self, other: &Self) -> Self {
        Product(self.0.clone() * other.0.clone())
    }
}

impl<T: std::ops::Mul<Output = T> + Clone + num_traits::One> Monoid for Product<T> {
    fn empty() -> Self {
        Product(T::one())
    }
}

impl<T: std::ops::Mul<Output = T> + Clone + num_traits::One> CommutativeMonoid for Product<T> {}

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

// Re-export derive macros when derive feature is enabled
#[cfg(feature = "derive")]
pub use algebra_core_derive::{
    AbelianGroup, BoundedJoinSemilattice, CommutativeMonoid, Group, JoinSemilattice, Monoid,
    Semigroup,
};

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

    // ============================================================
    // Standard library implementations
    // ============================================================

    #[test]
    fn hashset_semigroup_is_union() {
        use std::collections::HashSet;
        let a: HashSet<_> = [1, 2, 3].into_iter().collect();
        let b: HashSet<_> = [3, 4, 5].into_iter().collect();
        let result = a.combine(&b);
        let expected: HashSet<_> = [1, 2, 3, 4, 5].into_iter().collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn hashset_monoid_empty_is_empty_set() {
        use std::collections::HashSet;
        let empty = HashSet::<i32>::empty();
        assert!(empty.is_empty());
    }

    #[test]
    fn hashset_join_semilattice_is_union() {
        use std::collections::HashSet;
        let a: HashSet<_> = [1, 2].into_iter().collect();
        let b: HashSet<_> = [2, 3].into_iter().collect();
        assert_eq!(a.join(&b), [1, 2, 3].into_iter().collect());
    }

    #[test]
    fn hashset_bottom_is_empty() {
        use std::collections::HashSet;
        let bottom = HashSet::<i32>::bottom();
        assert!(bottom.is_empty());
    }

    #[test]
    fn btreeset_semigroup_is_union() {
        use std::collections::BTreeSet;
        let a: BTreeSet<_> = [1, 2, 3].into_iter().collect();
        let b: BTreeSet<_> = [3, 4, 5].into_iter().collect();
        let result = a.combine(&b);
        let expected: BTreeSet<_> = [1, 2, 3, 4, 5].into_iter().collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn btreeset_bottom_is_empty() {
        use std::collections::BTreeSet;
        let bottom = BTreeSet::<i32>::bottom();
        assert!(bottom.is_empty());
    }

    #[test]
    fn string_semigroup_is_concatenation() {
        let a = String::from("Hello");
        let b = String::from(" World");
        assert_eq!(a.combine(&b), "Hello World");
    }

    #[test]
    fn string_monoid_empty_is_empty_string() {
        assert_eq!(String::empty(), "");
    }

    #[test]
    fn option_semigroup_combines_inner_values() {
        let a = Some(Sum(3));
        let b = Some(Sum(5));
        assert_eq!(a.combine(&b), Some(Sum(8)));
    }

    #[test]
    fn option_semigroup_none_propagates() {
        let a: Option<Sum> = None;
        let b = Some(Sum(5));
        assert_eq!(a.combine(&b), Some(Sum(5)));
        assert_eq!(b.combine(&a), Some(Sum(5)));
    }

    #[test]
    fn option_monoid_empty_is_none() {
        assert_eq!(Option::<Sum>::empty(), None);
    }

    #[test]
    fn option_join_semilattice_joins_inner() {
        use std::collections::HashSet;
        let a: Option<HashSet<_>> = Some([1, 2].into_iter().collect());
        let b: Option<HashSet<_>> = Some([2, 3].into_iter().collect());
        let expected: Option<HashSet<_>> = Some([1, 2, 3].into_iter().collect());
        assert_eq!(a.join(&b), expected);
    }

    #[test]
    fn option_bottom_is_none() {
        use std::collections::HashSet;
        let bottom = Option::<HashSet<i32>>::bottom();
        assert_eq!(bottom, None);
    }

    // ============================================================
    // Sum<T> and Product<T> tests
    // ============================================================

    #[test]
    fn sum_wrapper_implements_semigroup() {
        use crate::Sum;
        let a = Sum(5);
        let b = Sum(3);
        assert_eq!(a.combine(&b), Sum(8));
    }

    #[test]
    fn sum_wrapper_monoid_empty_is_zero() {
        use crate::Sum;
        assert_eq!(Sum::<i32>::empty(), Sum(0));
    }

    #[test]
    fn product_wrapper_implements_semigroup() {
        use crate::Product;
        let a = Product(5);
        let b = Product(3);
        assert_eq!(a.combine(&b), Product(15));
    }

    #[test]
    fn product_wrapper_monoid_empty_is_one() {
        use crate::Product;
        assert_eq!(Product::<i32>::empty(), Product(1));
    }

    // Tests for derive macros on tuple structs
    #[cfg(feature = "derive")]
    mod derive_tuple_tests {
        use super::*;

        #[derive(Clone, Debug, PartialEq, Eq, Semigroup, Monoid)]
        struct TupleMonoid(Sum, HashSet<String>);

        #[test]
        fn tuple_monoid_empty_works() {
            let empty = TupleMonoid::empty();
            assert_eq!(empty.0, Sum(0));
            assert_eq!(empty.1, HashSet::<String>::new());
        }

        #[test]
        fn tuple_semigroup_combine_works() {
            let x = TupleMonoid(Sum(3), ["a".to_string()].into_iter().collect());
            let y = TupleMonoid(Sum(5), ["b".to_string()].into_iter().collect());
            let result = x.combine(&y);
            assert_eq!(result.0, Sum(8));
            assert_eq!(
                result.1,
                ["a".to_string(), "b".to_string()].into_iter().collect()
            );
        }

        #[derive(Clone, Debug, PartialEq, Eq, JoinSemilattice, BoundedJoinSemilattice)]
        struct TupleLattice(HashSet<i32>, HashSet<String>);

        #[test]
        fn tuple_lattice_bottom_works() {
            let bottom = TupleLattice::bottom();
            assert!(bottom.0.is_empty());
            assert!(bottom.1.is_empty());
        }

        #[test]
        fn tuple_lattice_join_works() {
            let x = TupleLattice(
                [1, 2].into_iter().collect(),
                ["a".to_string()].into_iter().collect(),
            );
            let y = TupleLattice(
                [2, 3].into_iter().collect(),
                ["b".to_string()].into_iter().collect(),
            );
            let result = x.join(&y);
            assert_eq!(result.0, [1, 2, 3].into_iter().collect());
            assert_eq!(
                result.1,
                ["a".to_string(), "b".to_string()].into_iter().collect()
            );
        }

        #[derive(Clone, Copy, Debug, PartialEq, Eq, Semigroup, Monoid, Group)]
        struct TupleGroup(AddInt, AddInt);

        #[test]
        fn tuple_group_inverse_works() {
            let x = TupleGroup(AddInt(3), AddInt(5));
            let inv = x.inverse();
            assert_eq!(inv.0, AddInt(-3));
            assert_eq!(inv.1, AddInt(-5));
            assert_eq!(x.combine(&inv), TupleGroup::empty());
        }
    }
}
