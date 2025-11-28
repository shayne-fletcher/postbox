use std::collections::BTreeSet;
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

    // In-place variant.
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

    // A by-reference variant to avoid moving items.
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
    fn bottom() -> Self;

    fn join_all_from_bottom<I>(it: I) -> Self
    where
        I: IntoIterator<Item = Self>,
    {
        it.into_iter().fold(Self::bottom(), |acc, x| acc.join(&x))
    }
}

// join = max

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

impl<A, B> JoinSemilattice for (A, B)
where
    A: JoinSemilattice + Clone,
    B: JoinSemilattice + Clone,
{
    fn join(&self, other: &Self) -> Self {
        (self.0.join(&other.0), self.1.join(&other.1))
    }
}

impl<A, B> BoundedJoinSemilattice for (A, B)
where
    A: BoundedJoinSemilattice + Clone,
    B: BoundedJoinSemilattice + Clone,
{
    fn bottom() -> Self {
        (A::bottom(), B::bottom())
    }
}

// JoinOf<L> (bounded)

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JoinOf<L>(pub L);

impl<L> JoinOf<L> {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonEmptyJoinOf<L>(pub L);

impl<L> NonEmptyJoinOf<L> {
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
}
