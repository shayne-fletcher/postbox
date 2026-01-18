//!
//! Fixed-point types and recursion schemes.
//! This module provides the building blocks for recursion schemes in Rust:
//!
//! - [`TypeApp`]: Higher-kinded type encoding
//! - [`Functor`]: Functor on one type parameter
//! - [`Fix`]: Least fixed point (μF)
//! - [`fold`]: Catamorphism / F-algebra eliminator
//!
//! # Example: Linked List
//!
//! ```rust
//! use algebra_core::fix::{TypeApp, Functor, Fix, fold};
//!
//! // Base functor for List<A>: ListF<A, X> = Nil | Cons(A, X)
//! enum ListF<A, X> {
//!     Nil,
//!     Cons(A, X),
//! }
//!
//! // Type-level tag for ListF<A, _>
//! struct ListTag<A>(std::marker::PhantomData<A>);
//!
//! impl<A> TypeApp for ListTag<A> {
//!     type Applied<X> = ListF<A, X>;
//! }
//!
//! impl<A: Clone> Functor for ListTag<A> {
//!     fn fmap<X, Y, G>(fx: ListF<A, X>, mut g: G) -> ListF<A, Y>
//!     where
//!         G: FnMut(X) -> Y,
//!     {
//!         match fx {
//!             ListF::Nil => ListF::Nil,
//!             ListF::Cons(a, x) => ListF::Cons(a.clone(), g(x)),
//!         }
//!     }
//! }
//!
//! // List<A> is the fixed point of ListF<A, _>
//! type List<A> = Fix<ListTag<A>>;
//!
//! // Smart constructors
//! fn nil<A: Clone>() -> List<A> {
//!     Fix::new(ListF::Nil)
//! }
//!
//! fn cons<A: Clone>(head: A, tail: List<A>) -> List<A> {
//!     Fix::new(ListF::Cons(head, tail))
//! }
//!
//! // Build a list: [1, 2, 3]
//! let list = cons(1, cons(2, cons(3, nil())));
//!
//! // Compute length via fold
//! let length: usize = fold(list, |layer| match layer {
//!     ListF::Nil => 0,
//!     ListF::Cons(_, n) => n + 1,
//! });
//! assert_eq!(length, 3);
//! ```

use std::fmt::Debug;

/// A **type constructor** encoding via associated types.
///
/// Rust lacks higher-kinded types, so we encode `F : Type → Type` as:
/// - A marker type `F` (the "tag")
/// - An associated type `Applied<X>` representing `F(X)`
///
/// # Example
///
/// ```rust
/// use algebra_core::fix::TypeApp;
///
/// // Option as a type constructor
/// struct OptionTag;
///
/// impl TypeApp for OptionTag {
///     type Applied<X> = Option<X>;
/// }
///
/// // Now OptionTag::Applied<i32> == Option<i32>
/// let x: <OptionTag as TypeApp>::Applied<i32> = Some(42);
/// ```
pub trait TypeApp {
    /// The result of applying this type constructor to `X`.
    type Applied<X>;
}

/// A **functor** for type constructors.
///
/// The `: TypeApp` bound is a kind signature — it says `Functor` is for
/// type constructors (`* -> *`), not a behavioral dependency.
///
/// This is a functor in the category-theoretic sense: it maps objects
/// (types) and morphisms (functions) while preserving identity and
/// composition.
///
/// Laws (not enforced by type system):
///
/// - **Identity**: `fmap id = id`
/// - **Composition**: `fmap g ∘ fmap f = fmap (g ∘ f)`
///
/// # Example
///
/// ```rust
/// use algebra_core::fix::{TypeApp, Functor};
///
/// // A simple container
/// struct Pair<X>(X, X);
///
/// struct PairTag;
///
/// impl TypeApp for PairTag {
///     type Applied<X> = Pair<X>;
/// }
///
/// impl Functor for PairTag {
///     fn fmap<X, Y, G>(fx: Pair<X>, mut g: G) -> Pair<Y>
///     where
///         G: FnMut(X) -> Y,
///     {
///         Pair(g(fx.0), g(fx.1))
///     }
/// }
/// ```
pub trait Functor: TypeApp {
    /// Map a function over the holes (type parameter positions).
    fn fmap<X, Y, G>(fx: Self::Applied<X>, g: G) -> Self::Applied<Y>
    where
        G: FnMut(X) -> Y;
}

/// The **least fixed point** (μF) of a functor F.
///
/// `Fix<F>` satisfies the isomorphism:
///
/// ```text
/// Fix<F> ≅ F(Fix<F>)
/// ```
///
/// This is the type-level analog of solving `x = f(x)`. Consider a base
/// functor `ListF<A, X> = Nil | Cons(A, X)` where `X` is a hole. Applying
/// `Fix` plugs `Fix<ListF<A, _>>` into that hole:
///
/// ```text
/// Fix<ListF<A, _>> ≅ ListF<A, Fix<ListF<A, _>>>
///                  = Nil | Cons(A, Fix<ListF<A, _>>)
/// ```
///
/// Calling this type `List<A>`, we get `List<A> = Nil | Cons(A, List<A>)`
/// — the recursive definition of a list.
///
/// The isomorphism is witnessed by two functions:
///
/// - [`Fix::new`]: `F(Fix<F>) → Fix<F>`
/// - [`Fix::out`]: `Fix<F> → F(Fix<F>)`
///
/// These are inverses: `new(out(x)) = x` and `out(new(y)) = y`.
///
/// The `Box` in our implementation is just indirection to make
/// the recursive type representable in memory — it doesn't affect
/// the math.
///
/// # Why "least"?
///
/// The "least" in "least fixed point" comes from domain theory. By
/// Kleene's fixed point theorem, μf can be computed as a join
/// (least upper bound) of an ascending chain:
///
/// ```text
/// μf = ⊔{fⁿ(⊥) | n ∈ ℕ} = ⊥ ⊔ f(⊥) ⊔ f(f(⊥)) ⊔ ...
/// ```
///
/// Concretely, for `ExprF<X> = Lit(i32) | Add(X, X)`:
///
/// - E₀ = ⊥ (empty)
/// - E₁ = Lit(n)
/// - E₂ = Lit(n) | Add(Lit, Lit)
/// - E₃ = Lit(n) | Add(E₂, E₂)
/// - ...
/// - μExprF = ⊔Eₙ = all finite expression trees
///
/// Each Eₙ is expressions of depth ≤ n. The fixed point is the join.
///
/// This connects recursion schemes to the lattice theory in
/// this crate.
///
/// # `Send` / `Sync`
///
/// `Fix<F>` is `Send` iff `F::Applied<Fix<F>>` is `Send`.
/// `Fix<F>` is `Sync` iff `F::Applied<Fix<F>>` is `Sync`.
///
/// # Example
///
/// ```rust
/// use algebra_core::fix::{TypeApp, Fix};
///
/// // Natural numbers: Nat = Zero | Succ(Nat)
/// enum NatF<X> {
///     Zero,
///     Succ(X),
/// }
///
/// struct NatTag;
///
/// impl TypeApp for NatTag {
///     type Applied<X> = NatF<X>;
/// }
///
/// type Nat = Fix<NatTag>;
///
/// // Smart constructors hide the Fix::new calls
/// fn zero() -> Nat { Fix::new(NatF::Zero) }
/// fn succ(n: Nat) -> Nat { Fix::new(NatF::Succ(n)) }
///
/// let two = succ(succ(zero()));
/// ```
#[repr(transparent)]
pub struct Fix<F: TypeApp>(Box<F::Applied<Fix<F>>>);

impl<F: TypeApp> Fix<F> {
    /// Construct a `Fix` from one layer of the functor.
    ///
    /// This is the "in" morphism: `F(Fix F) → Fix F`
    #[inline]
    pub fn new(node: F::Applied<Fix<F>>) -> Self {
        Fix(Box::new(node))
    }

    /// Unwrap one layer of the functor, consuming the `Fix`.
    ///
    /// This is the "out" morphism: `Fix F → F(Fix F)`
    #[inline]
    pub fn out(self) -> F::Applied<Fix<F>> {
        *self.0
    }

    /// Borrow one layer of the functor.
    #[inline]
    pub fn as_out(&self) -> &F::Applied<Fix<F>> {
        &self.0
    }
}

impl<F> Clone for Fix<F>
where
    F: TypeApp,
    F::Applied<Fix<F>>: Clone,
{
    fn clone(&self) -> Self {
        Self(Box::new((*self.0).clone()))
    }
}

impl<F> Debug for Fix<F>
where
    F: TypeApp,
    F::Applied<Fix<F>>: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Fix").field(&*self.0).finish()
    }
}

impl<F> PartialEq for Fix<F>
where
    F: TypeApp,
    F::Applied<Fix<F>>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        *self.0 == *other.0
    }
}

impl<F> Eq for Fix<F>
where
    F: TypeApp,
    F::Applied<Fix<F>>: Eq,
{
}

/// A **catamorphism**: fold a recursive structure using an F-algebra.
///
/// Given an algebra `alg : F(A) → A`, this produces a function
/// `Fix<F> → A` that recursively applies the algebra from the
/// leaves up.
///
/// # Stack Safety
///
/// This implementation uses direct recursion. The risk of stack
/// overflow is proportional to the **recursion depth**, not the total
/// size of the structure.
///
/// - **Balanced trees**: Typically safe (depth ≈ log(n))
/// - **Skewed structures** (e.g., long linked lists): May overflow
///
/// Stack overflow typically occurs around ~10k frames on common
/// platforms, but this varies. For very deep structures, consider
/// using the `stacker` crate or an iterative approach.
///
/// # Example
///
/// ```rust
/// use algebra_core::fix::{TypeApp, Functor, Fix, fold};
///
/// // Expression tree: Expr = Lit(i32) | Add(Expr, Expr)
/// enum ExprF<X> {
///     Lit(i32),
///     Add(X, X),
/// }
///
/// struct ExprTag;
///
/// impl TypeApp for ExprTag {
///     type Applied<X> = ExprF<X>;
/// }
///
/// impl Functor for ExprTag {
///     fn fmap<X, Y, G>(fx: ExprF<X>, mut g: G) -> ExprF<Y>
///     where
///         G: FnMut(X) -> Y,
///     {
///         match fx {
///             ExprF::Lit(n) => ExprF::Lit(n),
///             ExprF::Add(l, r) => ExprF::Add(g(l), g(r)),
///         }
///     }
/// }
///
/// type Expr = Fix<ExprTag>;
///
/// fn lit(n: i32) -> Expr {
///     Fix::new(ExprF::Lit(n))
/// }
///
/// fn add(l: Expr, r: Expr) -> Expr {
///     Fix::new(ExprF::Add(l, r))
/// }
///
/// // Build: (1 + 2) + 3
/// let expr = add(add(lit(1), lit(2)), lit(3));
///
/// // Evaluate via fold
/// let result: i32 = fold(expr, |layer| match layer {
///     ExprF::Lit(n) => n,
///     ExprF::Add(l, r) => l + r,
/// });
/// assert_eq!(result, 6);
/// ```
pub fn fold<F, A>(t: Fix<F>, alg: impl FnMut(F::Applied<A>) -> A) -> A
where
    F: Functor,
{
    // Internal helper avoids moving the closure on each recursive call
    fn go<F, A>(t: Fix<F>, alg: &mut impl FnMut(F::Applied<A>) -> A) -> A
    where
        F: Functor,
    {
        let node = t.out();
        let mapped = F::fmap(node, |child| go::<F, A>(child, alg));
        alg(mapped)
    }

    let mut alg = alg;
    go::<F, A>(t, &mut alg)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test fixtures: ListF as base functor

    // Note: We don't derive Clone/Debug/PartialEq on ListF because
    // the derived impls cause trait resolution overflow when used
    // with Fix<ListTag<A>> (recursive type). Instead, we implement
    // them manually or test without requiring these traits on Fix.
    enum ListF<A, X> {
        Nil,
        Cons(A, X),
    }

    // Manual implementations for non-recursive X (used in functor law
    // tests)
    impl<A: Clone, X: Clone> Clone for ListF<A, X> {
        fn clone(&self) -> Self {
            match self {
                ListF::Nil => ListF::Nil,
                ListF::Cons(a, x) => ListF::Cons(a.clone(), x.clone()),
            }
        }
    }

    impl<A: PartialEq, X: PartialEq> PartialEq for ListF<A, X> {
        fn eq(&self, other: &Self) -> bool {
            match (self, other) {
                (ListF::Nil, ListF::Nil) => true,
                (ListF::Cons(a1, x1), ListF::Cons(a2, x2)) => a1 == a2 && x1 == x2,
                _ => false,
            }
        }
    }

    impl<A: Eq, X: Eq> Eq for ListF<A, X> {}

    impl<A: Debug, X: Debug> Debug for ListF<A, X> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                ListF::Nil => write!(f, "Nil"),
                ListF::Cons(a, x) => f.debug_tuple("Cons").field(a).field(x).finish(),
            }
        }
    }

    struct ListTag<A>(std::marker::PhantomData<A>);

    impl<A> TypeApp for ListTag<A> {
        type Applied<X> = ListF<A, X>;
    }

    impl<A: Clone> Functor for ListTag<A> {
        fn fmap<X, Y, G>(fx: ListF<A, X>, mut g: G) -> ListF<A, Y>
        where
            G: FnMut(X) -> Y,
        {
            match fx {
                ListF::Nil => ListF::Nil,
                ListF::Cons(a, x) => ListF::Cons(a.clone(), g(x)),
            }
        }
    }

    type List<A> = Fix<ListTag<A>>;

    fn nil<A: Clone>() -> List<A> {
        Fix::new(ListF::Nil)
    }

    fn cons<A: Clone>(head: A, tail: List<A>) -> List<A> {
        Fix::new(ListF::Cons(head, tail))
    }

    fn from_vec<A: Clone>(v: Vec<A>) -> List<A> {
        v.into_iter().rev().fold(nil(), |acc, x| cons(x, acc))
    }

    // Helper to convert List back to Vec for comparison
    fn to_vec(list: List<i32>) -> Vec<i32> {
        fold(list, |layer: ListF<i32, Vec<i32>>| match layer {
            ListF::Nil => Vec::new(),
            ListF::Cons(a, mut acc) => {
                acc.insert(0, a);
                acc
            }
        })
    }

    // Fix tests

    #[test]
    fn fix_new_and_out_are_inverses() {
        // Test with a non-recursive inner type to avoid overflow
        let layer: ListF<i32, String> = ListF::Cons(42, "tail".to_string());
        let fixed: Fix<ListTag<i32>> = Fix(Box::new(ListF::Cons(42, nil())));

        // Verify we can unwrap and get back the expected structure
        match fixed.out() {
            ListF::Cons(head, _tail) => assert_eq!(head, 42),
            ListF::Nil => panic!("expected Cons"),
        }

        // Also test with Nil
        let nil_fixed: Fix<ListTag<i32>> = Fix::new(ListF::Nil);
        match nil_fixed.out() {
            ListF::Nil => {}
            ListF::Cons(_, _) => panic!("expected Nil"),
        }

        // Verify layer clone works for non-recursive case
        let cloned = layer.clone();
        assert_eq!(cloned, layer);
    }

    #[test]
    fn fix_as_out_borrows_layer() {
        let list = cons(1, cons(2, nil()));
        match list.as_out() {
            ListF::Cons(head, _) => assert_eq!(*head, 1),
            ListF::Nil => panic!("expected Cons"),
        }
    }

    #[test]
    fn fix_operations_preserve_structure() {
        // Build a list and verify via fold that structure is
        // preserved
        let list = cons(1, cons(2, cons(3, nil())));
        let vec = to_vec(list);
        assert_eq!(vec, vec![1, 2, 3]);
    }

    // fold tests

    #[test]
    fn fold_computes_length() {
        let list = from_vec(vec![1, 2, 3, 4, 5]);
        let length: usize = fold(list, |layer: ListF<i32, usize>| match layer {
            ListF::Nil => 0,
            ListF::Cons(_, n) => n + 1,
        });
        assert_eq!(length, 5);
    }

    #[test]
    fn fold_computes_sum() {
        let list = from_vec(vec![1, 2, 3, 4, 5]);
        let sum: i32 = fold(list, |layer: ListF<i32, i32>| match layer {
            ListF::Nil => 0,
            ListF::Cons(a, acc) => a + acc,
        });
        assert_eq!(sum, 15);
    }

    #[test]
    fn fold_converts_to_vec() {
        let list = from_vec(vec![1, 2, 3]);
        let vec = to_vec(list);
        assert_eq!(vec, vec![1, 2, 3]);
    }

    #[test]
    fn fold_empty_list() {
        let list: List<i32> = nil();
        let length: usize = fold(list, |layer: ListF<i32, usize>| match layer {
            ListF::Nil => 0,
            ListF::Cons(_, n) => n + 1,
        });
        assert_eq!(length, 0);
    }

    // Functor law tests (using non-recursive X to enable
    // Clone/PartialEq)

    #[test]
    fn functor_identity_law() {
        // fmap(fx, |x| x) == fx
        let layer: ListF<i32, String> = ListF::Cons(42, "tail".to_string());
        let mapped = ListTag::<i32>::fmap(layer.clone(), |x| x);
        assert_eq!(mapped, layer);
    }

    #[test]
    fn functor_identity_law_nil() {
        let layer: ListF<i32, String> = ListF::Nil;
        let mapped = ListTag::<i32>::fmap(layer.clone(), |x| x);
        assert_eq!(mapped, layer);
    }

    #[test]
    fn functor_composition_law() {
        // fmap(fmap(fx, f), g) == fmap(fx, |x| g(f(x)))
        let layer: ListF<i32, i32> = ListF::Cons(1, 10);

        let f = |x: i32| x * 2;
        let g = |x: i32| x + 1;

        // fmap twice
        let left = ListTag::<i32>::fmap(ListTag::<i32>::fmap(layer.clone(), f), g);

        // fmap once with composition
        let right = ListTag::<i32>::fmap(layer, |x| g(f(x)));

        assert_eq!(left, right);
    }

    #[test]
    fn functor_composition_law_nil() {
        let layer: ListF<i32, i32> = ListF::Nil;

        let f = |x: i32| x * 2;
        let g = |x: i32| x + 1;

        let left = ListTag::<i32>::fmap(ListTag::<i32>::fmap(layer.clone(), f), g);
        let right = ListTag::<i32>::fmap(layer, |x| g(f(x)));

        assert_eq!(left, right);
    }

    // Expression tree example (from docs)

    enum ExprF<X> {
        Lit(i32),
        Add(X, X),
    }

    struct ExprTag;

    impl TypeApp for ExprTag {
        type Applied<X> = ExprF<X>;
    }

    impl Functor for ExprTag {
        fn fmap<X, Y, G>(fx: ExprF<X>, mut g: G) -> ExprF<Y>
        where
            G: FnMut(X) -> Y,
        {
            match fx {
                ExprF::Lit(n) => ExprF::Lit(n),
                ExprF::Add(l, r) => ExprF::Add(g(l), g(r)),
            }
        }
    }

    type Expr = Fix<ExprTag>;

    fn lit(n: i32) -> Expr {
        Fix::new(ExprF::Lit(n))
    }

    fn add(l: Expr, r: Expr) -> Expr {
        Fix::new(ExprF::Add(l, r))
    }

    #[test]
    fn expr_eval_via_fold() {
        // (1 + 2) + 3 = 6
        let expr = add(add(lit(1), lit(2)), lit(3));
        let result: i32 = fold(expr, |layer: ExprF<i32>| match layer {
            ExprF::Lit(n) => n,
            ExprF::Add(l, r) => l + r,
        });
        assert_eq!(result, 6);
    }

    #[test]
    fn expr_count_nodes_via_fold() {
        // (1 + 2) + 3 has 5 nodes: 3 Lit, 2 Add
        let expr = add(add(lit(1), lit(2)), lit(3));
        let count: usize = fold(expr, |layer: ExprF<usize>| match layer {
            ExprF::Lit(_) => 1,
            ExprF::Add(l, r) => 1 + l + r,
        });
        assert_eq!(count, 5);
    }

    #[test]
    fn expr_to_string_via_fold() {
        let expr = add(add(lit(1), lit(2)), lit(3));
        let s: String = fold(expr, |layer: ExprF<String>| match layer {
            ExprF::Lit(n) => n.to_string(),
            ExprF::Add(l, r) => format!("({} + {})", l, r),
        });
        assert_eq!(s, "((1 + 2) + 3)");
    }
}
