//! # postbox — lattice basics + async join cells
//!
//! Core pieces:
//!
//! - [`join_semilattice`]: traits and helpers for
//!   **join-semilattices**
//! - [`crdt`]: classic **state-based CRDTs** built on those lattices:
//!   - [`crdt::GCounter`]: grow-only counter
//!   - [`crdt::GSet`]: grow-only set
//!   - [`crdt::PNCounter`]: increment/decrement counter built from
//!     two GCounters
//! - [`lvar`] *(feature = "async")*: an **LVar-style** cell whose
//!   state only grows by lattice join
//! - [`join_stream_ext`] *(feature = "async")*: stream adapters to
//!   fold `Stream<Item = L>` by lattice join
//! - [`mvar`] *(feature = "async")*: a classic **MVar** (single-slot
//!   put/take), separate from the monotone cell
//!
//! ## Concepts
//! A **join-semilattice** is a type `L` with an operation `join: &L ×
//! &L → L` that is associative, commutative, and idempotent. It
//! induces an order `x ≤ y` iff `x.join(y) == y`.
//!
//! - Common joins: `max` (on numbers), `union` (on sets), `min` (on
//!   the dual order).
//! - If a bottom element ⊥ exists, implement
//!   [`BoundedJoinSemilattice`] so you can fold from empty.
//!
//! ## Quick start
//! ```rust
//! use std::collections::HashSet;
//! use postbox::join_semilattice::BoundedJoinSemilattice;
//! use postbox::join_semilattice::JoinSemilattice;
//!
//! // Join = union on sets
//! let a: HashSet<_> = [1,2].into_iter().collect();
//! let b: HashSet<_> = [2,3].into_iter().collect();
//! let j = a.join(&b);
//! assert_eq!(j, HashSet::from([1,2,3]));
//! assert!(HashSet::<i32>::bottom().is_empty());
//! ```
//!
//! ### Async (enable the `async` feature)
//! ```toml
//! postbox = { version = "…", features = ["async"] }
//! ```
//! ```rust,ignore
//! use std::collections::HashSet;
//! use postbox::LVar;
//!
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() {
//! let cell = LVar::<HashSet<&'static str>>::new();
//! let target: HashSet<_> = ["a","b","c"].into_iter().collect();
//! let waiter = {
//!   let cell = cell.clone();
//!   tokio::spawn(async move { cell.await_at_least(&target).await })
//! };
//! cell.put_join(&HashSet::from(["a"]));
//! cell.put_join(&HashSet::from(["b","c"]));
//! let got = waiter.await.unwrap();
//! assert!(target.is_subset(&got));
//! # }
//! ```

// Make the current crate visible as `postbox` so the deive macros
// that use `::postbox::...` work both here and in downstream crates.
extern crate self as postbox;

/// Core algebra: join-semilattice traits and standard lattice
/// helpers.
pub mod join_semilattice;

/// CRDTs built on lattice combinators
pub mod crdt;

#[cfg(feature = "async")]
/// Join-only, monotone cell (LVar): state increases via lattice
/// `join`.
pub mod lvar;

#[cfg(feature = "async")]
/// Stream extensions for folding by lattice joins (uses
/// `futures::Stream`).
pub mod join_stream_ext;

#[cfg(feature = "async")]
/// A classic single-slot async **MVar** (blocking put/take). Not
/// monotone.
pub mod mvar;

// Re-export the traits
pub use join_semilattice::BoundedJoinSemilattice;
pub use join_semilattice::JoinSemilattice;

// Re-export derive macros when the feature is enabled
#[cfg(feature = "derive")]
pub use postbox_derive::BoundedJoinSemilattice;
pub use postbox_derive::JoinSemilattice;
