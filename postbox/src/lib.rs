#![deny(missing_docs)]
//! # postbox — lattice basics + async join cells
//!
//! **Part of the [postbox workspace](../index.html)**
//!
//! Core pieces:
//!
//! - [`join_semilattice`]: traits and helpers for
//!   **join-semilattices**
//! - [`crdt`]: classic **state-based CRDTs** built on those lattices:
//!   - [`crdt::GCounter`]: grow-only counter
//!   - [`crdt::PNCounter`]: increment/decrement counter built from
//!     two GCounters
//!   - [`crdt::GSet`]: grow-only set
//!   - [`crdt::TwoPSet`]: two-phase set (add + remove, no re-add)
//!   - [`crdt::ORSet`]: observed-remove set (supports re-add)
//!   - [`crdt::LWW`]: last-writer-wins register
//!   - [`crdt::MVRegister`]: multi-value register (keeps all
//!     concurrent writes)
//! - [`lvar`] *(feature = "async")*: an **LVar-style** cell whose
//!   state only grows by lattice join
//! - [`propagator`] *(feature = "async")*: **propagator networks** for
//!   accumulative computation with semigroup-valued cells
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
//! ## Features
//!
//! This crate has several optional features (all enabled by default):
//!
//! - **`async`** *(enabled by default)*: Enables async/await support,
//!   including:
//!   - [`lvar::LVar`]: monotone lattice variables with async waiting
//!   - [`mvar::MVar`]: classic single-slot async cells
//!   - [`join_stream_ext`]: stream folding with lattice join
//!   - Requires `tokio` and `futures` dependencies
//!
//! - **`derive`** *(enabled by default)*: Provides derive macros for
//!   automatic trait implementations:
//!   - `#[derive(JoinSemilattice)]`: derive join from field-wise
//!     joins
//!   - `#[derive(BoundedJoinSemilattice)]`: derive bottom from
//!     field-wise bottoms
//!
//! - **`bitflags`** *(enabled by default)*: Adds support for using
//!   [`BitOr`](join_semilattice::BitOr) with types from the
//!   `bitflags` crate
//!   - Enables testing and documentation for bitflags integration
//!
//! To use only the core lattice types without async or derives:
//! ```toml
//! postbox = { version = "…", default-features = false }
//! ```
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
//! ### Async example
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

#[cfg(feature = "async")]
/// Propagator networks for monotonic computation with
/// lattice-valued cells.
pub mod propagator;

// Re-export the traits (and derive macros when derive feature is enabled)
// The derive macros come from algebra_core, which re-exports them from algebra-core-derive
pub use join_semilattice::BoundedJoinSemilattice;
pub use join_semilattice::JoinSemilattice;
