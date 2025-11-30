#![cfg(feature = "async")]
//! Async **LVar-style** monotone cell built on join-semilattices.
//!
//! This module exposes [`crate::lvar::LVar`], a cell whose state lives in a
//! [`JoinSemilattice`] and is updated only via the lattice `join`.
//! Once information is added, it is never removed:
//!
//! - Writes go through [`crate::lvar::LVar::put_join`], which updates the state as
//!   `state := state ∨ delta` and notifies any waiters if the value
//!   changed.
//! - Readers can:
//!   - use [`crate::lvar::LVar::get`] for a synchronous snapshot, or
//!   - `await` monotone conditions with:
//!     - [`crate::lvar::LVar::await_at_least`] for `target ≤ current`
//!     - [`crate::lvar::LVar::await_monotone`] for an arbitrary monotone predicate
//!       `p` (if `p(x)` and `x ≤ y`, then `p(y)`).
//!
//! Internally, `LVar` combines a `Mutex<L>` for the authoritative
//! state with a `tokio::sync::watch::Sender<L>` to broadcast updates
//! to asynchronous subscribers. All evolution of the state is
//! monotone in the induced lattice order, which makes these cells a
//! good building block for async dataflow and CRDT-style
//! convergence.`

use std::sync::Arc;
use std::sync::Mutex;
use tokio::sync::watch;

use crate::join_semilattice::BoundedJoinSemilattice;
use crate::join_semilattice::JoinSemilattice;

/// A monotone cell (LVar): state only increases via lattice `join`.
pub struct LVar<L>
where
    L: JoinSemilattice + Clone + Send + Sync + PartialEq + 'static,
{
    inner: Arc<Inner<L>>,
}

struct Inner<L>
where
    L: JoinSemilattice + Clone + Send + Sync + PartialEq + 'static,
{
    state: Mutex<L>,
    tx: watch::Sender<L>,
}

impl<L> Clone for LVar<L>
where
    L: JoinSemilattice + Clone + Send + Sync + PartialEq + 'static,
{
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<L> LVar<L>
where
    L: BoundedJoinSemilattice + Clone + Send + Sync + PartialEq + 'static,
{
    /// Create a cell initialized to ⊥.
    pub fn new() -> Self {
        let bottom = L::bottom();
        let (tx, _rx) = watch::channel(bottom.clone());
        Self {
            inner: Arc::new(Inner {
                state: Mutex::new(bottom),
                tx,
            }),
        }
    }

    /// Read current value (clone).
    pub fn get(&self) -> L {
        self.inner.state.lock().unwrap().clone()
    }

    /// Monotone write: state := state ∨ delta. Returns true if it
    /// changed.
    pub fn put_join(&self, delta: &L) -> bool {
        let mut g = self.inner.state.lock().unwrap();
        let new_v = g.join(delta);
        if *g != new_v {
            *g = new_v.clone();
            let _ = self.inner.tx.send(new_v);
            true
        } else {
            false
        }
    }

    /// Wait until `target ≤ current`. Returns the (≥ target) value.
    pub async fn await_at_least(&self, target: &L) -> L {
        // Fast path: read the current state directly
        {
            let cur = self.inner.state.lock().unwrap().clone();
            if target.leq(&cur) {
                return cur;
            }
        }

        // Then subscribe for changes
        let mut rx = self.inner.tx.subscribe();
        loop {
            if rx.changed().await.is_err() {
                // sender dropped; return last seen value
                return rx.borrow().clone();
            }
            let cur = rx.borrow().clone();
            if target.leq(&cur) {
                return cur;
            }
        }
    }

    /// Wait until an arbitrary monotone predicate holds.
    ///
    /// The predicate `p` must be monotone: if `p(x)` is true and `x ≤ y`,
    /// then `p(y)` must also be true. This ensures the condition can only
    /// become true, never false again.
    ///
    /// Returns the first value for which `p` returns `true`.
    pub async fn await_monotone<F>(&self, mut p: F) -> L
    where
        F: FnMut(&L) -> bool + Send,
    {
        // Fast path via mutex
        {
            let cur = self.inner.state.lock().unwrap().clone();
            if p(&cur) {
                return cur;
            }
        }

        // Then subscribe
        let mut rx = self.inner.tx.subscribe();
        loop {
            if rx.changed().await.is_err() {
                return rx.borrow().clone();
            }
            let cur = rx.borrow().clone();
            if p(&cur) {
                return cur;
            }
        }
    }
}

#[cfg(all(test, feature = "async"))]
mod tests {
    use super::*;
    use crate::join_semilattice::Max;
    use std::collections::HashSet;
    use std::time::Duration;
    use tokio::time::timeout;

    fn set<T: Eq + std::hash::Hash + Clone>(xs: &[T]) -> HashSet<T> {
        xs.iter().cloned().collect()
    }

    #[tokio::test]
    async fn await_at_least_waits_for_union() {
        let cell = LVar::<HashSet<&'static str>>::new();

        let target = set(&["a", "b", "c"]);
        let target_in_task = target.clone();
        let waiter = {
            let cell = cell.clone();
            tokio::spawn(async move { cell.await_at_least(&target_in_task).await })
        };

        assert!(cell.put_join(&set(&["a"])));
        assert!(cell.put_join(&set(&["c"])));
        assert!(cell.put_join(&set(&["b"])));

        let got = waiter.await.unwrap();
        assert!(target.is_subset(&got));
    }

    #[tokio::test]
    async fn await_monotone_counter_reaches_threshold() {
        type C = Option<Max<i64>>;
        let cell = LVar::<C>::new();

        let done = {
            let cell = cell.clone();
            tokio::spawn(async move {
                cell.await_monotone(|v| matches!(v, Some(Max(n)) if *n >= 10))
                    .await
            })
        };

        assert!(cell.put_join(&Some(Max(3))));
        assert!(cell.put_join(&Some(Max(7))));
        assert!(!cell.put_join(&Some(Max(7))));
        assert!(cell.put_join(&Some(Max(12)))); // triggers

        let v = done.await.unwrap();
        assert!(matches!(v, Some(Max(n)) if n >= 10));
    }

    #[tokio::test]
    async fn fast_path_immediate_resolution() {
        let cell = LVar::<HashSet<i32>>::new();
        cell.put_join(&set(&[1, 2, 3, 4]));

        let target = set(&[2, 4]);
        let got = timeout(Duration::from_millis(50), cell.await_at_least(&target))
            .await
            .expect("timed out");
        assert!(target.is_subset(&got));
    }
}
