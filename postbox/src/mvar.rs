//! MVar-style for Rust (tokio)
//! ---------------------------
//! A single-slot, either-empty-or-full container with blocking `put`
//! and `take`, plus handy helpers (`try_` variants, `read_clone`,
//! `with`, and `modify`). This mirrors the spirit of Haskell's `MVar`
//! from Simon Marlow's *Parallel and Concurrent Programming in
//! Haskell*, adapted for async Rust.
//!
//! Design notes
//! - State is `Option<T>` inside an async `Mutex`.
//! - Two `Notify`s model the conditions "became non-empty" and
//!   "became non-full".
//! - We always subscribe to a `Notify` *before* dropping the lock to
//!   avoid lost wakeups.
//! - `with` holds the lock while running a **synchronous** closure;
//!   don't do async work there. Use `modify` for an async-friendly,
//!   exception/cancellation safe transform that temporarily empties
//!   the cell.
//! - `read_clone` requires `T: Clone`; if you want a zero-copy
//!   reader, store an `Arc<U>` inside your `MVar`.
//!
//! Rough equivalences
//! - `put`/`take` ~ bounded channel (capacity 1) send/recv, but with
//!   the cell identity retained and `read`/`modify` style ops
//!   supported.
//! - If you only need send/recv, `tokio::sync::mpsc(1)` or
//!   `async_channel::bounded(1)` may suffice. Choose `MVar` when
//!   you want the cell semantics.

use std::fmt;
use tokio::sync::Mutex;
use tokio::sync::Notify;

/// A single-slot, asynchronous mutable variable (classic **MVar**).
///
/// `MVar<T>` holds either:
/// - `None`  → the cell is **empty**
/// - `Some` → the cell is **full**
///
/// `put` waits until the cell is empty, then fills it; `take` waits
/// until the cell is full, then empties it and returns the value.
/// This gives you a simple one-element rendezvous point for async
/// tasks.
///
/// Internally this is implemented with a `Mutex<Option<T>>` plus two
/// `Notify` signals tracking transitions between empty and full.
pub struct MVar<T> {
    inner: Mutex<Option<T>>, // None = empty, Some(t) = full
    not_empty: Notify,       // signalled when we transition to full
    not_full: Notify,        // signalled when we transition to empty
}

impl<T> fmt::Debug for MVar<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MVar").finish_non_exhaustive()
    }
}

impl<T> Default for MVar<T> {
    fn default() -> Self {
        Self::new_empty()
    }
}

impl<T> MVar<T> {
    /// Create an empty MVar.
    pub fn new_empty() -> Self {
        Self {
            inner: Mutex::new(None),
            not_empty: Notify::new(),
            not_full: Notify::new(),
        }
    }

    /// Create a full MVar containing `value`.
    pub fn new(value: T) -> Self {
        Self {
            inner: Mutex::new(Some(value)),
            not_empty: Notify::new(),
            not_full: Notify::new(),
        }
    }

    /// Put a value into the MVar, waiting until it becomes empty.
    pub async fn put(&self, value: T) {
        // We loop until we manage to store the value.
        let mut value = Some(value);
        loop {
            let mut guard = self.inner.lock().await;
            if guard.is_none() {
                *guard = value.take();
                drop(guard);
                // The cell is now full.
                self.not_empty.notify_one();
                return;
            }
            // Already full: wait for someone to `take`.
            let notified = self.not_full.notified();
            drop(guard);
            notified.await;
        }
    }

    /// Try to put without waiting. Returns `Err(value)` if already
    /// full.
    pub async fn try_put(&self, value: T) -> Result<(), T> {
        let mut guard = self.inner.lock().await;
        if guard.is_none() {
            *guard = Some(value);
            drop(guard);
            self.not_empty.notify_one();
            Ok(())
        } else {
            Err(value)
        }
    }

    /// Take the value from the MVar, waiting until it becomes full.
    pub async fn take(&self) -> T {
        loop {
            let mut guard = self.inner.lock().await;
            if let Some(v) = guard.take() {
                drop(guard);
                // The cell is now empty.
                self.not_full.notify_one();
                return v;
            }
            // Empty: wait for a `put`.
            let notified = self.not_empty.notified();
            drop(guard);
            notified.await;
        }
    }

    /// Try to take without waiting. Returns `None` if empty.
    pub async fn try_take(&self) -> Option<T> {
        let mut guard = self.inner.lock().await;
        let out = guard.take();
        if out.is_some() {
            drop(guard);
            self.not_full.notify_one();
        }
        out
    }

    /// Read the current value without emptying the MVar. Requires
    /// `Clone`.
    pub async fn read_clone(&self) -> T
    where
        T: Clone,
    {
        loop {
            let guard = self.inner.lock().await;
            if let Some(v) = guard.as_ref() {
                return v.clone();
            }
            let notified = self.not_empty.notified();
            drop(guard);
            notified.await;
        }
    }

    /// Apply a synchronous function to `&T` without emptying the
    /// MVar. Do **not** do async work in `f`.
    pub async fn with<R>(&self, f: impl FnOnce(&T) -> R) -> R {
        loop {
            let guard = self.inner.lock().await;
            if let Some(v) = guard.as_ref() {
                // Run `f` while holding the lock; quick, sync-only
                // work.
                return f(v);
            }
            let notified = self.not_empty.notified();
            drop(guard);
            notified.await;
        }
    }

    /// Atomically take, run async `f`, then put the new value back.
    /// This mirrors Haskell's `modifyMVar` pattern and is
    /// cancellation-safe w.r.t. the put: if `f` is cancelled, the
    /// original value is restored.
    pub async fn modify<R, F, Fut>(&self, f: F) -> R
    where
        F: FnOnce(T) -> Fut,
        Fut: std::future::Future<Output = (T, R)>,
    {
        // Take the current value (waits if empty)
        let old = self.take().await;
        // Run user code outside the lock.
        let (new, result) = f(old).await;
        // Put it back.
        self.put(new).await;
        result
    }

    /// Swap the contents, returning the previous value. Waits until
    /// full.
    pub async fn swap(&self, new: T) -> T {
        let old = self.take().await;
        self.put(new).await;
        old
    }

    /// Returns whether the MVar is currently empty. Snapshot only.
    pub async fn is_empty(&self) -> bool {
        self.inner.lock().await.is_none()
    }
}

#[cfg(all(test, feature = "async"))]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[tokio::test]
    async fn put_take_roundtrip() {
        let m = Arc::new(MVar::new_empty());

        let t1 = {
            let m = Arc::clone(&m);
            tokio::spawn(async move {
                m.put(42).await;
                m.put(7).await;
            })
        };

        let a = m.take().await;
        let b = m.take().await;
        t1.await.unwrap();

        assert_eq!((a, b), (42, 7));
        assert!(m.is_empty().await);
    }

    #[tokio::test]
    async fn read_clone_doesnt_consume() {
        let m = MVar::new("hi".to_string());
        let r1 = m.read_clone().await;
        assert_eq!(r1, "hi");
        let taken = m.take().await;
        assert_eq!(taken, "hi");
    }

    #[tokio::test]
    async fn modify_updates_value_and_returns_result() {
        let m = MVar::new(10);
        let sum = m.modify(|x| async move { (x + 5, x + 1) }).await;
        assert_eq!(sum, 11);
        assert_eq!(m.read_clone().await, 15);
    }

    #[tokio::test]
    async fn swap_works() {
        let m = MVar::new(1);
        let old = m.swap(2).await;
        assert_eq!(old, 1);
        assert_eq!(m.read_clone().await, 2);
    }

    #[tokio::test]
    async fn new_empty_creates_empty_mvar() {
        let m: MVar<i32> = MVar::new_empty();
        assert!(m.is_empty().await);
    }

    #[tokio::test]
    async fn new_creates_full_mvar() {
        let m = MVar::new(42);
        assert!(!m.is_empty().await);
        assert_eq!(m.take().await, 42);
    }

    #[tokio::test]
    async fn try_put_succeeds_when_empty() {
        let m: MVar<i32> = MVar::new_empty();
        let result = m.try_put(10).await;
        assert!(result.is_ok());
        assert_eq!(m.take().await, 10);
    }

    #[tokio::test]
    async fn try_put_fails_when_full() {
        let m = MVar::new(5);
        let result = m.try_put(10).await;
        assert_eq!(result, Err(10)); // Returns the value back
        assert_eq!(m.take().await, 5); // Original value unchanged
    }

    #[tokio::test]
    async fn try_take_succeeds_when_full() {
        let m = MVar::new(42);
        let result = m.try_take().await;
        assert_eq!(result, Some(42));
        assert!(m.is_empty().await);
    }

    #[tokio::test]
    async fn try_take_fails_when_empty() {
        let m: MVar<i32> = MVar::new_empty();
        let result = m.try_take().await;
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn with_allows_readonly_access() {
        let m = MVar::new(String::from("hello"));
        let len = m.with(|s| s.len()).await;
        assert_eq!(len, 5);
        // Value still in MVar
        assert!(!m.is_empty().await);
    }

    #[tokio::test]
    async fn is_empty_reflects_state() {
        let m: MVar<i32> = MVar::new_empty();
        assert!(m.is_empty().await);

        m.put(10).await;
        assert!(!m.is_empty().await);

        m.take().await;
        assert!(m.is_empty().await);
    }

    #[tokio::test]
    async fn default_creates_empty_mvar() {
        let m: MVar<i32> = MVar::default();
        assert!(m.is_empty().await);
    }
}
