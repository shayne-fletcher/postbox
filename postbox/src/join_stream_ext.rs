//! Stream extensions for join-semilattices (feature = `"async"`).
//!
//! Adds `.join_all_lattice()` and `.join_all_from_bottom()` to any
//! `futures::Stream`, allowing you to fold a stream using your type's
//! `join` operation.
//!
//! For infinite streams, consider piping deltas into an
//! [`LVar`](crate::lvar::LVar) instead.

use async_trait::async_trait;

use futures::Stream;
use futures::StreamExt;

use crate::join_semilattice::BoundedJoinSemilattice;
use crate::join_semilattice::JoinSemilattice;

/// Extension trait for folding `Stream`s of lattice values.
///
/// This trait is automatically implemented for all `Stream` types and
/// provides methods to fold/reduce streams using lattice `join`:
///
/// - [`join_all_lattice`](JoinStreamExt::join_all_lattice): fold
///   starting from the first element, returning `None` for empty
///   streams.
/// - [`join_all_from_bottom`](JoinStreamExt::join_all_from_bottom):
///   fold starting from `⊥` (bottom), works even for empty streams.
///
/// # Example
///
/// ```rust,ignore
/// use futures::stream;
/// use std::collections::HashSet;
/// use postbox::join_stream_ext::JoinStreamExt;
///
/// let s = stream::iter(vec![
///     HashSet::from([1, 2]),
///     HashSet::from([2, 3]),
///     HashSet::from([4]),
/// ]);
/// let result = s.join_all_lattice().await;
/// assert_eq!(result.unwrap(), HashSet::from([1, 2, 3, 4]));
/// ```
#[async_trait]
pub trait JoinStreamExt: Stream + Sized + Unpin + Send {
    /// Join all items from a (possibly empty) stream.
    ///
    /// Returns `None` if the stream yields no items.
    async fn join_all_lattice(self) -> Option<Self::Item>
    where
        Self::Item: JoinSemilattice + Send,
    {
        let mut s = self;
        let first = s.next().await?; // None → empty stream
        Some(s.fold(first, |acc, x| async move { acc.join(&x) }).await)
    }

    /// Join all items from the stream starting at ⊥ (bottom).
    ///
    /// Works even if the stream is empty.
    async fn join_all_from_bottom(self) -> Self::Item
    where
        Self::Item: BoundedJoinSemilattice + Send,
    {
        self.fold(
            <Self::Item as BoundedJoinSemilattice>::bottom(),
            |acc, x| async move { acc.join(&x) },
        )
        .await
    }
}

#[async_trait]
impl<T> JoinStreamExt for T
where
    T: Stream + Sized + Unpin + Send,
    T::Item: Send,
{
    // Default method bodies from the trait are used.
}

#[cfg(all(test, feature = "async"))]
mod tests {
    use super::*;
    use crate::join_semilattice::BoundedJoinSemilattice;
    use futures::stream;
    use std::collections::HashSet;

    fn set<T: Eq + std::hash::Hash + Clone>(xs: &[T]) -> HashSet<T> {
        xs.iter().cloned().collect()
    }

    #[tokio::test]
    async fn join_all_lattice_unions_sets() {
        let s = stream::iter(vec![set(&[1, 2]), set(&[2, 3]), set(&[4])]);
        let joined = s.join_all_lattice().await;
        assert_eq!(joined.unwrap(), set(&[1, 2, 3, 4]));
    }

    #[tokio::test]
    async fn join_all_lattice_empty_stream_is_none() {
        let s = stream::iter(Vec::<HashSet<i32>>::new());
        let joined = s.join_all_lattice().await;
        assert!(joined.is_none());
    }

    #[tokio::test]
    async fn join_all_from_bottom_unions_sets() {
        let s = stream::iter(vec![set(&[10, 20]), set(&[20, 30])]);
        let joined = s.join_all_from_bottom().await;
        assert_eq!(joined, set(&[10, 20, 30]));
    }

    #[tokio::test]
    async fn join_all_from_bottom_empty_stream_returns_bottom() {
        let s = stream::iter(Vec::<HashSet<i32>>::new());
        let joined = s.join_all_from_bottom().await;
        assert!(joined.is_empty());
        let bottom = HashSet::<i32>::bottom();
        assert_eq!(joined, bottom);
    }
}
