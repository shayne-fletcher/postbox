# postbox [![Build and test](https://github.com/shayne-fletcher/postbox/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/shayne-fletcher/postbox/actions/workflows/build-and-test.yml)

Lattice-based state + async cells for Rust:

- `JoinSemilattice` and `BoundedJoinSemilattice` traits
- `LVar<L>`: monotone, join-only async cell
- `MVar<T>`: classic single-slot async cell (not monotone)
- `GCounter<Id>`: grow-only CRDT counter
- `GSet<T>`: grow-only CRDT set
- `PNCounter<Id>`: increment/decrement CRDT counter built from two GCounters
- `JoinStreamExt`: fold `Stream<Item = L>` by lattice join

```rust
use std::collections::HashSet;
use postbox::join_semilattice::BoundedJoinSemilattice
use postbox::join_semilattice::JoinSemilattice;

let a: HashSet<_> = [1, 2].into_iter().collect();
let b: HashSet<_> = [2, 3].into_iter().collect();
let j = a.join(&b);
assert_eq!(j, HashSet::from([1, 2, 3]));
```
