# postbox

[![Build and test](https://github.com/shayne-fletcher/postbox/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/shayne-fletcher/postbox/actions/workflows/build-and-test.yml)
[![codecov](https://codecov.io/gh/shayne-fletcher/postbox/branch/main/graph/badge.svg)](https://codecov.io/gh/shayne-fletcher/postbox)

<p align="center">
  <img src="ottie.jpg" alt="Ottie the Otter - postbox mascot" width="300">
</p>

Lattice-based state + async cells for Rust.

This workspace contains:
- **`algebra-core`**: Core algebraic abstractions (Semigroup, Monoid, Group, Semilattice)
- **`algebra-core-derive`**: Derive macros for algebraic traits
- **`postbox`**: Lattice-based state, LVars, MVars, and CRDTs built on algebra-core

## Getting Started

Add `postbox` to your `Cargo.toml`:

```toml
[dependencies]
postbox = "0.1"

# To use derive macros:
postbox = { version = "0.1", features = ["derive"] }
```

The `postbox` crate re-exports all traits from `algebra-core`, so you can use everything through `postbox`. If you only need the algebraic abstractions without async cells or CRDTs, you can depend on `algebra-core` directly.

## Features

### Core traits & helpers (from `algebra-core`):
- `JoinSemilattice` and `BoundedJoinSemilattice` traits
- `Max<T>`, `Min<T>`: lattice wrappers for max/min
- `Any`, `All`: boolean lattices (OR/AND)
- `BitOr<T>`, `BitAnd<T>`: bitwise lattices for bitflags and integer masks (OR/AND)
- `JoinOf<L>`, `NonEmptyJoinOf<L>`: collect iterators by lattice join
- `LatticeMap<K, V>`: pointwise map lattice (building block for CRDT states)

### Async cells (from `postbox`):
- 📬 `LVar<L>`: monotone, join-only async cell
- 📬 `MVar<T>`: classic single-slot async cell (not monotone)

### State-based CRDTs (from `postbox`):
- `GCounter<Id>`: grow-only CRDT counter
- `PNCounter<Id>`: increment/decrement CRDT counter built from two GCounters
- `GSet<T>`: grow-only CRDT set
- `TwoPSet<T>`: two-phase set (add + remove, but no re-add after remove)
- `ORSet<Id, T>`: observed-remove set (supports re-adding after remove)
- `LWW<T>`: last-writer-wins register
- `MVRegister<T, Ts, Id>`: multi-value register (keeps all concurrent writes)

### Stream extensions (from `postbox`):
- `JoinStreamExt`: fold `Stream<Item = L>` by lattice join

## Examples

### Basic usage

```rust
use std::collections::HashSet;
use postbox::join_semilattice::BoundedJoinSemilattice;
use postbox::join_semilattice::JoinSemilattice;

let a: HashSet<_> = [1, 2].into_iter().collect();
let b: HashSet<_> = [2, 3].into_iter().collect();
let j = a.join(&b);
assert_eq!(j, HashSet::from([1, 2, 3]));
```

### Deriving traits

Enable the `derive` feature to automatically implement algebraic traits for your types:

```rust
use std::collections::HashSet;
use postbox::join_semilattice::Max;
use postbox::{JoinSemilattice, BoundedJoinSemilattice};

#[derive(Debug, Clone, PartialEq, Eq, JoinSemilattice, BoundedJoinSemilattice)]
struct Foo {
    a: Max<i32>,       // join = max, bottom = i32::MIN
    b: HashSet<i32>,   // join = union, bottom = ∅
}
```

The derive macros work for both named structs and tuple structs, and support all algebraic traits: `Semigroup`, `Monoid`, `CommutativeMonoid`, `Group`, `AbelianGroup`, `JoinSemilattice`, and `BoundedJoinSemilattice`.
