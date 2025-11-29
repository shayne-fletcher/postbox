//! Small demo of the CRDT types and LVar from `postbox`.
//!
//! - Simulates two replicas (A, B) for `GCounter`, `PNCounter`, and
//!   `GSet`.
//! - Shows local updates, out-of-order merges, and convergence.
//! - When the `async` feature is enabled, also runs a tiny `LVar`
//!   demo.

use postbox::crdt::{GCounter, GSet, PNCounter};

fn gcounter_demo() {
    println!("=== GCounter demo ===");

    let mut a = GCounter::new("A");
    let mut b = GCounter::new("B");

    // Local updates on each replica.
    a.inc(3); // A: +3
    a.inc(2); // A: +2 (total 5)
    b.inc(5); // B: +5

    println!("A local value: {}", a.value());
    println!("B local value: {}", b.value());

    // Exchange states (in any order).
    let a_state = a.state().clone();
    let b_state = b.state().clone();

    println!("--- merging A <-> B ---");
    a.merge(&b_state);
    b.merge(&a_state);

    println!("A after merge: {}", a.value());
    println!("B after merge: {}", b.value());
    println!();
}

fn pncounter_demo() {
    println!("=== PNCounter demo ===");

    let mut a = PNCounter::new("A");
    let mut b = PNCounter::new("B");

    // Local updates:
    a.inc(10); // +10
    a.dec(3); // -3   => A thinks 7
    b.inc(4); // +4
    b.dec(1); // -1   => B thinks 3

    println!("A local value: {}", a.value());
    println!("B local value: {}", b.value());

    let a_state = a.state().clone();
    let b_state = b.state().clone();

    println!("--- merging A <-> B ---");
    a.merge(&b_state);
    b.merge(&a_state);

    println!("A after merge: {}", a.value());
    println!("B after merge: {}", b.value());
    println!("(expected: (10 + 4) - (3 + 1) = 10)");
    println!();
}

fn gset_demo() {
    println!("=== GSet demo ===");

    let mut a: GSet<&'static str> = GSet::new();
    let mut b: GSet<&'static str> = GSet::new();

    // Local adds (grow-only).
    a.insert("a");
    a.insert("b");

    b.insert("b");
    b.insert("c");

    println!("A local elems: {:?}", a.elements());
    println!("B local elems: {:?}", b.elements());

    let a_state = a.clone();
    let b_state = b.clone();

    println!("--- merging A <-> B ---");
    a.merge(&b_state);
    b.merge(&a_state);

    println!("A after merge: {:?}", a.elements());
    println!("B after merge: {:?}", b.elements());
    println!();
}

#[cfg(feature = "async")]
async fn lvar_demo() {
    use std::collections::HashSet;
    use std::time::Duration;
    use tokio::time::sleep;

    use postbox::join_semilattice::BoundedJoinSemilattice;
    use postbox::lvar::LVar;

    println!("=== LVar demo (requires `async` feature) ===");

    // L = HashSet<&'static str>, join = union, bottom = ∅
    let cell = LVar::<HashSet<&'static str>>::new();

    let target: HashSet<_> = ["a", "b", "c"].into_iter().collect();

    // Waiter task: waits until target ⊆ current.
    let waiter = {
        let cell = cell.clone();
        let target = target.clone();
        tokio::spawn(async move {
            let got = cell.await_at_least(&target).await;
            println!("[waiter] reached at-least {:?}, got {:?}", target, got);
        })
    };

    // Writer: dribble in elements with small delays.
    cell.put_join(&["a"].into_iter().collect());
    sleep(Duration::from_millis(100)).await;
    cell.put_join(&["c"].into_iter().collect());
    sleep(Duration::from_millis(100)).await;
    cell.put_join(&["b"].into_iter().collect());

    let _ = waiter.await;
    println!();
}

// Non-async main: just show CRDTs.
#[cfg(not(feature = "async"))]
fn main() {
    gcounter_demo();
    pncounter_demo();
    gset_demo();
}

// Async main when `async` feature is enabled: CRDTs + LVar demo.
#[cfg(feature = "async")]
#[tokio::main]
async fn main() {
    gcounter_demo();
    pncounter_demo();
    gset_demo();
    lvar_demo().await;
}
