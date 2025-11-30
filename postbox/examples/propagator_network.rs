//! Propagator network examples demonstrating both homogeneous and
//! heterogeneous networks.
//!
//! Propagator networks work with any semigroup type - from lattices
//! (Max, HashSet union) to general algebraic structures (addition,
//! string concatenation, etc.).
//!
//! Run with: `cargo run --example propagator_network`

use algebra_core::Semigroup;
use postbox::join_semilattice::Max;
use postbox::propagator::{CellId, Network, Propagator};
use std::collections::HashSet;

// ============================================================
// Example 1: Homogeneous network (all cells are Max<i32>)
// ============================================================
//
// WHAT IT DEMONSTRATES:
// - All cells have the same type (Max<i32>)
// - Multiple propagators can write to the same cell (values merge via semigroup combine)
// - Diamond-shaped computation graph
//
// HOW IT WORKS:
// - Creates cells a, b, c, d in a diamond: a → {b, c} → d
// - AddProp adds constants: b = a + 5, c = a + 3
// - MaxProp computes maximum: d = max(b, c)
// - All propagators run once per iteration
//
// WHAT WE LEARN:
// - With all propagators running per iteration, convergence is fast (1 iteration)
// - The final value d = 15 = max(10+5, 10+3)
// - Order of propagator activation doesn't matter (monotonicity guarantees this)

/// Propagator that sets target = max(source1, source2)
struct MaxProp {
    source1: CellId<Max<i32>>,
    source2: CellId<Max<i32>>,
    target: CellId<Max<i32>>,
}

impl Propagator for MaxProp {
    fn activate(&mut self, network: &mut Network) {
        let v1 = network.read(self.source1);
        let v2 = network.read(self.source2);
        let result = v1.combine(v2);
        network.merge(self.target, result);
    }
}

/// Propagator that sets target = source + constant
struct AddProp {
    source: CellId<Max<i32>>,
    target: CellId<Max<i32>>,
    constant: i32,
}

impl Propagator for AddProp {
    fn activate(&mut self, network: &mut Network) {
        let v = network.read(self.source);
        let result = Max(v.0 + self.constant);
        network.merge(self.target, result);
    }
}

fn homogeneous_example() {
    println!("=== Homogeneous Network Example ===");
    println!("Finding maximum value through a diamond-shaped graph\n");

    let mut net = Network::new();

    // Create cells for a diamond graph:
    //     a
    //    / \
    //   b   c
    //    \ /
    //     d
    let a = net.add_cell(Max(10));
    let b = net.add_cell(Max(i32::MIN));
    let c = net.add_cell(Max(i32::MIN));
    let d = net.add_cell(Max(i32::MIN));

    // Create propagators
    let mut props: Vec<Box<dyn Propagator>> = vec![
        Box::new(AddProp {
            source: a,
            target: b,
            constant: 5,
        }), // b = a + 5
        Box::new(AddProp {
            source: a,
            target: c,
            constant: 3,
        }), // c = a + 3
        Box::new(MaxProp {
            source1: b,
            source2: c,
            target: d,
        }), // d = max(b, c)
    ];

    println!("Initial state:");
    println!("  a = {}", net.read(a).0);
    println!("  b = {}", net.read(b).0);
    println!("  c = {}", net.read(c).0);
    println!("  d = {}\n", net.read(d).0);

    // Run propagators to fixed point
    let mut iterations = 0;
    let max_iterations = 10;
    loop {
        for prop in &mut props {
            prop.activate(&mut net);
        }
        iterations += 1;

        // Check convergence (simplified - just check expected final value)
        let current_d = net.read(d).0;
        if current_d == 15 || iterations >= max_iterations {
            break;
        }
    }

    println!("After {} iterations:", iterations);
    println!("  a = {}", net.read(a).0);
    println!("  b = {} (= a + 5)", net.read(b).0);
    println!("  c = {} (= a + 3)", net.read(c).0);
    println!("  d = {} (= max(b, c))", net.read(d).0);
    println!();
}

// ============================================================
// Example 2: Heterogeneous network (mixed cell types)
// ============================================================
//
// WHAT IT DEMONSTRATES:
// - Type-safe heterogeneous storage: different cell types in one network
// - Cross-type computation: HashSet → Max<i32> via size calculation
// - CellId<S> provides compile-time type safety despite runtime type erasure
//
// HOW IT WORKS:
// - Creates HashSet<String> cells (tags1, tags2, all_tags)
// - Creates Max<i32> cell (tag_count)
// - UnionProp merges sets: all_tags = tags1 ∪ tags2
// - SizeProp converts: tag_count = |all_tags|
// - Demonstrates that a single network can hold Mix<Max<i32>> and HashSet<String>
//
// WHAT WE LEARN:
// - The network uses Box<dyn Any> internally but CellId<S> maintains type safety
// - You can't accidentally use CellId<HashSet<String>> where CellId<Max<i32>> is expected
// - Propagators can connect cells of different types as long as the computation makes sense
// - Converges in 1 iteration (all propagators run together)

/// Propagator that merges two sets
struct UnionProp {
    source1: CellId<HashSet<String>>,
    source2: CellId<HashSet<String>>,
    target: CellId<HashSet<String>>,
}

impl Propagator for UnionProp {
    fn activate(&mut self, network: &mut Network) {
        let v1 = network.read(self.source1);
        let v2 = network.read(self.source2);
        let result = v1.combine(v2);
        network.merge(self.target, result);
    }
}

/// Propagator that converts set size to a Max value
struct SizeProp {
    source: CellId<HashSet<String>>,
    target: CellId<Max<i32>>,
}

impl Propagator for SizeProp {
    fn activate(&mut self, network: &mut Network) {
        let set = network.read(self.source);
        let size = Max(set.len() as i32);
        network.merge(self.target, size);
    }
}

fn heterogeneous_example() {
    println!("=== Heterogeneous Network Example ===");
    println!("Combining different lattice types in one network\n");

    let mut net = Network::new();

    // Create cells of different types
    let tags1 = net.add_cell(HashSet::from(["rust".to_string(), "async".to_string()]));
    let tags2 = net.add_cell(HashSet::from(["lattice".to_string(), "crdt".to_string()]));
    let all_tags = net.add_cell(HashSet::new());
    let tag_count = net.add_cell(Max(0));

    // Create propagators that work across types
    let mut props: Vec<Box<dyn Propagator>> = vec![
        Box::new(UnionProp {
            source1: tags1,
            source2: tags2,
            target: all_tags,
        }),
        Box::new(SizeProp {
            source: all_tags,
            target: tag_count,
        }),
    ];

    println!("Initial state:");
    println!("  tags1 = {:?}", net.read(tags1));
    println!("  tags2 = {:?}", net.read(tags2));
    println!("  all_tags = {:?}", net.read(all_tags));
    println!("  tag_count = {}\n", net.read(tag_count).0);

    // Run to fixed point
    let mut changed = true;
    let mut iterations = 0;
    while changed {
        changed = false;
        for prop in &mut props {
            prop.activate(&mut net);
        }
        iterations += 1;

        // Simple termination check
        if net.read(tag_count).0 == 4 || iterations > 10 {
            break;
        }
    }

    println!("After propagation:");
    println!("  tags1 = {:?}", net.read(tags1));
    println!("  tags2 = {:?}", net.read(tags2));
    println!("  all_tags = {:?}", net.read(all_tags));
    println!("  tag_count = {}", net.read(tag_count).0);
    println!("\nNote: all_tags is the union of tags1 and tags2");
    println!("      tag_count tracks the size of all_tags");
}

// ============================================================
// Example 3: Multi-iteration convergence
// ============================================================
//
// WHAT IT DEMONSTRATES:
// - How values propagate through a chain over multiple iterations
// - Iterative convergence to a fixed point
// - That propagator networks can require multiple rounds to reach stability
//
// HOW IT WORKS:
// - Creates a chain of 5 cells: a → b → c → d → e
// - Uses CopyProp to propagate values along the chain
// - Artificially runs ONE propagator per iteration (round-robin)
//   to force multi-iteration convergence
// - In a real system, you'd run all active propagators per iteration,
//   but this would converge in 1 iteration (all propagators cascade)
//
// WHAT WE LEARN:
// - With round-robin scheduling, convergence takes 4 iterations
//   (one per link in the chain)
// - Iteration 1: a→b, Iteration 2: b→c, Iteration 3: c→d, Iteration 4: d→e
// - This is ARTIFICIAL - we're constraining propagator activation to demonstrate
//   iteration count. Real propagator networks would run all propagators together
//   for efficiency.
// - True multi-iteration scenarios arise from:
//   * Cyclic dependencies (feedback loops in the graph)
//   * Incremental updates (new information added over time)
//   * Complex dependency structures that genuinely need multiple passes
// - This example is pedagogical: it shows HOW iteration works, not WHEN it's needed

/// Propagator that copies a value from source to target
struct CopyProp {
    source: CellId<Max<i32>>,
    target: CellId<Max<i32>>,
}

impl Propagator for CopyProp {
    fn activate(&mut self, network: &mut Network) {
        let value = *network.read(self.source);
        network.merge(self.target, value);
    }
}

fn multi_iteration_example() {
    println!("=== Multi-Iteration Example ===");
    println!("Propagating a value through a chain of cells\n");

    let mut net = Network::new();

    // Create a chain: a → b → c → d → e
    let a = net.add_cell(Max(100));
    let b = net.add_cell(Max(i32::MIN));
    let c = net.add_cell(Max(i32::MIN));
    let d = net.add_cell(Max(i32::MIN));
    let e = net.add_cell(Max(i32::MIN));

    // Create propagators for the chain
    let mut props: Vec<Box<dyn Propagator>> = vec![
        Box::new(CopyProp {
            source: a,
            target: b,
        }),
        Box::new(CopyProp {
            source: b,
            target: c,
        }),
        Box::new(CopyProp {
            source: c,
            target: d,
        }),
        Box::new(CopyProp {
            source: d,
            target: e,
        }),
    ];

    println!("Initial state:");
    println!("  a = {}", net.read(a).0);
    println!("  b = {}", net.read(b).0);
    println!("  c = {}", net.read(c).0);
    println!("  d = {}", net.read(d).0);
    println!("  e = {}\n", net.read(e).0);

    // Run to fixed point, one propagator per iteration (round-robin)
    // This demonstrates multi-iteration convergence
    let mut iterations = 0;
    let max_iterations = 20;
    let mut prop_index = 0;

    loop {
        // Activate one propagator per iteration (round-robin)
        props[prop_index].activate(&mut net);
        prop_index = (prop_index + 1) % props.len();
        iterations += 1;

        println!(
            "After iteration {}: a={}, b={}, c={}, d={}, e={}",
            iterations,
            net.read(a).0,
            net.read(b).0,
            net.read(c).0,
            net.read(d).0,
            net.read(e).0
        );

        // Check if we've reached the final state
        if net.read(e).0 == 100 || iterations >= max_iterations {
            break;
        }
    }

    println!("\nConverged after {} iterations", iterations);
    println!("Value propagated from a to e through the chain");
}

fn main() {
    homogeneous_example();
    println!("\n{}\n", "=".repeat(50));
    heterogeneous_example();
    println!("\n{}\n", "=".repeat(50));
    multi_iteration_example();
}
