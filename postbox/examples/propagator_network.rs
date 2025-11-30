//! Propagator network examples demonstrating both homogeneous and
//! heterogeneous networks.
//!
//! Run with: `cargo run --example propagator_network`

use postbox::join_semilattice::{JoinSemilattice, Max};
use postbox::propagator::{CellId, Network, Propagator};
use std::collections::HashSet;

// ============================================================
// Example 1: Homogeneous network (all cells are Max<i32>)
// ============================================================

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
        let result = v1.join(v2);
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
        let result = v1.join(v2);
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

fn main() {
    homogeneous_example();
    println!("\n{}\n", "=".repeat(50));
    heterogeneous_example();
}
