//! Propagator network examples demonstrating both homogeneous and
//! heterogeneous networks.
//!
//! Propagator networks work with any semigroup type - from lattices
//! (Max, HashSet union) to general algebraic structures (addition,
//! string concatenation, etc.).
//!
//! Run with: `cargo run --example propagator_network`

use algebra_core::{MonoidHom, Semigroup, SemigroupHom, Sum};
use postbox::lattice::Max;
use postbox::propagator::{CellId, HomProp, Network, Propagator};

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
// Example 2: Heterogeneous network with homomorphisms
// ============================================================
//
// WHAT IT DEMONSTRATES:
// - Type-safe heterogeneous storage: different cell types in one network
// - **TRUE monoid homomorphisms**: structure-preserving transformations
// - Using the library's generic HomProp<H> with custom homomorphisms
// - CellId<S> provides compile-time type safety despite runtime type erasure
//
// HOW IT WORKS:
// - Creates String cells (msg1, msg2, msg3, combined)
// - Creates Sum<usize> cell (length) for the character count
// - ConcatProp concatenates: combined = msg1 + msg2 + msg3
// - HomProp<StringLength> applies a TRUE homomorphism: String → Sum via length
// - StringLength satisfies: length(s1 + s2) = length(s1) + length(s2) ✓
//
// WHAT WE LEARN:
// - The network uses Box<dyn Any> internally but CellId<S> maintains type safety
// - You can't accidentally use CellId<String> where CellId<Sum> is expected
// - HomProp<H> is a reusable library component - you just define your homomorphisms
// - TRUE homomorphisms preserve algebraic structure exactly
// - This is the foundation for autodiff: gradient computation is a homomorphism!
// - Converges in 1 iteration (all propagators run together)

/// Propagator that concatenates strings
struct ConcatProp {
    sources: Vec<CellId<String>>,
    target: CellId<String>,
}

impl Propagator for ConcatProp {
    fn activate(&mut self, network: &mut Network) {
        let mut result = String::new();
        for &source in &self.sources {
            result = result.combine(network.read(source));
        }
        network.merge(self.target, result);
    }
}

// ============================================================
// Example homomorphism: StringLength (String → Sum via length)
// ============================================================
//
// This example demonstrates defining a TRUE monoid homomorphism and
// using it with the library's generic HomProp propagator.
//
// HomProp<H> is provided by the library - it works with ANY
// MonoidHom. You just define your domain-specific transformations.

/// True monoid homomorphism: String → Sum<usize> via length.
///
/// This IS a proper homomorphism because:
/// - length(s1 + s2) = length(s1) + length(s2) ✓
/// - length("") = 0 (identity preserved) ✓
struct StringLength;

impl SemigroupHom for StringLength {
    type Source = String;
    type Target = Sum<usize>;

    fn apply(&self, x: &Self::Source) -> Self::Target {
        Sum(x.len())
    }
}

impl MonoidHom for StringLength {
    // Homomorphism laws verified:
    // 1. length(s1 + s2) = len(s1) + len(s2) [structure preserved]
    // 2. length("") = 0 [identity preserved]
}

fn heterogeneous_example() {
    println!("=== Heterogeneous Network Example ===");
    println!("Combining different types with homomorphisms\n");

    let mut net = Network::new();

    // Create cells of different types
    let msg1 = net.add_cell(String::from("Hello"));
    let msg2 = net.add_cell(String::from(", "));
    let msg3 = net.add_cell(String::from("World"));
    let combined = net.add_cell(String::new());
    let length = net.add_cell(Sum(0));

    // Create propagators that work across types
    let mut props: Vec<Box<dyn Propagator>> = vec![
        // Concatenate strings: combined = msg1 + msg2 + msg3
        Box::new(ConcatProp {
            sources: vec![msg1, msg2, msg3],
            target: combined,
        }),
        // Apply homomorphism: length = |combined|
        Box::new(HomProp {
            hom: StringLength,
            source: combined,
            target: length,
        }),
    ];

    println!("Initial state:");
    println!("  msg1 = {:?}", net.read(msg1));
    println!("  msg2 = {:?}", net.read(msg2));
    println!("  msg3 = {:?}", net.read(msg3));
    println!("  combined = {:?}", net.read(combined));
    println!("  length = {}\n", net.read(length).0);

    // Run propagators once
    // Note: With non-idempotent semigroups like String concatenation,
    // repeated activation would keep accumulating values. For this
    // simple example, we just run once.
    for prop in &mut props {
        prop.activate(&mut net);
    }

    println!("After propagation:");
    println!("  msg1 = {:?}", net.read(msg1));
    println!("  msg2 = {:?}", net.read(msg2));
    println!("  msg3 = {:?}", net.read(msg3));
    println!("  combined = {:?}", net.read(combined));
    println!("  length = {}", net.read(length).0);
    println!("\nNote: combined = msg1 + msg2 + msg3");
    println!("      length uses the library's HomProp<StringLength>");
    println!("      StringLength is a TRUE homomorphism: length(s1+s2) = length(s1)+length(s2)");
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

// ============================================================
// Example 4: Non-lattice semigroups (String concatenation)
// ============================================================
//
// WHAT IT DEMONSTRATES:
// - Propagator networks work with ANY semigroup, not just lattices
// - String concatenation is a semigroup but NOT a lattice:
//   * No partial order (can't compare strings by ≤)
//   * Not idempotent (s + s ≠ s)
// - This validates the Semigroup generalization
//
// HOW IT WORKS:
// - Creates String cells (messages)
// - Propagators concatenate strings together
// - Unlike lattices, values keep growing (not idempotent)
//
// WHAT WE LEARN:
// - The accumulation model works for non-lattice semigroups
// - This is exactly how gradients accumulate in autodiff (addition, not max/union)
// - Opens the door to general algebraic computation beyond lattices

fn non_lattice_example() {
    println!("=== Non-Lattice Semigroup Example ===");
    println!("String concatenation: semigroup but not a lattice\n");

    let mut net = Network::new();

    // Create message cells
    let greeting = net.add_cell(String::from("Hello"));
    let punctuation = net.add_cell(String::from("!"));
    let message = net.add_cell(String::new());

    struct ConcatProp {
        source1: CellId<String>,
        source2: CellId<String>,
        target: CellId<String>,
    }

    impl Propagator for ConcatProp {
        fn activate(&mut self, network: &mut Network) {
            let s1 = network.read(self.source1).clone();
            let s2 = network.read(self.source2).clone();
            let result = s1.combine(&s2);
            network.merge(self.target, result);
        }
    }

    let mut concat_prop = ConcatProp {
        source1: greeting,
        source2: punctuation,
        target: message,
    };

    println!("Initial state:");
    println!("  greeting = {:?}", net.read(greeting));
    println!("  punctuation = {:?}", net.read(punctuation));
    println!("  message = {:?}\n", net.read(message));

    concat_prop.activate(&mut net);

    println!("After concatenation:");
    println!("  message = {:?}\n", net.read(message));

    // Unlike lattices, repeated accumulation keeps growing
    net.merge(greeting, String::from(", World"));
    concat_prop.activate(&mut net);

    println!("After updating greeting:");
    println!("  greeting = {:?}", net.read(greeting));
    println!("  message = {:?}\n", net.read(message));

    println!("Note: This is NOT idempotent like lattices.");
    println!("      Each merge adds more to the string.");
    println!("      This is exactly how gradient accumulation works!");
}

fn main() {
    homogeneous_example();
    println!("\n{}\n", "=".repeat(50));
    heterogeneous_example();
    println!("\n{}\n", "=".repeat(50));
    multi_iteration_example();
    println!("\n{}\n", "=".repeat(50));
    non_lattice_example();
}
