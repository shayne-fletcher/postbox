//! Propagator networks for accumulative computation with algebraic
//! cells.
//!
//! A **propagator** is a computational model where:
//!
//! - **Cells** hold semigroup values that accumulate via `combine`
//!   (one-way updates: never replace, only merge)
//! - **Propagators** are functions that read from cells and write to
//!   cells
//! - When a cell's value changes, propagators that depend on it are
//!   scheduled
//! - The system runs until reaching a fixed point (no more changes)
//!
//! For lattices (ordered semigroups), accumulation means **monotonic
//! growth**. For general semigroups, it means **one-way combination**
//! (e.g., string concatenation, gradient accumulation).
//!
//! This builds on the [`lvar`](crate::lvar) foundation, extending
//! individual accumulative cells to networks of interconnected
//! computations.
//!
//! ## Conceptual model
//!
//! ```text
//!   Cell<A>  Cell<B>
//!      │      │
//!      └──┬───┘
//!         ↓
//!    Propagator
//!         ↓
//!      Cell<C>
//! ```
//!
//! Arrows show data flow: the propagator **reads** from source cells
//! A and B, then **writes** to target cell C. When cells A or B
//! change, the propagator runs and may update cell C. Multiple
//! propagators can write to the same cell (values merge via the
//! semigroup operation).
//!
//! ## Status
//!
//! This module is under active development. Core types and traits are
//! currently defined.

use std::any::Any;
use std::collections::HashMap;
use std::marker::PhantomData;

use algebra_core::{MonoidHom, Semigroup};

/// A typed identifier for a cell holding values of type `S`.
///
/// The type parameter ensures compile-time safety: you cannot
/// accidentally use a `CellId<i32>` where a `CellId<String>` is
/// expected. At runtime, cells are stored in a type-erased container,
/// but the API maintains type safety through these phantom-typed
/// identifiers.
///
/// # Type Erasure Pattern
///
/// Propagator networks need to store cells with heterogeneous types
/// (e.g., `Cell<Max<i32>>`, `Cell<HashSet<String>>`, etc.) in the
/// same container. This is accomplished through a two-layer type
/// system:
///
/// - **Storage layer (runtime)**: Cells stored as `Box<dyn Any>`
///   (type-erased)
/// - **API layer (compile-time)**: `CellId<S>` carries the type
///   information
///
/// When you access a cell via its `CellId<S>`, the network can safely
/// downcast the type-erased storage back to the concrete type
/// `Cell<S>`.
///
/// # Example
///
/// ```rust
/// use postbox::propagator::CellId;
/// use postbox::join_semilattice::Max;
/// use std::collections::HashSet;
///
/// // Each cell ID carries its type
/// let id1: CellId<Max<i32>> = CellId::new(0);
/// let id2: CellId<Max<i32>> = CellId::new(1);
/// let id3: CellId<HashSet<String>> = CellId::new(0);
///
/// // Can compare IDs of the same type
/// assert_ne!(id1, id2);
/// assert_eq!(id1.raw(), 0);
/// assert_eq!(id2.raw(), 1);
///
/// // But CellId<Max<i32>> and CellId<HashSet<String>> are different types
/// // (this would be a compile error):
/// // assert_ne!(id1, id3);  // ← error: mismatched types
/// ```
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct CellId<S> {
    id: usize,
    _phantom: PhantomData<S>,
}

// Manually implement Clone and Copy. CellId is Copy regardless of
// whether S is Copy, since it only contains a usize and PhantomData
// (both always Copy). The derived implementations would incorrectly
// require S: Clone/Copy.
impl<S> Clone for CellId<S> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<S> Copy for CellId<S> {}

impl<S> CellId<S> {
    /// Create a new cell ID from a raw integer.
    ///
    /// This is typically called internally by the network when
    /// creating cells. The type parameter `S` captures what semigroup
    /// type this cell holds.
    ///
    /// # Example
    ///
    /// ```rust
    /// use postbox::propagator::CellId;
    /// use postbox::join_semilattice::Max;
    ///
    /// let id: CellId<Max<i32>> = CellId::new(42);
    /// assert_eq!(id.raw(), 42);
    /// ```
    pub fn new(id: usize) -> Self {
        Self {
            id,
            _phantom: PhantomData,
        }
    }

    /// Get the raw ID.
    ///
    /// This returns the underlying integer identifier, discarding
    /// type information. Useful for indexing into type-erased
    /// storage.
    ///
    /// # Example
    ///
    /// ```rust
    /// use postbox::propagator::CellId;
    ///
    /// let id: CellId<String> = CellId::new(7);
    /// assert_eq!(id.raw(), 7);
    /// ```
    pub fn raw(&self) -> usize {
        self.id
    }
}

/// A cell holding a semigroup value of type `S`.
///
/// Cells store values that accumulate via the semigroup's `combine`
/// operation. Updates never replace - they merge with existing values
/// via `old.combine(new)`. When a cell's value changes, dependent
/// propagators are notified (dependency tracking to be added).
///
/// # Accumulation invariant
///
/// The key invariant: values accumulate, never replace. If you merge
/// a value `v` into a cell holding `current`, the new value is
/// `current.combine(v)`. Since combine is associative, the order of
/// merges doesn't affect convergence - the cell will reach the same
/// final value regardless.
///
/// For lattices, this is **monotonic growth** (values increase in
/// partial order). For general semigroups, this is **one-way
/// accumulation** (e.g., concatenation).
///
/// # Example
///
/// ```rust
/// use postbox::propagator::Cell;
/// use std::collections::HashSet;
///
/// let mut cell = Cell::new(HashSet::<i32>::new());
///
/// // Merging adds elements (HashSet combine = union)
/// assert!(cell.merge(HashSet::from([1, 2])));  // changed
/// assert_eq!(cell.value(), &HashSet::from([1, 2]));
///
/// assert!(cell.merge(HashSet::from([2, 3])));  // changed
/// assert_eq!(cell.value(), &HashSet::from([1, 2, 3]));
///
/// // Merging a subset doesn't change the value
/// assert!(!cell.merge(HashSet::from([1])));    // unchanged
/// assert_eq!(cell.value(), &HashSet::from([1, 2, 3]));
/// ```
pub struct Cell<S: Semigroup> {
    value: S,
    // Future: dependency tracking (Vec<PropagatorId>)
}

impl<S: Semigroup> Cell<S> {
    /// Create a new cell with an initial value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use postbox::propagator::Cell;
    /// use postbox::join_semilattice::Max;
    ///
    /// let cell = Cell::new(Max(42));
    /// assert_eq!(cell.value(), &Max(42));
    /// ```
    pub fn new(value: S) -> Self {
        Self { value }
    }

    /// Get a reference to the current value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use postbox::propagator::Cell;
    /// use postbox::join_semilattice::Max;
    ///
    /// let cell = Cell::new(Max(10));
    /// assert_eq!(cell.value(), &Max(10));
    /// ```
    pub fn value(&self) -> &S {
        &self.value
    }

    /// Merge a new value into the cell via the semigroup combine
    /// operation.
    ///
    /// Returns `true` if the cell's value changed, `false` if it
    /// remained the same. A change indicates that the new value added
    /// information (the combine produced a new value).
    ///
    /// # Example
    ///
    /// ```rust
    /// use postbox::propagator::Cell;
    /// use postbox::join_semilattice::Max;
    ///
    /// let mut cell = Cell::new(Max(5));
    ///
    /// assert!(cell.merge(Max(10)));   // 5 → 10 (changed)
    /// assert!(!cell.merge(Max(7)));   // 10 → 10 (unchanged, 7 < 10)
    /// assert!(cell.merge(Max(15)));   // 10 → 15 (changed)
    /// ```
    pub fn merge(&mut self, new_value: S) -> bool
    where
        S: PartialEq,
    {
        let merged = self.value.combine(&new_value);
        if merged != self.value {
            self.value = merged;
            true
        } else {
            false
        }
    }
}

/// A propagator network containing cells with heterogeneous semigroup
/// types.
///
/// Networks store cells in a type-erased container (`Box<dyn Any>`)
/// while maintaining type safety through [`CellId<S>`]. This allows a
/// single network to contain cells of different types (e.g.,
/// `Cell<Max<i32>>`, `Cell<HashSet<String>>`, etc.) without requiring
/// all cells to share a common trait object type.
///
/// # Type Erasure
///
/// The network uses a two-layer type system:
/// - **Storage**: Cells stored as `Box<dyn Any>` (heterogeneous)
/// - **API**: `CellId<S>` carries type information (type-safe)
///
/// When you create a cell, you get back a `CellId<S>` that remembers
/// the type. When you access the cell, the network uses the type from
/// `CellId<S>` to safely downcast from `Any` back to the concrete
/// type.
///
/// # Example
///
/// ```rust
/// use postbox::propagator::Network;
/// use postbox::join_semilattice::Max;
/// use std::collections::HashSet;
///
/// let mut net = Network::new();
///
/// // Create cells with different types
/// let c1 = net.add_cell(Max(5));
/// let c2 = net.add_cell(HashSet::from([1, 2]));
///
/// // Type-safe access
/// assert_eq!(net.read(c1), &Max(5));
/// assert_eq!(net.read(c2), &HashSet::from([1, 2]));
///
/// // Monotonic updates
/// assert!(net.merge(c1, Max(10)));  // 5 → 10 (changed)
/// assert!(!net.merge(c1, Max(7)));  // 10 → 10 (unchanged)
///
/// assert!(net.merge(c2, HashSet::from([3])));
/// assert_eq!(net.read(c2), &HashSet::from([1, 2, 3]));
/// ```
pub struct Network {
    cells: HashMap<usize, Box<dyn Any>>,
    next_id: usize,
}

impl Network {
    /// Create a new empty propagator network.
    ///
    /// # Example
    ///
    /// ```rust
    /// use postbox::propagator::Network;
    ///
    /// let net = Network::new();
    /// ```
    pub fn new() -> Self {
        Self {
            cells: HashMap::new(),
            next_id: 0,
        }
    }

    /// Add a new cell to the network with an initial value.
    ///
    /// Returns a typed `CellId<S>` that can be used to access this
    /// cell. The type parameter `S` is inferred from the initial
    /// value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use postbox::propagator::Network;
    /// use postbox::join_semilattice::Max;
    ///
    /// let mut net = Network::new();
    /// let cell = net.add_cell(Max(42));
    ///
    /// assert_eq!(net.read(cell), &Max(42));
    /// ```
    pub fn add_cell<S: Semigroup + 'static>(&mut self, initial: S) -> CellId<S> {
        let id = self.next_id;
        self.next_id += 1;
        self.cells.insert(id, Box::new(Cell::new(initial)));
        CellId::new(id)
    }

    /// Read the current value of a cell.
    ///
    /// # Panics
    ///
    /// Panics if the cell ID is invalid or if there's a type mismatch
    /// (which should be impossible if you only use `CellId`s returned
    /// by this network).
    ///
    /// # Example
    ///
    /// ```rust
    /// use postbox::propagator::Network;
    /// use std::collections::HashSet;
    ///
    /// let mut net = Network::new();
    /// let cell = net.add_cell(HashSet::from([1, 2, 3]));
    ///
    /// assert_eq!(net.read(cell), &HashSet::from([1, 2, 3]));
    /// ```
    pub fn read<S: Semigroup + 'static>(&self, id: CellId<S>) -> &S {
        self.cells[&id.raw()]
            .downcast_ref::<Cell<S>>()
            .expect("type mismatch")
            .value()
    }

    /// Merge a value into a cell via semigroup combine.
    ///
    /// Returns `true` if the cell's value changed, `false` otherwise.
    /// A change means the new value added information (combine
    /// produced a new value).
    ///
    /// # Panics
    ///
    /// Panics if the cell ID is invalid or if there's a type
    /// mismatch.
    ///
    /// # Example
    ///
    /// ```rust
    /// use postbox::propagator::Network;
    /// use postbox::join_semilattice::Max;
    ///
    /// let mut net = Network::new();
    /// let cell = net.add_cell(Max(5));
    ///
    /// assert!(net.merge(cell, Max(10)));   // changed: 5 → 10
    /// assert!(!net.merge(cell, Max(7)));   // unchanged: 10 → 10
    /// assert_eq!(net.read(cell), &Max(10));
    /// ```
    pub fn merge<S: Semigroup + PartialEq + 'static>(&mut self, id: CellId<S>, value: S) -> bool {
        self.cells
            .get_mut(&id.raw())
            .expect("cell not found")
            .downcast_mut::<Cell<S>>()
            .expect("type mismatch")
            .merge(value)
    }
}

impl Default for Network {
    fn default() -> Self {
        Self::new()
    }
}

/// A propagator that performs accumulative computation on a network.
///
/// Propagators are functions that read values from cells and write
/// values to cells. When activated, a propagator reads its input
/// cells, performs some computation, and merges the result into
/// output cells. Since all updates accumulate via semigroup combine
/// (never replace), the order of propagator activations doesn't
/// affect the final result.
///
/// # Compatibility requirement
///
/// Propagators must be **compatible with accumulation**: if inputs
/// accumulate more information, outputs should accumulate compatible
/// information.
///
/// - For lattices: this means **monotonic** functions (inputs grow ⇒
///   outputs grow)
/// - For general semigroups: this means **accumulation-preserving**
///   functions (e.g., gradient propagation preserves addition
///   structure)
///
/// This ensures that the network converges to a unique fixed point
/// regardless of execution order.
///
/// # Example
///
/// ```rust
/// use postbox::propagator::{Propagator, Network, CellId};
/// use postbox::join_semilattice::Max;
/// use algebra_core::Semigroup;
///
/// // A propagator that computes max(a, b) → c
/// struct MaxPropagator {
///     a: CellId<Max<i32>>,
///     b: CellId<Max<i32>>,
///     c: CellId<Max<i32>>,
/// }
///
/// impl Propagator for MaxPropagator {
///     fn activate(&mut self, net: &mut Network) {
///         let a_val = *net.read(self.a);
///         let b_val = *net.read(self.b);
///         let result = a_val.combine(&b_val);
///         net.merge(self.c, result);
///     }
/// }
///
/// let mut net = Network::new();
/// let a = net.add_cell(Max(5));
/// let b = net.add_cell(Max(10));
/// let c = net.add_cell(Max(0));
///
/// let mut prop = MaxPropagator { a, b, c };
/// prop.activate(&mut net);
///
/// assert_eq!(net.read(c), &Max(10)); // max(5, 10) = 10
/// ```
pub trait Propagator {
    /// Activate this propagator, reading from and writing to the
    /// network.
    ///
    /// The propagator reads values from input cells, performs
    /// computation, and merges results into output cells via the
    /// semigroup's `combine` operation.
    ///
    /// **Important:** The propagator should compute a deterministic
    /// function of its inputs. The effect of repeated activation
    /// depends on the semigroup:
    /// - For idempotent semigroups (lattices): repeated calls with
    ///   unchanged inputs have no effect
    /// - For non-idempotent semigroups (String, gradients): repeated
    ///   calls keep accumulating
    fn activate(&mut self, network: &mut Network);
}

/// Generic propagator that applies a monoid homomorphism.
///
/// This propagator reads from a source cell, applies a
/// structure-preserving transformation (monoid homomorphism), and
/// writes the result to a target cell.
///
/// # Type Parameters
///
/// - `H`: A type implementing [`MonoidHom`](algebra_core::MonoidHom)
///
/// # Example
///
/// ```rust
/// use postbox::propagator::{HomProp, Network, CellId, Propagator};
/// use algebra_core::{MonoidHom, SemigroupHom, Sum};
///
/// // True monoid homomorphism: String → Sum<usize> via length
/// // Preserves structure: length(s1 + s2) = length(s1) + length(s2)
/// struct StringLength;
///
/// impl SemigroupHom for StringLength {
///     type Source = String;
///     type Target = Sum<usize>;
///
///     fn apply(&self, x: &Self::Source) -> Self::Target {
///         Sum(x.len())
///     }
/// }
///
/// impl MonoidHom for StringLength {}
/// // Homomorphism laws:
/// // 1. length("" + s) = length("") + length(s) = 0 + len(s) ✓
/// // 2. length(s1 + s2) = length(s1) + length(s2) ✓
///
/// let mut net = Network::new();
/// let source = net.add_cell(String::from("hello"));
/// let target = net.add_cell(Sum(0));
///
/// let mut prop = HomProp {
///     hom: StringLength,
///     source,
///     target,
/// };
///
/// prop.activate(&mut net);
/// assert_eq!(net.read(target), &Sum(5)); // length of "hello" is 5
/// ```
///
/// # Use Cases
///
/// - **Type transformations**: Convert between different semigroup types
///   while preserving algebraic structure
/// - **Autodiff foundation**: Gradient computation is a homomorphism from
///   the computation graph to the gradient space
/// - **Compositional pipelines**: Homomorphisms compose, enabling modular
///   transformation chains
pub struct HomProp<H: MonoidHom> {
    /// The monoid homomorphism to apply
    pub hom: H,
    /// Source cell to read from
    pub source: CellId<H::Source>,
    /// Target cell to write to
    pub target: CellId<H::Target>,
}

impl<H: MonoidHom> Propagator for HomProp<H>
where
    H::Source: 'static,
    H::Target: PartialEq + 'static,
{
    fn activate(&mut self, network: &mut Network) {
        let value = network.read(self.source);
        let result = self.hom.apply(value);
        network.merge(self.target, result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::join_semilattice::Max;
    use std::collections::HashSet;

    #[test]
    fn cell_id_is_copy() {
        // CellId should be Copy even when L is not Copy
        let id: CellId<HashSet<String>> = CellId::new(42);
        let id2 = id; // Copy
        assert_eq!(id, id2);
        assert_eq!(id.raw(), 42);
    }

    #[test]
    fn cell_id_equality() {
        let id1: CellId<i32> = CellId::new(0);
        let id2: CellId<i32> = CellId::new(0);
        let id3: CellId<i32> = CellId::new(1);

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn cell_monotonic_merge() {
        let mut cell = Cell::new(Max(5));

        // Growing updates
        assert!(cell.merge(Max(10)));
        assert_eq!(cell.value(), &Max(10));

        // Non-growing update (no change)
        assert!(!cell.merge(Max(7)));
        assert_eq!(cell.value(), &Max(10));

        // Growing again
        assert!(cell.merge(Max(15)));
        assert_eq!(cell.value(), &Max(15));
    }

    #[test]
    fn cell_hashset_union() {
        let mut cell = Cell::new(HashSet::<i32>::new());

        assert!(cell.merge(HashSet::from([1, 2])));
        assert_eq!(cell.value(), &HashSet::from([1, 2]));

        assert!(cell.merge(HashSet::from([2, 3])));
        assert_eq!(cell.value(), &HashSet::from([1, 2, 3]));

        // Subset doesn't change
        assert!(!cell.merge(HashSet::from([1])));
        assert_eq!(cell.value(), &HashSet::from([1, 2, 3]));
    }

    #[test]
    fn network_heterogeneous_cells() {
        let mut net = Network::new();

        // Create cells with different types
        let c1 = net.add_cell(Max(5));
        let c2 = net.add_cell(HashSet::from([1, 2]));
        let c3 = net.add_cell(Max(100));

        // Type-safe reads
        assert_eq!(net.read(c1), &Max(5));
        assert_eq!(net.read(c2), &HashSet::from([1, 2]));
        assert_eq!(net.read(c3), &Max(100));

        // Updates
        net.merge(c1, Max(10));
        net.merge(c2, HashSet::from([3]));

        assert_eq!(net.read(c1), &Max(10));
        assert_eq!(net.read(c2), &HashSet::from([1, 2, 3]));
    }

    #[test]
    fn network_merge_returns_changed() {
        let mut net = Network::new();
        let cell = net.add_cell(Max(5));

        assert!(net.merge(cell, Max(10))); // changed
        assert!(!net.merge(cell, Max(7))); // unchanged
        assert!(net.merge(cell, Max(15))); // changed
    }

    #[test]
    fn network_unique_ids() {
        let mut net = Network::new();

        let id1 = net.add_cell(Max(1));
        let id2 = net.add_cell(Max(2));
        let id3 = net.add_cell(Max(3));

        // Each cell gets a unique ID
        assert_ne!(id1.raw(), id2.raw());
        assert_ne!(id2.raw(), id3.raw());
        assert_ne!(id1.raw(), id3.raw());
    }

    #[test]
    #[should_panic(expected = "no entry found for key")]
    fn network_invalid_cell_id_panics() {
        let net = Network::new();
        let fake_id: CellId<Max<i32>> = CellId::new(999);
        net.read(fake_id); // Should panic
    }

    // Test that non-lattice semigroups work (String concatenation)
    #[test]
    fn network_string_concatenation() {
        let mut net = Network::new();

        // String is a Semigroup (concatenation) but NOT a lattice
        // (no partial order, no idempotency)
        let cell = net.add_cell(String::from("Hello"));

        // Each merge concatenates
        assert!(net.merge(cell, String::from(", ")));
        assert_eq!(net.read(cell), "Hello, ");

        assert!(net.merge(cell, String::from("World")));
        assert_eq!(net.read(cell), "Hello, World");

        // Note: Unlike lattices, repeated merges keep accumulating
        assert!(net.merge(cell, String::from("!")));
        assert_eq!(net.read(cell), "Hello, World!");
    }

    // Test gradient-like accumulation (Vec with element-wise addition)
    #[test]
    fn network_vector_addition() {
        // Wrapper for Vec<f64> with element-wise addition as Semigroup
        #[derive(Debug, Clone, PartialEq)]
        struct Gradient(Vec<f64>);

        impl algebra_core::Semigroup for Gradient {
            fn combine(&self, other: &Self) -> Self {
                assert_eq!(
                    self.0.len(),
                    other.0.len(),
                    "gradient dimensions must match"
                );
                Gradient(
                    self.0
                        .iter()
                        .zip(other.0.iter())
                        .map(|(a, b)| a + b)
                        .collect(),
                )
            }
        }

        let mut net = Network::new();
        let grad_cell = net.add_cell(Gradient(vec![0.0, 0.0, 0.0]));

        // Accumulate gradients (like backprop)
        assert!(net.merge(grad_cell, Gradient(vec![1.0, 2.0, 3.0])));
        assert_eq!(net.read(grad_cell).0, vec![1.0, 2.0, 3.0]);

        assert!(net.merge(grad_cell, Gradient(vec![0.5, 1.0, 1.5])));
        assert_eq!(net.read(grad_cell).0, vec![1.5, 3.0, 4.5]);

        // This is how gradient accumulation would work in autodiff
        assert!(net.merge(grad_cell, Gradient(vec![0.1, 0.2, 0.3])));
        assert_eq!(net.read(grad_cell).0, vec![1.6, 3.2, 4.8]);
    }

    // Example: A small propagator network that computes relationships
    // between cells. Demonstrates monotonic computation converging to
    // a fixed point.
    #[test]
    fn full_propagator_example() {
        use algebra_core::Semigroup;

        // Propagator: c = max(a, b)
        struct MaxProp {
            a: CellId<Max<i32>>,
            b: CellId<Max<i32>>,
            c: CellId<Max<i32>>,
        }

        impl Propagator for MaxProp {
            fn activate(&mut self, net: &mut Network) {
                let result = net.read(self.a).combine(net.read(self.b));
                net.merge(self.c, result);
            }
        }

        // Propagator: out = in + offset
        struct AddProp {
            input: CellId<Max<i32>>,
            offset: i32,
            output: CellId<Max<i32>>,
        }

        impl Propagator for AddProp {
            fn activate(&mut self, net: &mut Network) {
                let val = net.read(self.input).0;
                net.merge(self.output, Max(val + self.offset));
            }
        }

        // Build network: a, b → max → c → (+10) → d
        let mut net = Network::new();
        let a = net.add_cell(Max(5));
        let b = net.add_cell(Max(10));
        let c = net.add_cell(Max(0));
        let d = net.add_cell(Max(0));

        let mut max_prop = MaxProp { a, b, c };
        let mut add_prop = AddProp {
            input: c,
            offset: 10,
            output: d,
        };

        // Initial activation
        max_prop.activate(&mut net);
        assert_eq!(net.read(c), &Max(10)); // max(5, 10) = 10

        add_prop.activate(&mut net);
        assert_eq!(net.read(d), &Max(20)); // 10 + 10 = 20

        // Update inputs - propagators need reactivation
        net.merge(a, Max(15));
        max_prop.activate(&mut net);
        assert_eq!(net.read(c), &Max(15)); // max(15, 10) = 15

        add_prop.activate(&mut net);
        assert_eq!(net.read(d), &Max(25)); // 15 + 10 = 25

        // Monotonicity: smaller update doesn't change anything
        net.merge(b, Max(8));
        max_prop.activate(&mut net);
        assert_eq!(net.read(c), &Max(15)); // still 15 (max doesn't decrease)

        add_prop.activate(&mut net);
        assert_eq!(net.read(d), &Max(25)); // still 25
    }
}
