//! Propagator networks for monotonic computation with lattice-valued
//! cells.
//!
//! A **propagator** is a computational model where:

//! - **Cells** hold lattice values that can only grow (monotonic
//!    updates)
//! - **Propagators** are functions that read from cells and write to
//!   cells
//! - When a cell's value changes, propagators that depend on it are
//!   scheduled
//! - The system runs until reaching a fixed point (no more changes)
//!
//! This builds on the [`lvar`](crate::lvar) foundation, extending
//! individual monotonic cells to networks of interconnected
//! computations.
//!
//! ## Conceptual model
//!
//! ```text
//!   Cell<A>  Cell<B>
//!      │      │
//!      └──┬───┘
//!         │
//!    Propagator
//!         │
//!      Cell<C>
//! ```
//!
//! When cells A or B change, the propagator runs and may update cell
//! C. Multiple propagators can write to the same cell (values merge
//! via join).
//!
//! ## Status
//!
//! This module is under active development. Core types and traits are
//! curretly getting defined.

use std::any::Any;
use std::collections::HashMap;
use std::marker::PhantomData;

use crate::join_semilattice::JoinSemilattice;

/// A typed identifier for a cell holding values of lattice type `L`.
///
/// The type parameter ensures compile-time safety: you cannot
/// accidentally use a `CellId<i32>` where a `CellId<String>` is
/// expected. At runtime, cells are stored in a type-erased container,
/// but the API maintains type safety through these phantom-typed
/// identifiers.
///
/// # Type Erasure Pattern
///
/// Propagator networks need to store cells with heterogeneous lattice
/// types (e.g., `Cell<i32>`, `Cell<HashSet<String>>`, etc.) in the
/// same container.
/// This is accomplished through a two-layer type system:
///
/// - **Storage layer (runtime)**: Cells stored as `Box<dyn Any>`
///   (type-erased)
/// - **API layer (compile-time)**: `CellId<L>` carries the type
///   information
///
/// When you access a cell via its `CellId<L>`, the network can safely
/// downcast the type-erased storage back to the concrete type
/// `Cell<L>`.
///
/// # Example
///
/// ```rust
/// use postbox::propagator::CellId;
/// use std::collections::HashSet;
///
/// // Each cell ID carries its lattice type
/// let id1: CellId<i32> = CellId::new(0);
/// let id2: CellId<i32> = CellId::new(1);
/// let id3: CellId<HashSet<String>> = CellId::new(0);
///
/// // Can compare IDs of the same type
/// assert_ne!(id1, id2);
/// assert_eq!(id1.raw(), 0);
/// assert_eq!(id2.raw(), 1);
///
/// // But CellId<i32> and CellId<HashSet<String>> are different types
/// // (this would be a compile error):
/// // assert_ne!(id1, id3);  // ← error: mismatched types
/// ```
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct CellId<L> {
    id: usize,
    _phantom: PhantomData<L>,
}

// Manually implement Clone and Copy. CellId is Copy regardless of whether L
// is Copy, since it only contains a usize and PhantomData (both always Copy).
// The derived implementations would incorrectly require L: Clone/Copy.
impl<L> Clone for CellId<L> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<L> Copy for CellId<L> {}

impl<L> CellId<L> {
    /// Create a new cell ID from a raw integer.
    ///
    /// This is typically called internally by the network when creating cells.
    /// The type parameter `L` captures what lattice type this cell holds.
    ///
    /// # Example
    ///
    /// ```rust
    /// use postbox::propagator::CellId;
    ///
    /// let id: CellId<i32> = CellId::new(42);
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

/// A cell holding a lattice value of type `L`.
///
/// Cells store monotonically growing values. Updates merge via
/// lattice join, ensuring the value only increases in the lattice
/// order. When a cell's value changes, dependent propagators are
/// notified (dependency tracking to be added).
///
/// # Monotonicity
///
/// The key invariant: values can only grow. If you merge a value `v`
/// into a cell holding `current`, the new value is `current.join(v)`.
/// Since join is idempotent and associative, the order of merges
/// doesn't matter - the cell will converge to the same final value
/// regardless.
///
/// # Example
///
/// ```rust
/// use postbox::propagator::Cell;
/// use std::collections::HashSet;
///
/// let mut cell = Cell::new(HashSet::<i32>::new());
///
/// // Merging adds elements (HashSet join = union)
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
pub struct Cell<L: JoinSemilattice> {
    value: L,
    // Future: dependency tracking (Vec<PropagatorId>)
}

impl<L: JoinSemilattice> Cell<L> {
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
    pub fn new(value: L) -> Self {
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
    pub fn value(&self) -> &L {
        &self.value
    }

    /// Merge a new value into the cell via lattice join.
    ///
    /// Returns `true` if the cell's value changed, `false` if it
    /// remained the same. A change indicates that the new value added
    /// information (grew in the lattice).
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
    pub fn merge(&mut self, new_value: L) -> bool
    where
        L: PartialEq,
    {
        let merged = self.value.join(&new_value);
        if merged != self.value {
            self.value = merged;
            true
        } else {
            false
        }
    }
}

/// A propagator network containing cells with heterogeneous lattice types.
///
/// Networks store cells in a type-erased container (`Box<dyn Any>`) while
/// maintaining type safety through [`CellId<L>`]. This allows a single network
/// to contain cells of different types (e.g., `Cell<i32>`, `Cell<HashSet<String>>`,
/// etc.) without requiring all cells to share a common trait object type.
///
/// # Type Erasure
///
/// The network uses a two-layer type system:
/// - **Storage**: Cells stored as `Box<dyn Any>` (heterogeneous)
/// - **API**: `CellId<L>` carries type information (type-safe)
///
/// When you create a cell, you get back a `CellId<L>` that remembers the type.
/// When you access the cell, the network uses the type from `CellId<L>` to
/// safely downcast from `Any` back to the concrete type.
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
    /// Returns a typed `CellId<L>` that can be used to access this cell.
    /// The type parameter `L` is inferred from the initial value.
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
    pub fn add_cell<L: JoinSemilattice + 'static>(&mut self, initial: L) -> CellId<L> {
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
    /// (which should be impossible if you only use `CellId`s returned by this network).
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
    pub fn read<L: JoinSemilattice + 'static>(&self, id: CellId<L>) -> &L {
        self.cells[&id.raw()]
            .downcast_ref::<Cell<L>>()
            .expect("type mismatch")
            .value()
    }

    /// Merge a value into a cell via lattice join.
    ///
    /// Returns `true` if the cell's value changed, `false` otherwise.
    /// A change means the new value added information (grew in the lattice order).
    ///
    /// # Panics
    ///
    /// Panics if the cell ID is invalid or if there's a type mismatch.
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
    pub fn merge<L: JoinSemilattice + PartialEq + 'static>(
        &mut self,
        id: CellId<L>,
        value: L,
    ) -> bool {
        self.cells
            .get_mut(&id.raw())
            .expect("cell not found")
            .downcast_mut::<Cell<L>>()
            .expect("type mismatch")
            .merge(value)
    }
}

impl Default for Network {
    fn default() -> Self {
        Self::new()
    }
}

/// A propagator that performs monotonic computation on a network.
///
/// Propagators are functions that read values from cells and write values to cells.
/// When activated, a propagator reads its input cells, performs some computation,
/// and merges the result into output cells. Since all updates are monotonic (via
/// lattice join), the order of propagator activations doesn't affect the final result.
///
/// # Monotonicity
///
/// Propagators must be **monotonic**: if inputs grow (in the lattice order), outputs
/// can only grow (never shrink). This ensures that the network converges to a unique
/// fixed point regardless of execution order.
///
/// # Example
///
/// ```rust
/// use postbox::propagator::{Propagator, Network, CellId};
/// use postbox::join_semilattice::{Max, JoinSemilattice};
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
///         let result = a_val.join(&b_val);
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
    /// Activate this propagator, reading from and writing to the network.
    ///
    /// The propagator reads values from input cells, performs computation,
    /// and merges results into output cells. This method should be idempotent
    /// (calling it multiple times with the same cell values produces the same result).
    fn activate(&mut self, network: &mut Network);
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

        assert!(net.merge(cell, Max(10)));  // changed
        assert!(!net.merge(cell, Max(7)));  // unchanged
        assert!(net.merge(cell, Max(15)));  // changed
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

    // Example: A small propagator network that computes relationships
    // between cells. Demonstrates monotonic computation converging to
    // a fixed point.
    #[test]
    fn full_propagator_example() {
        use crate::join_semilattice::JoinSemilattice;

        // Propagator: c = max(a, b)
        struct MaxProp {
            a: CellId<Max<i32>>,
            b: CellId<Max<i32>>,
            c: CellId<Max<i32>>,
        }

        impl Propagator for MaxProp {
            fn activate(&mut self, net: &mut Network) {
                let result = net.read(self.a).join(net.read(self.b));
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
