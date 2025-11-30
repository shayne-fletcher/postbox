//! Propagator networks for monotonic computation with lattice-valued cells.
//!
//! A **propagator** is a computational model where:
//! - **Cells** hold lattice values that can only grow (monotonic updates)
//! - **Propagators** are functions that read from cells and write to cells
//! - When a cell's value changes, propagators that depend on it are scheduled
//! - The system runs until reaching a fixed point (no more changes)
//!
//! This builds on the [`lvar`](crate::lvar) foundation, extending individual
//! monotonic cells to networks of interconnected computations.
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
//! When cells A or B change, the propagator runs and may update cell C.
//! Multiple propagators can write to the same cell (values merge via join).
//!
//! ## Status
//!
//! This module is under active development. Core types and traits are not yet defined.
