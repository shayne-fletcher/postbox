//! Automatic differentiation for Rust.
//!
//! **Part of the [postbox workspace](../index.html)**
//!
//! This crate provides tools for computing derivatives automatically:
//!
//! - **Forward-mode AD**: Using dual numbers ([`Dual`])
//! - **Reverse-mode AD**: (planned)
//!
//! # Example
//!
//! ```
//! use autodiff::Dual;
//!
//! // Define a function
//! fn f(x: Dual<f64>) -> Dual<f64> {
//!     x * x + Dual::constant(2.0) * x
//! }
//!
//! // Compute f and f' at x=3
//! let y = f(Dual::variable(3.0));
//! assert_eq!(y.value, 15.0);   // f(3) = 15
//! assert_eq!(y.deriv, 8.0);    // f'(3) = 8
//! ```

pub mod dual;

pub use dual::Dual;
