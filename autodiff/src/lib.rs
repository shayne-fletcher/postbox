//! Automatic differentiation for Rust.
//!
//! **Part of the [postbox workspace](../index.html)**
//!
//! This crate provides tools for computing derivatives automatically:
//!
//! - **Forward-mode AD**: Using dual numbers ([`Dual`]) for single-variable functions
//! - **Multivariable gradients**: Using multi-component dual numbers ([`MultiDual`])
//! - **Reverse-mode AD**: (planned)
//!
//! # Single-variable differentiation
//!
//! Use [`Dual`] for computing derivatives of functions f: ℝ → ℝ:
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
//!
//! # Multivariable gradients
//!
//! Use [`MultiDual`] and [`gradient`] for computing gradients of functions f: ℝⁿ → ℝ
//! in a single forward pass:
//!
//! ```
//! use autodiff::{MultiDual, gradient};
//!
//! // f(x, y) = x² + 2xy + y²
//! let f = |vars: [MultiDual<f64, 2>; 2]| {
//!     let [x, y] = vars;
//!     let two = MultiDual::constant(2.0);
//!     x * x + two * x * y + y * y
//! };
//!
//! // Compute f and ∇f at (3, 4)
//! let (value, grad) = gradient(f, [3.0, 4.0]);
//! assert_eq!(value, 49.0);    // f(3, 4) = 49
//! assert_eq!(grad[0], 14.0);  // ∂f/∂x = 14
//! assert_eq!(grad[1], 14.0);  // ∂f/∂y = 14
//! ```

pub mod dual;
pub mod multidual;

pub use dual::Dual;
pub use multidual::{gradient, MultiDual};
