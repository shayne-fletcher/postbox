//! Dual numbers for forward-mode automatic differentiation.
//!
//! A dual number represents a value and its derivative simultaneously,
//! enabling automatic computation of derivatives through operator
//! overloading.
//!
//! # Mathematical Background
//!
//! A dual number has the form `a + a′·ε` where `ε² = 0` (and `a′` denotes
//! the derivative). Arithmetic operations on dual numbers follow these
//! algebraic rules:
//!
//! - `(a + a′·ε) + (b + b′·ε) = (a+b) + (a′+b′)·ε`
//! - `-(a + a′·ε) = -a + (-a′)·ε`
//! - `(a + a′·ε) - (b + b′·ε) = (a-b) + (a′-b′)·ε`
//! - `(a + a′·ε) * (b + b′·ε) = ab + (a′b + ab′)·ε`
//! - `1/(b + b′·ε) = (1/b) + (-b′/b²)·ε`
//! - `(a + a′·ε) / (b + b′·ε) = (a + a′·ε) * (1/(b + b′·ε))`
//!
//! The chain rule emerges implicitly from composing these operations—you
//! never write it down explicitly.
//!
//! This is **forward-mode** automatic differentiation: we compute the
//! derivative as we compute the function value.
//!
//! # Example
//!
//! ```
//! use algebra_core::Dual;
//!
//! // Compute f(x) = x² + 2x at x=3
//! let x = Dual::variable(3.0);  // x with derivative dx/dx = 1
//!
//! let f = x * x + Dual::constant(2.0) * x;
//!
//! assert_eq!(f.value, 15.0);    // f(3) = 9 + 6 = 15
//! assert_eq!(f.deriv, 8.0);     // f'(3) = 2*3 + 2 = 8
//! ```
//!
//! # Supported Operations
//!
//! - **Arithmetic**: `+`, `-`, `*`, `/`, negation
//! - **Transcendental**: `exp`, `ln`, `sin`, `cos`, `sqrt`
//! - Derivatives propagate automatically; chain rule emerges from composition
//!
//! # Use Cases
//!
//! - Computing derivatives of scalar functions
//! - Gradient-based optimization and neural networks
//! - Sensitivity analysis
//! - Physics simulations requiring derivatives

use num_traits::{Float, One, Zero};
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A dual number representing a value and its derivative.
///
/// `Dual(value, deriv)` represents `value + deriv·ε` where `ε² = 0`.
/// Arithmetic operations follow the algebraic rules of dual numbers;
/// derivatives propagate automatically.
///
/// # Type Parameter
///
/// - `T`: The numeric type (typically `f64` or `f32`)
///
/// # Examples
///
/// ## Basic Usage
///
/// ```
/// use algebra_core::Dual;
///
/// let x = Dual::variable(5.0);
/// let y = x * x;  // y = x²
///
/// assert_eq!(y.value, 25.0);  // 5² = 25
/// assert_eq!(y.deriv, 10.0);  // d/dx(x²) at x=5 is 2*5 = 10
/// ```
///
/// ## Chain Rule
///
/// ```
/// use algebra_core::Dual;
///
/// // f(x) = (x + 1) * (x + 2)
/// let x = Dual::variable(3.0);
/// let f = (x + Dual::constant(1.0)) * (x + Dual::constant(2.0));
///
/// assert_eq!(f.value, 20.0);  // (3+1)*(3+2) = 4*5 = 20
/// assert_eq!(f.deriv, 9.0);   // f'(x) = 2x+3, f'(3) = 9
/// ```
///
/// ## Multiple Operations
///
/// ```
/// use algebra_core::Dual;
///
/// // f(x) = x³ - 2x + 1
/// let x = Dual::variable(2.0);
/// let x2 = x * x;
/// let x3 = x2 * x;
/// let f = x3 - Dual::constant(2.0) * x + Dual::constant(1.0);
///
/// assert_eq!(f.value, 5.0);   // 8 - 4 + 1 = 5
/// assert_eq!(f.deriv, 10.0);  // f'(x) = 3x² - 2, f'(2) = 12 - 2 = 10
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual<T> {
    /// The primal value
    pub value: T,
    /// The derivative (tangent)
    pub deriv: T,
}

impl<T> Dual<T> {
    /// Create a new dual number with explicit value and derivative.
    ///
    /// # Example
    ///
    /// ```
    /// use algebra_core::Dual;
    ///
    /// let d = Dual::new(3.0, 1.0);
    /// assert_eq!(d.value, 3.0);
    /// assert_eq!(d.deriv, 1.0);
    /// ```
    pub fn new(value: T, deriv: T) -> Self {
        Dual { value, deriv }
    }

    /// Create a constant (derivative = 0).
    ///
    /// Use this for literal values in your computation.
    ///
    /// # Example
    ///
    /// ```
    /// use algebra_core::Dual;
    ///
    /// let c = Dual::constant(5.0);
    /// assert_eq!(c.value, 5.0);
    /// assert_eq!(c.deriv, 0.0);
    /// ```
    pub fn constant(value: T) -> Self
    where
        T: Zero,
    {
        Dual {
            value,
            deriv: T::zero(),
        }
    }

    /// Create a variable (derivative = 1).
    ///
    /// Use this for the input variable you're differentiating with
    /// respect to.
    ///
    /// # Example
    ///
    /// ```
    /// use algebra_core::Dual;
    ///
    /// let x = Dual::variable(3.0);
    /// assert_eq!(x.value, 3.0);
    /// assert_eq!(x.deriv, 1.0);  // dx/dx = 1
    /// ```
    pub fn variable(value: T) -> Self
    where
        T: One,
    {
        Dual {
            value,
            deriv: T::one(),
        }
    }

    /// Reciprocal (multiplicative inverse).
    ///
    /// For `g = b + b′·ε`, computes `1/g = (1/b) + (-b′/b²)·ε`.
    ///
    /// This encodes the derivative of `1/x`: `d/dx(1/x) = -1/x²`.
    ///
    /// # Example
    ///
    /// ```
    /// use algebra_core::Dual;
    ///
    /// // f(x) = 1/x at x=2
    /// let x = Dual::variable(2.0);
    /// let f = x.recip();
    ///
    /// assert_eq!(f.value, 0.5);      // 1/2 = 0.5
    /// assert_eq!(f.deriv, -0.25);    // d/dx(1/x) at x=2 is -1/4
    /// ```
    pub fn recip(self) -> Self
    where
        T: One + Div<Output = T> + Mul<Output = T> + Neg<Output = T> + Clone,
    {
        let b = self.value.clone();
        let b_squared = b.clone() * b.clone();

        Dual {
            value: T::one() / b.clone(),
            deriv: -(self.deriv / b_squared),
        }
    }

    /// Exponential function.
    ///
    /// For `f = a + a′·ε`, computes `e^f = e^a + (a′·e^a)·ε`.
    ///
    /// This encodes the derivative: `d/dx(e^f) = f′·e^f`.
    ///
    /// # Example
    ///
    /// ```
    /// use algebra_core::Dual;
    ///
    /// // f(x) = e^x at x=0
    /// let x = Dual::variable(0.0);
    /// let f = x.exp();
    ///
    /// assert_eq!(f.value, 1.0);      // e^0 = 1
    /// assert_eq!(f.deriv, 1.0);      // d/dx(e^x) at x=0 is e^0 = 1
    /// ```
    pub fn exp(self) -> Self
    where
        T: Float,
    {
        let exp_val = self.value.exp();
        Dual {
            value: exp_val,
            deriv: self.deriv * exp_val,
        }
    }

    /// Natural logarithm.
    ///
    /// For `f = a + a′·ε`, computes `ln(f) = ln(a) + (a′/a)·ε`.
    ///
    /// This encodes the derivative: `d/dx(ln f) = f′/f`.
    ///
    /// # Example
    ///
    /// ```
    /// use algebra_core::Dual;
    ///
    /// // f(x) = ln(x) at x=1
    /// let x = Dual::variable(1.0);
    /// let f = x.ln();
    ///
    /// assert_eq!(f.value, 0.0);      // ln(1) = 0
    /// assert_eq!(f.deriv, 1.0);      // d/dx(ln x) at x=1 is 1/1 = 1
    /// ```
    pub fn ln(self) -> Self
    where
        T: Float,
    {
        Dual {
            value: self.value.ln(),
            deriv: self.deriv / self.value,
        }
    }

    /// Sine function.
    ///
    /// For `f = a + a′·ε`, computes `sin(f) = sin(a) + (a′·cos(a))·ε`.
    ///
    /// This encodes the derivative: `d/dx(sin f) = f′·cos f`.
    ///
    /// # Example
    ///
    /// ```
    /// use algebra_core::Dual;
    ///
    /// // f(x) = sin(x) at x=0
    /// let x = Dual::variable(0.0);
    /// let f = x.sin();
    ///
    /// assert_eq!(f.value, 0.0);      // sin(0) = 0
    /// assert_eq!(f.deriv, 1.0);      // d/dx(sin x) at x=0 is cos(0) = 1
    /// ```
    pub fn sin(self) -> Self
    where
        T: Float,
    {
        Dual {
            value: self.value.sin(),
            deriv: self.deriv * self.value.cos(),
        }
    }

    /// Cosine function.
    ///
    /// For `f = a + a′·ε`, computes `cos(f) = cos(a) + (-a′·sin(a))·ε`.
    ///
    /// This encodes the derivative: `d/dx(cos f) = -f′·sin f`.
    ///
    /// # Example
    ///
    /// ```
    /// use algebra_core::Dual;
    ///
    /// // f(x) = cos(x) at x=0
    /// let x = Dual::variable(0.0);
    /// let f = x.cos();
    ///
    /// assert_eq!(f.value, 1.0);      // cos(0) = 1
    /// assert_eq!(f.deriv, 0.0);      // d/dx(cos x) at x=0 is -sin(0) = 0
    /// ```
    pub fn cos(self) -> Self
    where
        T: Float,
    {
        Dual {
            value: self.value.cos(),
            deriv: -self.deriv * self.value.sin(),
        }
    }

    /// Square root.
    ///
    /// For `f = a + a′·ε`, computes `√f = √a + (a′/(2√a))·ε`.
    ///
    /// This encodes the derivative: `d/dx(√f) = f′/(2√f)`.
    ///
    /// # Example
    ///
    /// ```
    /// use algebra_core::Dual;
    ///
    /// // f(x) = √x at x=4
    /// let x = Dual::variable(4.0);
    /// let f = x.sqrt();
    ///
    /// assert_eq!(f.value, 2.0);      // √4 = 2
    /// assert_eq!(f.deriv, 0.25);     // d/dx(√x) at x=4 is 1/(2*2) = 0.25
    /// ```
    pub fn sqrt(self) -> Self
    where
        T: Float,
    {
        let sqrt_val = self.value.sqrt();
        Dual {
            value: sqrt_val,
            deriv: self.deriv / (sqrt_val + sqrt_val),
        }
    }
}

/// Addition: (a + a′·ε) + (b + b′·ε) = (a+b) + (a′+b′)·ε
impl<T: Add<Output = T>> Add for Dual<T> {
    type Output = Dual<T>;

    fn add(self, rhs: Self) -> Self::Output {
        Dual {
            value: self.value + rhs.value,
            deriv: self.deriv + rhs.deriv,
        }
    }
}

/// Subtraction: (a + a′·ε) - (b + b′·ε) = (a-b) + (a′-b′)·ε
impl<T: Sub<Output = T>> Sub for Dual<T> {
    type Output = Dual<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        Dual {
            value: self.value - rhs.value,
            deriv: self.deriv - rhs.deriv,
        }
    }
}

/// Multiplication: (a + a′·ε) * (b + b′·ε) = ab + (a′b + ab′)·ε
///
/// This implements the product rule: d/dx(f·g) = f′·g + f·g′
impl<T: Mul<Output = T> + Add<Output = T> + Clone> Mul for Dual<T> {
    type Output = Dual<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        Dual {
            value: self.value.clone() * rhs.value.clone(),
            // Product rule: f′·g + f·g′
            deriv: self.deriv * rhs.value + self.value * rhs.deriv,
        }
    }
}

/// Division: `f / g = f * (1/g)`.
///
/// The quotient rule emerges automatically from the product rule
/// (in `Mul`) composed with the reciprocal rule (in `recip`).
#[allow(clippy::suspicious_arithmetic_impl)]
impl<T> Div for Dual<T>
where
    T: One + Div<Output = T> + Mul<Output = T> + Add<Output = T> + Neg<Output = T> + Clone,
{
    type Output = Dual<T>;

    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.recip()
    }
}

/// Negation: -(a + a′·ε) = -a + (-a′)·ε
impl<T: Neg<Output = T>> Neg for Dual<T> {
    type Output = Dual<T>;

    fn neg(self) -> Self::Output {
        Dual {
            value: -self.value,
            deriv: -self.deriv,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_has_zero_derivative() {
        let c = Dual::constant(5.0);
        assert_eq!(c.value, 5.0);
        assert_eq!(c.deriv, 0.0);
    }

    #[test]
    fn variable_has_unit_derivative() {
        let x = Dual::variable(3.0);
        assert_eq!(x.value, 3.0);
        assert_eq!(x.deriv, 1.0);
    }

    #[test]
    fn addition_works() {
        let x = Dual::variable(3.0);
        let c = Dual::constant(5.0);
        let y = x + c;

        assert_eq!(y.value, 8.0);
        assert_eq!(y.deriv, 1.0); // d/dx(x + 5) = 1
    }

    #[test]
    fn multiplication_implements_product_rule() {
        // f(x) = x * x = x²
        let x = Dual::variable(3.0);
        let y = x * x;

        assert_eq!(y.value, 9.0);
        assert_eq!(y.deriv, 6.0); // d/dx(x²) at x=3 is 2*3 = 6
    }

    #[test]
    fn recip_implements_inverse_rule() {
        // f(x) = 1/x at x=2
        let x = Dual::variable(2.0);
        let y = x.recip();

        assert_eq!(y.value, 0.5);
        assert_eq!(y.deriv, -0.25); // d/dx(1/x) at x=2 is -1/4
    }

    #[test]
    fn division_via_recip() {
        // f(x) = 1/x at x=2, using division operator
        let x = Dual::variable(2.0);
        let one = Dual::constant(1.0);
        let y = one / x;

        assert_eq!(y.value, 0.5);
        assert_eq!(y.deriv, -0.25); // d/dx(1/x) at x=2 is -1/4
    }

    #[test]
    fn division_quotient_rule() {
        // f(x) = (x+1)/(x+2) at x=3
        // f(x) = 4/5 = 0.8
        // f'(x) = [(x+2) - (x+1)]/(x+2)² = 1/(x+2)² = 1/25 = 0.04
        let x = Dual::variable(3.0);
        let num = x + Dual::constant(1.0);
        let den = x + Dual::constant(2.0);
        let y = num / den;

        assert_eq!(y.value, 0.8);
        assert!((y.deriv - 0.04_f64).abs() < 1e-10); // floating point tolerance
    }

    #[test]
    fn chain_rule_example() {
        // f(x) = (x + 1) * (x + 2)
        let x = Dual::variable(3.0);
        let f = (x + Dual::constant(1.0)) * (x + Dual::constant(2.0));

        assert_eq!(f.value, 20.0); // (3+1)*(3+2) = 20
        assert_eq!(f.deriv, 9.0); // f'(x) = 2x+3, f'(3) = 9
    }

    #[test]
    fn polynomial_example() {
        // f(x) = x³ - 2x + 1 at x=2
        let x = Dual::variable(2.0);
        let x2 = x * x;
        let x3 = x2 * x;
        let f = x3 - Dual::constant(2.0) * x + Dual::constant(1.0);

        assert_eq!(f.value, 5.0); // 8 - 4 + 1 = 5
        assert_eq!(f.deriv, 10.0); // f'(x) = 3x² - 2, f'(2) = 10
    }

    #[test]
    fn negation_works() {
        let x = Dual::variable(3.0);
        let y = -x;

        assert_eq!(y.value, -3.0);
        assert_eq!(y.deriv, -1.0);
    }

    #[test]
    fn subtraction_works() {
        let x = Dual::variable(5.0);
        let c = Dual::constant(2.0);
        let y = x - c;

        assert_eq!(y.value, 3.0);
        assert_eq!(y.deriv, 1.0); // d/dx(x - 2) = 1
    }

    #[test]
    fn exp_works() {
        // f(x) = e^x at x=0
        let x = Dual::variable(0.0);
        let f = x.exp();

        assert_eq!(f.value, 1.0); // e^0 = 1
        assert_eq!(f.deriv, 1.0); // d/dx(e^x) at x=0 is e^0 = 1
    }

    #[test]
    fn ln_works() {
        // f(x) = ln(x) at x=1
        let x = Dual::variable(1.0);
        let f = x.ln();

        assert_eq!(f.value, 0.0); // ln(1) = 0
        assert_eq!(f.deriv, 1.0); // d/dx(ln x) at x=1 is 1/1 = 1
    }

    #[test]
    fn sin_works() {
        // f(x) = sin(x) at x=0
        let x = Dual::variable(0.0);
        let f = x.sin();

        assert_eq!(f.value, 0.0); // sin(0) = 0
        assert_eq!(f.deriv, 1.0); // d/dx(sin x) at x=0 is cos(0) = 1
    }

    #[test]
    fn cos_works() {
        // f(x) = cos(x) at x=0
        let x = Dual::variable(0.0);
        let f = x.cos();

        assert_eq!(f.value, 1.0); // cos(0) = 1
        assert_eq!(f.deriv, 0.0); // d/dx(cos x) at x=0 is -sin(0) = 0
    }

    #[test]
    fn sqrt_works() {
        // f(x) = √x at x=4
        let x = Dual::variable(4.0);
        let f = x.sqrt();

        assert_eq!(f.value, 2.0); // √4 = 2
        assert_eq!(f.deriv, 0.25); // d/dx(√x) at x=4 is 1/(2*2) = 0.25
    }

    #[test]
    fn chain_rule_with_transcendentals() {
        // f(x) = sin(2x) at x=0
        // f'(x) = 2*cos(2x), f'(0) = 2*1 = 2
        let x = Dual::variable(0.0);
        let two_x = Dual::constant(2.0) * x;
        let f = two_x.sin();

        assert_eq!(f.value, 0.0);
        assert_eq!(f.deriv, 2.0);
    }

    #[test]
    fn exp_of_polynomial() {
        // f(x) = e^(x²) at x=1
        // f'(x) = 2x * e^(x²), f'(1) = 2 * e
        let x = Dual::variable(1.0);
        let x_squared = x * x;
        let f = x_squared.exp();

        let e = 1.0_f64.exp();
        assert_eq!(f.value, e);
        assert!((f.deriv - 2.0 * e).abs() < 1e-10);
    }
}
