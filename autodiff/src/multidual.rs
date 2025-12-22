//! Multi-component dual numbers for multivariable automatic
//! differentiation.
//!
//! A multi-component dual number tracks a value and multiple partial
//! derivatives simultaneously, enabling computation of gradients in a
//! **single forward pass**.
//!
//! # Mathematical Background
//!
//! For a function f: ℝⁿ → ℝ, `MultiDual<T, N>` represents a value and
//! its gradient ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ] simultaneously.
//!
//! Arithmetic operations extend naturally from single-variable dual
//! numbers:
//!
//! - `(a + ∇a) + (b + ∇b) = (a+b) + (∇a+∇b)`
//! - `-(a + ∇a) = -a + (-∇a)`
//! - `(a + ∇a) - (b + ∇b) = (a-b) + (∇a-∇b)`
//! - `(a + ∇a) * (b + ∇b) = ab + (b∇a + a∇b)`
//! - `1/(b + ∇b) = (1/b) + (-∇b/b²)`
//!
//! Each operation updates all N derivative components at once,
//! computing the full gradient in a single pass through the
//! computation.
//!
//! # Example
//!
//! ```
//! use autodiff::{MultiDual, gradient};
//!
//! // Compute ∇f for f(x, y) = x² + 2xy + y² at (3, 4)
//! let f = |vars: [MultiDual<f64, 2>; 2]| {
//!     let [x, y] = vars;
//!     let two = MultiDual::constant(2.0);
//!     x * x + two * x * y + y * y
//! };
//!
//! let point = [3.0, 4.0];
//! let (value, grad) = gradient(f, point);
//!
//! assert_eq!(value, 49.0);   // f(3, 4) = 9 + 24 + 16 = 49
//! assert_eq!(grad[0], 14.0); // ∂f/∂x = 2x + 2y = 14
//! assert_eq!(grad[1], 14.0); // ∂f/∂y = 2x + 2y = 14
//! ```
//!
//! # Efficiency
//!
//! Computing the gradient requires **1 forward pass** with
//! `MultiDual<T, N>`, compared to n passes if using `Dual<T>` to
//! compute each partial derivative separately. For n=10 inputs, this
//! is a 10x speedup.
//!
//! # Use Cases
//!

//! - Gradient-based optimization (gradient descent, Newton's method)
//! - Neural network backpropagation alternatives
//! - Sensitivity analysis with multiple parameters
//! - Scientific computing with multivariable functions

use num_traits::{Float, One, Zero};
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A multi-component dual number representing a value and N partial
/// derivatives.
///
/// `MultiDual<T, N>` represents a value along with its gradient
/// [∂f/∂x₁, ..., ∂f/∂xₙ] for forward-mode automatic differentiation
/// of multivariable functions.
///
/// # Type Parameters
///
/// - `T`: The numeric type (typically `f64` or `f32`)
/// - `N`: The number of input variables (compile-time constant)
///
/// # Examples
///
/// ## Creating Variables
///
/// ```
/// use autodiff::MultiDual;
///
/// // Create the first variable x with value 3.0 (∂/∂x = 1, ∂/∂y = 0)
/// let x = MultiDual::<f64, 2>::variable(3.0, 0);
/// assert_eq!(x.value, 3.0);
/// assert_eq!(x.derivs, [1.0, 0.0]);
///
/// // Create the second variable y with value 4.0 (∂/∂x = 0, ∂/∂y = 1)
/// let y = MultiDual::<f64, 2>::variable(4.0, 1);
/// assert_eq!(y.value, 4.0);
/// assert_eq!(y.derivs, [0.0, 1.0]);
///
/// // Create a constant (∂/∂x = 0, ∂/∂y = 0)
/// let c = MultiDual::<f64, 2>::constant(2.0);
/// assert_eq!(c.value, 2.0);
/// assert_eq!(c.derivs, [0.0, 0.0]);
/// ```
///
/// ## Computing Gradients
///
/// ```
/// use autodiff::MultiDual;
///
/// // f(x, y) = x² + y²
/// let x = MultiDual::<f64, 2>::variable(3.0, 0);
/// let y = MultiDual::<f64, 2>::variable(4.0, 1);
///
/// let f = x * x + y * y;
///
/// assert_eq!(f.value, 25.0);      // 3² + 4² = 25
/// assert_eq!(f.derivs[0], 6.0);   // ∂f/∂x = 2x = 6
/// assert_eq!(f.derivs[1], 8.0);   // ∂f/∂y = 2y = 8
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiDual<T, const N: usize> {
    /// The primal value (function output)
    pub value: T,
    /// The partial derivatives [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
    pub derivs: [T; N],
}

impl<T, const N: usize> MultiDual<T, N>
where
    T: Copy,
{
    /// Create a dual number with explicit value and derivatives.
    ///
    /// # Examples
    ///
    /// ```
    /// use autodiff::MultiDual;
    ///
    /// let d = MultiDual::new(5.0, [1.0, 2.0, 3.0]);
    /// assert_eq!(d.value, 5.0);
    /// assert_eq!(d.derivs, [1.0, 2.0, 3.0]);
    /// ```
    pub fn new(value: T, derivs: [T; N]) -> Self {
        Self { value, derivs }
    }

    /// Create a constant (all partial derivatives are zero).
    ///
    /// # Examples
    ///
    /// ```
    /// use autodiff::MultiDual;
    ///
    /// let c = MultiDual::<f64, 3>::constant(42.0);
    /// assert_eq!(c.value, 42.0);
    /// assert_eq!(c.derivs, [0.0, 0.0, 0.0]);
    /// ```
    pub fn constant(value: T) -> Self
    where
        T: Zero,
    {
        Self {
            value,
            derivs: [T::zero(); N],
        }
    }

    /// Create the i-th input variable.
    ///
    /// Sets `derivs[index] = 1` and all other derivatives to zero,
    /// representing ∂xᵢ/∂xⱼ = δᵢⱼ (Kronecker delta).
    ///
    /// # Panics
    ///
    /// Panics if `index >= N`.
    ///
    /// # Examples
    ///
    /// ```
    /// use autodiff::MultiDual;
    ///
    /// // Create x (first variable)
    /// let x = MultiDual::<f64, 2>::variable(3.0, 0);
    /// assert_eq!(x.value, 3.0);
    /// assert_eq!(x.derivs, [1.0, 0.0]);
    ///
    /// // Create y (second variable)
    /// let y = MultiDual::<f64, 2>::variable(4.0, 1);
    /// assert_eq!(y.value, 4.0);
    /// assert_eq!(y.derivs, [0.0, 1.0]);
    /// ```
    pub fn variable(value: T, index: usize) -> Self
    where
        T: Zero + One,
    {
        assert!(
            index < N,
            "Variable index {} out of bounds for N={}",
            index,
            N
        );
        let mut derivs = [T::zero(); N];
        derivs[index] = T::one();
        Self { value, derivs }
    }
}

/// Addition: `(a + ∇a) + (b + ∇b) = (a+b) + (∇a+∇b)`
///
/// # Examples
///
/// ```
/// use autodiff::MultiDual;
///
/// let x = MultiDual::new(3.0, [1.0, 0.0]);
/// let y = MultiDual::new(4.0, [0.0, 1.0]);
/// let sum = x + y;
///
/// assert_eq!(sum.value, 7.0);
/// assert_eq!(sum.derivs, [1.0, 1.0]);
/// ```
impl<T, const N: usize> Add for MultiDual<T, N>
where
    T: Add<Output = T> + Copy,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let mut derivs = self.derivs;
        for (deriv, rhs_deriv) in derivs.iter_mut().zip(rhs.derivs.iter()) {
            *deriv = *deriv + *rhs_deriv;
        }
        Self {
            value: self.value + rhs.value,
            derivs,
        }
    }
}

/// Subtraction: `(a + ∇a) - (b + ∇b) = (a-b) + (∇a-∇b)`
///
/// # Examples
///
/// ```
/// use autodiff::MultiDual;
///
/// let x = MultiDual::new(7.0, [2.0, 3.0]);
/// let y = MultiDual::new(4.0, [1.0, 2.0]);
/// let diff = x - y;
///
/// assert_eq!(diff.value, 3.0);
/// assert_eq!(diff.derivs, [1.0, 1.0]);
/// ```
impl<T, const N: usize> Sub for MultiDual<T, N>
where
    T: Sub<Output = T> + Copy,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        let mut derivs = self.derivs;
        for (deriv, rhs_deriv) in derivs.iter_mut().zip(rhs.derivs.iter()) {
            *deriv = *deriv - *rhs_deriv;
        }
        Self {
            value: self.value - rhs.value,
            derivs,
        }
    }
}

/// Multiplication: `(a + ∇a) * (b + ∇b) = ab + (b∇a + a∇b)`
///
/// This implements the product rule automatically.
///
/// # Examples
///
/// ```
/// use autodiff::MultiDual;
///
/// // f(x, y) = x * y at (3, 4)
/// let x = MultiDual::<f64, 2>::variable(3.0, 0);
/// let y = MultiDual::<f64, 2>::variable(4.0, 1);
/// let product = x * y;
///
/// assert_eq!(product.value, 12.0);      // 3 * 4
/// assert_eq!(product.derivs[0], 4.0);   // ∂(xy)/∂x = y = 4
/// assert_eq!(product.derivs[1], 3.0);   // ∂(xy)/∂y = x = 3
/// ```
impl<T, const N: usize> Mul for MultiDual<T, N>
where
    T: Mul<Output = T> + Add<Output = T> + Copy,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let mut derivs = [self.value; N];
        for (deriv, (self_deriv, rhs_deriv)) in derivs
            .iter_mut()
            .zip(self.derivs.iter().zip(rhs.derivs.iter()))
        {
            // Product rule: (f*g)' = f'*g + f*g'
            *deriv = *self_deriv * rhs.value + self.value * *rhs_deriv;
        }
        Self {
            value: self.value * rhs.value,
            derivs,
        }
    }
}

/// Negation: `-(a + ∇a) = -a + (-∇a)`
///
/// # Examples
///
/// ```
/// use autodiff::MultiDual;
///
/// let x = MultiDual::new(3.0, [1.0, 2.0]);
/// let neg_x = -x;
///
/// assert_eq!(neg_x.value, -3.0);
/// assert_eq!(neg_x.derivs, [-1.0, -2.0]);
/// ```
impl<T, const N: usize> Neg for MultiDual<T, N>
where
    T: Neg<Output = T> + Copy,
{
    type Output = Self;

    fn neg(self) -> Self {
        let mut derivs = self.derivs;
        for deriv in &mut derivs {
            *deriv = -*deriv;
        }
        Self {
            value: -self.value,
            derivs,
        }
    }
}

impl<T, const N: usize> MultiDual<T, N>
where
    T: Float,
{
    /// Reciprocal: `1/(b + ∇b) = (1/b) + (-∇b/b²)`
    ///
    /// Implements the rule: (1/g)′ = -g′/g² (note: g ≠ 0)
    ///
    /// # Examples
    ///
    /// ```
    /// use autodiff::MultiDual;
    ///
    /// // f(x, y) = 1/x at (2, 3)
    /// let x = MultiDual::<f64, 2>::variable(2.0, 0);
    /// let recip_x = x.recip();
    ///
    /// assert_eq!(recip_x.value, 0.5);        // 1/2
    /// assert!((recip_x.derivs[0] + 0.25).abs() < 1e-10);  // -1/4
    /// assert_eq!(recip_x.derivs[1], 0.0);    // ∂(1/x)/∂y = 0
    /// ```
    pub fn recip(self) -> Self {
        let recip_val = self.value.recip();
        let recip_val_sq = recip_val * recip_val;

        let mut derivs = self.derivs;
        for deriv in &mut derivs {
            *deriv = -*deriv * recip_val_sq;
        }

        Self {
            value: recip_val,
            derivs,
        }
    }

    /// Exponential function: `exp(a + ∇a) = exp(a) + (exp(a) · ∇a)`
    ///
    /// Implements the chain rule: `∂/∂xᵢ(exp(f)) = exp(f) · ∂f/∂xᵢ`
    ///
    /// # Examples
    ///
    /// ```
    /// use autodiff::MultiDual;
    ///
    /// // f(x, y) = exp(x) at (0, 1)
    /// let x = MultiDual::<f64, 2>::variable(0.0, 0);
    /// let f = x.exp();
    ///
    /// assert_eq!(f.value, 1.0);       // exp(0) = 1
    /// assert_eq!(f.derivs[0], 1.0);   // ∂exp(x)/∂x at x=0 is exp(0) = 1
    /// assert_eq!(f.derivs[1], 0.0);   // ∂exp(x)/∂y = 0
    /// ```
    pub fn exp(self) -> Self {
        let exp_val = self.value.exp();
        let mut derivs = self.derivs;
        for deriv in &mut derivs {
            *deriv = *deriv * exp_val;
        }
        Self {
            value: exp_val,
            derivs,
        }
    }

    /// Natural logarithm: `ln(a + ∇a) = ln(a) + (∇a / a)`
    ///
    /// Implements the chain rule: `∂/∂xᵢ(ln(f)) = (∂f/∂xᵢ) / f`
    ///
    /// # Examples
    ///
    /// ```
    /// use autodiff::MultiDual;
    ///
    /// // f(x, y) = ln(x) at (1, 2)
    /// let x = MultiDual::<f64, 2>::variable(1.0, 0);
    /// let f = x.ln();
    ///
    /// assert!((f.value - 0.0_f64).abs() < 1e-12);    // ln(1) = 0
    /// assert!((f.derivs[0] - 1.0_f64).abs() < 1e-12); // ∂ln(x)/∂x at x=1 is 1/1 = 1
    /// assert_eq!(f.derivs[1], 0.0);               // ∂ln(x)/∂y = 0
    /// ```
    pub fn ln(self) -> Self {
        let ln_val = self.value.ln();
        let mut derivs = self.derivs;
        for deriv in &mut derivs {
            *deriv = *deriv / self.value;
        }
        Self {
            value: ln_val,
            derivs,
        }
    }

    /// Sine function: `sin(a + ∇a) = sin(a) + (cos(a) · ∇a)`
    ///
    /// Implements the chain rule: `∂/∂xᵢ(sin(f)) = cos(f) · ∂f/∂xᵢ`
    ///
    /// # Examples
    ///
    /// ```
    /// use autodiff::MultiDual;
    ///
    /// // f(x, y) = sin(x) at (0, 1)
    /// let x = MultiDual::<f64, 2>::variable(0.0, 0);
    /// let f = x.sin();
    ///
    /// assert!((f.value - 0.0_f64).abs() < 1e-12);    // sin(0) = 0
    /// assert!((f.derivs[0] - 1.0_f64).abs() < 1e-12); // ∂sin(x)/∂x at x=0 is cos(0) = 1
    /// assert_eq!(f.derivs[1], 0.0);               // ∂sin(x)/∂y = 0
    /// ```
    pub fn sin(self) -> Self {
        let sin_val = self.value.sin();
        let cos_val = self.value.cos();
        let mut derivs = self.derivs;
        for deriv in &mut derivs {
            *deriv = *deriv * cos_val;
        }
        Self {
            value: sin_val,
            derivs,
        }
    }

    /// Cosine function: `cos(a + ∇a) = cos(a) + (-sin(a) · ∇a)`
    ///
    /// Implements the chain rule: `∂/∂xᵢ(cos(f)) = -sin(f) · ∂f/∂xᵢ`
    ///
    /// # Examples
    ///
    /// ```
    /// use autodiff::MultiDual;
    ///
    /// // f(x, y) = cos(x) at (0, 1)
    /// let x = MultiDual::<f64, 2>::variable(0.0, 0);
    /// let f = x.cos();
    ///
    /// assert!((f.value - 1.0_f64).abs() < 1e-12);    // cos(0) = 1
    /// assert!((f.derivs[0] - 0.0_f64).abs() < 1e-12); // ∂cos(x)/∂x at x=0 is -sin(0) = 0
    /// assert_eq!(f.derivs[1], 0.0);               // ∂cos(x)/∂y = 0
    /// ```
    pub fn cos(self) -> Self {
        let cos_val = self.value.cos();
        let sin_val = self.value.sin();
        let mut derivs = self.derivs;
        for deriv in &mut derivs {
            *deriv = -*deriv * sin_val;
        }
        Self {
            value: cos_val,
            derivs,
        }
    }

    /// Square root: `sqrt(a + ∇a) = sqrt(a) + (∇a / (2·sqrt(a)))`
    ///
    /// Implements the chain rule: `∂/∂xᵢ(√f) = (∂f/∂xᵢ) / (2√f)`
    ///
    /// # Examples
    ///
    /// ```
    /// use autodiff::MultiDual;
    ///
    /// // f(x, y) = √x at (4, 1)
    /// let x = MultiDual::<f64, 2>::variable(4.0, 0);
    /// let f = x.sqrt();
    ///
    /// assert_eq!(f.value, 2.0);       // √4 = 2
    /// assert_eq!(f.derivs[0], 0.25);  // ∂√x/∂x at x=4 is 1/(2·2) = 0.25
    /// assert_eq!(f.derivs[1], 0.0);   // ∂√x/∂y = 0
    /// ```
    pub fn sqrt(self) -> Self {
        let sqrt_val = self.value.sqrt();
        let two_sqrt = sqrt_val + sqrt_val; // 2 * sqrt(value)
        let mut derivs = self.derivs;
        for deriv in &mut derivs {
            *deriv = *deriv / two_sqrt;
        }
        Self {
            value: sqrt_val,
            derivs,
        }
    }
}

/// Compute the gradient of a scalar multivariable function in a
/// single forward pass.
///
/// Given a function `f: ℝⁿ → ℝ` and a point in ℝⁿ, computes both the
/// function value and its gradient ∇f = [∂f/∂x₁, ..., ∂f/∂xₙ] at that
/// point.
///
/// This is the primary high-level API for computing gradients with
/// MultiDual. It automatically seeds the input variables and
/// evaluates the function once.
///
/// # Type Parameters
///
/// - `T`: The numeric type (typically `f64` or `f32`)
/// - `F`: A function that takes N `MultiDual` inputs and returns a
///   `MultiDual` output
/// - `N`: The number of input variables (compile-time constant)
///
/// # Arguments
///
/// - `f`: The function to differentiate
/// - `point`: The point at which to evaluate the gradient
///
/// # Returns
///
/// A tuple `(value, gradient)` where:
/// - `value`: The function value f(point)
/// - `gradient`: The gradient ∇f evaluated at point
///
/// # Examples
///
/// ## Quadratic Function
///
/// ```
/// use autodiff::{MultiDual, gradient};
///
/// // f(x, y) = x² + 2xy + y² at (3, 4)
/// let f = |vars: [MultiDual<f64, 2>; 2]| {
///     let [x, y] = vars;
///     let two = MultiDual::constant(2.0);
///     x * x + two * x * y + y * y
/// };
///
/// let point = [3.0, 4.0];
/// let (value, grad) = gradient(f, point);
///
/// assert_eq!(value, 49.0);    // f(3, 4) = 9 + 24 + 16
/// assert_eq!(grad[0], 14.0);  // ∂f/∂x = 2x + 2y = 14
/// assert_eq!(grad[1], 14.0);  // ∂f/∂y = 2x + 2y = 14
/// ```
///
/// ## With Transcendental Functions
///
/// ```
/// use autodiff::{MultiDual, gradient};
///
/// // f(x, y, z) = x² + y·exp(z) at (1, 2, 0)
/// let f = |vars: [MultiDual<f64, 3>; 3]| {
///     let [x, y, z] = vars;
///     x * x + y * z.exp()
/// };
///
/// let point = [1.0, 2.0, 0.0];
/// let (value, grad) = gradient(f, point);
///
/// assert_eq!(value, 3.0);     // 1 + 2·1 = 3
/// assert_eq!(grad[0], 2.0);   // ∂f/∂x = 2x = 2
/// assert_eq!(grad[1], 1.0);   // ∂f/∂y = exp(z) = 1
/// assert_eq!(grad[2], 2.0);   // ∂f/∂z = y·exp(z) = 2
/// ```
///
/// ## Rosenbrock Function (optimization benchmark)
///
/// ```
/// use autodiff::{MultiDual, gradient};
///
/// // Rosenbrock: f(x, y) = (1-x)² + 100(y-x²)²
/// let rosenbrock = |vars: [MultiDual<f64, 2>; 2]| {
///     let [x, y] = vars;
///     let one = MultiDual::constant(1.0);
///     let hundred = MultiDual::constant(100.0);
///
///     let term1 = one - x;
///     let term2 = y - x * x;
///     term1 * term1 + hundred * term2 * term2
/// };
///
/// let point = [1.0, 1.0];  // Global minimum
/// let (value, grad) = gradient(rosenbrock, point);
///
/// assert_eq!(value, 0.0);           // Minimum value is 0
/// assert_eq!(grad[0], 0.0);         // Gradient is zero at minimum
/// assert_eq!(grad[1], 0.0);
/// ```
pub fn gradient<T, F, const N: usize>(f: F, point: [T; N]) -> (T, [T; N])
where
    T: Float,
    F: Fn([MultiDual<T, N>; N]) -> MultiDual<T, N>,
{
    // Seed input variables: each gets its value from point with the
    // appropriate unit vector for derivatives
    let vars = std::array::from_fn(|i| MultiDual::variable(point[i], i));

    // Single forward pass through the computation
    let result = f(vars);

    // Return both the function value and the gradient
    (result.value, result.derivs)
}

/// Division: `(a + ∇a) / (b + ∇b) = (a + ∇a) * (1/(b + ∇b))`
///
/// Implements division via reciprocal (composition of operations).
///
/// # Examples
///
/// ```
/// use autodiff::MultiDual;
///
/// // f(x, y) = x / y at (6, 2)
/// let x = MultiDual::<f64, 2>::variable(6.0, 0);
/// let y = MultiDual::<f64, 2>::variable(2.0, 1);
/// let quotient = x / y;
///
/// assert_eq!(quotient.value, 3.0);         // 6/2
/// assert_eq!(quotient.derivs[0], 0.5);     // ∂(x/y)/∂x = 1/y = 0.5
/// assert_eq!(quotient.derivs[1], -1.5);    // ∂(x/y)/∂y = -x/y² = -1.5
/// ```
#[allow(clippy::suspicious_arithmetic_impl)]
impl<T, const N: usize> Div for MultiDual<T, N>
where
    T: Float,
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        self * rhs.recip()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_has_zero_derivatives() {
        let c = MultiDual::<f64, 3>::constant(42.0);
        assert_eq!(c.value, 42.0);
        assert_eq!(c.derivs, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn variable_sets_correct_derivative() {
        let x = MultiDual::<f64, 3>::variable(3.0, 0);
        assert_eq!(x.value, 3.0);
        assert_eq!(x.derivs, [1.0, 0.0, 0.0]);

        let y = MultiDual::<f64, 3>::variable(4.0, 1);
        assert_eq!(y.value, 4.0);
        assert_eq!(y.derivs, [0.0, 1.0, 0.0]);

        let z = MultiDual::<f64, 3>::variable(5.0, 2);
        assert_eq!(z.value, 5.0);
        assert_eq!(z.derivs, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn addition_works() {
        let x = MultiDual::new(3.0, [1.0, 0.0]);
        let y = MultiDual::new(4.0, [0.0, 1.0]);
        let sum = x + y;

        assert_eq!(sum.value, 7.0);
        assert_eq!(sum.derivs, [1.0, 1.0]);
    }

    #[test]
    fn subtraction_works() {
        let x = MultiDual::new(7.0, [2.0, 3.0]);
        let y = MultiDual::new(4.0, [1.0, 2.0]);
        let diff = x - y;

        assert_eq!(diff.value, 3.0);
        assert_eq!(diff.derivs, [1.0, 1.0]);
    }

    #[test]
    fn negation_works() {
        let x = MultiDual::new(3.0, [1.0, 2.0, 3.0]);
        let neg_x = -x;

        assert_eq!(neg_x.value, -3.0);
        assert_eq!(neg_x.derivs, [-1.0, -2.0, -3.0]);
    }

    #[test]
    fn multiplication_implements_product_rule() {
        // f(x, y) = x * y at (3, 4)
        let x = MultiDual::<f64, 2>::variable(3.0, 0);
        let y = MultiDual::<f64, 2>::variable(4.0, 1);
        let product = x * y;

        assert_eq!(product.value, 12.0); // 3 * 4
        assert_eq!(product.derivs[0], 4.0); // ∂(xy)/∂x = y = 4
        assert_eq!(product.derivs[1], 3.0); // ∂(xy)/∂y = x = 3
    }

    #[test]
    fn recip_implements_inverse_rule() {
        // f(x, y) = 1/x at (2, 3)
        let x = MultiDual::<f64, 2>::variable(2.0, 0);
        let recip_x = x.recip();

        assert_eq!(recip_x.value, 0.5); // 1/2
        assert!((recip_x.derivs[0] + 0.25).abs() < 1e-10); // -1/4
        assert_eq!(recip_x.derivs[1], 0.0); // ∂(1/x)/∂y = 0
    }

    #[test]
    fn division_quotient_rule() {
        // f(x, y) = x / y at (6, 2)
        let x = MultiDual::<f64, 2>::variable(6.0, 0);
        let y = MultiDual::<f64, 2>::variable(2.0, 1);
        let quotient = x / y;

        assert_eq!(quotient.value, 3.0); // 6/2
        assert_eq!(quotient.derivs[0], 0.5); // ∂(x/y)/∂x = 1/y = 0.5
        assert_eq!(quotient.derivs[1], -1.5); // ∂(x/y)/∂y = -x/y² = -1.5
    }

    #[test]
    fn polynomial_gradient() {
        // f(x, y) = x² + 2xy + y² at (3, 4)
        // ∂f/∂x = 2x + 2y = 14, ∂f/∂y = 2x + 2y = 14
        let x = MultiDual::<f64, 2>::variable(3.0, 0);
        let y = MultiDual::<f64, 2>::variable(4.0, 1);
        let two = MultiDual::<f64, 2>::constant(2.0);

        let f = x * x + two * x * y + y * y;

        assert_eq!(f.value, 49.0); // 9 + 24 + 16
        assert_eq!(f.derivs[0], 14.0); // 2x + 2y
        assert_eq!(f.derivs[1], 14.0); // 2x + 2y
    }

    #[test]
    fn three_variable_sum_of_squares() {
        // f(x, y, z) = x² + y² + z² at (1, 2, 3)
        let x = MultiDual::<f64, 3>::variable(1.0, 0);
        let y = MultiDual::<f64, 3>::variable(2.0, 1);
        let z = MultiDual::<f64, 3>::variable(3.0, 2);

        let f = x * x + y * y + z * z;

        assert_eq!(f.value, 14.0); // 1 + 4 + 9
        assert_eq!(f.derivs[0], 2.0); // ∂f/∂x = 2x = 2
        assert_eq!(f.derivs[1], 4.0); // ∂f/∂y = 2y = 4
        assert_eq!(f.derivs[2], 6.0); // ∂f/∂z = 2z = 6
    }

    #[test]
    fn exp_gradient() {
        // f(x, y) = exp(x) at (0, 1)
        let x = MultiDual::<f64, 2>::variable(0.0, 0);
        let f = x.exp();

        assert_eq!(f.value, 1.0); // exp(0) = 1
        assert_eq!(f.derivs[0], 1.0); // ∂exp(x)/∂x at x=0 is exp(0) = 1
        assert_eq!(f.derivs[1], 0.0); // ∂exp(x)/∂y = 0

        // f(x, y) = exp(x + y) at (1, 2)
        let x = MultiDual::<f64, 2>::variable(1.0, 0);
        let y = MultiDual::<f64, 2>::variable(2.0, 1);
        let f = (x + y).exp();

        let exp_3 = 3.0_f64.exp();
        assert!((f.value - exp_3).abs() < 1e-10); // exp(3)
        assert!((f.derivs[0] - exp_3).abs() < 1e-10); // ∂exp(x+y)/∂x = exp(x+y)
        assert!((f.derivs[1] - exp_3).abs() < 1e-10); // ∂exp(x+y)/∂y = exp(x+y)
    }

    #[test]
    fn ln_gradient() {
        // f(x, y) = ln(x) at (1, 2)
        let x = MultiDual::<f64, 2>::variable(1.0, 0);
        let f = x.ln();

        assert!((f.value - 0.0).abs() < 1e-12); // ln(1) = 0
        assert!((f.derivs[0] - 1.0).abs() < 1e-12); // ∂ln(x)/∂x at x=1 is 1/1 = 1
        assert_eq!(f.derivs[1], 0.0); // ∂ln(x)/∂y = 0

        // f(x, y) = ln(x * y) at (2, 3)
        let x = MultiDual::<f64, 2>::variable(2.0, 0);
        let y = MultiDual::<f64, 2>::variable(3.0, 1);
        let f = (x * y).ln();

        assert!((f.value - 6.0_f64.ln()).abs() < 1e-10); // ln(6)
        assert!((f.derivs[0] - 0.5).abs() < 1e-10); // ∂ln(xy)/∂x = 1/x = 0.5
        assert!((f.derivs[1] - 1.0 / 3.0).abs() < 1e-10); // ∂ln(xy)/∂y = 1/y = 1/3
    }

    #[test]
    fn sin_gradient() {
        // f(x, y) = sin(x) at (0, 1)
        let x = MultiDual::<f64, 2>::variable(0.0, 0);
        let f = x.sin();

        assert!((f.value - 0.0).abs() < 1e-12); // sin(0) = 0
        assert!((f.derivs[0] - 1.0).abs() < 1e-12); // ∂sin(x)/∂x at x=0 is cos(0) = 1
        assert_eq!(f.derivs[1], 0.0); // ∂sin(x)/∂y = 0
    }

    #[test]
    fn cos_gradient() {
        // f(x, y) = cos(x) at (0, 1)
        let x = MultiDual::<f64, 2>::variable(0.0, 0);
        let f = x.cos();

        assert!((f.value - 1.0).abs() < 1e-12); // cos(0) = 1
        assert!((f.derivs[0] - 0.0).abs() < 1e-12); // ∂cos(x)/∂x at x=0 is -sin(0) = 0
        assert_eq!(f.derivs[1], 0.0); // ∂cos(x)/∂y = 0
    }

    #[test]
    fn sqrt_gradient() {
        // f(x, y) = √x at (4, 1)
        let x = MultiDual::<f64, 2>::variable(4.0, 0);
        let f = x.sqrt();

        assert_eq!(f.value, 2.0); // √4 = 2
        assert_eq!(f.derivs[0], 0.25); // ∂√x/∂x at x=4 is 1/(2·2) = 0.25
        assert_eq!(f.derivs[1], 0.0); // ∂√x/∂y = 0
    }

    #[test]
    fn mixed_transcendental_gradient() {
        // f(x, y) = sin(x) * exp(y) at (0, 0)
        // ∂f/∂x = cos(x) * exp(y) at (0, 0) = 1 * 1 = 1
        // ∂f/∂y = sin(x) * exp(y) at (0, 0) = 0 * 1 = 0
        let x = MultiDual::<f64, 2>::variable(0.0, 0);
        let y = MultiDual::<f64, 2>::variable(0.0, 1);
        let f = x.sin() * y.exp();

        assert!((f.value - 0.0).abs() < 1e-12); // sin(0) * exp(0) = 0 * 1 = 0
        assert!((f.derivs[0] - 1.0).abs() < 1e-12); // cos(0) * exp(0) = 1
        assert!((f.derivs[1] - 0.0).abs() < 1e-12); // sin(0) * exp(0) = 0
    }

    #[test]
    fn chain_rule_with_transcendentals() {
        // f(x, y) = exp(x²) at (1, 0)
        // ∂f/∂x = 2x * exp(x²) at x=1 is 2 * e
        let x = MultiDual::<f64, 2>::variable(1.0, 0);
        let f = (x * x).exp();

        let e = 1.0_f64.exp();
        assert!((f.value - e).abs() < 1e-10); // exp(1)
        assert!((f.derivs[0] - 2.0 * e).abs() < 1e-10); // 2 * exp(1)
        assert_eq!(f.derivs[1], 0.0);

        // f(x, y, z) = √(x² + y² + z²) at (3, 4, 0)
        // This is the Euclidean norm
        // ∂f/∂x = x / √(x² + y² + z²) = 3/5
        // ∂f/∂y = y / √(x² + y² + z²) = 4/5
        // ∂f/∂z = z / √(x² + y² + z²) = 0/5 = 0
        let x = MultiDual::<f64, 3>::variable(3.0, 0);
        let y = MultiDual::<f64, 3>::variable(4.0, 1);
        let z = MultiDual::<f64, 3>::variable(0.0, 2);
        let f = (x * x + y * y + z * z).sqrt();

        assert_eq!(f.value, 5.0); // √(9 + 16 + 0) = 5
        assert_eq!(f.derivs[0], 0.6); // 3/5
        assert_eq!(f.derivs[1], 0.8); // 4/5
        assert_eq!(f.derivs[2], 0.0); // 0/5
    }

    #[test]
    fn gradient_quadratic_2d() {
        use crate::gradient;

        // f(x, y) = x² + 2xy + y² at (3, 4)
        let f = |vars: [MultiDual<f64, 2>; 2]| {
            let [x, y] = vars;
            let two = MultiDual::constant(2.0);
            x * x + two * x * y + y * y
        };

        let point = [3.0, 4.0];
        let (value, grad) = gradient(f, point);

        assert_eq!(value, 49.0); // f(3, 4) = 9 + 24 + 16
        assert_eq!(grad[0], 14.0); // ∂f/∂x = 2x + 2y = 14
        assert_eq!(grad[1], 14.0); // ∂f/∂y = 2x + 2y = 14
    }

    #[test]
    fn gradient_with_transcendentals() {
        use crate::gradient;

        // f(x, y, z) = x² + y·exp(z) at (1, 2, 0)
        let f = |vars: [MultiDual<f64, 3>; 3]| {
            let [x, y, z] = vars;
            x * x + y * z.exp()
        };

        let point = [1.0, 2.0, 0.0];
        let (value, grad) = gradient(f, point);

        assert_eq!(value, 3.0); // 1 + 2·1 = 3
        assert_eq!(grad[0], 2.0); // ∂f/∂x = 2x = 2
        assert_eq!(grad[1], 1.0); // ∂f/∂y = exp(z) = 1
        assert_eq!(grad[2], 2.0); // ∂f/∂z = y·exp(z) = 2
    }

    #[test]
    fn gradient_rosenbrock() {
        use crate::gradient;

        // Rosenbrock: f(x, y) = (1-x)² + 100(y-x²)²
        let rosenbrock = |vars: [MultiDual<f64, 2>; 2]| {
            let [x, y] = vars;
            let one = MultiDual::constant(1.0);
            let hundred = MultiDual::constant(100.0);

            let term1 = one - x;
            let term2 = y - x * x;
            term1 * term1 + hundred * term2 * term2
        };

        // Test at global minimum (1, 1)
        let point = [1.0, 1.0];
        let (value, grad) = gradient(rosenbrock, point);

        assert_eq!(value, 0.0); // Minimum value is 0
        assert_eq!(grad[0], 0.0); // Gradient is zero at minimum
        assert_eq!(grad[1], 0.0);

        // Test at another point (0, 0)
        let point = [0.0, 0.0];
        let (value, grad) = gradient(rosenbrock, point);

        assert_eq!(value, 1.0); // f(0, 0) = 1 + 0 = 1
        assert_eq!(grad[0], -2.0); // ∂f/∂x at (0,0) = -2(1-x) - 400x(y-x²) = -2
        assert_eq!(grad[1], 0.0); // ∂f/∂y at (0,0) = 200(y-x²) = 0
    }

    #[test]
    fn gradient_euclidean_norm() {
        use crate::gradient;

        // f(x, y, z) = √(x² + y² + z²) at (3, 4, 0)
        let euclidean_norm = |vars: [MultiDual<f64, 3>; 3]| {
            let [x, y, z] = vars;
            (x * x + y * y + z * z).sqrt()
        };

        let point = [3.0, 4.0, 0.0];
        let (value, grad) = gradient(euclidean_norm, point);

        assert_eq!(value, 5.0); // √(9 + 16 + 0) = 5
        assert_eq!(grad[0], 0.6); // x/‖x‖ = 3/5
        assert_eq!(grad[1], 0.8); // y/‖x‖ = 4/5
        assert_eq!(grad[2], 0.0); // z/‖x‖ = 0/5
    }

    #[test]
    fn gradient_single_variable() {
        use crate::gradient;

        // f(x) = x³ at x=2
        // f'(x) = 3x² = 12
        let f = |vars: [MultiDual<f64, 1>; 1]| {
            let [x] = vars;
            x * x * x
        };

        let point = [2.0];
        let (value, grad) = gradient(f, point);

        assert_eq!(value, 8.0); // 2³ = 8
        assert_eq!(grad[0], 12.0); // 3 * 2² = 12
    }

    #[test]
    fn gradient_high_dimensional() {
        use crate::gradient;

        // f(x₁, x₂, x₃, x₄, x₅) = Σᵢ xᵢ² at (1, 2, 3, 4, 5)
        // ∂f/∂xᵢ = 2xᵢ
        let f = |vars: [MultiDual<f64, 5>; 5]| {
            let [x1, x2, x3, x4, x5] = vars;
            x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4 + x5 * x5
        };

        let point = [1.0, 2.0, 3.0, 4.0, 5.0];
        let (value, grad) = gradient(f, point);

        assert_eq!(value, 55.0); // 1 + 4 + 9 + 16 + 25
        assert_eq!(grad[0], 2.0); // 2 * 1
        assert_eq!(grad[1], 4.0); // 2 * 2
        assert_eq!(grad[2], 6.0); // 2 * 3
        assert_eq!(grad[3], 8.0); // 2 * 4
        assert_eq!(grad[4], 10.0); // 2 * 5
    }

    #[test]
    fn gradient_mixed_operations() {
        use crate::gradient;

        // f(x, y) = sin(x) * exp(y) + ln(x + y) at (1, 0)
        let f = |vars: [MultiDual<f64, 2>; 2]| {
            let [x, y] = vars;
            x.sin() * y.exp() + (x + y).ln()
        };

        let point = [1.0, 0.0];
        let (value, grad) = gradient(f, point);

        let sin_1 = 1.0_f64.sin();
        let cos_1 = 1.0_f64.cos();

        assert!((value - (sin_1 + 0.0)).abs() < 1e-10); // sin(1)*1 + ln(1) = sin(1)
                                                        // ∂f/∂x = cos(x)*exp(y) + 1/(x+y) at (1,0) = cos(1) + 1
        assert!((grad[0] - (cos_1 + 1.0)).abs() < 1e-10);
        // ∂f/∂y = sin(x)*exp(y) + 1/(x+y) at (1,0) = sin(1) + 1
        assert!((grad[1] - (sin_1 + 1.0)).abs() < 1e-10);
    }
}
