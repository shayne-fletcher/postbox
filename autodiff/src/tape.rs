//! Tape-based reverse-mode automatic differentiation.
//!
//! Reverse-mode AD computes gradients by recording operations during
//! a forward pass, then propagating gradients backward through the
//! recorded tape. This is efficient for functions f: ℝⁿ → ℝ where n
//! is large (e.g., neural networks with millions of parameters).
//!
//! # How It Works
//!
//! 1. Create a [`Tape`] to record operations
//! 2. Create variables with [`Tape::var`]
//! 3. Compute using arithmetic operations (recorded on the tape)
//! 4. Call [`Var::backward`] to propagate gradients
//! 5. Query gradients with [`Gradients::get`]
//!
//! # Example
//!
//! ```
//! use autodiff::Tape;
//!
//! let tape = Tape::new();
//! let x = tape.var(3.0);
//! let y = x.clone() * x.clone();  // y = x²
//!
//! let grads = y.backward();
//! assert_eq!(y.value(), 9.0);
//! assert_eq!(grads.get(&x), 6.0);  // dy/dx = 2x = 6
//! ```
//!
//! # Functional API
//!
//! For simple cases, use [`reverse_diff`] or [`reverse_gradient`]:
//!
//! ```
//! use autodiff::reverse_diff;
//!
//! let (val, deriv) = reverse_diff(|x| x.clone() * x, 3.0);
//! assert_eq!(val, 9.0);
//! assert_eq!(deriv, 6.0);
//! ```
//!
//! # When to Use Reverse-Mode
//!
//! - **Reverse-mode** (this module): Efficient when outputs << inputs
//!   (e.g., loss function with many parameters)
//! - **Forward-mode** ([`crate::dual`]): Efficient when inputs <<
//!   outputs (e.g., sensitivity of many outputs to one parameter)

use num_traits::Float;
use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

/// The computation tape that records operations for reverse-mode AD.
///
/// Create a tape with [`Tape::new`], then create variables on it with
/// [`Tape::var`].
///
/// # Examples
///
/// ```
/// use autodiff::Tape;
///
/// let tape = Tape::new();
/// let x = tape.var(3.0);
/// let y = x.clone() * x.clone();  // y = x²
///
/// let grads = y.backward();
/// assert_eq!(y.value(), 9.0);
/// assert_eq!(grads.get(&x), 6.0);  // dy/dx = 2x = 6
/// ```
#[derive(Clone)]
pub struct Tape<T> {
    inner: Rc<RefCell<TapeInner<T>>>,
}

impl<T> Tape<T> {
    /// Creates a new empty tape.
    pub fn new() -> Self {
        Self {
            inner: Rc::new(RefCell::new(TapeInner::new())),
        }
    }
}

impl<T: Float> Tape<T> {
    /// Creates a differentiable variable on this tape.
    pub fn var(&self, value: T) -> Var<T> {
        let idx = self.inner.borrow_mut().push_value(value);
        Var {
            tape: self.clone(),
            idx,
        }
    }

    fn constant(&self, value: T) -> Var<T> {
        let idx = self.inner.borrow_mut().push_value(value);
        Var {
            tape: self.clone(),
            idx,
        }
    }
}

/// A differentiable variable for reverse-mode automatic
/// differentiation.
///
/// # Examples
///
/// Using [`reverse_diff`] for reusable functions:
///
/// ```
/// use autodiff::{reverse_diff, Var};
///
/// // Define f(x) = (x+1)(x-1) = x² - 1
/// let f = |x: Var<f64>| (x.clone() + 1.0) * (x - 1.0);
///
/// let (val, deriv) = reverse_diff(f, 3.0);
/// assert_eq!(val, 8.0);    // f(3) = 8
/// assert_eq!(deriv, 6.0);  // f'(3) = 2x = 6
/// ```
///
/// ```
/// use autodiff::{reverse_diff, Var};
///
/// // Define f(x) = x² + x
/// let f = |x: Var<f64>| x.clone() * x.clone() + x;
///
/// let (val, deriv) = reverse_diff(f, 3.0);
/// assert_eq!(val, 12.0);   // f(3) = 12
/// assert_eq!(deriv, 7.0);  // f'(3) = 2x + 1 = 7
/// ```
#[derive(Clone)]
pub struct Var<T> {
    tape: Tape<T>,
    idx: usize,
}

impl<T: Copy> Var<T> {
    /// Returns the value of this variable.
    pub fn value(&self) -> T {
        self.tape.inner.borrow().vals[self.idx]
    }
}

impl<T: Float> Var<T> {
    /// Computes gradients by backpropagation from this variable.
    ///
    /// Returns a [`Gradients`] object that can be queried for the
    /// gradient with respect to any variable.
    ///
    /// # Examples
    ///
    /// ```
    /// use autodiff::Tape;
    ///
    /// let tape = Tape::new();
    /// let x = tape.var(3.0);
    /// let y = x.clone() * x.clone();  // y = x²
    ///
    /// let grads = y.backward();
    /// assert_eq!(grads.get(&x), 6.0);  // dy/dx = 2x = 6
    /// ```
    pub fn backward(&self) -> Gradients<T> {
        self.tape.inner.borrow_mut().backward_from(self.idx);
        Gradients {
            tape: self.tape.clone(),
        }
    }

    /// Computes the reciprocal `1/self`.
    pub fn recip(self) -> Self {
        unary(self, OpKind::Recip, |a| T::one() / a)
    }

    /// Computes `e^self`.
    pub fn exp(self) -> Self {
        unary(self, OpKind::Exp, |a| a.exp())
    }

    /// Computes `sin(self)`.
    pub fn sin(self) -> Self {
        unary(self, OpKind::Sin, |a| a.sin())
    }

    /// Computes `cos(self)`.
    pub fn cos(self) -> Self {
        unary(self, OpKind::Cos, |a| a.cos())
    }

    /// Computes `ln(self)`.
    pub fn ln(self) -> Self {
        unary(self, OpKind::Ln, |a| a.ln())
    }

    /// Computes `sqrt(self)`.
    pub fn sqrt(self) -> Self {
        unary(self, OpKind::Sqrt, |a| a.sqrt())
    }
}

/// The gradients computed by [`Var::backward`].
///
/// Query individual gradients using [`get`](Gradients::get).
pub struct Gradients<T> {
    tape: Tape<T>,
}

impl<T: Copy> Gradients<T> {
    /// Returns the gradient with respect to the given variable.
    pub fn get(&self, var: &Var<T>) -> T {
        self.tape.inner.borrow().grads[var.idx]
    }
}

struct TapeInner<T> {
    vals: Vec<T>,
    grads: Vec<T>,
    ops: Vec<Op>,
}

impl<T> TapeInner<T> {
    fn new() -> Self {
        Self {
            vals: Vec::new(),
            grads: Vec::new(),
            ops: Vec::new(),
        }
    }

    fn push_op(&mut self, op: Op) {
        self.ops.push(op)
    }
}

impl<T: Float> TapeInner<T> {
    fn push_value(&mut self, v: T) -> usize {
        let idx = self.vals.len();
        self.vals.push(v);
        self.grads.push(T::zero());
        idx
    }

    fn backward_from(&mut self, out: usize) {
        for g in &mut self.grads {
            *g = T::zero();
        }
        self.grads[out] = T::one();

        for op in self.ops.iter().rev() {
            let go = self.grads[op.out];

            match op.kind {
                OpKind::Add => {
                    self.grads[op.a] = self.grads[op.a] + go;
                    self.grads[op.b] = self.grads[op.b] + go;
                }
                OpKind::Sub => {
                    self.grads[op.a] = self.grads[op.a] + go;
                    self.grads[op.b] = self.grads[op.b] - go;
                }
                OpKind::Mul => {
                    let a = self.vals[op.a];
                    let b = self.vals[op.b];
                    self.grads[op.a] = self.grads[op.a] + go * b;
                    self.grads[op.b] = self.grads[op.b] + go * a;
                }
                OpKind::Div => {
                    let a = self.vals[op.a];
                    let b = self.vals[op.b];
                    self.grads[op.a] = self.grads[op.a] + go / b;
                    self.grads[op.b] = self.grads[op.b] + go * (-(a) / (b * b));
                }
                OpKind::Neg => {
                    self.grads[op.a] = self.grads[op.a] - go;
                }
                OpKind::Recip => {
                    let a = self.vals[op.a];
                    self.grads[op.a] = self.grads[op.a] + go * (-(T::one()) / (a * a));
                }
                OpKind::Exp => {
                    let out = self.vals[op.out];
                    self.grads[op.a] = self.grads[op.a] + go * out;
                }
                OpKind::Sin => {
                    let a = self.vals[op.a];
                    self.grads[op.a] = self.grads[op.a] + go * a.cos();
                }
                OpKind::Cos => {
                    let a = self.vals[op.a];
                    self.grads[op.a] = self.grads[op.a] - go * a.sin();
                }
                OpKind::Ln => {
                    let a = self.vals[op.a];
                    self.grads[op.a] = self.grads[op.a] + go / a;
                }
                OpKind::Sqrt => {
                    let out = self.vals[op.out];
                    self.grads[op.a] = self.grads[op.a] + go / (out + out);
                }
            }
        }
    }
}

struct Op {
    kind: OpKind,
    out: usize,
    a: usize,
    b: usize,
}

enum OpKind {
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Recip,
    Exp,
    Sin,
    Cos,
    Ln,
    Sqrt,
}

fn unary<T: Float>(x: Var<T>, kind: OpKind, f: impl FnOnce(T) -> T) -> Var<T> {
    let tape = x.tape.clone();
    let outv = {
        let t = tape.inner.borrow();
        f(t.vals[x.idx])
    };

    let out = {
        let mut t = tape.inner.borrow_mut();
        let out = t.push_value(outv);
        t.push_op(Op {
            kind,
            out,
            a: x.idx,
            b: 0,
        });
        out
    };

    Var { tape, idx: out }
}

fn binary<T: Float>(lhs: Var<T>, rhs: Var<T>, kind: OpKind, f: impl FnOnce(T, T) -> T) -> Var<T> {
    assert!(
        Rc::ptr_eq(&lhs.tape.inner, &rhs.tape.inner),
        "Vars must share a tape"
    );
    let tape = lhs.tape.clone();

    let (a, b, outv) = {
        let t = tape.inner.borrow();
        let av = t.vals[lhs.idx];
        let bv = t.vals[rhs.idx];
        (lhs.idx, rhs.idx, f(av, bv))
    };

    let out = {
        let mut t = tape.inner.borrow_mut();
        let out = t.push_value(outv);
        t.push_op(Op { kind, out, a, b });
        out
    };

    Var { tape, idx: out }
}

impl<T: Float> Add for Var<T> {
    type Output = Var<T>;
    fn add(self, rhs: Self) -> Self::Output {
        binary(self, rhs, OpKind::Add, |a, b| a + b)
    }
}

impl<T: Float> Sub for Var<T> {
    type Output = Var<T>;
    fn sub(self, rhs: Self) -> Self::Output {
        binary(self, rhs, OpKind::Sub, |a, b| a - b)
    }
}

impl<T: Float> Mul for Var<T> {
    type Output = Var<T>;
    fn mul(self, rhs: Self) -> Self::Output {
        binary(self, rhs, OpKind::Mul, |a, b| a * b)
    }
}

impl<T: Float> Div for Var<T> {
    type Output = Var<T>;
    fn div(self, rhs: Self) -> Self::Output {
        binary(self, rhs, OpKind::Div, |a, b| a / b)
    }
}

impl<T: Float> Neg for Var<T> {
    type Output = Var<T>;
    fn neg(self) -> Self::Output {
        unary(self, OpKind::Neg, |a| -a)
    }
}

impl<T: Float> Add<T> for Var<T> {
    type Output = Var<T>;
    fn add(self, c: T) -> Self::Output {
        let cvar = self.tape.constant(c);
        self + cvar
    }
}

impl<T: Float> Sub<T> for Var<T> {
    type Output = Var<T>;
    fn sub(self, c: T) -> Self::Output {
        let cvar = self.tape.constant(c);
        self - cvar
    }
}

impl<T: Float> Mul<T> for Var<T> {
    type Output = Var<T>;
    fn mul(self, c: T) -> Self::Output {
        let cvar = self.tape.constant(c);
        self * cvar
    }
}

impl<T: Float> Div<T> for Var<T> {
    type Output = Var<T>;
    fn div(self, c: T) -> Self::Output {
        let cvar = self.tape.constant(c);
        self / cvar
    }
}

/// Computes the value and derivative of a function using reverse-mode
/// AD.
///
/// This is the reverse-mode equivalent of forward-mode
/// differentiation. The function `f` is evaluated at `x`, and both
/// the value and derivative are returned.
///
/// # Examples
///
/// ```
/// use autodiff::reverse_diff;
///
/// // f(x) = x² at x = 3
/// let (val, deriv) = reverse_diff(|x| x.clone() * x, 3.0);
/// assert_eq!(val, 9.0);    // f(3) = 9
/// assert_eq!(deriv, 6.0);  // f'(3) = 2x = 6
/// ```
///
/// Reuse the same function at different points:
///
/// ```
/// use autodiff::{reverse_diff, Var};
///
/// let f = |x: Var<f64>| x.clone() * x.clone() - x;
///
/// let (v1, d1) = reverse_diff(f, 2.0);
/// let (v2, d2) = reverse_diff(f, 5.0);
///
/// assert_eq!((v1, d1), (2.0, 3.0));   // f(2) = 2, f'(2) = 3
/// assert_eq!((v2, d2), (20.0, 9.0));  // f(5) = 20, f'(5) = 9
/// ```
pub fn reverse_diff<T, F>(f: F, x: T) -> (T, T)
where
    T: Float,
    F: FnOnce(Var<T>) -> Var<T>,
{
    let tape = Tape::new();
    let var = tape.var(x);
    let var_clone = var.clone();
    let result = f(var);
    let grads = result.backward();
    (result.value(), grads.get(&var_clone))
}

/// Computes the value and gradient of a multivariable function using
/// reverse-mode AD.
///
/// This is the reverse-mode equivalent of
/// [`gradient`](crate::gradient) for functions f: ℝⁿ → ℝ.
///
/// # Examples
///
/// ```
/// use autodiff::{reverse_gradient, Var};
///
/// // f(x, y) = x² + x*y at (3, 4)
/// let f = |[x, y]: [Var<f64>; 2]| x.clone() * x.clone() + x * y;
///
/// let (val, grad) = reverse_gradient(f, [3.0, 4.0]);
/// assert_eq!(val, 21.0);       // f(3, 4) = 9 + 12 = 21
/// assert_eq!(grad[0], 10.0);   // ∂f/∂x = 2x + y = 10
/// assert_eq!(grad[1], 3.0);    // ∂f/∂y = x = 3
/// ```
pub fn reverse_gradient<T, F, const N: usize>(f: F, point: [T; N]) -> (T, [T; N])
where
    T: Float,
    F: FnOnce([Var<T>; N]) -> Var<T>,
{
    let tape = Tape::new();
    let vars: [Var<T>; N] = std::array::from_fn(|i| tape.var(point[i]));
    let vars_clone = vars.clone();
    let result = f(vars);
    let grads = result.backward();
    (
        result.value(),
        std::array::from_fn(|i| grads.get(&vars_clone[i])),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // === Basic operations ===

    #[test]
    fn addition_works() {
        // f(x) = x + 5 at x=3
        let (val, deriv) = reverse_diff(|x| x + 5.0, 3.0);
        assert_eq!(val, 8.0);
        assert_eq!(deriv, 1.0);
    }

    #[test]
    fn subtraction_works() {
        // f(x) = x - 2 at x=5
        let (val, deriv) = reverse_diff(|x| x - 2.0, 5.0);
        assert_eq!(val, 3.0);
        assert_eq!(deriv, 1.0);
    }

    #[test]
    fn multiplication_works() {
        // f(x) = x * x = x² at x=3
        let (val, deriv) = reverse_diff(|x| x.clone() * x, 3.0);
        assert_eq!(val, 9.0);
        assert_eq!(deriv, 6.0); // d/dx(x²) = 2x = 6
    }

    #[test]
    fn division_works() {
        // f(x) = x / 2 at x=6
        let (val, deriv) = reverse_diff(|x| x / 2.0, 6.0);
        assert_eq!(val, 3.0);
        assert_eq!(deriv, 0.5); // d/dx(x/2) = 0.5
    }

    #[test]
    fn negation_works() {
        // f(x) = -x at x=3
        let (val, deriv) = reverse_diff(|x| -x, 3.0);
        assert_eq!(val, -3.0);
        assert_eq!(deriv, -1.0);
    }

    // === Var-Var operations ===

    #[test]
    fn var_var_addition() {
        // f(x) = x + x = 2x at x=3
        let (val, deriv) = reverse_diff(|x| x.clone() + x, 3.0);
        assert_eq!(val, 6.0);
        assert_eq!(deriv, 2.0);
    }

    #[test]
    fn var_var_subtraction() {
        // f(x) = x - x = 0 at x=3
        let (val, deriv) = reverse_diff(|x| x.clone() - x, 3.0);
        assert_eq!(val, 0.0);
        assert_eq!(deriv, 0.0);
    }

    #[test]
    fn var_var_division() {
        // f(x) = x / x = 1 at x=3
        // f'(x) = 0 (derivative of constant 1)
        let (val, deriv) = reverse_diff(|x| x.clone() / x, 3.0);
        assert_eq!(val, 1.0);
        assert!((deriv - 0.0).abs() < 1e-10);
    }

    // === Scalar operations (Var op T) ===

    #[test]
    fn var_add_scalar() {
        let (val, deriv) = reverse_diff(|x| x + 10.0, 5.0);
        assert_eq!(val, 15.0);
        assert_eq!(deriv, 1.0);
    }

    #[test]
    fn var_sub_scalar() {
        let (val, deriv) = reverse_diff(|x| x - 3.0, 10.0);
        assert_eq!(val, 7.0);
        assert_eq!(deriv, 1.0);
    }

    #[test]
    fn var_mul_scalar() {
        let (val, deriv) = reverse_diff(|x| x * 3.0, 4.0);
        assert_eq!(val, 12.0);
        assert_eq!(deriv, 3.0);
    }

    #[test]
    fn var_div_scalar() {
        let (val, deriv) = reverse_diff(|x| x / 4.0, 12.0);
        assert_eq!(val, 3.0);
        assert_eq!(deriv, 0.25);
    }

    // === Unary functions ===

    #[test]
    fn recip_works() {
        // f(x) = 1/x at x=2
        let (val, deriv) = reverse_diff(|x| x.recip(), 2.0);
        assert_eq!(val, 0.5);
        assert_eq!(deriv, -0.25); // d/dx(1/x) = -1/x² = -0.25
    }

    #[test]
    fn exp_works() {
        // f(x) = e^x at x=0
        let (val, deriv) = reverse_diff(|x| x.exp(), 0.0);
        assert_eq!(val, 1.0);
        assert_eq!(deriv, 1.0); // d/dx(e^x) = e^x = 1 at x=0
    }

    #[test]
    fn exp_at_one() {
        // f(x) = e^x at x=1
        let (val, deriv) = reverse_diff(|x| x.exp(), 1.0);
        let e = 1.0_f64.exp();
        assert!((val - e).abs() < 1e-10);
        assert!((deriv - e).abs() < 1e-10);
    }

    #[test]
    fn sin_works() {
        // f(x) = sin(x) at x=0
        let (val, deriv) = reverse_diff(|x| x.sin(), 0.0);
        assert_eq!(val, 0.0);
        assert_eq!(deriv, 1.0); // cos(0) = 1
    }

    #[test]
    fn cos_works() {
        // f(x) = cos(x) at x=0
        let (val, deriv) = reverse_diff(|x| x.cos(), 0.0);
        assert_eq!(val, 1.0);
        assert_eq!(deriv, 0.0); // -sin(0) = 0
    }

    #[test]
    fn ln_works() {
        // f(x) = ln(x) at x=1
        let (val, deriv) = reverse_diff(|x| x.ln(), 1.0);
        assert_eq!(val, 0.0);
        assert_eq!(deriv, 1.0); // 1/x = 1 at x=1
    }

    #[test]
    fn ln_at_e() {
        // f(x) = ln(x) at x=e
        let e = 1.0_f64.exp();
        let (val, deriv) = reverse_diff(|x| x.ln(), e);
        assert!((val - 1.0).abs() < 1e-10);
        assert!((deriv - 1.0 / e).abs() < 1e-10);
    }

    #[test]
    fn sqrt_works() {
        // f(x) = √x at x=4
        let (val, deriv) = reverse_diff(|x| x.sqrt(), 4.0);
        assert_eq!(val, 2.0);
        assert_eq!(deriv, 0.25); // 1/(2√x) = 1/4
    }

    // === Fan-out (variable used multiple times) ===

    #[test]
    fn fan_out_addition() {
        // f(x) = x + x + x = 3x at x=2
        let (val, deriv) = reverse_diff(|x| x.clone() + x.clone() + x, 2.0);
        assert_eq!(val, 6.0);
        assert_eq!(deriv, 3.0);
    }

    #[test]
    fn fan_out_mixed() {
        // f(x) = x² + x at x=3
        // f'(x) = 2x + 1 = 7
        let (val, deriv) = reverse_diff(|x| x.clone() * x.clone() + x, 3.0);
        assert_eq!(val, 12.0);
        assert_eq!(deriv, 7.0);
    }

    #[test]
    fn fan_out_product() {
        // f(x) = x * x * x = x³ at x=2
        // f'(x) = 3x² = 12
        let (val, deriv) = reverse_diff(|x| x.clone() * x.clone() * x, 2.0);
        assert_eq!(val, 8.0);
        assert_eq!(deriv, 12.0);
    }

    // === Complex expressions ===

    #[test]
    fn polynomial() {
        // f(x) = x³ - 2x + 1 at x=2
        // f'(x) = 3x² - 2 = 10
        let (val, deriv) = reverse_diff(
            |x| {
                let x2 = x.clone() * x.clone();
                let x3 = x2 * x.clone();
                x3 - x * 2.0 + 1.0
            },
            2.0,
        );
        assert_eq!(val, 5.0);
        assert_eq!(deriv, 10.0);
    }

    #[test]
    fn quotient_rule() {
        // f(x) = (x+1)/(x+2) at x=3
        // f(3) = 4/5 = 0.8
        // f'(x) = 1/(x+2)² = 1/25 = 0.04
        let (val, deriv) = reverse_diff(
            |x| {
                let num = x.clone() + 1.0;
                let den = x + 2.0;
                num / den
            },
            3.0,
        );
        assert_eq!(val, 0.8);
        assert!((deriv - 0.04).abs() < 1e-10);
    }

    #[test]
    fn chain_rule_product() {
        // f(x) = (x+1)(x-1) = x² - 1 at x=3
        // f'(x) = 2x = 6
        let (val, deriv) = reverse_diff(|x| (x.clone() + 1.0) * (x - 1.0), 3.0);
        assert_eq!(val, 8.0);
        assert_eq!(deriv, 6.0);
    }

    #[test]
    fn chain_rule_transcendental() {
        // f(x) = sin(2x) at x=0
        // f'(x) = 2*cos(2x) = 2 at x=0
        let (val, deriv) = reverse_diff(|x| (x * 2.0).sin(), 0.0);
        assert_eq!(val, 0.0);
        assert_eq!(deriv, 2.0);
    }

    #[test]
    fn exp_of_square() {
        // f(x) = e^(x²) at x=1
        // f'(x) = 2x * e^(x²) = 2e at x=1
        let (val, deriv) = reverse_diff(|x| (x.clone() * x).exp(), 1.0);
        let e = 1.0_f64.exp();
        assert!((val - e).abs() < 1e-10);
        assert!((deriv - 2.0 * e).abs() < 1e-10);
    }

    #[test]
    fn ln_of_square() {
        // f(x) = ln(x²) = 2*ln(x) at x=e
        // f'(x) = 2/x
        let e = 1.0_f64.exp();
        let (val, deriv) = reverse_diff(|x| (x.clone() * x).ln(), e);
        assert!((val - 2.0).abs() < 1e-10);
        assert!((deriv - 2.0 / e).abs() < 1e-10);
    }

    #[test]
    fn sqrt_of_sum() {
        // f(x) = √(x+5) at x=4
        // f'(x) = 1/(2√(x+5)) = 1/6
        let (val, deriv) = reverse_diff(|x| (x + 5.0).sqrt(), 4.0);
        assert_eq!(val, 3.0);
        assert!((deriv - 1.0 / 6.0).abs() < 1e-10);
    }

    // === reverse_gradient tests ===

    #[test]
    fn gradient_sum() {
        // f(x, y) = x + y at (3, 4)
        // ∂f/∂x = 1, ∂f/∂y = 1
        let (val, grad) = reverse_gradient(|[x, y]| x + y, [3.0, 4.0]);
        assert_eq!(val, 7.0);
        assert_eq!(grad, [1.0, 1.0]);
    }

    #[test]
    fn gradient_product() {
        // f(x, y) = x * y at (3, 4)
        // ∂f/∂x = y = 4, ∂f/∂y = x = 3
        let (val, grad) = reverse_gradient(|[x, y]| x * y, [3.0, 4.0]);
        assert_eq!(val, 12.0);
        assert_eq!(grad, [4.0, 3.0]);
    }

    #[test]
    fn gradient_mixed() {
        // f(x, y) = x² + x*y at (3, 4)
        // ∂f/∂x = 2x + y = 10, ∂f/∂y = x = 3
        let (val, grad) = reverse_gradient(|[x, y]| x.clone() * x.clone() + x * y, [3.0, 4.0]);
        assert_eq!(val, 21.0);
        assert_eq!(grad, [10.0, 3.0]);
    }

    #[test]
    fn gradient_three_vars() {
        // f(x, y, z) = x*y + y*z + z*x at (1, 2, 3)
        // f = 2 + 6 + 3 = 11
        // ∂f/∂x = y + z = 5
        // ∂f/∂y = x + z = 4
        // ∂f/∂z = y + x = 3
        let (val, grad) = reverse_gradient(
            |[x, y, z]| x.clone() * y.clone() + y.clone() * z.clone() + z * x,
            [1.0, 2.0, 3.0],
        );
        assert_eq!(val, 11.0);
        assert_eq!(grad, [5.0, 4.0, 3.0]);
    }

    #[test]
    fn gradient_with_transcendental() {
        // f(x, y) = sin(x) * cos(y) at (0, 0)
        // f = 0 * 1 = 0
        // ∂f/∂x = cos(x) * cos(y) = 1
        // ∂f/∂y = sin(x) * (-sin(y)) = 0
        let (val, grad) = reverse_gradient(|[x, y]| x.sin() * y.cos(), [0.0, 0.0]);
        assert_eq!(val, 0.0);
        assert_eq!(grad, [1.0, 0.0]);
    }

    // === Reusable function pattern ===

    #[test]
    fn reusable_function() {
        let f = |x: Var<f64>| x.clone() * x.clone() + x * 2.0;

        // Evaluate at multiple points
        let (v1, d1) = reverse_diff(f, 3.0);
        assert_eq!(v1, 15.0); // 9 + 6
        assert_eq!(d1, 8.0); // 2*3 + 2

        let (v2, d2) = reverse_diff(f, 5.0);
        assert_eq!(v2, 35.0); // 25 + 10
        assert_eq!(d2, 12.0); // 2*5 + 2

        let (v3, d3) = reverse_diff(f, 0.0);
        assert_eq!(v3, 0.0);
        assert_eq!(d3, 2.0);
    }

    // === Comparison with forward-mode ===

    #[test]
    fn matches_forward_mode_polynomial() {
        // Verify reverse-mode matches forward-mode for x³ + 2x² - x at x=2
        use crate::dual::Dual;

        let forward = {
            let x = Dual::variable(2.0);
            let y = x * x * x + Dual::constant(2.0) * x * x - x;
            (y.value, y.deriv)
        };

        let reverse = reverse_diff(
            |x| {
                let x2 = x.clone() * x.clone();
                let x3 = x2.clone() * x.clone();
                x3 + x2 * 2.0 - x
            },
            2.0,
        );

        assert_eq!(forward.0, reverse.0);
        assert!((forward.1 - reverse.1).abs() < 1e-10);
    }

    #[test]
    fn matches_forward_mode_transcendental() {
        // Verify reverse-mode matches forward-mode for e^(sin(x)) at x=1
        use crate::dual::Dual;

        let forward = {
            let x = Dual::variable(1.0);
            let y = x.sin().exp();
            (y.value, y.deriv)
        };

        let reverse = reverse_diff(|x| x.sin().exp(), 1.0);

        assert!((forward.0 - reverse.0).abs() < 1e-10);
        assert!((forward.1 - reverse.1).abs() < 1e-10);
    }
}
