use num_traits::Float;
use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

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
pub struct Var<T: Float> {
    tape: Rc<RefCell<Tape<T>>>,
    idx: usize,
}

impl<T: Float> Var<T> {
    pub fn tape() -> Rc<RefCell<Tape<T>>> {
        Rc::new(RefCell::new(Tape::new()))
    }

    pub fn variable_on(tape: Rc<RefCell<Tape<T>>>, value: T) -> Self {
        let idx = tape.borrow_mut().push_value(value);
        Self { tape, idx }
    }

    pub fn constant_on(tape: Rc<RefCell<Tape<T>>>, value: T) -> Self {
        let idx = tape.borrow_mut().push_value(value);
        Self { tape, idx }
    }

    pub fn value(&self) -> T {
        self.tape.borrow().vals[self.idx]
    }

    pub fn grad(&self) -> T {
        self.tape.borrow().grads[self.idx]
    }

    pub fn backward(&self) {
        self.tape.borrow_mut().backward_from(self.idx)
    }

    /// Computes the reciprocal `1/self`.
    pub fn recip(self) -> Self {
        unary(self, OpKind::Recip)
    }
}

/// The computation tape that records operations for reverse-mode AD.
pub struct Tape<T: Float> {
    vals: Vec<T>,
    grads: Vec<T>,
    ops: Vec<Op>,
}

impl<T: Float> Tape<T> {
    fn new() -> Self {
        Self {
            vals: Vec::new(),
            grads: Vec::new(),
            ops: Vec::new(),
        }
    }

    fn push_value(&mut self, v: T) -> usize {
        let idx = self.vals.len();
        self.vals.push(v);
        self.grads.push(T::zero());
        idx
    }

    fn push_op(&mut self, op: Op) {
        self.ops.push(op)
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
}

fn unary<T: Float>(x: Var<T>, kind: OpKind) -> Var<T> {
    let tape = x.tape.clone();
    let outv = {
        let t = tape.borrow();
        let a = t.vals[x.idx];
        match kind {
            OpKind::Neg => -a,
            OpKind::Recip => T::one() / a,
            _ => unreachable!(),
        }
    };

    let out = {
        let mut t = tape.borrow_mut();
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
    assert!(Rc::ptr_eq(&lhs.tape, &rhs.tape), "Vars must share a tape");
    let tape = lhs.tape.clone();

    let (a, b, outv) = {
        let t = tape.borrow();
        let av = t.vals[lhs.idx];
        let bv = t.vals[rhs.idx];
        (lhs.idx, rhs.idx, f(av, bv))
    };

    let out = {
        let mut t = tape.borrow_mut();
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
        unary(self, OpKind::Neg)
    }
}

impl<T: Float> Add<T> for Var<T> {
    type Output = Var<T>;
    fn add(self, c: T) -> Self::Output {
        let cvar = Var::constant_on(self.tape.clone(), c);
        self + cvar
    }
}

impl<T: Float> Sub<T> for Var<T> {
    type Output = Var<T>;
    fn sub(self, c: T) -> Self::Output {
        let cvar = Var::constant_on(self.tape.clone(), c);
        self - cvar
    }
}

impl<T: Float> Mul<T> for Var<T> {
    type Output = Var<T>;
    fn mul(self, c: T) -> Self::Output {
        let cvar = Var::constant_on(self.tape.clone(), c);
        self * cvar
    }
}

impl<T: Float> Div<T> for Var<T> {
    type Output = Var<T>;
    fn div(self, c: T) -> Self::Output {
        let cvar = Var::constant_on(self.tape.clone(), c);
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
    let tape = Var::<T>::tape();
    let var = Var::variable_on(tape, x);
    let var_clone = var.clone();
    let result = f(var);
    result.backward();
    (result.value(), var_clone.grad())
}
