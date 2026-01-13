# Reverse-mode automatic differentiation (from scratch)

This note explains reverse-mode automatic differentiation (AD) for scalar
functions, using only the chain rule and a concrete "tape" model.

---

## Notation

We write adjoints using bar notation:

$$
\bar{x}_i \;\equiv\; \frac{\partial y}{\partial x_i}
$$

That is, $\bar{x}_i$ measures how the final output $y$ depends on the
intermediate value $x_i$.

---

## Core idea

Reverse-mode AD consists of two phases:

1. **Forward pass**
   Execute the program normally, recording each primitive operation
   (the *tape*).

2. **Backward pass**
   Traverse the tape in reverse order, applying the chain rule locally
   and accumulating contributions to adjoints.

Key invariant:

> When backpropagating through an operation, the adjoint of its output
> is already complete.

---

## Primitive rules (local Jacobians)

Each primitive operation defines **one local partial derivative per input**.

### Addition
$$
x_{\text{out}} = a + b
$$

Local partials:
$$
\frac{\partial x_{\text{out}}}{\partial a} = 1,
\qquad
\frac{\partial x_{\text{out}}}{\partial b} = 1
$$

Backprop rule:
$$
\bar{a} \;{+}{=}\; \bar{x}_{\text{out}},
\qquad
\bar{b} \;{+}{=}\; \bar{x}_{\text{out}}
$$

---

### Multiplication
$$
x_{\text{out}} = a \cdot b
$$

Local partials:
$$
\frac{\partial x_{\text{out}}}{\partial a} = b,
\qquad
\frac{\partial x_{\text{out}}}{\partial b} = a
$$

Backprop rule:
$$
\bar{a} \;{+}{=}\; \bar{x}_{\text{out}} \cdot b,
\qquad
\bar{b} \;{+}{=}\; \bar{x}_{\text{out}} \cdot a
$$

Note: if $a = b$, these two contributions accumulate into the same adjoint.

---

## Worked example 1: $(x+1)(x-1)$

### Forward pass (program decomposition)

$$
\begin{aligned}
x_1 &= x + 1 \\
x_2 &= x - 1 \\
x_3 &= x_1 \cdot x_2 \\
y &= x_3
\end{aligned}
$$

---

### Backward pass

Seed the output:
$$
\bar{x}_3 = 1
$$

Backprop through multiplication:
$$
\bar{x}_1 \;{+}{=}\; \bar{x}_3 \cdot x_2,
\qquad
\bar{x}_2 \;{+}{=}\; \bar{x}_3 \cdot x_1
$$

Backprop through subtraction:
$$
\bar{x} \;{+}{=}\; \bar{x}_2
$$

Backprop through addition:
$$
\bar{x} \;{+}{=}\; \bar{x}_1
$$

Final result:
$$
\bar{x} = x_2 + x_1 = (x-1) + (x+1) = 2x
$$

---

## Worked example 2: $x^2 + x$ (accumulation from multiple paths)

### Forward pass

$$
\begin{aligned}
x_0 &= x \\
x_1 &= x_0 \cdot x_0 \\
x_2 &= x_1 + x_0 \\
y &= x_2
\end{aligned}
$$

---

### Backward pass

Seed:
$$
\bar{x}_2 = 1
$$

Backprop through addition:
$$
\bar{x}_1 \;{+}{=}\; \bar{x}_2,
\qquad
\bar{x}_0 \;{+}{=}\; \bar{x}_2
$$

Backprop through multiplication:
$$
\bar{x}_0 \;{+}{=}\; \bar{x}_1 \cdot x_0
$$
$$
\bar{x}_0 \;{+}{=}\; \bar{x}_1 \cdot x_0
$$

Final result:
$$
\bar{x}_0 = 1 + 2x_0
$$

Since $x_0 = x$:
$$
\frac{df}{dx} = 1 + 2x
$$

---

## Extending the primitive set

Additional operations (negation, division, transcendental functions) are
handled by supplying their **local partial derivatives** and applying the
same reverse accumulation rule.

No changes to the reverse-mode algorithm are required.

---

### General unary rule

For a unary operation:

$$
x_i = f(x_j)
$$

the reverse-mode update is:

$$
\bar{x}_j \;{+}{=}\; \bar{x}_i \cdot \frac{\partial x_i}{\partial x_j}
$$

The local derivative is evaluated using values already computed during
the forward pass.

---

### Example: exponential

Consider the tape node:

$$
x_i = \exp(x_j)
$$

#### Forward pass

During forward execution we compute and store:

$$
x_i = \exp(x_j)
$$

#### Backward pass invariant

When we reach this node in the reverse sweep, the adjoint
$\bar{x}_i = \partial y / \partial x_i$ is already known.

#### Local derivative

The local partial derivative is:

$$
\frac{\partial x_i}{\partial x_j} = \exp(x_j)
$$

Since $x_i = \exp(x_j)$ was already computed, we reuse it:

$$
\frac{\partial x_i}{\partial x_j} = x_i
$$

#### Reverse accumulation

Applying the unary reverse rule:

$$
\boxed{
\bar{x}_j \;{+}{=}\; \bar{x}_i \cdot x_i
}
$$

This is the complete backpropagation rule for `exp`.

---

### Remarks

- No symbolic differentiation is performed during backpropagation.
- The exponential is evaluated only once, during the forward pass.
- Reverse-mode AD reuses the stored primal value to compute the gradient.
- This pattern applies uniformly to all unary primitives.

For example:

- Negation: `x_i = -x_j`
  $$
  \bar{x}_j \;{+}{=}\; -\bar{x}_i
  $$

- Sine: `x_i = \sin(x_j)`
  $$
  \bar{x}_j \;{+}{=}\; \bar{x}_i \cdot \cos(x_j)
  $$

Each new primitive is added by specifying its local derivative and
plugging it into the same reverse accumulation rule.

---

## Summary

- Reverse-mode AD applies the chain rule **backwards over executed code**
- Each primitive contributes local partial derivatives
- Gradients accumulate via `+=` because multiple paths may contribute
- Coefficients like `2` arise from **path multiplicity**, not symbolic rules
- For scalar outputs, reverse mode computes the full gradient in one backward pass
