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

Additional operations (negation, division, transcendental functions)
are handled by supplying their local partial derivatives and applying
the same reverse accumulation pattern.

For example:

- Negation: `out = -a`
  $$ \bar{a} += \bar{out} \cdot (-1) $$

- Sine: `out = sin(a)`
  $$ \bar{a} += \bar{out} \cdot \cos(a) $$

No changes to the algorithm are required.

---

## Summary

- Reverse-mode AD applies the chain rule **backwards over executed code**
- Each primitive contributes local partial derivatives
- Gradients accumulate via `+=` because multiple paths may contribute
- Coefficients like `2` arise from **path multiplicity**, not symbolic rules
- For scalar outputs, reverse mode computes the full gradient in one backward pass
