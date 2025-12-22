//! Example demonstrating multivariable automatic differentiation.
//!
//! This showcases how `MultiDual` computes gradients of scalar multivariable
//! functions (f: ℝⁿ → ℝ) in a **single forward pass**.
//!
//! Run with: `cargo run --example multivar_gradient`

use autodiff::{gradient, MultiDual};

fn main() {
    println!("=== Multivariable Automatic Differentiation ===\n");

    // Example 1: Simple quadratic
    // f(x, y) = x² + 2xy + y² at (3, 4)
    println!("Example 1: f(x, y) = x² + 2xy + y² at (3, 4)");
    let f = |vars: [MultiDual<f64, 2>; 2]| {
        let [x, y] = vars;
        let two = MultiDual::constant(2.0);
        x * x + two * x * y + y * y
    };
    let (value, grad) = gradient(f, [3.0, 4.0]);
    println!("  f(3, 4) = {}", value);
    println!("  ∂f/∂x = {} (expected: 2x + 2y = 14)", grad[0]);
    println!("  ∂f/∂y = {} (expected: 2x + 2y = 14)", grad[1]);
    println!();

    // Example 2: Rosenbrock function (optimization benchmark)
    // f(x, y) = (1-x)² + 100(y-x²)²
    println!("Example 2: Rosenbrock f(x, y) = (1-x)² + 100(y-x²)²");
    let rosenbrock = |vars: [MultiDual<f64, 2>; 2]| {
        let [x, y] = vars;
        let one = MultiDual::constant(1.0);
        let hundred = MultiDual::constant(100.0);
        let term1 = one - x;
        let term2 = y - x * x;
        term1 * term1 + hundred * term2 * term2
    };

    println!("  At minimum (1, 1):");
    let (value, grad) = gradient(rosenbrock, [1.0, 1.0]);
    println!("    f(1, 1) = {}", value);
    println!(
        "    ∇f = [{}, {}] (gradient is zero at minimum)",
        grad[0], grad[1]
    );

    println!("  At starting point (0, 0):");
    let (value, grad) = gradient(rosenbrock, [0.0, 0.0]);
    println!("    f(0, 0) = {}", value);
    println!("    ∇f = [{}, {}]", grad[0], grad[1]);
    println!();

    // Example 3: Euclidean norm (important for gradient descent)
    // f(x, y, z) = √(x² + y² + z²) at (3, 4, 0)
    println!("Example 3: Euclidean norm f(x, y, z) = √(x² + y² + z²) at (3, 4, 0)");
    let euclidean_norm = |vars: [MultiDual<f64, 3>; 3]| {
        let [x, y, z] = vars;
        (x * x + y * y + z * z).sqrt()
    };
    let (value, grad) = gradient(euclidean_norm, [3.0, 4.0, 0.0]);
    println!("  ‖(3, 4, 0)‖ = {}", value);
    println!("  ∇f = [{}, {}, {}]", grad[0], grad[1], grad[2]);
    println!("  (Gradient points in direction of steepest ascent)");
    println!();

    // Example 4: With transcendental functions
    // f(x, y, z) = x² + y·exp(z) at (1, 2, 0)
    println!("Example 4: f(x, y, z) = x² + y·exp(z) at (1, 2, 0)");
    let f = |vars: [MultiDual<f64, 3>; 3]| {
        let [x, y, z] = vars;
        x * x + y * z.exp()
    };
    let (value, grad) = gradient(f, [1.0, 2.0, 0.0]);
    println!("  f(1, 2, 0) = {}", value);
    println!("  ∂f/∂x = {} (expected: 2x = 2)", grad[0]);
    println!("  ∂f/∂y = {} (expected: exp(z) = 1)", grad[1]);
    println!("  ∂f/∂z = {} (expected: y·exp(z) = 2)", grad[2]);
    println!();

    // Example 5: Mixed transcendentals
    // f(x, y) = sin(x)·exp(y) + ln(x + y) at (1, 0)
    println!("Example 5: f(x, y) = sin(x)·exp(y) + ln(x + y) at (1, 0)");
    let f = |vars: [MultiDual<f64, 2>; 2]| {
        let [x, y] = vars;
        x.sin() * y.exp() + (x + y).ln()
    };
    let (value, grad) = gradient(f, [1.0, 0.0]);
    println!("  f(1, 0) = {:.6}", value);
    println!("  ∂f/∂x = {:.6}", grad[0]);
    println!("  ∂f/∂y = {:.6}", grad[1]);
    println!("  (Product rule and chain rule applied automatically!)");
    println!();

    // Example 6: Higher dimensional (5D)
    // f(x₁, x₂, x₃, x₄, x₅) = Σᵢ xᵢ² at (1, 2, 3, 4, 5)
    println!("Example 6: Sum of squares in 5D");
    let f = |vars: [MultiDual<f64, 5>; 5]| {
        let [x1, x2, x3, x4, x5] = vars;
        x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4 + x5 * x5
    };
    let (value, grad) = gradient(f, [1.0, 2.0, 3.0, 4.0, 5.0]);
    println!("  f(1, 2, 3, 4, 5) = {} (expected: 55)", value);
    println!("  ∇f = {:?}", grad);
    println!("  (Each ∂f/∂xᵢ = 2xᵢ)");
    println!();

    // Example 7: Chain rule demonstration
    // f(x, y) = exp(x²) at (1, 0)
    println!("Example 7: Chain rule f(x, y) = exp(x²) at (1, 0)");
    let f = |vars: [MultiDual<f64, 2>; 2]| {
        let [x, _y] = vars;
        (x * x).exp()
    };
    let (value, grad) = gradient(f, [1.0, 0.0]);
    let e = 1.0_f64.exp();
    println!("  f(1, 0) = {:.6} (expected: e ≈ {:.6})", value, e);
    println!(
        "  ∂f/∂x = {:.6} (expected: 2x·exp(x²) = 2e ≈ {:.6})",
        grad[0],
        2.0 * e
    );
    println!("  ∂f/∂y = {} (independent of y)", grad[1]);
    println!();

    // Example 8: Softmax-like (for ML)
    // f(x, y) = x / (x + y) at (3, 1)
    println!("Example 8: Ratio f(x, y) = x/(x+y) at (3, 1)");
    let f = |vars: [MultiDual<f64, 2>; 2]| {
        let [x, y] = vars;
        x / (x + y)
    };
    let (value, grad) = gradient(f, [3.0, 1.0]);
    println!("  f(3, 1) = {}", value);
    println!("  ∂f/∂x = {} (expected: y/(x+y)² = 0.0625)", grad[0]);
    println!("  ∂f/∂y = {} (expected: -x/(x+y)² = -0.1875)", grad[1]);
    println!();

    println!("=== Key Insights ===");
    println!("• MultiDual computes ALL partial derivatives in a single forward pass");
    println!("• For n inputs, this is n× faster than using n passes with Dual<T>");
    println!("• Perfect for gradient-based optimization (gradient descent, Newton's method)");
    println!("• Works seamlessly with transcendental functions and complex compositions");
    println!("• Compile-time dimension checking via const generics (N)");
}
