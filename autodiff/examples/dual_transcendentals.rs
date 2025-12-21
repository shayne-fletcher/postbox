//! Example demonstrating automatic differentiation with transcendental functions.
//!
//! This showcases how dual numbers automatically compute derivatives of
//! complex expressions involving exp, log, sin, cos, and sqrt.
//!
//! Run with: `cargo run --example dual_transcendentals`

use autodiff::Dual;

fn main() {
    println!("=== Dual Numbers: Transcendental Functions ===\n");

    // Example 1: Exponential function
    // f(x) = e^x at x=0
    println!("Example 1: f(x) = e^x at x=0");
    let x = Dual::variable(0.0);
    let f = x.exp();
    println!("  f(0) = {}", f.value);
    println!("  f'(0) = {} (expected: 1.0)", f.deriv);
    println!();

    // Example 2: Natural logarithm
    // f(x) = ln(x) at x=2
    println!("Example 2: f(x) = ln(x) at x=2");
    let x = Dual::variable(2.0);
    let f = x.ln();
    println!("  f(2) = {}", f.value);
    println!("  f'(2) = {} (expected: 0.5)", f.deriv);
    println!();

    // Example 3: Sine function
    // f(x) = sin(x) at x=π/2
    println!("Example 3: f(x) = sin(x) at x=π/2");
    let x = Dual::variable(std::f64::consts::PI / 2.0);
    let f = x.sin();
    println!("  f(π/2) = {}", f.value);
    println!("  f'(π/2) = {} (expected: ~0.0)", f.deriv);
    println!();

    // Example 4: Cosine function
    // f(x) = cos(x) at x=π
    println!("Example 4: f(x) = cos(x) at x=π");
    let x = Dual::variable(std::f64::consts::PI);
    let f = x.cos();
    println!("  f(π) = {}", f.value);
    println!("  f'(π) = {} (expected: ~0.0)", f.deriv);
    println!();

    // Example 5: Square root
    // f(x) = √x at x=9
    println!("Example 5: f(x) = √x at x=9");
    let x = Dual::variable(9.0);
    let f = x.sqrt();
    println!("  f(9) = {}", f.value);
    println!("  f'(9) = {} (expected: 1/6 ≈ 0.1667)", f.deriv);
    println!();

    // Example 6: Sigmoid function (ML activation)
    // σ(x) = 1 / (1 + e^(-x)) at x=0
    println!("Example 6: Sigmoid σ(x) = 1/(1 + e^(-x)) at x=0");
    let x = Dual::variable(0.0);
    let sigmoid = Dual::constant(1.0) / (Dual::constant(1.0) + (-x).exp());
    println!("  σ(0) = {}", sigmoid.value);
    println!("  σ'(0) = {} (expected: 0.25)", sigmoid.deriv);
    println!();

    // Example 7: Gaussian function
    // f(x) = e^(-x²) at x=0
    println!("Example 7: Gaussian f(x) = e^(-x²) at x=0");
    let x = Dual::variable(0.0);
    let gaussian = (-(x * x)).exp();
    println!("  f(0) = {}", gaussian.value);
    println!("  f'(0) = {} (expected: 0.0)", gaussian.deriv);
    println!();

    // Example 8: More complex: f(x) = sin(x²) at x=√(π/2)
    println!("Example 8: f(x) = sin(x²) at x=√(π/2)");
    let x = Dual::variable((std::f64::consts::PI / 2.0).sqrt());
    let f = (x * x).sin();
    println!("  f(√(π/2)) = {}", f.value);
    println!("  f'(√(π/2)) = {}", f.deriv);
    println!("  (Chain rule: f'(x) = 2x·cos(x²))");
    println!();

    // Example 9: Composition of multiple functions
    // f(x) = ln(sin(e^x)) at x=0
    println!("Example 9: f(x) = ln(sin(e^x)) at x=0");
    let x = Dual::variable(0.0);
    let f = x.exp().sin().ln();
    println!("  f(0) = ln(sin(1)) = {}", f.value);
    println!("  f'(0) = {}", f.deriv);
    println!("  (Triple chain rule automatically applied!)");
    println!();

    println!("=== Key Insights ===");
    println!("• Dual numbers compute derivatives automatically via operator overloading");
    println!("• Chain rule emerges naturally from function composition");
    println!("• Perfect for gradient-based optimization and neural networks");
    println!("• Forward-mode AD: compute derivative alongside function value");
}
