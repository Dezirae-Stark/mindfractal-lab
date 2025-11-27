# 2D Fractal Explorer

Explore the fractal structure of consciousness dynamics in 2D parameter space.

<script src="https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js"></script>
<link rel="stylesheet" href="../site/interactive/css/interactive.css">
<link rel="stylesheet" href="../site/interactive/css/interactive-mobile.css">
<script src="../site/interactive/js/ui_common.js"></script>
<script src="../site/interactive/js/pyodide_bootstrap.js"></script>
<script src="../site/interactive/js/fractal_viewer.js"></script>

<div id="fractal-viewer-container" class="interactive-demo"></div>

<script>
document.addEventListener('DOMContentLoaded', function() {
    new FractalViewer('fractal-viewer-container');
});
</script>

## About This Tool

The 2D Fractal Explorer visualizes how system behavior changes across the parameter space $(c_1, c_2)$.

### Basin of Attraction

The **basin of attraction** shows which initial conditions converge to the same final state. Different colors represent different attractors.

### Lyapunov Map

The **Lyapunov exponent** measures sensitivity to initial conditions:

- **Negative** (blue): Stable, converging dynamics
- **Zero** (white): Periodic or quasi-periodic
- **Positive** (red): Chaotic, sensitive dependence

### The Dynamical System

$$
\mathbf{x}_{n+1} = A\mathbf{x}_n + B \tanh(W\mathbf{x}_n) + \mathbf{c}
$$

The parameter $\mathbf{c} = (c_1, c_2)$ controls which region of the fractal boundary you explore.

## Usage Tips

1. **Start with Basin mode** to see the overall structure
2. **Switch to Lyapunov mode** to identify chaotic regions
3. **Adjust resolution** higher for more detail (slower computation)
4. **Parameters near the boundary** often show the most interesting dynamics
