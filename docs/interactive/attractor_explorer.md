# 3D Attractor Explorer

Visualize strange attractors and 3D dynamics of the fractal consciousness model.

<script src="https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js"></script>
<link rel="stylesheet" href="../site/interactive/css/interactive.css">
<link rel="stylesheet" href="../site/interactive/css/interactive-mobile.css">
<script src="../site/interactive/js/ui_common.js"></script>
<script src="../site/interactive/js/pyodide_bootstrap.js"></script>
<script src="../site/interactive/js/attractor_viewer.js"></script>

<div id="attractor-viewer-container" class="interactive-demo"></div>

<script>
document.addEventListener('DOMContentLoaded', function() {
    new AttractorViewer('attractor-viewer-container');
});
</script>

## About This Tool

The 3D Attractor Explorer extends the fractal dynamics model to three dimensions, revealing the rich geometric structure of strange attractors.

### Attractor Types

The system can exhibit several types of attractors:

| Type | Lyapunov Signature | Behavior |
|------|-------------------|----------|
| **Fixed Point** | $(-, -, -)$ | Converges to single point |
| **Limit Cycle** | $(0, -, -)$ | Periodic oscillation |
| **Torus** | $(0, 0, -)$ | Quasi-periodic, two frequencies |
| **Strange Attractor** | $(+, 0, -)$ | Chaotic, fractal structure |

### Poincare Section

The **Poincare section** shows where the trajectory crosses a chosen plane. For chaotic attractors, this reveals:

- **Single point**: Fixed point attractor
- **Finite points**: Periodic orbit
- **Continuous curve**: Quasi-periodic (torus)
- **Fractal dust**: Strange attractor

### 3D Dynamical System

$$
\mathbf{x}_{n+1} = A\mathbf{x}_n + B \tanh(W\mathbf{x}_n) + \mathbf{c}
$$

With $\mathbf{x}, \mathbf{c} \in \mathbb{R}^3$ and $A, B, W \in \mathbb{R}^{3\times 3}$.

## Usage Tips

1. **Start with default parameters** and observe the attractor type
2. **Vary $c_1, c_2, c_3$** to explore different dynamical regimes
3. **Use Scan Types** to see a map of attractor types in parameter space
4. **Poincare Section** helps distinguish chaos from quasi-periodicity
5. **Adjust view angle** to see 3D structure from different perspectives
