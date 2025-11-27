# CY Slice Viewer

Explore Calabi-Yau inspired complex dynamics and classic fractals.

<script src="https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js"></script>
<link rel="stylesheet" href="../site/interactive/css/interactive.css">
<link rel="stylesheet" href="../site/interactive/css/interactive-mobile.css">
<script src="../site/interactive/js/ui_common.js"></script>
<script src="../site/interactive/js/pyodide_bootstrap.js"></script>
<script src="../site/interactive/js/cy_slice_viewer.js"></script>

<div id="cy-viewer-container" class="interactive-demo"></div>

<script>
document.addEventListener('DOMContentLoaded', function() {
    new CYSliceViewer('cy-viewer-container');
});
</script>

## About This Tool

The CY Slice Viewer explores complex dynamics inspired by Calabi-Yau manifolds, alongside classic Mandelbrot and Julia sets.

### Mandelbrot Set

The classic **Mandelbrot set** shows which values of $c$ produce bounded orbits for:

$$
z_{n+1} = z_n^2 + c, \quad z_0 = 0
$$

Points inside the set (black) remain bounded; colors indicate escape time.

### Julia Sets

**Julia sets** fix $c$ and vary the initial condition $z_0$:

$$
z_{n+1} = z_n^2 + c
$$

Each point in the Mandelbrot set corresponds to a connected Julia set. Try the presets for famous examples:

- **Dendrite**: $c = 0 + i$ — tree-like branching
- **Spiral**: $c = -0.7463 + 0.1102i$ — spiral arms
- **Rabbit**: $c = -0.123 + 0.745i$ — three-lobed structure
- **Seahorse**: $c = -0.75 + 0.11i$ — valley with seahorse shapes

### CY Slices

**Calabi-Yau slices** use a modified iteration inspired by higher-dimensional geometry:

$$
z_{n+1} = U z_n + \epsilon (z_n \odot z_n) + c
$$

The parameter $k$ controls the degree, and $\epsilon$ controls the perturbation strength.

## Mathematical Connection

These complex dynamics connect to consciousness modeling through:

1. **Bifurcation structure**: Analogous to phase transitions in mind states
2. **Self-similarity**: Fractal boundaries mirror hierarchical organization
3. **Julia/Mandelbrot duality**: Parameter space vs. state space perspectives

## Usage Tips

1. **Start with Mandelbrot** to see the overall parameter space
2. **Click Julia presets** to explore famous Julia sets
3. **Adjust c parameters** to find your own interesting Julia sets
4. **Try CY Slice mode** to see the higher-dimensional generalization
5. **Increase resolution** for more detail (slower computation)
