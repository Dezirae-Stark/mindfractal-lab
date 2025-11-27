# CY Slice Viewer

Visualize complex Calabi-Yau dynamics through 2D slice projections.

!!! info "Interactive Demo"
    Explore Julia/Mandelbrot-like structures from CY dynamics.

## Visualization Types

### Julia Sets

Fix parameter $c$, vary initial $z_0$:

$$
\mathcal{J}_c = \{z_0 : \text{orbit bounded}\}
$$

### Mandelbrot-like Sets

Fix $z_0$, vary parameter $c$:

$$
\mathcal{M} = \{c : \text{orbit bounded}\}
$$

### CY Slices

2D cuts through the complex manifold:

- Real-Imaginary plane
- Modulus-Phase representation
- Parameter cross-sections

## Controls

| Control | Action |
|:--------|:-------|
| Resolution | Grid density |
| Max iterations | Escape time |
| Color map | Visualization scheme |
| Slice plane | Which 2D projection |

## Python Equivalent

```python
from extensions.cy_extension.visualize import CYSliceViewer

viewer = CYSliceViewer(model)
viewer.julia_set(c=0.3 + 0.5j, resolution=500)
viewer.mandelbrot_slice(z0=0, resolution=500)
```

## Gallery

Explore these parameter regions:

- **c = -0.7 + 0.27i** — Dendrite structure
- **c = 0.285 + 0.01i** — Rabbit fractal
- **c = -0.8 + 0.156i** — Spiral arms

<div id="cy-viewer-container">
  <p><em>CY viewer loads here when JavaScript is enabled.</em></p>
</div>
