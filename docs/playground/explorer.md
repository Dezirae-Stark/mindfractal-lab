# 2D/3D Fractal Explorer

Interactive visualization of fractal dynamics.

!!! info "Interactive Demo"
    This page hosts a browser-based fractal explorer powered by Pyodide.

## Features

- **Real-time orbit simulation**
- **Parameter sliders** for A, B, W, c matrices
- **Basin of attraction visualization**
- **Lyapunov exponent computation**

## Usage

### Controls

| Control | Action |
|:--------|:-------|
| Click | Set initial condition |
| Drag | Pan view |
| Scroll | Zoom |
| Sliders | Adjust parameters |

### Visualization Modes

1. **Orbit** — Single trajectory
2. **Phase Portrait** — Vector field
3. **Basin** — Attractor classification
4. **Lyapunov** — Stability map

## Python Equivalent

```python
from mindfractal.visualize import InteractiveExplorer

explorer = InteractiveExplorer(model)
explorer.show()  # Opens matplotlib interactive window
```

## Parameters

Explore different dynamical regimes:

- **Low damping** — extended transients
- **Strong coupling** — chaotic behavior
- **Asymmetric weights** — complex basin boundaries

<div id="fractal-explorer-container">
  <p><em>Interactive explorer loads here when JavaScript is enabled.</em></p>
</div>
