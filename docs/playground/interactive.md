# Interactive Demo

A unified interactive interface combining all visualization tools.

!!! info "Browser-Based Computation"
    Powered by Pyodide — Python running in WebAssembly.

## Features

This demo combines:

- [2D/3D Explorer](explorer.md)
- [CY Slice Viewer](cy-viewer.md)
- [Possibility Navigator](navigator.md)

## Quick Start

1. **Select mode** from the tabs
2. **Adjust parameters** using sliders
3. **Click to interact** with the visualization
4. **Export results** as images or data

## Pyodide Backend

The interactive demos use [Pyodide](https://pyodide.org/) to run Python code directly in your browser.

### Loading Process

1. Download Pyodide runtime (~10MB)
2. Install numpy, matplotlib
3. Load MindFractal core modules
4. Initialize visualization

### Performance Notes

- First load takes 5-10 seconds
- Subsequent interactions are fast
- Complex computations may lag on mobile

## Fallback

If JavaScript is disabled, use the Python API:

```python
from mindfractal import FractalDynamicsModel
from mindfractal.visualize import plot_orbit, plot_basin_of_attraction

model = FractalDynamicsModel()
plot_orbit(model, x0=[0.5, 0.5])
plot_basin_of_attraction(model, resolution=200)
```

## Source Code

The interactive modules are in:

```
docs/site/interactive/
├── js/
│   ├── pyodide_bootstrap.js
│   └── fractal_viewer.js
└── py/
    ├── fractal_core.py
    ├── cy_core.py
    └── possibility_core.py
```

<div id="interactive-container">
  <p><em>Interactive demo loads here when JavaScript is enabled.</em></p>
</div>
