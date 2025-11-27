# API Reference

Complete API documentation for MindFractal Lab.

## Package Structure

```
mindfractal/
├── model.py          # Core FractalDynamicsModel
├── simulate.py       # Simulation functions
├── visualize.py      # Plotting utilities
├── fractal_map.py    # Parameter space fractals
└── mindfractal_cli.py  # CLI interface
```

## Quick Links

- [Core Module](core.md) — `FractalDynamicsModel` class
- [Simulation](simulation.md) — `simulate_orbit`, `iterate`
- [Visualization](visualization.md) — `plot_orbit`, `plot_basin_of_attraction`
- [Extensions](extensions.md) — 3D model, CY dynamics, trait mapping

## Installation

```bash
pip install mindfractal
```

## Basic Usage

```python
from mindfractal import FractalDynamicsModel, simulate_orbit, plot_orbit

# Create model
model = FractalDynamicsModel()

# Simulate
trajectory = simulate_orbit(model, x0=[0.5, 0.5], n_steps=1000)

# Visualize
plot_orbit(model, x0=[0.5, 0.5], save_path='orbit.png')
```

## Type Hints

The package uses type hints throughout:

```python
def simulate_orbit(
    model: FractalDynamicsModel,
    x0: np.ndarray,
    n_steps: int = 1000
) -> np.ndarray:
    ...
```

## Version

Current version: **1.0.0**

See [Changelog](https://github.com/Dezirae-Stark/mindfractal-lab/releases) for release history.
