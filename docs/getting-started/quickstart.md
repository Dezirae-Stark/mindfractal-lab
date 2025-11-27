# Quick Start

Get up and running with MindFractal Lab in minutes.

## Basic Usage

```python
import numpy as np
from mindfractal import FractalDynamicsModel, simulate_orbit, plot_orbit

# Create model with default parameters
model = FractalDynamicsModel()

# Define initial condition
x0 = np.array([0.5, 0.5])

# Simulate orbit
trajectory = simulate_orbit(model, x0, n_steps=1000)

# Visualize
plot_orbit(model, x0, save_path='orbit.png')
```

## Command Line Interface

```bash
# Simulate trajectory
python -m mindfractal.mindfractal_cli simulate --x0 0.5 0.5 --steps 1000

# Generate visualization
python -m mindfractal.mindfractal_cli visualize --mode orbit --output orbit.png

# Generate fractal map
python -m mindfractal.mindfractal_cli fractal --resolution 500 --output fractal.png
```

## Next Steps

- Explore the [Tutorials](tutorials.md) for detailed examples
- Read the [Mathematical Foundations](../math/overview.md) to understand the theory
- Try the [Interactive Playground](../playground/explorer.md)
