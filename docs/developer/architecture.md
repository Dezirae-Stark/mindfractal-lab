# Architecture

System architecture and design principles.

## Overview

```
mindfractal-lab/
├── mindfractal/           # Core package
│   ├── model.py           # FractalDynamicsModel
│   ├── simulate.py        # Simulation engine
│   ├── visualize.py       # Plotting functions
│   ├── fractal_map.py     # Parameter space analysis
│   └── mindfractal_cli.py # CLI interface
├── extensions/            # Optional modules
│   ├── state3d/           # 3D dynamics
│   ├── cy_extension/      # Complex CY dynamics
│   ├── psychomapping/     # Trait mapping
│   ├── tenth_dimension_possibility/  # Possibility Manifold
│   ├── gui_kivy/          # Android/desktop GUI
│   ├── webapp/            # FastAPI web interface
│   └── cpp_backend/       # C++ acceleration
├── tests/                 # Unit tests
├── docs/                  # Documentation
└── examples/              # Usage examples
```

## Design Principles

### 1. Modularity

Each component is independent:

- Core package has zero extension dependencies
- Extensions import core, not each other
- Clean interfaces between layers

### 2. Pure Python Core

The `mindfractal/` package requires only:

- NumPy
- Matplotlib

No GPU, no complex dependencies.

### 3. Optional Extensions

Extensions add functionality without bloating core:

```python
# Core only
from mindfractal import FractalDynamicsModel

# With extension
from extensions.state3d.model_3d import FractalDynamicsModel3D
```

### 4. Android Compatibility

Designed for Termux and PyDroid 3:

- Pure CPU computation
- Minimal dependencies
- CLI-first interface

## Data Flow

```
User Input
    ↓
CLI/GUI/API
    ↓
FractalDynamicsModel
    ↓
simulate_orbit()
    ↓
Trajectory Data
    ↓
Visualization/Analysis
    ↓
Output (PNG, data, interactive)
```

## Key Classes

### FractalDynamicsModel

Central class encapsulating:

- Parameters (A, B, W, c matrices)
- Iteration logic
- Jacobian computation
- Lyapunov estimation

### Simulation Functions

Stateless functions operating on models:

- `simulate_orbit(model, x0, n_steps)`
- `simulate_batch(model, x0_batch, n_steps)`
- `find_fixed_points(model)`

### Visualization Functions

Generate matplotlib figures:

- `plot_orbit(model, x0)`
- `plot_basin_of_attraction(model)`
- `plot_phase_portrait(model)`

## Extension Points

### Adding New Dynamics

1. Create class inheriting structure of `FractalDynamicsModel`
2. Implement `iterate()` and `jacobian()`
3. Place in `extensions/your_extension/`

### Adding Visualizations

1. Add function to `visualize.py` or create new module
2. Follow existing signature patterns
3. Return matplotlib Figure object

### Adding CLI Commands

1. Edit `mindfractal_cli.py`
2. Add new subcommand with argparse
3. Delegate to appropriate function
