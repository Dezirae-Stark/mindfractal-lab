# API Reference

Complete API documentation for MindFractal Lab.

---

## Core Module: `mindfractal`

### `FractalDynamicsModel`

```python
from mindfractal import FractalDynamicsModel

model = FractalDynamicsModel(A=None, B=None, W=None, c=None)
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `A` | `ndarray` | `[[0.9, 0], [0, 0.9]]` | Linear feedback matrix |
| `B` | `ndarray` | `[[0.2, 0.3], [0.3, 0.2]]` | Nonlinear coupling matrix |
| `W` | `ndarray` | `[[1.0, 0.1], [0.1, 1.0]]` | Weight matrix |
| `c` | `ndarray` | `[0.1, 0.1]` | External drive vector |

**Methods:**

```python
# Single iteration step
x_next = model.step(x)

# Compute Jacobian at point x
J = model.jacobian(x)

# Estimate largest Lyapunov exponent
lyap = model.lyapunov_exponent_estimate(x0, n_steps=1000, transient=200)
```

---

### `simulate_orbit`

```python
from mindfractal import simulate_orbit

trajectory = simulate_orbit(model, x0, n_steps)
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `FractalDynamicsModel` | Model instance |
| `x0` | `ndarray` | Initial condition |
| `n_steps` | `int` | Number of iterations |

**Returns:** `ndarray` of shape `(n_steps, 2)`

---

### `find_fixed_points`

```python
from mindfractal import find_fixed_points

fixed_points = find_fixed_points(model, n_guesses=10, tolerance=1e-8)
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `FractalDynamicsModel` | — | Model instance |
| `n_guesses` | `int` | `10` | Number of random initial guesses |
| `tolerance` | `float` | `1e-8` | Convergence tolerance |

**Returns:** List of `ndarray` fixed points

---

### `compute_attractor_type`

```python
from mindfractal import compute_attractor_type

attractor_type = compute_attractor_type(model, x0, n_steps=1000)
```

**Returns:** String: `'fixed_point'`, `'limit_cycle'`, `'chaotic'`, or `'unbounded'`

---

## Visualization Module: `mindfractal.visualize`

### `plot_orbit`

```python
from mindfractal.visualize import plot_orbit

plot_orbit(model, x0, n_steps=1000, save_path=None)
```

### `plot_basin_of_attraction`

```python
from mindfractal.visualize import plot_basin_of_attraction

plot_basin_of_attraction(model, x_range=(-3, 3), y_range=(-3, 3),
                         resolution=200, n_steps=500, save_path=None)
```

### `plot_bifurcation_diagram`

```python
from mindfractal.visualize import plot_bifurcation_diagram

plot_bifurcation_diagram(model, param_name='c1', param_range=(-2, 2),
                         n_params=200, save_path=None)
```

---

## Fractal Map Module: `mindfractal.fractal_map`

### `generate_fractal_map`

```python
from mindfractal.fractal_map import generate_fractal_map

fractal_data = generate_fractal_map(c1_range=(-2, 2), c2_range=(-2, 2),
                                    resolution=500, n_steps=100)
```

**Returns:** `ndarray` of shape `(resolution, resolution)` with iteration counts

### `zoom_fractal_map`

```python
from mindfractal.fractal_map import zoom_fractal_map

zoomed_data = zoom_fractal_map(c1_center, c2_center, zoom_factor,
                               resolution=500, n_steps=100)
```

---

## Extensions

### 3D Model: `extensions.state3d.model_3d`

```python
from extensions.state3d.model_3d import FractalDynamicsModel3D

model_3d = FractalDynamicsModel3D(A=None, B=None, W=None, c=None)
```

Same interface as 2D model, but with 3×3 matrices and 3D state vector.

### Trait Mapping: `extensions.psychomapping.trait_to_c`

```python
from extensions.psychomapping.trait_to_c import traits_to_parameters

traits = {
    'openness': 0.8,      # [0, 1]
    'volatility': 0.3,    # [0, 1]
    'integration': 0.7,   # [0, 1]
    'focus': 0.6          # [0, 1]
}
c = traits_to_parameters(traits)
```

**Returns:** `ndarray` of shape `(2,)`

---

## CLI Module: `mindfractal.mindfractal_cli`

```bash
# Show help
python -m mindfractal.mindfractal_cli --help

# Commands
python -m mindfractal.mindfractal_cli simulate --x0 0.5 0.5 --steps 1000
python -m mindfractal.mindfractal_cli visualize --mode orbit --output orbit.png
python -m mindfractal.mindfractal_cli fractal --resolution 500 --output fractal.png
```

---

*See [docs/developer.md](../docs/developer.md) for architecture details.*
