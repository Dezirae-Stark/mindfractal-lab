# MindFractal Lab - Developer Guide

## API Reference

### Core Package: `mindfractal`

#### `mindfractal.model.FractalDynamicsModel`

**Constructor:**
```python
FractalDynamicsModel(A=None, B=None, W=None, c=None)
```

**Parameters:**
- `A` (ndarray, optional): 2×2 feedback matrix. Default: `diag(0.9, 0.9)`
- `B` (ndarray, optional): 2×2 coupling matrix. Default: `[[0.2, 0.3], [0.3, 0.2]]`
- `W` (ndarray, optional): 2×2 weight matrix. Default: `eye(2) + 0.1 * random`
- `c` (ndarray, optional): 2D external drive vector. Default: `[0.1, 0.1]`

**Methods:**

##### `step(x: ndarray) -> ndarray`
Single dynamics iteration: `x_{n+1} = A x_n + B tanh(W x_n) + c`

**Parameters:**
- `x` (ndarray): Current state vector (2D)

**Returns:**
- `ndarray`: Next state vector (2D)

##### `jacobian(x: ndarray) -> ndarray`
Compute Jacobian matrix at state `x`

**Parameters:**
- `x` (ndarray): State vector (2D)

**Returns:**
- `ndarray`: 2×2 Jacobian matrix

**Formula:**
```
J(x) = A + B * diag(sech²(W x)) * W
```

##### `lyapunov_exponent_estimate(x0, n_steps=5000, n_transient=1000) -> float`
Estimate largest Lyapunov exponent

**Parameters:**
- `x0` (ndarray): Initial condition (2D)
- `n_steps` (int): Number of iterations
- `n_transient` (int): Transient steps to discard

**Returns:**
- `float`: Estimated λ (positive → chaos, zero → periodic, negative → stable)

##### `energy(x: ndarray) -> float`
Heuristic energy function (experimental)

**Parameters:**
- `x` (ndarray): State vector (2D)

**Returns:**
- `float`: Energy value

---

#### `mindfractal.simulate`

##### `simulate_orbit(model, x0, n_steps=1000, return_all=True)`
Generate trajectory from initial condition

**Parameters:**
- `model` (FractalDynamicsModel): Dynamics model
- `x0` (ndarray): Initial state (2D)
- `n_steps` (int): Number of steps to simulate
- `return_all` (bool): If True, return full trajectory; if False, return final state only

**Returns:**
- `ndarray`: Trajectory array of shape `(n_steps, 2)` or final state `(2,)`

##### `find_fixed_points(model, n_trials=10, tolerance=1e-6, max_iter=100)`
Find fixed points using Newton's method from random initializations

**Parameters:**
- `model` (FractalDynamicsModel): Dynamics model
- `n_trials` (int): Number of random starting points
- `tolerance` (float): Convergence criterion
- `max_iter` (int): Max Newton iterations

**Returns:**
- `list`: List of `(fixed_point, is_stable)` tuples

##### `compute_attractor_type(trajectory, tolerance=1e-3)`
Classify attractor type from trajectory

**Parameters:**
- `trajectory` (ndarray): Orbit array `(n_steps, 2)`
- `tolerance` (float): Threshold for fixed point detection

**Returns:**
- `str`: One of `'fixed_point'`, `'limit_cycle'`, `'chaotic'`, `'unbounded'`

##### `basin_of_attraction_sample(model, x_range, y_range, resolution=100, criterion='attractor_type')`
Compute basin of attraction on a grid

**Parameters:**
- `model` (FractalDynamicsModel): Dynamics model
- `x_range` (tuple): `(x_min, x_max)` for state space
- `y_range` (tuple): `(y_min, y_max)` for state space
- `resolution` (int): Grid resolution
- `criterion` (str): `'attractor_type'`, `'divergence_time'`, or `'final_norm'`

**Returns:**
- `ndarray`: Basin map of shape `(resolution, resolution)`

##### `poincare_section(trajectory, axis=0, value=0.0, tolerance=0.01)`
Extract Poincaré section crossings

**Parameters:**
- `trajectory` (ndarray): Orbit array `(n_steps, 2)`
- `axis` (int): Axis to slice (0 or 1)
- `value` (float): Section value
- `tolerance` (float): Crossing detection threshold

**Returns:**
- `ndarray`: Array of crossing points

---

#### `mindfractal.visualize`

##### `plot_orbit(model, x0, n_steps=1000, save_path=None)`
Generate 4-panel phase portrait with time series

**Parameters:**
- `model` (FractalDynamicsModel): Dynamics model
- `x0` (ndarray): Initial condition (2D)
- `n_steps` (int): Simulation length
- `save_path` (str, optional): File path to save figure

**Returns:**
- `Figure`: Matplotlib figure object

**Layout:**
- Top-left: Phase portrait (x1 vs x2)
- Top-right: x1 time series
- Bottom-left: x2 time series
- Bottom-right: Lyapunov exponent estimate

##### `plot_fractal_map(fractal_data, c1_range, c2_range, save_path=None)`
Visualize parameter-space fractal

**Parameters:**
- `fractal_data` (ndarray): Fractal map array
- `c1_range` (tuple): `(c1_min, c1_max)`
- `c2_range` (tuple): `(c2_min, c2_max)`
- `save_path` (str, optional): File path to save figure

**Returns:**
- `Figure`: Matplotlib figure object

##### `plot_basin_of_attraction(model, resolution=200, x_range=(-2, 2), y_range=(-2, 2), save_path=None)`
Visualize basin of attraction with fractal boundaries

**Parameters:**
- `model` (FractalDynamicsModel): Dynamics model
- `resolution` (int): Grid resolution
- `x_range` (tuple): State space x bounds
- `y_range` (tuple): State space y bounds
- `save_path` (str, optional): File path to save figure

**Returns:**
- `Figure`: Matplotlib figure object

##### `plot_bifurcation_diagram(model, param_name='c1', param_range=(-1.0, 1.0), n_points=200, n_steps=500, n_plot=100, save_path=None)`
Generate bifurcation diagram varying a parameter

**Parameters:**
- `model` (FractalDynamicsModel): Dynamics model
- `param_name` (str): Parameter to vary (`'c1'` or `'c2'`)
- `param_range` (tuple): Parameter range
- `n_points` (int): Number of parameter values to sample
- `n_steps` (int): Simulation steps per parameter value
- `n_plot` (int): Number of final steps to plot
- `save_path` (str, optional): File path to save figure

**Returns:**
- `Figure`: Matplotlib figure object

##### `plot_lyapunov_spectrum(model, x0, n_steps=5000, window_size=500, save_path=None)`
Plot Lyapunov exponent evolution over time

**Parameters:**
- `model` (FractalDynamicsModel): Dynamics model
- `x0` (ndarray): Initial condition (2D)
- `n_steps` (int): Total simulation steps
- `window_size` (int): Sliding window for λ estimation
- `save_path` (str, optional): File path to save figure

**Returns:**
- `Figure`: Matplotlib figure object

---

#### `mindfractal.fractal_map`

##### `generate_fractal_map(model=None, c1_range=(-1.0, 1.0), c2_range=(-1.0, 1.0), resolution=500, x0=None, n_steps=500, max_steps=1000, criterion='divergence_time')`
Generate parameter-space fractal map

**Parameters:**
- `model` (FractalDynamicsModel, optional): Base model (ignored if varying c)
- `c1_range` (tuple): Range for c1 parameter
- `c2_range` (tuple): Range for c2 parameter
- `resolution` (int): Grid resolution (warning: high values slow)
- `x0` (ndarray, optional): Fixed initial condition. Default: `[0.5, 0.5]`
- `n_steps` (int): Simulation steps
- `max_steps` (int): Maximum steps before declaring divergence
- `criterion` (str): One of:
  - `'divergence_time'`: Steps until ||x|| > 10
  - `'final_norm'`: ||x_final||
  - `'lyapunov'`: Estimated λ
  - `'attractor_type'`: Categorical (0=fixed, 1=cycle, 2=chaos, 3=unbounded)

**Returns:**
- `ndarray`: Fractal map of shape `(resolution, resolution)`

##### `zoom_fractal_map(center, zoom_factor=2.0, resolution=500, **kwargs)`
Zoom into fractal boundary region

**Parameters:**
- `center` (tuple): `(c1, c2)` center point
- `zoom_factor` (float): Magnification factor
- `resolution` (int): Grid resolution
- `**kwargs`: Passed to `generate_fractal_map()`

**Returns:**
- `ndarray`: Zoomed fractal map

##### `adaptive_fractal_map(c1_range, c2_range, resolution=500, refinement_levels=2, **kwargs)`
Generate fractal map with adaptive refinement at boundaries

**Parameters:**
- `c1_range` (tuple): Range for c1
- `c2_range` (tuple): Range for c2
- `resolution` (int): Base resolution
- `refinement_levels` (int): Number of adaptive refinement passes
- `**kwargs`: Passed to `generate_fractal_map()`

**Returns:**
- `ndarray`: Adaptively refined fractal map

---

### Extension: `extensions.state3d`

#### `extensions.state3d.model_3d.FractalDynamicsModel3D`

Same interface as 2D model, but with 3D state vectors and 3×3 matrices.

**Key differences:**
- `x ∈ ℝ³` (instead of ℝ²)
- `A, B, W ∈ ℝ^{3×3}` (instead of ℝ^{2×2})
- `c ∈ ℝ³` (instead of ℝ²)

#### `extensions.state3d.simulate_3d.lyapunov_spectrum_3d(model_3d, x0, n_steps=5000)`
Compute full 3-exponent Lyapunov spectrum

**Returns:**
- `tuple`: `(λ1, λ2, λ3)` sorted in descending order

---

### Extension: `extensions.psychomapping`

#### `extensions.psychomapping.trait_to_c.traits_to_parameters(traits)`
Map psychological traits to parameter vector

**Parameters:**
- `traits` (dict): Dictionary with keys:
  - `'openness'`: float ∈ [0, 1]
  - `'volatility'`: float ∈ [0, 1]
  - `'integration'`: float ∈ [0, 1]
  - `'focus'`: float ∈ [0, 1]

**Returns:**
- `ndarray`: Parameter vector `c` of shape `(2,)`

**Mapping Formula:**
```python
c1 = -1.0 + 2.0 * traits['openness'] + 0.5 * (traits['volatility'] - 0.5)
c2 = -1.0 + 2.0 * traits['integration'] + 0.5 * (traits['focus'] - 0.5)
```

#### `extensions.psychomapping.trait_to_c.load_trait_profile(profile_name)`
Load pre-defined trait profile from `traits.json`

**Parameters:**
- `profile_name` (str): One of `'balanced'`, `'creative_explorer'`, `'stable_focused'`, `'chaotic_fragmented'`, `'meditative'`

**Returns:**
- `dict`: Trait dictionary

---

## Architecture

### Module Dependency Graph

```
mindfractal/
├── model.py          (no internal deps)
├── simulate.py       (depends: model)
├── visualize.py      (depends: model, simulate)
├── fractal_map.py    (depends: model, simulate)
└── mindfractal_cli.py (depends: all)

extensions/
├── state3d/          (depends: core mindfractal)
├── psychomapping/    (depends: core mindfractal)
├── gui_kivy/         (depends: core + psychomapping)
├── webapp/           (depends: core + psychomapping)
└── cpp_backend/      (no Python deps, pure C++)
```

### Data Flow

```
User Input → Model Parameters → Simulation → Trajectory → Visualization
                ↓
         Trait Mapping (optional)
```

### Design Patterns

1. **Strategy Pattern**: `criterion` parameter in `generate_fractal_map()` allows different coloring strategies
2. **Factory Pattern**: Model constructors with default parameters
3. **Template Method**: `simulate_orbit()` as general-purpose trajectory generator
4. **Adapter Pattern**: C++ backend wraps Python API via pybind11

---

## Development Workflow

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/mindfractal-lab.git
cd mindfractal-lab

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (if using)
pip install pre-commit
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=mindfractal tests/

# Run specific test file
pytest tests/test_model.py

# Run specific test function
pytest tests/test_model.py::test_step_function
```

### Code Style

- **PEP 8**: Follow Python style guide
- **Type Hints**: Add type annotations to all public functions
- **Docstrings**: Use NumPy-style docstrings

**Example:**
```python
def simulate_orbit(
    model: FractalDynamicsModel,
    x0: np.ndarray,
    n_steps: int = 1000,
    return_all: bool = True
) -> np.ndarray:
    """
    Generate trajectory from initial condition.

    Parameters
    ----------
    model : FractalDynamicsModel
        Dynamics model instance
    x0 : np.ndarray
        Initial state vector (2D)
    n_steps : int, optional
        Number of simulation steps (default: 1000)
    return_all : bool, optional
        If True, return full trajectory; else return final state (default: True)

    Returns
    -------
    np.ndarray
        Trajectory array of shape (n_steps, 2) or final state (2,)

    Examples
    --------
    >>> model = FractalDynamicsModel()
    >>> x0 = np.array([0.5, 0.5])
    >>> trajectory = simulate_orbit(model, x0, n_steps=1000)
    >>> trajectory.shape
    (1000, 2)
    """
    # Implementation...
```

### Testing Guidelines

1. **Unit Tests**: Test individual functions in isolation
2. **Integration Tests**: Test module interactions
3. **Property Tests**: Use hypothesis for property-based testing (optional)
4. **Regression Tests**: Prevent fixed bugs from reoccurring

**Test Coverage Goals:**
- Core modules: 90%+ coverage
- Extensions: 70%+ coverage

---

## Performance Optimization

### Bottlenecks

1. **Fractal map generation**: O(resolution²) simulations
2. **Lyapunov exponent estimation**: O(n_steps) Jacobian evaluations
3. **Basin of attraction**: O(resolution²) full simulations

### Optimization Strategies

1. **Use C++ backend**: 10-100× speedup for orbit simulation
2. **Reduce resolution**: Use 200×200 instead of 500×500 for prototyping
3. **Parallelize**: Use `multiprocessing` for embarrassingly parallel tasks (fractal maps)
4. **Adaptive sampling**: Refine only near boundaries
5. **Caching**: Store computed trajectories for repeated queries

### Example: Parallel Fractal Map

```python
from multiprocessing import Pool

def compute_row(args):
    """Compute single row of fractal map."""
    row_idx, c1_values, c2, model, x0, n_steps = args
    # ... compute row ...
    return row_idx, row_data

# Parallelize over rows
with Pool() as pool:
    results = pool.map(compute_row, row_args)
```

---

## Extension Development

### Creating a New Extension

1. Create directory: `extensions/my_extension/`
2. Add `__init__.py`
3. Implement functionality
4. Document in `README.md`
5. Add tests in `tests/test_my_extension.py`
6. Update main `README.md` with extension description

### Example Extension Structure

```
extensions/my_extension/
├── __init__.py
├── my_module.py
├── README.md
└── requirements.txt  (if extra dependencies needed)
```

---

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.

**Quick checklist:**
- [ ] Fork the repository
- [ ] Create feature branch
- [ ] Add tests for new functionality
- [ ] Update documentation
- [ ] Ensure all tests pass
- [ ] Submit pull request

---

## Troubleshooting

### Common Issues

**Issue: `ImportError: No module named 'mindfractal'`**

Solution: Install in editable mode: `pip install -e .`

**Issue: Matplotlib backend errors on Android**

Solution: Code already sets `Agg` backend. If issues persist, check:
```python
import matplotlib
matplotlib.use('Agg')  # Must be before pyplot import
```

**Issue: Slow fractal map generation**

Solutions:
1. Reduce resolution: `resolution=200`
2. Decrease steps: `n_steps=300`
3. Use C++ backend (if compiled)
4. Parallelize with multiprocessing

**Issue: Memory errors on Android**

Solutions:
1. Use `return_all=False` in `simulate_orbit()`
2. Process in batches
3. Reduce grid resolution
4. Clear variables: `del large_array`

---

## API Stability

**Version 0.1.0**: API is experimental and may change.

**Future compatibility:**
- Semantic versioning: MAJOR.MINOR.PATCH
- Breaking changes only in MAJOR versions
- Deprecation warnings before removal

---

## Contact

- **Issues**: https://github.com/YOUR_USERNAME/mindfractal-lab/issues
- **Discussions**: https://github.com/YOUR_USERNAME/mindfractal-lab/discussions

---

**Version**: 0.1.0
**Last Updated**: 2025-11-17
