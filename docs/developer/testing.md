# Testing

Testing strategy and guidelines for MindFractal Lab.

## Test Structure

```
tests/
├── test_model.py        # Core model tests
├── test_simulate.py     # Simulation tests
├── test_visualize.py    # Visualization tests
├── test_cli.py          # CLI tests
└── test_extensions/     # Extension tests
    ├── test_state3d.py
    ├── test_cy.py
    └── test_psychomapping.py
```

## Running Tests

### All Tests

```bash
pytest tests/ -v
```

### With Coverage

```bash
pytest tests/ -v --cov=mindfractal --cov-report=term-missing
```

### Specific File

```bash
pytest tests/test_model.py -v
```

### Specific Test

```bash
pytest tests/test_model.py::test_iteration -v
```

## Writing Tests

### Basic Test

```python
import numpy as np
import pytest
from mindfractal import FractalDynamicsModel

def test_model_creation():
    """Test default model creation."""
    model = FractalDynamicsModel()
    assert model.dim == 2
    assert model.A.shape == (2, 2)

def test_iteration():
    """Test single iteration."""
    model = FractalDynamicsModel()
    x = np.array([0.5, 0.5])
    x_next = model.iterate(x)
    assert x_next.shape == (2,)
    assert not np.allclose(x, x_next)  # State should change
```

### Parametrized Tests

```python
@pytest.mark.parametrize("x0", [
    np.array([0.0, 0.0]),
    np.array([1.0, 1.0]),
    np.array([-1.0, 0.5]),
])
def test_iteration_various_starts(x0):
    """Test iteration from various starting points."""
    model = FractalDynamicsModel()
    x_next = model.iterate(x0)
    assert np.isfinite(x_next).all()
```

### Fixtures

```python
@pytest.fixture
def model():
    """Create default model fixture."""
    return FractalDynamicsModel()

@pytest.fixture
def chaotic_model():
    """Create model in chaotic regime."""
    return FractalDynamicsModel(
        c=np.array([0.5, 0.5])
    )

def test_with_fixture(model):
    """Test using fixture."""
    assert model.dim == 2
```

## Test Categories

### Unit Tests

Test individual functions in isolation:

```python
def test_jacobian_shape():
    model = FractalDynamicsModel()
    J = model.jacobian(np.array([0.5, 0.5]))
    assert J.shape == (2, 2)
```

### Integration Tests

Test components working together:

```python
def test_simulate_and_visualize():
    model = FractalDynamicsModel()
    trajectory = simulate_orbit(model, [0.5, 0.5], 100)
    fig = plot_orbit(model, [0.5, 0.5], show=False)
    assert fig is not None
```

### Property-Based Tests

Test mathematical properties:

```python
def test_fixed_point_is_fixed():
    """Fixed points should map to themselves."""
    model = FractalDynamicsModel()
    fps = find_fixed_points(model)
    for fp in fps:
        fp_next = model.iterate(fp)
        assert np.allclose(fp, fp_next, atol=1e-6)
```

## Continuous Integration

Tests run automatically on:

- Push to main/develop
- Pull requests
- Multiple Python versions (3.8-3.12)
- Multiple OS (Ubuntu, macOS, Windows)

See `.github/workflows/tests.yml` for configuration.
