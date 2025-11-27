# Simulation Module

Functions for simulating fractal dynamics.

## simulate_orbit

```python
def simulate_orbit(
    model: FractalDynamicsModel,
    x0: np.ndarray,
    n_steps: int = 1000
) -> np.ndarray:
    """
    Simulate orbit from initial condition.

    Args:
        model: FractalDynamicsModel instance
        x0: Initial state vector
        n_steps: Number of iterations

    Returns:
        Array of shape (n_steps+1, dim) containing trajectory
    """
```

### Example

```python
from mindfractal import FractalDynamicsModel, simulate_orbit

model = FractalDynamicsModel()
x0 = np.array([0.5, 0.5])
trajectory = simulate_orbit(model, x0, n_steps=1000)

print(f"Shape: {trajectory.shape}")  # (1001, 2)
print(f"Final state: {trajectory[-1]}")
```

## simulate_batch

```python
def simulate_batch(
    model: FractalDynamicsModel,
    x0_batch: np.ndarray,
    n_steps: int = 1000
) -> np.ndarray:
    """
    Simulate multiple orbits in parallel.

    Args:
        model: FractalDynamicsModel instance
        x0_batch: Initial conditions, shape (n_orbits, dim)
        n_steps: Number of iterations

    Returns:
        Array of shape (n_orbits, n_steps+1, dim)
    """
```

### Example

```python
# Simulate 100 orbits
x0_batch = np.random.randn(100, 2)
trajectories = simulate_batch(model, x0_batch, n_steps=500)
print(f"Shape: {trajectories.shape}")  # (100, 501, 2)
```

## find_fixed_points

```python
def find_fixed_points(
    model: FractalDynamicsModel,
    n_trials: int = 100,
    tol: float = 1e-8
) -> List[np.ndarray]:
    """
    Find fixed points via Newton iteration.

    Args:
        model: Model instance
        n_trials: Number of random starting points
        tol: Convergence tolerance

    Returns:
        List of fixed point arrays (duplicates removed)
    """
```

## analyze_stability

```python
def analyze_stability(
    model: FractalDynamicsModel,
    fixed_point: np.ndarray
) -> dict:
    """
    Analyze stability of a fixed point.

    Args:
        model: Model instance
        fixed_point: Fixed point to analyze

    Returns:
        Dictionary with:
        - eigenvalues: Jacobian eigenvalues
        - stable: True if all |λ| < 1
        - node_type: 'stable_node', 'unstable_node', 'saddle', 'spiral'
    """
```
