# Core Module

The `FractalDynamicsModel` class — heart of MindFractal Lab.

## FractalDynamicsModel

```python
class FractalDynamicsModel:
    """
    2D Fractal Dynamical System Model.

    Implements the discrete-time map:
        x_{n+1} = A @ x_n + B @ tanh(W @ x_n) + c
    """
```

### Constructor

```python
FractalDynamicsModel(
    A: Optional[np.ndarray] = None,
    B: Optional[np.ndarray] = None,
    W: Optional[np.ndarray] = None,
    c: Optional[np.ndarray] = None
)
```

**Parameters:**

| Name | Type | Default | Description |
|:-----|:-----|:--------|:------------|
| `A` | `np.ndarray` | `[[0.9, 0.1], [-0.1, 0.9]]` | Linear feedback matrix |
| `B` | `np.ndarray` | `[[0.5, 0.0], [0.0, 0.5]]` | Nonlinear coupling matrix |
| `W` | `np.ndarray` | `[[1.0, 0.5], [0.5, 1.0]]` | Weight matrix |
| `c` | `np.ndarray` | `[0.1, 0.05]` | External drive vector |

### Methods

#### `iterate`

```python
def iterate(self, x: np.ndarray) -> np.ndarray:
    """
    Apply one iteration of the map.

    Args:
        x: Current state vector (2D)

    Returns:
        Next state vector
    """
```

#### `jacobian`

```python
def jacobian(self, x: np.ndarray) -> np.ndarray:
    """
    Compute Jacobian matrix at state x.

    Args:
        x: State vector

    Returns:
        2x2 Jacobian matrix
    """
```

#### `lyapunov_exponent_estimate`

```python
def lyapunov_exponent_estimate(
    self,
    x0: np.ndarray,
    n_steps: int = 5000,
    n_transient: int = 1000
) -> float:
    """
    Estimate maximal Lyapunov exponent.

    Args:
        x0: Initial condition
        n_steps: Number of iterations
        n_transient: Transient to discard

    Returns:
        Estimated Lyapunov exponent
    """
```

### Properties

```python
@property
def dim(self) -> int:
    """State space dimension (always 2 for this model)."""

@property
def params(self) -> dict:
    """Dictionary of all parameters."""
```

### Example

```python
import numpy as np
from mindfractal import FractalDynamicsModel

# Custom parameters
A = np.array([[0.8, 0.2], [-0.2, 0.8]])
model = FractalDynamicsModel(A=A)

# Single iteration
x = np.array([0.5, 0.5])
x_next = model.iterate(x)

# Jacobian
J = model.jacobian(x)

# Lyapunov exponent
lyap = model.lyapunov_exponent_estimate(x, n_steps=5000)
print(f"λ = {lyap:.4f}")
```
