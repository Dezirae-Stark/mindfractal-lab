# Visualization Module

Plotting functions for fractal dynamics.

## plot_orbit

```python
def plot_orbit(
    model: FractalDynamicsModel,
    x0: np.ndarray,
    n_steps: int = 1000,
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs
) -> matplotlib.figure.Figure:
    """
    Plot trajectory in state space.

    Args:
        model: Model instance
        x0: Initial condition
        n_steps: Simulation length
        save_path: Path to save figure (optional)
        show: Whether to display figure
        **kwargs: Additional matplotlib options

    Returns:
        Matplotlib Figure object
    """
```

### Example

```python
from mindfractal.visualize import plot_orbit

fig = plot_orbit(model, x0=[0.5, 0.5], n_steps=2000,
                 save_path='orbit.png', color='blue', alpha=0.7)
```

## plot_basin_of_attraction

```python
def plot_basin_of_attraction(
    model: FractalDynamicsModel,
    xlim: Tuple[float, float] = (-2, 2),
    ylim: Tuple[float, float] = (-2, 2),
    resolution: int = 200,
    n_steps: int = 500,
    save_path: Optional[str] = None,
    cmap: str = 'viridis'
) -> matplotlib.figure.Figure:
    """
    Generate basin of attraction plot.

    Each pixel is colored by which attractor it converges to.

    Args:
        model: Model instance
        xlim, ylim: State space bounds
        resolution: Grid resolution
        n_steps: Iterations per point
        save_path: Path to save figure
        cmap: Colormap name

    Returns:
        Matplotlib Figure object
    """
```

## plot_phase_portrait

```python
def plot_phase_portrait(
    model: FractalDynamicsModel,
    xlim: Tuple[float, float] = (-2, 2),
    ylim: Tuple[float, float] = (-2, 2),
    resolution: int = 20,
    save_path: Optional[str] = None
) -> matplotlib.figure.Figure:
    """
    Plot vector field (phase portrait).

    Shows arrows indicating direction of flow at each grid point.

    Args:
        model: Model instance
        xlim, ylim: State space bounds
        resolution: Number of arrows per axis
        save_path: Path to save figure

    Returns:
        Matplotlib Figure object
    """
```

## plot_lyapunov_map

```python
def plot_lyapunov_map(
    param_ranges: dict,
    resolution: int = 100,
    n_steps: int = 1000,
    save_path: Optional[str] = None,
    cmap: str = 'RdBu_r'
) -> matplotlib.figure.Figure:
    """
    Generate Lyapunov exponent heatmap over parameter space.

    Args:
        param_ranges: Dict specifying parameter sweep
        resolution: Grid resolution
        n_steps: Iterations for Lyapunov estimate
        save_path: Path to save figure
        cmap: Colormap (diverging recommended)

    Returns:
        Matplotlib Figure object
    """
```

## animate_trajectory

```python
def animate_trajectory(
    model: FractalDynamicsModel,
    x0: np.ndarray,
    n_steps: int = 500,
    save_path: str = 'trajectory.gif',
    fps: int = 30
) -> None:
    """
    Create animated GIF of trajectory evolution.

    Args:
        model: Model instance
        x0: Initial condition
        n_steps: Total frames
        save_path: Output GIF path
        fps: Frames per second
    """
```
