"""
Fractal Core — Pyodide-compatible fractal computation module
MindFractal Lab

This module provides lightweight fractal computation for browser-based
visualization using Pyodide (Python in WebAssembly).
"""

import numpy as np
from typing import Tuple, Optional
import base64
from io import BytesIO

# Default parameters
DEFAULT_A = np.array([[0.9, 0.0], [0.0, 0.9]])
DEFAULT_B = np.array([[0.2, 0.3], [0.3, 0.2]])
DEFAULT_W = np.array([[1.0, 0.1], [0.1, 1.0]])


def compute_orbit(
    x0: np.ndarray,
    c: np.ndarray,
    n_steps: int = 500,
    A: np.ndarray = DEFAULT_A,
    B: np.ndarray = DEFAULT_B,
    W: np.ndarray = DEFAULT_W
) -> np.ndarray:
    """
    Compute trajectory from initial condition.

    Parameters
    ----------
    x0 : np.ndarray
        Initial state (2D vector)
    c : np.ndarray
        Parameter vector
    n_steps : int
        Number of iterations
    A, B, W : np.ndarray
        System matrices

    Returns
    -------
    np.ndarray
        Trajectory array of shape (n_steps, 2)
    """
    trajectory = np.zeros((n_steps, 2))
    x = x0.copy()

    for i in range(n_steps):
        trajectory[i] = x
        x = A @ x + B @ np.tanh(W @ x) + c

    return trajectory


def compute_lyapunov(
    x0: np.ndarray,
    c: np.ndarray,
    n_steps: int = 1000,
    n_transient: int = 100,
    A: np.ndarray = DEFAULT_A,
    B: np.ndarray = DEFAULT_B,
    W: np.ndarray = DEFAULT_W
) -> float:
    """
    Compute largest Lyapunov exponent.

    Parameters
    ----------
    x0 : np.ndarray
        Initial state
    c : np.ndarray
        Parameter vector
    n_steps : int
        Number of iterations for averaging
    n_transient : int
        Transient iterations to discard

    Returns
    -------
    float
        Estimated largest Lyapunov exponent
    """
    x = x0.copy()
    v = np.random.randn(2)
    v = v / np.linalg.norm(v)

    # Transient
    for _ in range(n_transient):
        x = A @ x + B @ np.tanh(W @ x) + c

    # Accumulate Lyapunov
    lyap_sum = 0.0
    for _ in range(n_steps):
        Wx = W @ x
        sech2 = 1 - np.tanh(Wx)**2
        J = A + B @ np.diag(sech2) @ W

        v = J @ v
        norm_v = np.linalg.norm(v)
        lyap_sum += np.log(norm_v)
        v = v / norm_v

        x = A @ x + B @ np.tanh(W @ x) + c

    return lyap_sum / n_steps


def compute_basin(
    c: np.ndarray,
    resolution: int = 100,
    x_range: Tuple[float, float] = (-2, 2),
    y_range: Tuple[float, float] = (-2, 2),
    max_iter: int = 200,
    escape_radius: float = 10.0,
    A: np.ndarray = DEFAULT_A,
    B: np.ndarray = DEFAULT_B,
    W: np.ndarray = DEFAULT_W
) -> np.ndarray:
    """
    Compute basin of attraction map.

    Parameters
    ----------
    c : np.ndarray
        Parameter vector
    resolution : int
        Grid resolution
    x_range, y_range : tuple
        Bounds for state space
    max_iter : int
        Maximum iterations
    escape_radius : float
        Escape threshold

    Returns
    -------
    np.ndarray
        Classification array of shape (resolution, resolution)
    """
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)

    result = np.zeros((resolution, resolution))

    for i, y in enumerate(y_vals):
        for j, x in enumerate(x_vals):
            state = np.array([x, y])

            for k in range(max_iter):
                state = A @ state + B @ np.tanh(W @ state) + c

                if np.linalg.norm(state) > escape_radius:
                    result[i, j] = k / max_iter  # Escape time
                    break
            else:
                result[i, j] = 1.0  # Bounded

    return result


def compute_lyapunov_map(
    resolution: int = 50,
    c1_range: Tuple[float, float] = (-2, 2),
    c2_range: Tuple[float, float] = (-2, 2),
    x0: Optional[np.ndarray] = None,
    n_steps: int = 500,
    A: np.ndarray = DEFAULT_A,
    B: np.ndarray = DEFAULT_B,
    W: np.ndarray = DEFAULT_W
) -> np.ndarray:
    """
    Compute Lyapunov exponent map over parameter space.

    Parameters
    ----------
    resolution : int
        Grid resolution
    c1_range, c2_range : tuple
        Bounds for c parameters
    x0 : np.ndarray, optional
        Initial state (default: origin)
    n_steps : int
        Iterations for Lyapunov computation

    Returns
    -------
    np.ndarray
        Lyapunov map of shape (resolution, resolution)
    """
    if x0 is None:
        x0 = np.array([0.1, 0.1])

    c1_vals = np.linspace(c1_range[0], c1_range[1], resolution)
    c2_vals = np.linspace(c2_range[0], c2_range[1], resolution)

    result = np.zeros((resolution, resolution))

    for i, c2 in enumerate(c2_vals):
        for j, c1 in enumerate(c1_vals):
            c = np.array([c1, c2])
            result[i, j] = compute_lyapunov(x0.copy(), c, n_steps, A=A, B=B, W=W)

    return result


def render_to_base64(
    data: np.ndarray,
    cmap: str = 'viridis',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
) -> str:
    """
    Render numpy array to base64-encoded PNG.

    Parameters
    ----------
    data : np.ndarray
        2D array to render
    cmap : str
        Colormap name
    vmin, vmax : float, optional
        Color scale limits

    Returns
    -------
    str
        Base64-encoded PNG image
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.imshow(data, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
    ax.axis('off')

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def compute_and_render_basin(
    c1: float = 0.1,
    c2: float = 0.1,
    resolution: int = 100
) -> str:
    """
    Convenience function: compute basin and return as base64 image.
    """
    c = np.array([c1, c2])
    basin = compute_basin(c, resolution=resolution)
    return render_to_base64(basin, cmap='magma')


def compute_and_render_lyapunov(
    resolution: int = 50,
    c1_range: Tuple[float, float] = (-2, 2),
    c2_range: Tuple[float, float] = (-2, 2)
) -> str:
    """
    Convenience function: compute Lyapunov map and return as base64 image.
    """
    lyap_map = compute_lyapunov_map(resolution, c1_range, c2_range)
    return render_to_base64(lyap_map, cmap='RdBu_r', vmin=-0.5, vmax=0.5)


# Export functions for Pyodide
__all__ = [
    'compute_orbit',
    'compute_lyapunov',
    'compute_basin',
    'compute_lyapunov_map',
    'render_to_base64',
    'compute_and_render_basin',
    'compute_and_render_lyapunov'
]
