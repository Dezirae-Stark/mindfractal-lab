"""
CY Core — Calabi-Yau Inspired Complex Dynamics
MindFractal Lab

Pyodide-compatible module for complex high-dimensional fractal computation.
"""

import numpy as np
from typing import Tuple, Optional
import base64
from io import BytesIO


def random_unitary(k: int) -> np.ndarray:
    """Generate a random k×k unitary matrix."""
    # QR decomposition of random complex matrix
    Z = np.random.randn(k, k) + 1j * np.random.randn(k, k)
    Q, R = np.linalg.qr(Z)
    # Make diagonal of R positive for uniqueness
    D = np.diag(R)
    Ph = np.diag(D / np.abs(D))
    return Q @ Ph


def cy_iterate(
    z: np.ndarray,
    c: np.ndarray,
    U: np.ndarray,
    eps: complex = 1.0
) -> np.ndarray:
    """
    Single CY iteration: z_{n+1} = U @ z + eps * (z * z) + c

    Parameters
    ----------
    z : np.ndarray
        Complex state vector
    c : np.ndarray
        Complex parameter vector
    U : np.ndarray
        Unitary matrix
    eps : complex
        Nonlinearity strength

    Returns
    -------
    np.ndarray
        Next state
    """
    return U @ z + eps * (z * z) + c


def compute_cy_orbit(
    z0: np.ndarray,
    c: np.ndarray,
    U: np.ndarray,
    n_steps: int = 500,
    eps: complex = 1.0
) -> np.ndarray:
    """
    Compute CY trajectory.

    Parameters
    ----------
    z0 : np.ndarray
        Initial complex state
    c : np.ndarray
        Complex parameter
    U : np.ndarray
        Unitary matrix
    n_steps : int
        Number of iterations
    eps : complex
        Nonlinearity strength

    Returns
    -------
    np.ndarray
        Trajectory of shape (n_steps, k) where k = len(z0)
    """
    k = len(z0)
    trajectory = np.zeros((n_steps, k), dtype=complex)
    z = z0.copy()

    for i in range(n_steps):
        trajectory[i] = z
        z = cy_iterate(z, c, U, eps)

        # Check for escape
        if np.linalg.norm(z) > 100:
            trajectory[i+1:] = np.nan
            break

    return trajectory


def compute_mandelbrot_slice(
    resolution: int = 200,
    c_re_range: Tuple[float, float] = (-2.5, 1.0),
    c_im_range: Tuple[float, float] = (-1.5, 1.5),
    max_iter: int = 100,
    escape_radius: float = 2.0
) -> np.ndarray:
    """
    Compute classic Mandelbrot set (k=1, U=1, eps=1).

    Parameters
    ----------
    resolution : int
        Grid resolution
    c_re_range, c_im_range : tuple
        Complex plane bounds
    max_iter : int
        Maximum iterations
    escape_radius : float
        Escape threshold

    Returns
    -------
    np.ndarray
        Escape time array
    """
    c_re = np.linspace(c_re_range[0], c_re_range[1], resolution)
    c_im = np.linspace(c_im_range[0], c_im_range[1], resolution)

    result = np.zeros((resolution, resolution))

    for i, im in enumerate(c_im):
        for j, re in enumerate(c_re):
            c = complex(re, im)
            z = 0.0 + 0.0j

            for k in range(max_iter):
                z = z * z + c

                if abs(z) > escape_radius:
                    # Smooth coloring
                    nu = k + 1 - np.log(np.log(abs(z))) / np.log(2)
                    result[i, j] = nu
                    break
            else:
                result[i, j] = max_iter

    return result


def compute_julia_slice(
    c: complex,
    resolution: int = 200,
    z_re_range: Tuple[float, float] = (-2.0, 2.0),
    z_im_range: Tuple[float, float] = (-2.0, 2.0),
    max_iter: int = 100,
    escape_radius: float = 2.0
) -> np.ndarray:
    """
    Compute Julia set for fixed c.

    Parameters
    ----------
    c : complex
        Fixed complex parameter
    resolution : int
        Grid resolution
    z_re_range, z_im_range : tuple
        Initial condition bounds
    max_iter : int
        Maximum iterations
    escape_radius : float
        Escape threshold

    Returns
    -------
    np.ndarray
        Escape time array
    """
    z_re = np.linspace(z_re_range[0], z_re_range[1], resolution)
    z_im = np.linspace(z_im_range[0], z_im_range[1], resolution)

    result = np.zeros((resolution, resolution))

    for i, im in enumerate(z_im):
        for j, re in enumerate(z_re):
            z = complex(re, im)

            for k in range(max_iter):
                z = z * z + c

                if abs(z) > escape_radius:
                    nu = k + 1 - np.log(np.log(abs(z))) / np.log(2)
                    result[i, j] = nu
                    break
            else:
                result[i, j] = max_iter

    return result


def compute_cy_slice(
    resolution: int = 100,
    c1_re_range: Tuple[float, float] = (-2.0, 2.0),
    c1_im_range: Tuple[float, float] = (-2.0, 2.0),
    k: int = 2,
    max_iter: int = 50,
    escape_radius: float = 10.0,
    eps: complex = 0.5
) -> np.ndarray:
    """
    Compute CY-style slice (k-dimensional complex dynamics, slice through c[0]).

    Parameters
    ----------
    resolution : int
        Grid resolution
    c1_re_range, c1_im_range : tuple
        Bounds for c[0] component
    k : int
        Complex dimension
    max_iter : int
        Maximum iterations
    escape_radius : float
        Escape threshold
    eps : complex
        Nonlinearity strength

    Returns
    -------
    np.ndarray
        Escape time array
    """
    c_re = np.linspace(c1_re_range[0], c1_re_range[1], resolution)
    c_im = np.linspace(c1_im_range[0], c1_im_range[1], resolution)

    # Fixed unitary matrix
    np.random.seed(42)  # Reproducible
    U = random_unitary(k)

    result = np.zeros((resolution, resolution))

    for i, im in enumerate(c_im):
        for j, re in enumerate(c_re):
            # Parameter vector with varying first component
            c = np.zeros(k, dtype=complex)
            c[0] = complex(re, im)

            # Initial state
            z = np.zeros(k, dtype=complex)

            for m in range(max_iter):
                z = cy_iterate(z, c, U, eps)

                if np.linalg.norm(z) > escape_radius:
                    result[i, j] = m / max_iter
                    break
            else:
                result[i, j] = 1.0

    return result


def render_to_base64(
    data: np.ndarray,
    cmap: str = 'twilight',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
) -> str:
    """
    Render numpy array to base64-encoded PNG.
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


def compute_and_render_mandelbrot(
    resolution: int = 200,
    c_re_range: Tuple[float, float] = (-2.5, 1.0),
    c_im_range: Tuple[float, float] = (-1.5, 1.5)
) -> str:
    """Compute Mandelbrot and return as base64 image."""
    data = compute_mandelbrot_slice(resolution, c_re_range, c_im_range)
    return render_to_base64(data, cmap='hot')


def compute_and_render_julia(
    c_re: float = -0.4,
    c_im: float = 0.6,
    resolution: int = 200
) -> str:
    """Compute Julia set and return as base64 image."""
    c = complex(c_re, c_im)
    data = compute_julia_slice(c, resolution)
    return render_to_base64(data, cmap='twilight')


def compute_and_render_cy_slice(
    resolution: int = 100,
    k: int = 2,
    eps: float = 0.5
) -> str:
    """Compute CY slice and return as base64 image."""
    data = compute_cy_slice(resolution, k=k, eps=complex(eps, 0))
    return render_to_base64(data, cmap='magma')


# Export for Pyodide
__all__ = [
    'random_unitary',
    'cy_iterate',
    'compute_cy_orbit',
    'compute_mandelbrot_slice',
    'compute_julia_slice',
    'compute_cy_slice',
    'render_to_base64',
    'compute_and_render_mandelbrot',
    'compute_and_render_julia',
    'compute_and_render_cy_slice'
]
