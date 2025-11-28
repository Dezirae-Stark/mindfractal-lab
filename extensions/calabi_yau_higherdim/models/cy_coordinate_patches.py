"""
Coordinate Patch Utilities

Tools for mapping between local coordinates and global representations,
slicing, and projection operations for CY state spaces.
"""

import numpy as np
from typing import Tuple, List, Optional, Callable


def slice_2d(
    z: np.ndarray,
    indices: Tuple[int, int] = (0, 1)
) -> np.ndarray:
    """
    Extract 2D slice from high-dimensional complex vector.

    Parameters:
        z (ndarray): Complex vector (k,)
        indices (tuple): Which two indices to extract

    Returns:
        ndarray: 2D complex vector
    """
    return z[list(indices)]


def project_to_real_2d(
    z: np.ndarray,
    method: str = 'real_imag',
    indices: Tuple[int, int] = (0, 1)
) -> np.ndarray:
    """
    Project complex vector to ℝ².

    Parameters:
        z (ndarray): Complex vector (k,)
        method (str): Projection method
            - 'real_imag': [Re(z[i]), Im(z[j])] for indices i,j
            - 'magnitude_phase': [|z[i]|, arg(z[j])]
            - 'pca': PCA on real representation
        indices (tuple): Which indices to use

    Returns:
        ndarray: 2D real vector
    """
    if method == 'real_imag':
        i, j = indices
        return np.array([np.real(z[i]), np.imag(z[j])])

    elif method == 'magnitude_phase':
        i, j = indices
        return np.array([np.abs(z[i]), np.angle(z[j])])

    elif method == 'pca':
        # Convert complex to real
        z_real = np.concatenate([np.real(z), np.imag(z)])

        # Simple PCA: first two components
        # (for single vector, just return first two)
        return z_real[:2]

    else:
        raise ValueError(f"Unknown method: {method}")


def project_to_real_3d(
    z: np.ndarray,
    method: str = 'real_imag',
    indices: Tuple[int, int, int] = (0, 1, 2)
) -> np.ndarray:
    """
    Project complex vector to ℝ³.

    Parameters:
        z (ndarray): Complex vector (k,)
        method (str): Projection method
            - 'real_imag': [Re(z[i]), Im(z[j]), Re(z[k])]
            - 'magnitude': [|z[i]|, |z[j]|, |z[k]|]
        indices (tuple): Which indices to use

    Returns:
        ndarray: 3D real vector
    """
    if method == 'real_imag':
        i, j, k = indices
        return np.array([np.real(z[i]), np.imag(z[j]), np.real(z[k])])

    elif method == 'magnitude':
        i, j, k = indices
        return np.array([np.abs(z[i]), np.abs(z[j]), np.abs(z[k])])

    else:
        raise ValueError(f"Unknown method: {method}")


def trajectory_to_real_2d(
    trajectory: np.ndarray,
    method: str = 'real_imag',
    indices: Tuple[int, int] = (0, 1)
) -> np.ndarray:
    """
    Project complex trajectory to ℝ² for visualization.

    Parameters:
        trajectory (ndarray): (n_steps, k) complex array
        method (str): Projection method
        indices (tuple): Which indices to use

    Returns:
        ndarray: (n_steps, 2) real array
    """
    n_steps = trajectory.shape[0]
    result = np.zeros((n_steps, 2))

    for i in range(n_steps):
        result[i] = project_to_real_2d(trajectory[i], method=method, indices=indices)

    return result


def trajectory_to_real_3d(
    trajectory: np.ndarray,
    method: str = 'real_imag',
    indices: Tuple[int, int, int] = (0, 1, 2)
) -> np.ndarray:
    """
    Project complex trajectory to ℝ³ for visualization.

    Parameters:
        trajectory (ndarray): (n_steps, k) complex array
        method (str): Projection method
        indices (tuple): Which indices to use

    Returns:
        ndarray: (n_steps, 3) real array
    """
    n_steps = trajectory.shape[0]
    result = np.zeros((n_steps, 3))

    for i in range(n_steps):
        result[i] = project_to_real_3d(trajectory[i], method=method, indices=indices)

    return result


def create_parameter_slice(
    k: int,
    fixed_indices: List[int],
    fixed_values: List[complex],
    var_index_1: int,
    var_index_2: int,
    var_range_1: Tuple[float, float],
    var_range_2: Tuple[float, float],
    resolution: int = 100
) -> np.ndarray:
    """
    Create a 2D slice through parameter space ℂ^k.

    Parameters:
        k (int): Dimension
        fixed_indices (list): Indices to hold constant
        fixed_values (list): Values for fixed indices
        var_index_1 (int): First varying index
        var_index_2 (int): Second varying index
        var_range_1 (tuple): (min, max) for first variable (real part)
        var_range_2 (tuple): (min, max) for second variable (real part)
        resolution (int): Grid resolution

    Returns:
        ndarray: (resolution, resolution, k) array of parameter vectors
    """
    # Create meshgrid
    v1 = np.linspace(var_range_1[0], var_range_1[1], resolution)
    v2 = np.linspace(var_range_2[0], var_range_2[1], resolution)
    V1, V2 = np.meshgrid(v1, v2)

    # Initialize parameter array
    params = np.zeros((resolution, resolution, k), dtype=np.complex128)

    # Set fixed values
    for idx, val in zip(fixed_indices, fixed_values):
        params[:, :, idx] = val

    # Set varying values
    params[:, :, var_index_1] = V1 + 0j
    params[:, :, var_index_2] = V2 + 0j

    return params


def random_projection_matrix(
    k: int,
    target_dim: int = 2
) -> np.ndarray:
    """
    Generate random projection matrix for dimensionality reduction.

    Parameters:
        k (int): Source dimension
        target_dim (int): Target dimension

    Returns:
        ndarray: (target_dim, k) projection matrix
    """
    # Random Gaussian matrix
    P = np.random.randn(target_dim, k) + 1j * np.random.randn(target_dim, k)

    # Normalize rows
    for i in range(target_dim):
        P[i] = P[i] / np.linalg.norm(P[i])

    return P


def apply_projection(
    z: np.ndarray,
    P: np.ndarray
) -> np.ndarray:
    """
    Apply projection matrix to vector.

    Parameters:
        z (ndarray): (k,) complex vector
        P (ndarray): (m, k) projection matrix

    Returns:
        ndarray: (m,) projected vector
    """
    return P @ z


def stereographic_projection(
    z: np.ndarray,
    pole_index: int = -1
) -> np.ndarray:
    """
    Stereographic projection from ℂ^k to ℂ^{k-1}.

    Projects from north pole onto equatorial hyperplane.

    Parameters:
        z (ndarray): (k,) complex vector
        pole_index (int): Index to project from (default: last)

    Returns:
        ndarray: (k-1,) projected vector
    """
    k = len(z)

    if pole_index == -1:
        pole_index = k - 1

    # Denominator
    denom = 1.0 - z[pole_index]

    # Project
    indices = [i for i in range(k) if i != pole_index]
    z_proj = z[indices] / denom

    return z_proj


def inverse_stereographic_projection(
    w: np.ndarray,
    k: int,
    pole_index: int = -1
) -> np.ndarray:
    """
    Inverse stereographic projection from ℂ^{k-1} to ℂ^k.

    Parameters:
        w (ndarray): (k-1,) complex vector
        k (int): Target dimension
        pole_index (int): Where to insert projected coordinate

    Returns:
        ndarray: (k,) vector on unit sphere
    """
    if pole_index == -1:
        pole_index = k - 1

    # Denominator
    norm_sq = np.sum(np.abs(w)**2)
    denom = 1.0 + norm_sq

    # Construct full vector
    z = np.zeros(k, dtype=np.complex128)

    indices = [i for i in range(k) if i != pole_index]
    z[indices] = 2 * w / denom
    z[pole_index] = (norm_sq - 1) / denom

    return z


def local_coordinate_frame(
    z: np.ndarray,
    n_vectors: int = 2
) -> np.ndarray:
    """
    Construct local orthonormal frame at point z.

    Parameters:
        z (ndarray): (k,) base point
        n_vectors (int): Number of frame vectors

    Returns:
        ndarray: (n_vectors, k) orthonormal vectors
    """
    k = len(z)

    if n_vectors > k:
        raise ValueError(f"Cannot construct {n_vectors} frame vectors in dimension {k}")

    # Start with random vectors
    frame = np.random.randn(n_vectors, k) + 1j * np.random.randn(n_vectors, k)

    # Gram-Schmidt orthogonalization
    for i in range(n_vectors):
        for j in range(i):
            frame[i] -= np.vdot(frame[j], frame[i]) * frame[j]

        frame[i] /= np.linalg.norm(frame[i])

    return frame
