"""
Update Rule Construction for CY Systems

Utilities for constructing unitary matrices, nonlinear terms,
and parameter vectors for CY-inspired dynamics.
"""

import numpy as np
from typing import Optional, Tuple
from .cy_complex_dynamics import CYSystem


def generate_unitary_matrix(
    k: int,
    method: str = 'random',
    **kwargs
) -> np.ndarray:
    """
    Generate a k×k unitary matrix.

    Parameters:
        k (int): Matrix dimension
        method (str): Generation method
            - 'random': Random unitary via QR decomposition
            - 'rotation': Composition of 2D rotations
            - 'diagonal': Diagonal with phases
        **kwargs: Method-specific parameters

    Returns:
        ndarray: k×k unitary matrix
    """
    if method == 'random':
        return _generate_random_unitary(k)
    elif method == 'rotation':
        return _generate_rotation_unitary(k, **kwargs)
    elif method == 'diagonal':
        return _generate_diagonal_unitary(k, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


def _generate_random_unitary(k: int) -> np.ndarray:
    """Generate random unitary via QR decomposition"""
    A = np.random.randn(k, k) + 1j * np.random.randn(k, k)
    Q, R = np.linalg.qr(A)
    d = np.diagonal(R)
    ph = d / np.abs(d)
    Q = Q @ np.diag(ph)
    return Q


def _generate_rotation_unitary(
    k: int,
    angles: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Generate unitary as composition of 2D rotations.

    Parameters:
        k (int): Dimension (must be even)
        angles (ndarray, optional): Rotation angles (k//2,)

    Returns:
        ndarray: k×k unitary matrix
    """
    if k % 2 != 0:
        raise ValueError("Rotation method requires even dimension")

    if angles is None:
        angles = 2 * np.pi * np.random.rand(k // 2)

    U = np.eye(k, dtype=np.complex128)

    for i, theta in enumerate(angles):
        idx1, idx2 = 2*i, 2*i+1
        c, s = np.cos(theta), np.sin(theta)
        U[idx1, idx1] = c + 1j*s
        U[idx1, idx2] = 0
        U[idx2, idx1] = 0
        U[idx2, idx2] = c - 1j*s

    return U


def _generate_diagonal_unitary(
    k: int,
    phases: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Generate diagonal unitary matrix.

    Parameters:
        k (int): Dimension
        phases (ndarray, optional): Phase angles (k,)

    Returns:
        ndarray: k×k diagonal unitary
    """
    if phases is None:
        phases = 2 * np.pi * np.random.rand(k)

    return np.diag(np.exp(1j * phases))


def construct_cy_system(
    k: int,
    epsilon: float = 0.01,
    unitary_method: str = 'random',
    c_scale: float = 0.1,
    c_value: Optional[np.ndarray] = None,
    **unitary_kwargs
) -> CYSystem:
    """
    Construct a CY system with specified parameters.

    Parameters:
        k (int): Dimension
        epsilon (float): Nonlinearity strength
        unitary_method (str): Method for generating U
        c_scale (float): Scale for random c (if c_value not provided)
        c_value (ndarray, optional): Explicit c vector
        **unitary_kwargs: Passed to generate_unitary_matrix

    Returns:
        CYSystem: Constructed system
    """
    U = generate_unitary_matrix(k, method=unitary_method, **unitary_kwargs)

    if c_value is None:
        c = c_scale * (np.random.randn(k) + 1j * np.random.randn(k))
    else:
        c = np.array(c_value, dtype=np.complex128)

    return CYSystem(k=k, U=U, epsilon=epsilon, c=c)


def construct_symmetric_system(
    k: int,
    epsilon: float = 0.01
) -> CYSystem:
    """
    Construct a CY system with additional symmetry.

    Uses diagonal unitary with equally-spaced phases.

    Parameters:
        k (int): Dimension
        epsilon (float): Nonlinearity strength

    Returns:
        CYSystem: Symmetric system
    """
    phases = 2 * np.pi * np.arange(k) / k
    U = np.diag(np.exp(1j * phases))

    # Symmetric c: real and equal magnitude
    c = 0.1 * np.ones(k, dtype=np.complex128)

    return CYSystem(k=k, U=U, epsilon=epsilon, c=c)


def perturb_unitary(
    U: np.ndarray,
    perturbation_strength: float = 0.01
) -> np.ndarray:
    """
    Perturb a unitary matrix slightly.

    Adds a small Hermitian perturbation and re-unitarizes via matrix exponential.

    Parameters:
        U (ndarray): k×k unitary matrix
        perturbation_strength (float): Strength of perturbation

    Returns:
        ndarray: Perturbed unitary matrix
    """
    k = U.shape[0]

    # Generate Hermitian perturbation
    H = np.random.randn(k, k) + 1j * np.random.randn(k, k)
    H = (H + H.conj().T) / 2  # Make Hermitian

    # Perturb
    U_pert = U @ (np.eye(k) + perturbation_strength * H)

    # Re-unitarize via polar decomposition
    U_final, _ = np.linalg.qr(U_pert)

    return U_final


def verify_unitarity(U: np.ndarray, tol: float = 1e-10) -> Tuple[bool, float]:
    """
    Verify that U is unitary.

    Checks if U† U ≈ I.

    Parameters:
        U (ndarray): Matrix to check
        tol (float): Tolerance

    Returns:
        (bool, float): (is_unitary, max_error)
    """
    k = U.shape[0]
    I = np.eye(k, dtype=np.complex128)
    UdaggerU = U.conj().T @ U

    error = np.max(np.abs(UdaggerU - I))
    is_unitary = error < tol

    return is_unitary, float(error)
