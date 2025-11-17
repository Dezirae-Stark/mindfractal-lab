"""3D Simulation Functions"""

import numpy as np
from .model_3d import FractalDynamicsModel3D


def simulate_orbit_3d(
    model: FractalDynamicsModel3D,
    x0: np.ndarray,
    n_steps: int = 1000
) -> np.ndarray:
    """Simulate 3D orbit"""
    x = np.array(x0, dtype=np.float64)
    trajectory = np.zeros((n_steps, 3))
    trajectory[0] = x

    for i in range(1, n_steps):
        x = model.step(x)
        trajectory[i] = x

    return trajectory


def lyapunov_spectrum_3d(
    model: FractalDynamicsModel3D,
    x0: np.ndarray,
    n_steps: int = 5000,
    transient: int = 1000
) -> np.ndarray:
    """Compute full Lyapunov spectrum (3 exponents)"""
    x = np.array(x0, dtype=np.float64)

    # Discard transient
    for _ in range(transient):
        x = model.step(x)

    # Initialize orthonormal basis
    Q = np.eye(3)
    lyap_sum = np.zeros(3)

    for _ in range(n_steps):
        J = model.jacobian(x)

        # Evolve tangent vectors
        Q = J @ Q

        # QR decomposition for orthonormalization
        Q, R = np.linalg.qr(Q)

        # Accumulate log of diagonal elements
        lyap_sum += np.log(np.abs(np.diag(R)))

        x = model.step(x)

    return lyap_sum / n_steps
