"""
Boundary Explorer for CY Parameter Space

Tools for probing boundaries between bounded/unbounded regions.
"""

import numpy as np
from typing import List, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.cy_complex_dynamics import CYSystem
from models.cy_update_rules import construct_cy_system


class BoundaryExplorer:
    """
    Explores boundaries in parameter space.
    """

    def __init__(self, k: int, epsilon: float = 0.01):
        self.k = k
        self.epsilon = epsilon

    def bisect_boundary(
        self,
        c_bounded: np.ndarray,
        c_unbounded: np.ndarray,
        tolerance: float = 1e-3,
        max_iter: int = 20,
        n_steps: int = 500,
        z0: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Find boundary point via bisection.

        Parameters:
            c_bounded (ndarray): Parameter in bounded region
            c_unbounded (ndarray): Parameter in unbounded region
            tolerance (float): Convergence tolerance
            max_iter (int): Maximum bisection iterations
            n_steps (int): Steps for boundedness test
            z0 (ndarray, optional): Initial state

        Returns:
            ndarray: Approximate boundary parameter
        """
        if z0 is None:
            z0 = np.zeros(self.k, dtype=np.complex128)

        c_low = c_bounded.copy()
        c_high = c_unbounded.copy()

        for _ in range(max_iter):
            if np.linalg.norm(c_high - c_low) < tolerance:
                break

            c_mid = (c_low + c_high) / 2

            system = construct_cy_system(self.k, epsilon=self.epsilon, c_value=c_mid)
            bounded, _ = system.is_bounded(z0, n_steps=n_steps)

            if bounded:
                c_low = c_mid
            else:
                c_high = c_mid

        return (c_low + c_high) / 2

    def sample_boundary(
        self,
        n_samples: int = 100,
        seed_points: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    ) -> List[np.ndarray]:
        """
        Sample multiple boundary points.

        Parameters:
            n_samples (int): Number of samples
            seed_points (list, optional): List of (bounded, unbounded) pairs

        Returns:
            list: Boundary points
        """
        if seed_points is None:
            # Generate random seed points
            seed_points = []
            for _ in range(n_samples):
                c_bounded = 0.1 * (np.random.randn(self.k) + 1j * np.random.randn(self.k))
                c_unbounded = 5.0 * (np.random.randn(self.k) + 1j * np.random.randn(self.k))
                seed_points.append((c_bounded, c_unbounded))

        boundary_points = []
        for c_b, c_u in seed_points:
            c_boundary = self.bisect_boundary(c_b, c_u)
            boundary_points.append(c_boundary)

        return boundary_points


def find_boundary_points(
    k: int,
    n_points: int = 10,
    **kwargs
) -> List[np.ndarray]:
    """
    Quick boundary point finding.

    Parameters:
        k (int): Dimension
        n_points (int): Number of boundary points
        **kwargs: Passed to BoundaryExplorer.sample_boundary

    Returns:
        list: Boundary points
    """
    explorer = BoundaryExplorer(k)
    return explorer.sample_boundary(n_samples=n_points, **kwargs)
