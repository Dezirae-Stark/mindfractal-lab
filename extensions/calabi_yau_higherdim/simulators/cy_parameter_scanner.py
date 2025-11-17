"""
Parameter Space Scanner for CY Systems

Tools for scanning parameter space and identifying interesting regions.
"""

import numpy as np
from typing import Dict, Callable, Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.cy_complex_dynamics import CYSystem
from models.cy_update_rules import construct_cy_system


class ParameterScanner:
    """
    Scans parameter space (c-space) for CY systems.
    """

    def __init__(self, k: int, epsilon: float = 0.01):
        """
        Initialize scanner.

        Parameters:
            k (int): Dimension
            epsilon (float): Nonlinearity parameter
        """
        self.k = k
        self.epsilon = epsilon

    def scan_2d_slice(
        self,
        fixed_indices: list,
        fixed_values: list,
        var_index_1: int,
        var_index_2: int,
        ranges: Tuple[Tuple[float, float], Tuple[float, float]],
        resolution: int = 100,
        criterion: str = 'escape_time',
        n_steps: int = 500,
        z0: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Scan a 2D slice of parameter space.

        Parameters:
            fixed_indices (list): Indices to fix
            fixed_values (list): Values for fixed indices
            var_index_1 (int): First varying index
            var_index_2 (int): Second varying index
            ranges (tuple): ((min1, max1), (min2, max2))
            resolution (int): Grid resolution
            criterion (str): 'escape_time', 'final_norm', 'bounded'
            n_steps (int): Steps to simulate
            z0 (ndarray, optional): Initial condition

        Returns:
            ndarray: (resolution, resolution) scan result
        """
        if z0 is None:
            z0 = 0.1 * np.ones(self.k, dtype=np.complex128)

        (min1, max1), (min2, max2) = ranges

        v1_vals = np.linspace(min1, max1, resolution)
        v2_vals = np.linspace(min2, max2, resolution)

        result = np.zeros((resolution, resolution))

        for i, v1 in enumerate(v1_vals):
            for j, v2 in enumerate(v2_vals):
                # Construct c vector
                c = np.zeros(self.k, dtype=np.complex128)
                for idx, val in zip(fixed_indices, fixed_values):
                    c[idx] = val
                c[var_index_1] = v1
                c[var_index_2] = v2

                # Create system
                system = construct_cy_system(self.k, epsilon=self.epsilon, c_value=c)

                # Compute criterion
                if criterion == 'escape_time':
                    bounded, escape_t = system.is_bounded(z0, n_steps=n_steps)
                    result[j, i] = escape_t
                elif criterion == 'final_norm':
                    traj = system.trajectory(z0, n_steps, return_states=False)
                    result[j, i] = np.linalg.norm(traj[-1])
                elif criterion == 'bounded':
                    bounded, _ = system.is_bounded(z0, n_steps=n_steps)
                    result[j, i] = 1.0 if bounded else 0.0

        return result


def scan_2d_slice(
    k: int,
    var_index_1: int = 0,
    var_index_2: int = 1,
    ranges: Tuple[Tuple[float, float], Tuple[float, float]] = ((-1.0, 1.0), (-1.0, 1.0)),
    resolution: int = 100,
    **kwargs
) -> np.ndarray:
    """
    Quick 2D parameter scan.

    Parameters:
        k (int): Dimension
        var_index_1 (int): First varying index
        var_index_2 (int): Second varying index
        ranges (tuple): Parameter ranges
        resolution (int): Grid resolution
        **kwargs: Passed to ParameterScanner.scan_2d_slice

    Returns:
        ndarray: Scan result
    """
    scanner = ParameterScanner(k)
    fixed_indices = [i for i in range(k) if i not in [var_index_1, var_index_2]]
    fixed_values = [0.0] * len(fixed_indices)

    return scanner.scan_2d_slice(
        fixed_indices, fixed_values,
        var_index_1, var_index_2,
        ranges, resolution,
        **kwargs
    )
