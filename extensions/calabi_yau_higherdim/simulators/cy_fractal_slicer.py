"""
Fractal Slicer for CY Parameter Space

Generates Mandelbrot-like visualizations of parameter space slices.
"""

import numpy as np
from typing import Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.cy_complex_dynamics import CYSystem
from models.cy_update_rules import construct_cy_system


class FractalSlicer:
    """
    Generates fractal slices through high-dimensional parameter space.
    """

    def __init__(self, k: int, epsilon: float = 0.01):
        self.k = k
        self.epsilon = epsilon

    def generate_slice(
        self,
        slice_indices: Tuple[int, int] = (0, 1),
        c_ranges: Tuple[Tuple[float, float], Tuple[float, float]] = ((-2, 2), (-2, 2)),
        resolution: int = 500,
        max_iter: int = 100,
        escape_radius: float = 10.0,
        z0: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate fractal slice.

        Parameters:
            slice_indices (tuple): Which c indices to vary
            c_ranges (tuple): Ranges for the two varying dimensions
            resolution (int): Grid resolution
            max_iter (int): Maximum iterations
            escape_radius (float): Escape threshold
            z0 (ndarray, optional): Initial state

        Returns:
            ndarray: (resolution, resolution) escape time map
        """
        if z0 is None:
            z0 = np.zeros(self.k, dtype=np.complex128)

        idx1, idx2 = slice_indices
        (c1_min, c1_max), (c2_min, c2_max) = c_ranges

        c1_vals = np.linspace(c1_min, c1_max, resolution)
        c2_vals = np.linspace(c2_min, c2_max, resolution)

        fractal = np.zeros((resolution, resolution))

        for i, c1 in enumerate(c1_vals):
            for j, c2 in enumerate(c2_vals):
                c = np.zeros(self.k, dtype=np.complex128)
                c[idx1] = c1
                c[idx2] = c2

                system = construct_cy_system(self.k, epsilon=self.epsilon, c_value=c)
                bounded, escape_t = system.is_bounded(
                    z0, n_steps=max_iter,
                    escape_radius=escape_radius,
                    check_interval=1
                )

                fractal[j, i] = escape_t

        return fractal


def generate_fractal_slice(
    k: int = 3,
    slice_indices: Tuple[int, int] = (0, 1),
    resolution: int = 500,
    **kwargs
) -> np.ndarray:
    """
    Quick fractal slice generation.

    Parameters:
        k (int): Dimension
        slice_indices (tuple): Which indices to vary
        resolution (int): Grid resolution
        **kwargs: Passed to FractalSlicer.generate_slice

    Returns:
        ndarray: Fractal map
    """
    slicer = FractalSlicer(k)
    return slicer.generate_slice(slice_indices=slice_indices, resolution=resolution, **kwargs)
