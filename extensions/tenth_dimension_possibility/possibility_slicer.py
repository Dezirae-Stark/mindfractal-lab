"""
Possibility Manifold Timeline Slicer

Extracts "timelines" (orbit branches) from the Possibility Manifold.
This is the mathematical realization of "slicing through all possible realities".

A timeline is a continuous path through 𝒫, representing a connected
sequence of dynamical states across parameter variations.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .possibility_manifold import ParameterPoint, PossibilityManifold


@dataclass
class OrbitBranch:
    """A timeline/branch through the possibility space"""

    points: List[ParameterPoint]
    orbits: List[np.ndarray]
    branch_id: int
    parent_id: Optional[int] = None


class TimelineSlicer:
    """
    Extract and analyze timeline branches from 𝒫

    Conceptually: "slicing through all possible realities"
    Mathematically: parametric curves through the manifold
    """

    def __init__(self, manifold: PossibilityManifold):
        self.manifold = manifold
        self.branches: List[OrbitBranch] = []

    def slice_parameter_line(
        self, start: ParameterPoint, end: ParameterPoint, n_steps: int = 20
    ) -> OrbitBranch:
        """
        Create a timeline by linearly interpolating parameters

        Parameters:
        -----------
        start, end : ParameterPoint
            Endpoints of parameter path
        n_steps : int
            Number of intermediate points

        Returns:
        --------
        branch : OrbitBranch
            The timeline branch
        """
        points = []
        orbits = []

        for t in np.linspace(0, 1, n_steps):
            # Interpolate point
            point = self._interpolate_points(start, end, t)

            # Compute orbit
            orbit = self.manifold.compute_orbit(point, steps=100)

            points.append(point)
            orbits.append(orbit)

        branch = OrbitBranch(points=points, orbits=orbits, branch_id=len(self.branches))
        self.branches.append(branch)
        return branch

    def slice_random_walk(
        self, start: ParameterPoint, n_steps: int = 20, step_size: float = 0.1
    ) -> OrbitBranch:
        """Random walk through parameter space"""
        points = [start]
        orbits = [self.manifold.compute_orbit(start, steps=100)]

        current = start.copy()
        for _ in range(n_steps - 1):
            # Random step in parameter space
            delta_c = step_size * (
                np.random.randn(self.manifold.dim) + 1j * np.random.randn(self.manifold.dim)
            )
            current.c = current.c + delta_c

            orbit = self.manifold.compute_orbit(current, steps=100)
            points.append(current.copy())
            orbits.append(orbit)

        branch = OrbitBranch(points=points, orbits=orbits, branch_id=len(self.branches))
        self.branches.append(branch)
        return branch

    def _interpolate_points(
        self, p1: ParameterPoint, p2: ParameterPoint, t: float
    ) -> ParameterPoint:
        """Linear interpolation between two points"""
        point = p1.copy()
        point.z0 = (1 - t) * p1.z0 + t * p2.z0
        point.c = (1 - t) * p1.c + t * p2.c
        if p1.A is not None and p2.A is not None:
            point.A = (1 - t) * p1.A + t * p2.A
        if p1.B is not None and p2.B is not None:
            point.B = (1 - t) * p1.B + t * p2.B
        if p1.W is not None and p2.W is not None:
            point.W = (1 - t) * p1.W + t * p2.W
        return point
