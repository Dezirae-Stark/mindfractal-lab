"""
Orbit Simulation for CY Systems

Tools for simulating long orbits, analyzing trajectories,
and characterizing attractor behavior.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.cy_complex_dynamics import CYSystem, CYState


def simulate_orbit(
    system: CYSystem,
    z0: np.ndarray,
    n_steps: int = 1000,
    return_format: str = 'array'
) -> np.ndarray:
    """
    Simulate a single orbit.

    Parameters:
        system (CYSystem): Dynamical system
        z0 (ndarray): Initial condition (k,)
        n_steps (int): Number of steps
        return_format (str): 'array' or 'states'

    Returns:
        ndarray or list: Trajectory
    """
    return system.trajectory(z0, n_steps, return_states=(return_format=='states'))


def simulate_multiple_orbits(
    system: CYSystem,
    initial_conditions: List[np.ndarray],
    n_steps: int = 1000
) -> List[np.ndarray]:
    """
    Simulate multiple orbits in parallel.

    Parameters:
        system (CYSystem): Dynamical system
        initial_conditions (list): List of initial states
        n_steps (int): Steps per orbit

    Returns:
        list: List of trajectory arrays
    """
    orbits = []
    for z0 in initial_conditions:
        orbit = simulate_orbit(system, z0, n_steps, return_format='array')
        orbits.append(orbit)
    return orbits


class OrbitAnalyzer:
    """
    Analyzes properties of simulated orbits.
    """

    def __init__(self, trajectory: np.ndarray):
        """
        Initialize analyzer with trajectory.

        Parameters:
            trajectory (ndarray): (n_steps, k) complex array
        """
        self.trajectory = trajectory
        self.n_steps, self.k = trajectory.shape

    def compute_norms(self) -> np.ndarray:
        """Compute ||z(t)|| for all t"""
        return np.linalg.norm(self.trajectory, axis=1)

    def is_bounded(self, threshold: float = 10.0) -> bool:
        """Check if orbit remains bounded"""
        norms = self.compute_norms()
        return np.all(norms < threshold)

    def escape_time(self, threshold: float = 10.0) -> Optional[int]:
        """Find first time ||z|| > threshold"""
        norms = self.compute_norms()
        escape_indices = np.where(norms > threshold)[0]
        return int(escape_indices[0]) if len(escape_indices) > 0 else None

    def final_state_statistics(self, window: int = 100) -> Dict[str, float]:
        """Compute statistics of final window"""
        final_window = self.trajectory[-window:]
        norms = np.linalg.norm(final_window, axis=1)

        return {
            'mean_norm': float(np.mean(norms)),
            'std_norm': float(np.std(norms)),
            'min_norm': float(np.min(norms)),
            'max_norm': float(np.max(norms))
        }

    def estimate_period(self, tolerance: float = 0.1) -> Optional[int]:
        """
        Estimate period for periodic orbits.

        Returns:
            int or None: Period if detected
        """
        # Check last half of trajectory
        half = self.n_steps // 2
        traj = self.trajectory[half:]

        # Try periods from 2 to 100
        for period in range(2, min(101, len(traj) // 2)):
            # Check if traj[i] ≈ traj[i + period]
            diffs = []
            for i in range(len(traj) - period):
                diff = np.linalg.norm(traj[i] - traj[i + period])
                diffs.append(diff)

            if np.mean(diffs) < tolerance:
                return period

        return None

    def classify_attractor(self) -> str:
        """
        Classify attractor type.

        Returns:
            str: 'fixed_point', 'periodic', 'chaotic', or 'unbounded'
        """
        if not self.is_bounded():
            return 'unbounded'

        stats = self.final_state_statistics(window=100)
        if stats['std_norm'] < 0.01:
            return 'fixed_point'

        period = self.estimate_period()
        if period is not None:
            return 'periodic'

        return 'chaotic'
