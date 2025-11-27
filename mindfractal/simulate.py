"""
Simulation Engine for Fractal Dynamics

This module provides functions for:
- Simulating orbits (trajectories) of the dynamical system
- Finding fixed points
- Analyzing stability
- Computing basin of attraction boundaries
"""

from typing import List, Optional, Tuple

import numpy as np

from .model import FractalDynamicsModel


def simulate_orbit(
    model: FractalDynamicsModel,
    x0: np.ndarray,
    n_steps: int = 1000,
    return_all: bool = True,
) -> np.ndarray:
    """
    Simulate an orbit starting from initial condition x0.

    Args:
        model: FractalDynamicsModel instance
        x0: Initial state (2D vector)
        n_steps: Number of time steps to simulate
        return_all: If True, return full trajectory; if False, return only final state

    Returns:
        If return_all=True: array of shape (n_steps, 2) containing full trajectory
        If return_all=False: final state (2D vector)

    Example:
        >>> model = FractalDynamicsModel()
        >>> x0 = np.array([0.5, 0.5])
        >>> trajectory = simulate_orbit(model, x0, n_steps=1000)
        >>> print(trajectory.shape)  # (1000, 2)
    """
    x = np.array(x0, dtype=np.float64)

    if return_all:
        trajectory = np.zeros((n_steps, model.dim))
        trajectory[0] = x

        for i in range(1, n_steps):
            x = model.step(x)
            trajectory[i] = x

        return trajectory
    else:
        for _ in range(n_steps):
            x = model.step(x)
        return x


def find_fixed_points(
    model: FractalDynamicsModel,
    initial_guesses: Optional[List[np.ndarray]] = None,
    tolerance: float = 1e-6,
    max_iter: int = 1000,
) -> List[Tuple[np.ndarray, bool]]:
    """
    Find fixed points of the dynamical system using Newton's method.

    A fixed point satisfies: x* = f(x*)

    Args:
        model: FractalDynamicsModel instance
        initial_guesses: List of initial guesses for Newton's method
                        If None, use a default grid of guesses
        tolerance: Convergence tolerance
        max_iter: Maximum iterations for Newton's method

    Returns:
        List of tuples: (fixed_point, is_stable)
        where is_stable is determined by eigenvalues of Jacobian

    Example:
        >>> model = FractalDynamicsModel()
        >>> fixed_points = find_fixed_points(model)
        >>> for fp, stable in fixed_points:
        ...     print(f"Fixed point: {fp}, Stable: {stable}")
    """
    if initial_guesses is None:
        # Default grid of initial guesses
        initial_guesses = [
            np.array([x, y])
            for x in np.linspace(-2, 2, 5)
            for y in np.linspace(-2, 2, 5)
        ]

    fixed_points = []
    found_points = []

    for x0 in initial_guesses:
        x = np.array(x0, dtype=np.float64)

        # Newton's method: solve f(x) - x = 0
        for iteration in range(max_iter):
            fx = model.step(x)
            residual = fx - x

            if np.linalg.norm(residual) < tolerance:
                # Check if we already found this fixed point
                is_new = True
                for existing_fp, _ in found_points:
                    if np.linalg.norm(x - existing_fp) < tolerance * 10:
                        is_new = False
                        break

                if is_new:
                    # Determine stability from Jacobian eigenvalues
                    J = model.jacobian(x)
                    eigenvalues = np.linalg.eigvals(J)
                    is_stable = np.all(np.abs(eigenvalues) < 1.0)

                    found_points.append((x.copy(), is_stable))
                break

            # Newton step: x := x - (J - I)^{-1} (f(x) - x)
            J = model.jacobian(x)
            A = J - np.eye(model.dim)

            try:
                delta = np.linalg.solve(A, residual)
                x = x - delta
            except np.linalg.LinAlgError:
                # Singular matrix, try different initial guess
                break

    return found_points


def compute_attractor_type(
    model: FractalDynamicsModel,
    x0: np.ndarray,
    n_steps: int = 5000,
    transient: int = 1000,
    tolerance: float = 1e-3,
) -> str:
    """
    Classify the type of attractor reached from initial condition x0.

    Types:
        - "fixed_point": trajectory converges to a fixed point
        - "limit_cycle": trajectory forms a periodic orbit
        - "chaotic": trajectory exhibits sensitive dependence (positive Lyapunov)
        - "unbounded": trajectory diverges

    Args:
        model: FractalDynamicsModel instance
        x0: Initial condition
        n_steps: Total steps to simulate
        transient: Steps to discard before analysis
        tolerance: Tolerance for fixed point detection

    Returns:
        String describing attractor type
    """
    # Simulate with transient removal
    trajectory = simulate_orbit(model, x0, n_steps=n_steps)
    trajectory_post_transient = trajectory[transient:]

    # Check for unbounded growth
    max_norm = np.max(np.linalg.norm(trajectory_post_transient, axis=1))
    if max_norm > 100.0:
        return "unbounded"

    # Check for fixed point
    final_state = trajectory[-1]
    next_state = model.step(final_state)
    if np.linalg.norm(next_state - final_state) < tolerance:
        return "fixed_point"

    # Estimate Lyapunov exponent
    lyap = model.lyapunov_exponent_estimate(x0, n_steps=1000, transient=transient)

    if lyap > 0.01:
        return "chaotic"
    elif lyap < -0.01:
        return "fixed_point"  # Converging
    else:
        return "limit_cycle"  # Near-zero Lyapunov


def basin_of_attraction_sample(
    model: FractalDynamicsModel,
    x_range: Tuple[float, float] = (-2.0, 2.0),
    y_range: Tuple[float, float] = (-2.0, 2.0),
    resolution: int = 100,
    n_steps: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample the basin of attraction on a grid.

    For each initial condition, determine which attractor it reaches.

    Args:
        model: FractalDynamicsModel instance
        x_range: Range for x-coordinate
        y_range: Range for y-coordinate
        resolution: Grid resolution
        n_steps: Number of steps to simulate for each initial condition

    Returns:
        Tuple of (X, Y, basin_labels) where:
            X, Y: meshgrid coordinates
            basin_labels: array of attractor labels (integers)
    """
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x_vals, y_vals)

    basin_labels = np.zeros((resolution, resolution), dtype=int)

    # Find fixed points first
    fixed_points_data = find_fixed_points(model)
    fixed_points = [fp for fp, _ in fixed_points_data]

    for i in range(resolution):
        for j in range(resolution):
            x0 = np.array([X[i, j], Y[i, j]])

            # Simulate to final state
            final_state = simulate_orbit(model, x0, n_steps=n_steps, return_all=False)

            # Determine which fixed point it's closest to
            if len(fixed_points) > 0:
                distances = [np.linalg.norm(final_state - fp) for fp in fixed_points]
                basin_labels[i, j] = np.argmin(distances)
            else:
                basin_labels[i, j] = 0

    return X, Y, basin_labels


def poincare_section(
    model: FractalDynamicsModel,
    x0: np.ndarray,
    n_steps: int = 10000,
    plane_coord: int = 0,
    plane_value: float = 0.0,
    direction: int = 1,
) -> np.ndarray:
    """
    Compute Poincaré section for analyzing periodic/chaotic orbits.

    Records states when trajectory crosses a hyperplane.

    Args:
        model: FractalDynamicsModel instance
        x0: Initial condition
        n_steps: Number of steps to simulate
        plane_coord: Coordinate index defining the plane (0 or 1)
        plane_value: Value of the plane
        direction: Crossing direction (+1 or -1)

    Returns:
        Array of Poincaré section points
    """
    trajectory = simulate_orbit(model, x0, n_steps=n_steps)
    poincare_points = []

    for i in range(1, len(trajectory)):
        x_prev = trajectory[i - 1]
        x_curr = trajectory[i]

        # Check if crossing occurred
        val_prev = x_prev[plane_coord] - plane_value
        val_curr = x_curr[plane_coord] - plane_value

        if direction * val_prev < 0 and direction * val_curr >= 0:
            poincare_points.append(x_curr)

    return np.array(poincare_points) if poincare_points else np.array([])
