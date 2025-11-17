"""
Core 2D Fractal Dynamical Consciousness Model

This module implements the fundamental discrete-time nonlinear dynamical system:

    x_{n+1} = A x_n + B tanh(W x_n) + c

where:
    x ∈ ℝ²     : state vector (consciousness state)
    A ∈ ℝ²ˣ²   : linear feedback matrix
    B ∈ ℝ²ˣ²   : nonlinear coupling matrix
    W ∈ ℝ²ˣ²   : weight matrix for tanh activation
    c ∈ ℝ²     : external drive / parameter vector

The model exhibits:
- Fixed points (stable consciousness states)
- Limit cycles (oscillatory states)
- Chaotic attractors (fragmented / fluid states)
- Fractal basin boundaries (metastability regions)
"""

import numpy as np
from typing import Tuple, Optional


class FractalDynamicsModel:
    """
    2D Fractal Dynamical Consciousness Model

    This class encapsulates the core dynamical system with configurable parameters.
    Default parameters are chosen to produce rich dynamics including chaos and
    fractal basin boundaries.

    Attributes:
        A (np.ndarray): 2x2 linear feedback matrix
        B (np.ndarray): 2x2 nonlinear coupling matrix
        W (np.ndarray): 2x2 weight matrix for tanh activation
        c (np.ndarray): 2D external drive vector
        dim (int): Dimensionality (always 2 for this model)
    """

    def __init__(
        self,
        A: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        W: Optional[np.ndarray] = None,
        c: Optional[np.ndarray] = None
    ):
        """
        Initialize the 2D fractal dynamics model.

        Args:
            A: 2x2 linear feedback matrix (default: 0.9 * I)
            B: 2x2 nonlinear coupling matrix (default: small off-diagonal coupling)
            W: 2x2 weight matrix (default: identity-like with small perturbation)
            c: 2D external drive vector (default: small positive values)

        The default parameters are scientifically chosen to produce:
        - Weak linear damping (A slightly < I)
        - Moderate nonlinear coupling
        - Rich attractor structure
        - Fractal basin boundaries
        """
        self.dim = 2

        # Default A: weak damping
        if A is None:
            self.A = np.array([
                [0.9, 0.0],
                [0.0, 0.9]
            ], dtype=np.float64)
        else:
            self.A = np.array(A, dtype=np.float64)
            assert self.A.shape == (2, 2), "A must be 2x2"

        # Default B: off-diagonal coupling
        if B is None:
            self.B = np.array([
                [0.2, 0.3],
                [0.3, 0.2]
            ], dtype=np.float64)
        else:
            self.B = np.array(B, dtype=np.float64)
            assert self.B.shape == (2, 2), "B must be 2x2"

        # Default W: near-identity with perturbation
        if W is None:
            self.W = np.array([
                [1.0, 0.1],
                [0.1, 1.0]
            ], dtype=np.float64)
        else:
            self.W = np.array(W, dtype=np.float64)
            assert self.W.shape == (2, 2), "W must be 2x2"

        # Default c: small external drive
        if c is None:
            self.c = np.array([0.1, 0.1], dtype=np.float64)
        else:
            self.c = np.array(c, dtype=np.float64)
            assert self.c.shape == (2,), "c must be 2D vector"

    def step(self, x: np.ndarray) -> np.ndarray:
        """
        Compute one step of the dynamics: x_{n+1} = f(x_n)

        Args:
            x: Current state vector (2D)

        Returns:
            Next state vector (2D)

        Mathematical formula:
            x_{n+1} = A x_n + B tanh(W x_n) + c
        """
        x = np.array(x, dtype=np.float64)
        assert x.shape == (2,), "State must be 2D vector"

        # Linear term
        linear_term = self.A @ x

        # Nonlinear term: B tanh(W x)
        wx = self.W @ x
        nonlinear_term = self.B @ np.tanh(wx)

        # External drive
        x_next = linear_term + nonlinear_term + self.c

        return x_next

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the Jacobian matrix at state x.

        J(x) = A + B diag(sech²(W x)) W

        This is used for stability analysis and fixed point classification.

        Args:
            x: State vector (2D)

        Returns:
            2x2 Jacobian matrix
        """
        x = np.array(x, dtype=np.float64)
        wx = self.W @ x

        # sech²(u) = 1 - tanh²(u)
        tanh_wx = np.tanh(wx)
        sech_sq = 1.0 - tanh_wx**2

        # Diagonal matrix of sech²(W x)
        D = np.diag(sech_sq)

        # J = A + B D W
        J = self.A + self.B @ D @ self.W

        return J

    def lyapunov_exponent_estimate(
        self,
        x0: np.ndarray,
        n_steps: int = 5000,
        transient: int = 1000
    ) -> float:
        """
        Estimate the largest Lyapunov exponent using the Jacobian method.

        A positive Lyapunov exponent indicates chaotic dynamics.

        Args:
            x0: Initial state
            n_steps: Number of iterations for estimation
            transient: Number of transient steps to discard

        Returns:
            Estimated largest Lyapunov exponent
        """
        x = np.array(x0, dtype=np.float64)

        # Discard transient
        for _ in range(transient):
            x = self.step(x)

        # Accumulate log of Jacobian norms
        log_sum = 0.0
        v = np.array([1.0, 0.0])  # Initial tangent vector

        for _ in range(n_steps):
            J = self.jacobian(x)
            v = J @ v
            norm_v = np.linalg.norm(v)

            if norm_v > 0:
                log_sum += np.log(norm_v)
                v = v / norm_v  # Renormalize

            x = self.step(x)

        return log_sum / n_steps

    def energy(self, x: np.ndarray) -> float:
        """
        Compute a Lyapunov-like energy function for the system.

        This is a heuristic energy function that can be used for
        analyzing attractor basins.

        Args:
            x: State vector

        Returns:
            Energy value (scalar)
        """
        # Quadratic form: E(x) = ½ xᵀ x + potential term
        kinetic = 0.5 * np.dot(x, x)

        # Potential from tanh nonlinearity
        wx = self.W @ x
        potential = -np.sum(np.log(np.cosh(wx)))

        return kinetic + potential

    def __repr__(self) -> str:
        return (
            f"FractalDynamicsModel(dim={self.dim}, "
            f"A_trace={np.trace(self.A):.3f}, "
            f"c_norm={np.linalg.norm(self.c):.3f})"
        )
