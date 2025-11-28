"""
3D State Space Extension

Extends the fractal dynamics model to 3D:
    x ∈ ℝ³
    c ∈ ℝ³
    x_{n+1} = A x_n + B tanh(W x_n) + c

This provides richer dynamics and additional dimensions for consciousness modeling.
"""

from typing import Optional

import numpy as np


class FractalDynamicsModel3D:
    """3D Fractal Dynamical System"""

    def __init__(
        self,
        A: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        W: Optional[np.ndarray] = None,
        c: Optional[np.ndarray] = None,
    ):
        self.dim = 3

        self.A = A if A is not None else 0.9 * np.eye(3)
        self.B = (
            B if B is not None else np.array([[0.2, 0.1, 0.1], [0.1, 0.2, 0.1], [0.1, 0.1, 0.2]])
        )
        self.W = W if W is not None else np.eye(3) + 0.1 * np.random.randn(3, 3)
        self.c = c if c is not None else np.array([0.1, 0.1, 0.1])

        assert self.A.shape == (3, 3)
        assert self.B.shape == (3, 3)
        assert self.W.shape == (3, 3)
        assert self.c.shape == (3,)

    def step(self, x: np.ndarray) -> np.ndarray:
        """Single dynamics step"""
        x = np.array(x, dtype=np.float64)
        assert x.shape == (3,)

        linear = self.A @ x
        nonlinear = self.B @ np.tanh(self.W @ x)
        return linear + nonlinear + self.c

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian matrix at x"""
        wx = self.W @ x
        sech_sq = 1.0 - np.tanh(wx) ** 2
        D = np.diag(sech_sq)
        return self.A + self.B @ D @ self.W

    def __repr__(self):
        return f"FractalDynamicsModel3D(dim=3, c_norm={np.linalg.norm(self.c):.3f})"
