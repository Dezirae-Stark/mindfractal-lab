"""
Calabi-Yau Inspired Complex Dynamics

Implements a higher-dimensional complex-valued dynamical system
inspired by the mathematical structure of Calabi-Yau manifolds.

Mathematical Model:
    z_{n+1} = U z_n + ε (z_n ⊙ z_n) + c

Where:
    - z ∈ ℂ^k (k-dimensional complex state space)
    - c ∈ ℂ^k (complex parameter vector)
    - U: k×k unitary (or approximately unitary) matrix
    - ε: small real scalar (controls nonlinearity strength)
    - ⊙: element-wise (Hadamard) product

DISCLAIMER: This is a CONCEPTUAL DYNAMICAL MODEL for research purposes.
It is NOT a physical theory of spacetime or consciousness.
"""

import numpy as np
from typing import Optional, Tuple, Union


class CYState:
    """
    Represents a state in the CY-inspired complex state space.

    Attributes:
        z (ndarray): Complex state vector of shape (k,)
        k (int): Dimension of the complex space
    """

    def __init__(self, z: np.ndarray):
        """
        Initialize a CY state.

        Parameters:
            z (ndarray): Complex vector of shape (k,)
        """
        if not np.iscomplexobj(z):
            # Convert to complex if needed
            z = z.astype(np.complex128)

        self.z = np.array(z, dtype=np.complex128)
        self.k = len(self.z)

    def __repr__(self):
        return f"CYState(k={self.k}, ||z||={self.norm():.4f})"

    def norm(self) -> float:
        """Compute the Euclidean norm ||z||"""
        return np.linalg.norm(self.z)

    def copy(self):
        """Return a deep copy of this state"""
        return CYState(self.z.copy())

    def to_array(self) -> np.ndarray:
        """Return the underlying complex array"""
        return self.z.copy()


class CYSystem:
    """
    CY-inspired complex dynamical system.

    Implements the update rule:
        z_{n+1} = U z_n + ε (z_n ⊙ z_n) + c

    Attributes:
        k (int): Dimension of complex state space
        U (ndarray): k×k unitary (or near-unitary) matrix
        epsilon (float): Nonlinearity strength parameter
        c (ndarray): Complex parameter vector (k,)
    """

    def __init__(
        self,
        k: int = 3,
        U: Optional[np.ndarray] = None,
        epsilon: float = 0.01,
        c: Optional[np.ndarray] = None
    ):
        """
        Initialize CY system.

        Parameters:
            k (int): Dimension of complex space
            U (ndarray, optional): k×k matrix (default: random unitary)
            epsilon (float): Nonlinearity strength
            c (ndarray, optional): Parameter vector (default: small random)
        """
        self.k = k
        self.epsilon = epsilon

        # Initialize U as unitary if not provided
        if U is None:
            U = self._generate_random_unitary(k)
        else:
            U = np.array(U, dtype=np.complex128)
            if U.shape != (k, k):
                raise ValueError(f"U must be {k}×{k}, got {U.shape}")

        self.U = U

        # Initialize c as small random if not provided
        if c is None:
            c = 0.1 * (np.random.randn(k) + 1j * np.random.randn(k))
        else:
            c = np.array(c, dtype=np.complex128)
            if c.shape != (k,):
                raise ValueError(f"c must have shape ({k},), got {c.shape}")

        self.c = c

    def _generate_random_unitary(self, n: int) -> np.ndarray:
        """
        Generate a random unitary matrix using QR decomposition.

        Parameters:
            n (int): Matrix dimension

        Returns:
            ndarray: n×n unitary matrix
        """
        # Generate random complex matrix
        A = np.random.randn(n, n) + 1j * np.random.randn(n, n)

        # QR decomposition gives unitary Q
        Q, R = np.linalg.qr(A)

        # Adjust phases to ensure proper distribution
        d = np.diagonal(R)
        ph = d / np.abs(d)
        Q = Q @ np.diag(ph)

        return Q

    def step(self, z: Union[np.ndarray, CYState]) -> CYState:
        """
        Perform one iteration of the CY dynamics.

        Parameters:
            z (ndarray or CYState): Current state

        Returns:
            CYState: Next state z_{n+1}
        """
        # Extract array if CYState
        if isinstance(z, CYState):
            z_array = z.z
        else:
            z_array = np.array(z, dtype=np.complex128)

        # Update rule: z_{n+1} = U z_n + ε (z_n ⊙ z_n) + c
        z_next = self.U @ z_array + self.epsilon * (z_array * z_array) + self.c

        return CYState(z_next)

    def trajectory(
        self,
        z0: Union[np.ndarray, CYState],
        n_steps: int = 1000,
        return_states: bool = True
    ) -> Union[np.ndarray, list]:
        """
        Generate a trajectory from initial condition.

        Parameters:
            z0 (ndarray or CYState): Initial state
            n_steps (int): Number of steps to simulate
            return_states (bool): If True, return list of CYState objects;
                                  if False, return array of shape (n_steps, k)

        Returns:
            list or ndarray: Trajectory
        """
        if isinstance(z0, CYState):
            z = z0.copy()
        else:
            z = CYState(z0)

        if return_states:
            trajectory = [z.copy()]
            for _ in range(n_steps - 1):
                z = self.step(z)
                trajectory.append(z.copy())
            return trajectory
        else:
            trajectory = np.zeros((n_steps, self.k), dtype=np.complex128)
            trajectory[0] = z.z
            for i in range(1, n_steps):
                z = self.step(z)
                trajectory[i] = z.z
            return trajectory

    def is_bounded(
        self,
        z0: Union[np.ndarray, CYState],
        n_steps: int = 1000,
        escape_radius: float = 10.0,
        check_interval: int = 10
    ) -> Tuple[bool, int]:
        """
        Check if orbit remains bounded.

        Parameters:
            z0 (ndarray or CYState): Initial state
            n_steps (int): Maximum steps to check
            escape_radius (float): Threshold for escape
            check_interval (int): Check norm every N steps

        Returns:
            (bool, int): (is_bounded, escape_time)
                        escape_time = n_steps if bounded
        """
        if isinstance(z0, CYState):
            z = z0.copy()
        else:
            z = CYState(z0)

        for i in range(n_steps):
            if i % check_interval == 0:
                if z.norm() > escape_radius:
                    return False, i
            z = self.step(z)

        return True, n_steps

    def jacobian(self, z: Union[np.ndarray, CYState]) -> np.ndarray:
        """
        Compute the Jacobian matrix at state z.

        For the update rule z_{n+1} = U z + ε (z ⊙ z) + c,
        the Jacobian is:
            J = U + ε diag(2z)

        Note: This is the complex Jacobian. For real-valued analysis,
        consider the real and imaginary parts separately.

        Parameters:
            z (ndarray or CYState): State at which to compute Jacobian

        Returns:
            ndarray: k×k complex Jacobian matrix
        """
        if isinstance(z, CYState):
            z_array = z.z
        else:
            z_array = np.array(z, dtype=np.complex128)

        # J = U + ε diag(2z)
        J = self.U + self.epsilon * np.diag(2 * z_array)

        return J

    def lyapunov_exponent_estimate(
        self,
        z0: Union[np.ndarray, CYState],
        n_steps: int = 5000,
        n_transient: int = 1000
    ) -> float:
        """
        Estimate the largest Lyapunov exponent.

        Uses the tangent space method with periodic renormalization.

        Parameters:
            z0 (ndarray or CYState): Initial condition
            n_steps (int): Number of steps for estimation
            n_transient (int): Transient steps to discard

        Returns:
            float: Estimated largest Lyapunov exponent
        """
        if isinstance(z0, CYState):
            z = z0.copy()
        else:
            z = CYState(z0)

        # Initialize tangent vector
        v = np.random.randn(self.k) + 1j * np.random.randn(self.k)
        v = v / np.linalg.norm(v)

        lyap_sum = 0.0

        # Transient
        for _ in range(n_transient):
            z = self.step(z)

        # Accumulate
        for _ in range(n_steps):
            J = self.jacobian(z)
            v = J @ v
            norm_v = np.linalg.norm(v)
            lyap_sum += np.log(norm_v)
            v = v / norm_v
            z = self.step(z)

        return lyap_sum / n_steps

    def energy(self, z: Union[np.ndarray, CYState]) -> float:
        """
        Compute a heuristic 'energy' function.

        E(z) = ||z||² + ε/3 Σ|z_i|⁴

        This is NOT derived from a physical Hamiltonian;
        it's a convenient diagnostic tool.

        Parameters:
            z (ndarray or CYState): State

        Returns:
            float: Energy value
        """
        if isinstance(z, CYState):
            z_array = z.z
        else:
            z_array = np.array(z, dtype=np.complex128)

        E = np.sum(np.abs(z_array)**2) + (self.epsilon / 3.0) * np.sum(np.abs(z_array)**4)

        return np.real(E)
