"""
Possibility Manifold Core Implementation

The Possibility Manifold 𝒫 represents the complete space of all possible
dynamical states, trajectories, and system configurations.

Mathematical Definition:
    𝒫 = { (z₀, c, F) : z₀ ∈ ℂⁿ, c ∈ ℂⁿ, F: ℂⁿ → ℂⁿ, orbit bounded }

where:
    - z₀ is the initial state
    - c is the parameter vector
    - F is the update rule family
    - orbit must remain defined (no divergence to infinity)

This is the formal mathematical analogue of the "10th dimension" metaphor:
the space containing all possible timelines, versions, and realities.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Tuple, Dict
from enum import Enum


class UpdateRuleFamily(Enum):
    """Types of dynamical update rules"""
    TANH_2D = "tanh_2d"  # x_{n+1} = A x_n + B tanh(W x_n) + c
    SIGMOID_2D = "sigmoid_2d"  # x_{n+1} = A x_n + B σ(W x_n) + c
    STATE_3D = "state_3d"  # 3-dimensional state space extension
    CALABI_YAU = "calabi_yau"  # Complex manifold dynamics
    CUSTOM = "custom"  # User-defined update rule


class StabilityRegion(Enum):
    """Classification of manifold regions by stability"""
    STABLE_ATTRACTOR = "stable"  # Converges to fixed point/cycle
    CHAOTIC = "chaotic"  # Sensitive dependence on init conditions
    DIVERGENT = "divergent"  # Orbit escapes to infinity
    BOUNDARY = "boundary"  # Near bifurcation point
    UNKNOWN = "unknown"  # Not yet classified


@dataclass
class ParameterPoint:
    """
    A point in the Possibility Manifold 𝒫

    Represents a specific configuration: (initial state, parameters, update rule)
    """
    z0: np.ndarray  # Initial state
    c: np.ndarray  # Parameter vector
    rule_family: UpdateRuleFamily
    A: Optional[np.ndarray] = None  # Linear component matrix
    B: Optional[np.ndarray] = None  # Nonlinear scaling matrix
    W: Optional[np.ndarray] = None  # Weight matrix
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Ensure arrays are numpy arrays"""
        self.z0 = np.asarray(self.z0, dtype=complex)
        self.c = np.asarray(self.c, dtype=complex)
        if self.A is not None:
            self.A = np.asarray(self.A, dtype=complex)
        if self.B is not None:
            self.B = np.asarray(self.B, dtype=complex)
        if self.W is not None:
            self.W = np.asarray(self.W, dtype=complex)

    @property
    def dimension(self) -> int:
        """Dimensionality of the state space"""
        return len(self.z0)

    def copy(self) -> 'ParameterPoint':
        """Create a deep copy of this point"""
        return ParameterPoint(
            z0=self.z0.copy(),
            c=self.c.copy(),
            rule_family=self.rule_family,
            A=self.A.copy() if self.A is not None else None,
            B=self.B.copy() if self.B is not None else None,
            W=self.W.copy() if self.W is not None else None,
            metadata=self.metadata.copy()
        )


class PossibilityManifold:
    """
    The Possibility Manifold 𝒫

    Represents the complete space of all possible dynamical system configurations.
    This is the mathematical formalization of the "tenth dimension" - the space
    containing all possible timelines, trajectories, and system versions.

    Features:
    ---------
    - Sample points uniformly from the manifold
    - Define custom update rule families
    - Classify stability regions
    - Extract orbit branches ("timelines")
    - Compute distances and measures on 𝒫

    Example:
    --------
    >>> manifold = PossibilityManifold(dim=2)
    >>> point = manifold.sample_point(rule_family=UpdateRuleFamily.TANH_2D)
    >>> orbit = manifold.compute_orbit(point, steps=100)
    >>> stability = manifold.classify_stability(orbit)
    """

    def __init__(self, dim: int = 2, bounds: Optional[Tuple[float, float]] = None):
        """
        Initialize the Possibility Manifold

        Parameters:
        -----------
        dim : int
            Dimensionality of the state space
        bounds : tuple, optional
            (min, max) bounds for sampling parameters
            Default: (-2.0, 2.0)
        """
        self.dim = dim
        self.bounds = bounds or (-2.0, 2.0)
        self.sampled_points: List[ParameterPoint] = []
        self.stability_map: Dict[int, StabilityRegion] = {}

    def sample_point(self,
                    rule_family: UpdateRuleFamily = UpdateRuleFamily.TANH_2D,
                    z0: Optional[np.ndarray] = None,
                    c: Optional[np.ndarray] = None) -> ParameterPoint:
        """
        Sample a random point from the manifold

        Parameters:
        -----------
        rule_family : UpdateRuleFamily
            Type of update rule to use
        z0 : ndarray, optional
            Initial state (random if not provided)
        c : ndarray, optional
            Parameter vector (random if not provided)

        Returns:
        --------
        point : ParameterPoint
            A point in 𝒫
        """
        if z0 is None:
            z0 = self._random_complex_vector(self.dim)
        if c is None:
            c = self._random_complex_vector(self.dim)

        # Generate matrices based on rule family
        if rule_family in [UpdateRuleFamily.TANH_2D, UpdateRuleFamily.SIGMOID_2D]:
            A = self._random_matrix(self.dim, self.dim, scale=0.5)
            B = self._random_matrix(self.dim, self.dim, scale=0.3)
            W = self._random_matrix(self.dim, self.dim, scale=1.0)
        elif rule_family == UpdateRuleFamily.STATE_3D:
            # 3D requires dim=3
            if self.dim != 3:
                raise ValueError("STATE_3D requires dim=3")
            A = self._random_matrix(3, 3, scale=0.5)
            B = self._random_matrix(3, 3, scale=0.3)
            W = self._random_matrix(3, 3, scale=1.0)
        elif rule_family == UpdateRuleFamily.CALABI_YAU:
            # Calabi-Yau requires even dimension
            if self.dim % 2 != 0:
                raise ValueError("CALABI_YAU requires even dimension")
            A = self._random_hermitian_matrix(self.dim)
            B = self._random_matrix(self.dim, self.dim, scale=0.3)
            W = self._random_unitary_matrix(self.dim)
        else:
            A = B = W = None

        point = ParameterPoint(z0=z0, c=c, rule_family=rule_family, A=A, B=B, W=W)
        self.sampled_points.append(point)
        return point

    def compute_orbit(self, point: ParameterPoint, steps: int = 100,
                     max_radius: float = 1e6) -> np.ndarray:
        """
        Compute the orbit (trajectory) for a given point in 𝒫

        Parameters:
        -----------
        point : ParameterPoint
            Point in the manifold
        steps : int
            Number of iteration steps
        max_radius : float
            Maximum allowed distance from origin (divergence threshold)

        Returns:
        --------
        orbit : ndarray, shape (steps, dim)
            The trajectory through state space
        """
        orbit = np.zeros((steps, self.dim), dtype=complex)
        orbit[0] = point.z0

        for i in range(1, steps):
            z = orbit[i-1]

            # Apply update rule based on family
            if point.rule_family == UpdateRuleFamily.TANH_2D:
                z_next = point.A @ z + point.B @ np.tanh(point.W @ z) + point.c
            elif point.rule_family == UpdateRuleFamily.SIGMOID_2D:
                z_next = point.A @ z + point.B / (1 + np.exp(-point.W @ z)) + point.c
            elif point.rule_family == UpdateRuleFamily.STATE_3D:
                z_next = point.A @ z + point.B @ np.tanh(point.W @ z) + point.c
            elif point.rule_family == UpdateRuleFamily.CALABI_YAU:
                # Hermitian evolution with nonlinear coupling
                z_next = point.A @ z + point.B @ np.tanh(point.W @ z) + point.c
            else:
                raise ValueError(f"Unsupported rule family: {point.rule_family}")

            # Check for divergence
            if np.abs(z_next).max() > max_radius:
                # Fill rest with NaN
                orbit[i:] = np.nan
                break

            orbit[i] = z_next

        return orbit

    def classify_stability(self, orbit: np.ndarray,
                          threshold_lyap: float = 0.1) -> StabilityRegion:
        """
        Classify the stability of an orbit

        Parameters:
        -----------
        orbit : ndarray
            Trajectory to classify
        threshold_lyap : float
            Threshold for Lyapunov exponent classification

        Returns:
        --------
        region : StabilityRegion
            Classification of the orbit's stability
        """
        # Check for divergence
        if np.isnan(orbit).any():
            return StabilityRegion.DIVERGENT

        # Compute approximate largest Lyapunov exponent
        n_steps = len(orbit)
        if n_steps < 10:
            return StabilityRegion.UNKNOWN

        # Use differences between adjacent points
        diffs = np.diff(orbit, axis=0)
        norms = np.linalg.norm(diffs, axis=1)

        # Avoid log(0)
        norms = norms[norms > 1e-12]
        if len(norms) == 0:
            return StabilityRegion.STABLE_ATTRACTOR

        log_norms = np.log(norms + 1e-12)
        lyap_approx = np.mean(log_norms)

        # Classify based on Lyapunov exponent
        if lyap_approx < -threshold_lyap:
            return StabilityRegion.STABLE_ATTRACTOR
        elif lyap_approx > threshold_lyap:
            return StabilityRegion.CHAOTIC
        else:
            return StabilityRegion.BOUNDARY

    def distance(self, p1: ParameterPoint, p2: ParameterPoint) -> float:
        """
        Compute distance between two points in the manifold

        Uses a weighted metric combining state, parameter, and matrix distances.

        Parameters:
        -----------
        p1, p2 : ParameterPoint
            Points to compare

        Returns:
        --------
        dist : float
            Distance in 𝒫
        """
        # State distance
        d_state = np.linalg.norm(p1.z0 - p2.z0)

        # Parameter distance
        d_param = np.linalg.norm(p1.c - p2.c)

        # Matrix distances (if available)
        d_matrix = 0.0
        if p1.A is not None and p2.A is not None:
            d_matrix += np.linalg.norm(p1.A - p2.A, 'fro')
        if p1.B is not None and p2.B is not None:
            d_matrix += np.linalg.norm(p1.B - p2.B, 'fro')
        if p1.W is not None and p2.W is not None:
            d_matrix += np.linalg.norm(p1.W - p2.W, 'fro')

        # Weighted combination
        return np.sqrt(d_state**2 + d_param**2 + 0.1*d_matrix**2)

    def _random_complex_vector(self, n: int) -> np.ndarray:
        """Generate random complex vector in bounds"""
        real = np.random.uniform(self.bounds[0], self.bounds[1], n)
        imag = np.random.uniform(self.bounds[0], self.bounds[1], n)
        return real + 1j * imag

    def _random_matrix(self, m: int, n: int, scale: float = 1.0) -> np.ndarray:
        """Generate random complex matrix"""
        real = np.random.randn(m, n) * scale
        imag = np.random.randn(m, n) * scale
        return real + 1j * imag

    def _random_hermitian_matrix(self, n: int) -> np.ndarray:
        """Generate random Hermitian matrix (for Calabi-Yau)"""
        M = self._random_matrix(n, n, scale=0.5)
        return (M + M.conj().T) / 2

    def _random_unitary_matrix(self, n: int) -> np.ndarray:
        """Generate random unitary matrix via QR decomposition"""
        M = self._random_matrix(n, n, scale=1.0)
        Q, R = np.linalg.qr(M)
        # Make Q truly unitary
        return Q @ np.diag(np.exp(1j * np.random.uniform(0, 2*np.pi, n)))

    def get_statistics(self) -> Dict:
        """
        Get statistics about sampled points

        Returns:
        --------
        stats : dict
            Statistics about the manifold sampling
        """
        if not self.sampled_points:
            return {"num_points": 0}

        rule_counts = {}
        for point in self.sampled_points:
            rule = point.rule_family.value
            rule_counts[rule] = rule_counts.get(rule, 0) + 1

        stability_counts = {}
        for region in self.stability_map.values():
            stability_counts[region.value] = stability_counts.get(region.value, 0) + 1

        return {
            "num_points": len(self.sampled_points),
            "dimension": self.dim,
            "bounds": self.bounds,
            "rule_distribution": rule_counts,
            "stability_distribution": stability_counts
        }
