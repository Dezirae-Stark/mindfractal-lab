"""
Possibility Manifold Metrics

Defines measures, distances, and stability classification on the
Possibility Manifold 𝒫.

Metrics include:
- Frobenius distance on parameter matrices
- Lyapunov exponent estimation
- Basin of attraction measures
- Bifurcation proximity
- Information-theoretic divergence
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .possibility_manifold import ParameterPoint, PossibilityManifold, StabilityRegion


@dataclass
class StabilityMetrics:
    """Container for stability analysis results"""

    lyapunov_exponent: float
    attractor_dimension: float
    basin_volume_estimate: float
    bifurcation_distance: float
    region: StabilityRegion


class ManifoldMetrics:
    """
    Metrics and measures on the Possibility Manifold

    Provides mathematical tools for analyzing the geometry and
    topology of 𝒫, including distances, volumes, and curvature.
    """

    def __init__(self, manifold: PossibilityManifold):
        """
        Initialize metrics for a given manifold

        Parameters:
        -----------
        manifold : PossibilityManifold
            The manifold to analyze
        """
        self.manifold = manifold

    def frobenius_distance(self, p1: ParameterPoint, p2: ParameterPoint) -> float:
        """
        Frobenius norm distance between parameter matrices

        d_F(p1, p2) = √(‖A1-A2‖²_F + ‖B1-B2‖²_F + ‖W1-W2‖²_F)

        Parameters:
        -----------
        p1, p2 : ParameterPoint
            Points to compare

        Returns:
        --------
        distance : float
            Frobenius distance
        """
        dist_sq = 0.0

        if p1.A is not None and p2.A is not None:
            dist_sq += np.linalg.norm(p1.A - p2.A, "fro") ** 2
        if p1.B is not None and p2.B is not None:
            dist_sq += np.linalg.norm(p1.B - p2.B, "fro") ** 2
        if p1.W is not None and p2.W is not None:
            dist_sq += np.linalg.norm(p1.W - p2.W, "fro") ** 2

        return np.sqrt(dist_sq)

    def parameter_distance(self, p1: ParameterPoint, p2: ParameterPoint) -> float:
        """
        Euclidean distance in parameter space

        d_c(p1, p2) = ‖c1 - c2‖

        Parameters:
        -----------
        p1, p2 : ParameterPoint
            Points to compare

        Returns:
        --------
        distance : float
            Parameter vector distance
        """
        return np.linalg.norm(p1.c - p2.c)

    def state_distance(self, p1: ParameterPoint, p2: ParameterPoint) -> float:
        """
        Initial state distance

        d_z(p1, p2) = ‖z0,1 - z0,2‖

        Parameters:
        -----------
        p1, p2 : ParameterPoint
            Points to compare

        Returns:
        --------
        distance : float
            Initial state distance
        """
        return np.linalg.norm(p1.z0 - p2.z0)

    def manifold_distance(
        self,
        p1: ParameterPoint,
        p2: ParameterPoint,
        weights: Optional[Tuple[float, float, float]] = None,
    ) -> float:
        """
        Combined distance metric on 𝒫

        d_𝒫(p1, p2) = √(w1·d_z² + w2·d_c² + w3·d_F²)

        Parameters:
        -----------
        p1, p2 : ParameterPoint
            Points to compare
        weights : tuple, optional
            (w_state, w_param, w_matrix) weights
            Default: (1.0, 1.0, 0.1)

        Returns:
        --------
        distance : float
            Weighted manifold distance
        """
        if weights is None:
            weights = (1.0, 1.0, 0.1)

        w_state, w_param, w_matrix = weights

        d_z = self.state_distance(p1, p2)
        d_c = self.parameter_distance(p1, p2)
        d_F = self.frobenius_distance(p1, p2)

        return np.sqrt(w_state * d_z**2 + w_param * d_c**2 + w_matrix * d_F**2)

    def lyapunov_exponent(self, orbit: np.ndarray, method: str = "tangent") -> float:
        """
        Estimate largest Lyapunov exponent

        Uses either tangent vector or nearby orbit method

        Parameters:
        -----------
        orbit : ndarray
            Trajectory to analyze
        method : str
            'tangent' or 'nearby' (default: 'tangent')

        Returns:
        --------
        lambda : float
            Estimated largest Lyapunov exponent
        """
        if method == "tangent":
            return self._lyapunov_tangent(orbit)
        elif method == "nearby":
            return self._lyapunov_nearby(orbit)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _lyapunov_tangent(self, orbit: np.ndarray) -> float:
        """Estimate Lyapunov using finite difference tangent vectors"""
        n_steps = len(orbit)
        if n_steps < 10:
            return 0.0

        # Compute differences (approximate tangent vectors)
        diffs = np.diff(orbit, axis=0)
        norms = np.linalg.norm(diffs, axis=1)

        # Avoid log(0)
        norms = norms[norms > 1e-12]
        if len(norms) == 0:
            return -np.inf  # Converged to fixed point

        # Average log of growth rates
        log_growth = np.log(norms + 1e-12)
        return np.mean(log_growth)

    def _lyapunov_nearby(self, orbit: np.ndarray, epsilon: float = 1e-6) -> float:
        """Estimate Lyapunov using nearby orbit separation"""
        # This requires recomputing orbit with perturbed initial condition
        # For now, delegate to tangent method
        return self._lyapunov_tangent(orbit)

    def attractor_dimension(self, orbit: np.ndarray) -> float:
        """
        Estimate correlation dimension of attractor

        Uses Grassberger-Procaccia algorithm

        Parameters:
        -----------
        orbit : ndarray
            Trajectory on attractor

        Returns:
        --------
        dim : float
            Estimated correlation dimension
        """
        n_points = len(orbit)
        if n_points < 100:
            return 0.0

        # Sample distances between points
        n_sample = min(1000, n_points)
        indices = np.random.choice(n_points, n_sample, replace=False)
        points = orbit[indices]

        # Compute pairwise distances
        dists = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                d = np.linalg.norm(points[i] - points[j])
                if d > 1e-12:  # Avoid identical points
                    dists.append(d)

        if len(dists) == 0:
            return 0.0

        dists = np.array(dists)

        # Correlation integral at different scales
        scales = np.logspace(np.log10(dists.min()), np.log10(dists.max()), 20)
        correlations = []
        for r in scales:
            C_r = np.sum(dists < r) / len(dists)
            if C_r > 0:
                correlations.append((r, C_r))

        if len(correlations) < 2:
            return 0.0

        # Estimate slope in log-log plot
        log_r = np.log([c[0] for c in correlations])
        log_C = np.log([c[1] for c in correlations])

        # Linear fit
        coeffs = np.polyfit(log_r, log_C, 1)
        return coeffs[0]  # Slope is correlation dimension


class StabilityClassifier:
    """
    Classify regions of the Possibility Manifold by stability

    Determines whether a given point leads to stable attractors,
    chaos, divergence, or is near a bifurcation boundary.
    """

    def __init__(self, manifold: PossibilityManifold):
        """
        Initialize classifier

        Parameters:
        -----------
        manifold : PossibilityManifold
            The manifold to classify
        """
        self.manifold = manifold
        self.metrics = ManifoldMetrics(manifold)

    def classify_point(self, point: ParameterPoint, steps: int = 500) -> StabilityMetrics:
        """
        Full stability classification of a manifold point

        Parameters:
        -----------
        point : ParameterPoint
            Point to classify
        steps : int
            Number of orbit iterations

        Returns:
        --------
        metrics : StabilityMetrics
            Complete stability analysis
        """
        # Compute orbit
        orbit = self.manifold.compute_orbit(point, steps=steps)

        # Classify stability region
        region = self.manifold.classify_stability(orbit)

        # Compute metrics
        lyap = self.metrics.lyapunov_exponent(orbit)
        dim = self.metrics.attractor_dimension(orbit)

        # Estimate basin volume (rough approximation)
        basin_volume = self._estimate_basin_volume(point, region)

        # Bifurcation distance (how close to boundary)
        bif_dist = self._bifurcation_distance(point, lyap)

        return StabilityMetrics(
            lyapunov_exponent=lyap,
            attractor_dimension=dim,
            basin_volume_estimate=basin_volume,
            bifurcation_distance=bif_dist,
            region=region,
        )

    def _estimate_basin_volume(self, point: ParameterPoint, region: StabilityRegion) -> float:
        """
        Rough estimate of basin of attraction volume

        Samples nearby points and checks if they converge to same attractor
        """
        if region == StabilityRegion.DIVERGENT:
            return 0.0

        n_samples = 50
        converge_count = 0
        epsilon = 0.1

        # Get reference attractor
        ref_orbit = self.manifold.compute_orbit(point, steps=200)
        if np.isnan(ref_orbit).any():
            return 0.0

        ref_attractor = ref_orbit[-50:]  # Last 50 points

        # Sample nearby initial conditions
        for _ in range(n_samples):
            # Perturb initial condition
            perturbed_point = point.copy()
            perturbed_point.z0 = point.z0 + epsilon * self._random_unit_vector(self.manifold.dim)

            # Compute orbit
            orbit = self.manifold.compute_orbit(perturbed_point, steps=200)
            if np.isnan(orbit).any():
                continue

            # Check if converges to same attractor
            test_attractor = orbit[-50:]
            dist = np.mean(
                [np.linalg.norm(ref_attractor[i] - test_attractor[i]) for i in range(50)]
            )

            if dist < epsilon:
                converge_count += 1

        # Basin volume estimate (very rough)
        return (converge_count / n_samples) * (epsilon**self.manifold.dim)

    def _bifurcation_distance(self, point: ParameterPoint, lyap: float) -> float:
        """
        Estimate distance to nearest bifurcation

        A bifurcation occurs when Lyapunov exponent changes sign
        """
        # Lyapunov near zero suggests proximity to bifurcation
        return np.abs(lyap)

    def _random_unit_vector(self, n: int) -> np.ndarray:
        """Generate random unit vector in C^n"""
        v = np.random.randn(n) + 1j * np.random.randn(n)
        return v / np.linalg.norm(v)

    def map_stability_landscape(
        self, param_range: Tuple[float, float], resolution: int = 50
    ) -> Dict:
        """
        Create a 2D slice of the stability landscape

        Varies two parameters while holding others fixed

        Parameters:
        -----------
        param_range : tuple
            (min, max) range for parameters
        resolution : int
            Grid resolution

        Returns:
        --------
        landscape : dict
            Grid of stability classifications
        """
        c1_vals = np.linspace(param_range[0], param_range[1], resolution)
        c2_vals = np.linspace(param_range[0], param_range[1], resolution)

        stability_grid = np.zeros((resolution, resolution), dtype=int)
        lyap_grid = np.zeros((resolution, resolution))

        for i, c1 in enumerate(c1_vals):
            for j, c2 in enumerate(c2_vals):
                # Create point with varying first two parameters
                point = self.manifold.sample_point()
                point.c[0] = c1
                point.c[1] = c2 if self.manifold.dim > 1 else 0

                # Classify
                metrics = self.classify_point(point, steps=200)

                # Store results
                stability_grid[i, j] = self._region_to_int(metrics.region)
                lyap_grid[i, j] = metrics.lyapunov_exponent

        return {
            "c1_vals": c1_vals,
            "c2_vals": c2_vals,
            "stability_grid": stability_grid,
            "lyapunov_grid": lyap_grid,
        }

    @staticmethod
    def _region_to_int(region: StabilityRegion) -> int:
        """Convert stability region to integer for plotting"""
        mapping = {
            StabilityRegion.STABLE_ATTRACTOR: 0,
            StabilityRegion.CHAOTIC: 1,
            StabilityRegion.DIVERGENT: 2,
            StabilityRegion.BOUNDARY: 3,
            StabilityRegion.UNKNOWN: 4,
        }
        return mapping.get(region, 4)
