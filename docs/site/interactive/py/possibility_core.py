"""
Possibility Core — Possibility Manifold Sampling and Exploration
MindFractal Lab

Pyodide-compatible module for exploring the Possibility Manifold.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
import base64
from io import BytesIO


# Default system matrices
DEFAULT_A = np.array([[0.9, 0.0], [0.0, 0.9]])
DEFAULT_B = np.array([[0.2, 0.3], [0.3, 0.2]])
DEFAULT_W = np.array([[1.0, 0.1], [0.1, 1.0]])


class PossibilityPoint:
    """
    A point in the Possibility Manifold.

    Attributes
    ----------
    z0 : np.ndarray
        Initial state
    c : np.ndarray
        Parameter vector
    rule : str
        Update rule identifier
    """

    def __init__(
        self,
        z0: np.ndarray,
        c: np.ndarray,
        rule: str = 'tanh'
    ):
        self.z0 = np.array(z0)
        self.c = np.array(c)
        self.rule = rule

    def to_dict(self) -> Dict:
        return {
            'z0': self.z0.tolist(),
            'c': self.c.tolist(),
            'rule': self.rule
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'PossibilityPoint':
        return cls(
            z0=np.array(d['z0']),
            c=np.array(d['c']),
            rule=d.get('rule', 'tanh')
        )


def iterate_point(
    p: PossibilityPoint,
    n_steps: int = 500,
    A: np.ndarray = DEFAULT_A,
    B: np.ndarray = DEFAULT_B,
    W: np.ndarray = DEFAULT_W
) -> np.ndarray:
    """
    Compute orbit for a possibility point.
    """
    trajectory = np.zeros((n_steps, 2))
    x = p.z0.copy()

    for i in range(n_steps):
        trajectory[i] = x
        if p.rule == 'tanh':
            x = A @ x + B @ np.tanh(W @ x) + p.c
        elif p.rule == 'sigmoid':
            x = A @ x + B * (1 / (1 + np.exp(-W @ x))) + p.c
        else:
            x = A @ x + B @ np.tanh(W @ x) + p.c

        if np.linalg.norm(x) > 100:
            trajectory[i+1:] = np.nan
            break

    return trajectory


def compute_lyapunov_point(
    p: PossibilityPoint,
    n_steps: int = 500,
    n_transient: int = 100,
    A: np.ndarray = DEFAULT_A,
    B: np.ndarray = DEFAULT_B,
    W: np.ndarray = DEFAULT_W
) -> float:
    """
    Compute Lyapunov exponent for a possibility point.
    """
    x = p.z0.copy()
    v = np.random.randn(2)
    v = v / np.linalg.norm(v)

    # Transient
    for _ in range(n_transient):
        x = A @ x + B @ np.tanh(W @ x) + p.c
        if np.linalg.norm(x) > 100:
            return np.inf  # Escaped

    # Accumulate
    lyap_sum = 0.0
    for _ in range(n_steps):
        Wx = W @ x
        sech2 = 1 - np.tanh(Wx)**2
        J = A + B @ np.diag(sech2) @ W

        v = J @ v
        norm_v = np.linalg.norm(v)
        if norm_v == 0:
            return -np.inf

        lyap_sum += np.log(norm_v)
        v = v / norm_v

        x = A @ x + B @ np.tanh(W @ x) + p.c
        if np.linalg.norm(x) > 100:
            return np.inf

    return lyap_sum / n_steps


def classify_point(
    p: PossibilityPoint,
    threshold: float = 0.01
) -> str:
    """
    Classify a possibility point by its Lyapunov exponent.

    Returns
    -------
    str
        One of: 'stable', 'chaotic', 'periodic', 'divergent'
    """
    lyap = compute_lyapunov_point(p)

    if np.isinf(lyap) and lyap > 0:
        return 'divergent'
    elif lyap < -threshold:
        return 'stable'
    elif lyap > threshold:
        return 'chaotic'
    else:
        return 'periodic'


def sample_possibility_manifold(
    n_samples: int = 100,
    z0_range: Tuple[float, float] = (-1, 1),
    c_range: Tuple[float, float] = (-2, 2),
    rules: List[str] = None,
    bounded_only: bool = True
) -> List[PossibilityPoint]:
    """
    Sample random points from the Possibility Manifold.

    Parameters
    ----------
    n_samples : int
        Target number of samples
    z0_range : tuple
        Bounds for initial condition components
    c_range : tuple
        Bounds for parameter components
    rules : list
        Update rules to sample from
    bounded_only : bool
        If True, reject divergent points

    Returns
    -------
    list
        List of PossibilityPoint objects
    """
    if rules is None:
        rules = ['tanh']

    samples = []
    attempts = 0
    max_attempts = n_samples * 10

    while len(samples) < n_samples and attempts < max_attempts:
        attempts += 1

        z0 = np.random.uniform(z0_range[0], z0_range[1], 2)
        c = np.random.uniform(c_range[0], c_range[1], 2)
        rule = np.random.choice(rules)

        p = PossibilityPoint(z0, c, rule)

        if bounded_only:
            classification = classify_point(p)
            if classification != 'divergent':
                samples.append(p)
        else:
            samples.append(p)

    return samples


def possibility_distance(
    p1: PossibilityPoint,
    p2: PossibilityPoint,
    w_z0: float = 1.0,
    w_c: float = 1.0,
    w_rule: float = 1.0
) -> float:
    """
    Compute weighted distance between two possibility points.
    """
    d_z0 = np.linalg.norm(p1.z0 - p2.z0)
    d_c = np.linalg.norm(p1.c - p2.c)
    d_rule = 0.0 if p1.rule == p2.rule else 1.0

    return np.sqrt(w_z0 * d_z0**2 + w_c * d_c**2 + w_rule * d_rule**2)


def interpolate_timeline(
    p1: PossibilityPoint,
    p2: PossibilityPoint,
    t: float
) -> PossibilityPoint:
    """
    Linear interpolation between two possibility points.

    Parameters
    ----------
    p1, p2 : PossibilityPoint
        Start and end points
    t : float
        Interpolation parameter in [0, 1]

    Returns
    -------
    PossibilityPoint
        Interpolated point
    """
    z0 = (1 - t) * p1.z0 + t * p2.z0
    c = (1 - t) * p1.c + t * p2.c
    rule = p1.rule if t < 0.5 else p2.rule

    return PossibilityPoint(z0, c, rule)


def compute_timeline_orbits(
    p1: PossibilityPoint,
    p2: PossibilityPoint,
    n_points: int = 10,
    n_steps: int = 200
) -> List[np.ndarray]:
    """
    Compute orbits along a timeline between two points.

    Returns
    -------
    list
        List of trajectory arrays
    """
    orbits = []

    for i in range(n_points):
        t = i / (n_points - 1)
        p = interpolate_timeline(p1, p2, t)
        orbit = iterate_point(p, n_steps)
        orbits.append(orbit)

    return orbits


def render_timeline_to_base64(
    p1: PossibilityPoint,
    p2: PossibilityPoint,
    n_points: int = 5
) -> str:
    """
    Render timeline orbits to base64 image.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    orbits = compute_timeline_orbits(p1, p2, n_points)

    fig, axes = plt.subplots(1, n_points, figsize=(3*n_points, 3))
    if n_points == 1:
        axes = [axes]

    for i, (ax, orbit) in enumerate(zip(axes, orbits)):
        t = i / (n_points - 1)
        valid = ~np.isnan(orbit[:, 0])
        ax.plot(orbit[valid, 0], orbit[valid, 1], 'b-', lw=0.5, alpha=0.7)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_title(f't = {t:.2f}')
        ax.set_aspect('equal')
        ax.axis('off')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)

    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def scan_stability_region(
    resolution: int = 50,
    c1_range: Tuple[float, float] = (-2, 2),
    c2_range: Tuple[float, float] = (-2, 2),
    z0: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Scan parameter space for stability classification.

    Returns
    -------
    dict
        Contains 'lyapunov' and 'classification' arrays
    """
    if z0 is None:
        z0 = np.array([0.1, 0.1])

    c1_vals = np.linspace(c1_range[0], c1_range[1], resolution)
    c2_vals = np.linspace(c2_range[0], c2_range[1], resolution)

    lyap_map = np.zeros((resolution, resolution))
    class_map = np.zeros((resolution, resolution))

    for i, c2 in enumerate(c2_vals):
        for j, c1 in enumerate(c1_vals):
            p = PossibilityPoint(z0.copy(), np.array([c1, c2]))
            lyap = compute_lyapunov_point(p, n_steps=200, n_transient=50)

            lyap_map[i, j] = np.clip(lyap, -1, 1)

            if np.isinf(lyap):
                class_map[i, j] = 3  # Divergent
            elif lyap < -0.01:
                class_map[i, j] = 0  # Stable
            elif lyap > 0.01:
                class_map[i, j] = 2  # Chaotic
            else:
                class_map[i, j] = 1  # Periodic

    return {
        'lyapunov': lyap_map,
        'classification': class_map
    }


# Export for Pyodide
__all__ = [
    'PossibilityPoint',
    'iterate_point',
    'compute_lyapunov_point',
    'classify_point',
    'sample_possibility_manifold',
    'possibility_distance',
    'interpolate_timeline',
    'compute_timeline_orbits',
    'render_timeline_to_base64',
    'scan_stability_region'
]
