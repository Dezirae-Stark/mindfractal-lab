"""
Attractor Core — 3D Attractor Visualization and Analysis
MindFractal Lab

Pyodide-compatible module for exploring 3D attractors and strange attractors.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
import base64
from io import BytesIO


# Default 3D system matrices
DEFAULT_A_3D = np.array([
    [0.9, 0.0, 0.0],
    [0.0, 0.9, 0.0],
    [0.0, 0.0, 0.9]
])

DEFAULT_B_3D = np.array([
    [0.2, 0.3, 0.1],
    [0.3, 0.2, 0.1],
    [0.1, 0.1, 0.2]
])

DEFAULT_W_3D = np.array([
    [1.0, 0.1, 0.05],
    [0.1, 1.0, 0.05],
    [0.05, 0.05, 1.0]
])


def compute_orbit_3d(
    x0: np.ndarray,
    c: np.ndarray,
    n_steps: int = 1000,
    A: np.ndarray = None,
    B: np.ndarray = None,
    W: np.ndarray = None
) -> np.ndarray:
    """
    Compute 3D orbit trajectory.

    Parameters
    ----------
    x0 : np.ndarray
        Initial condition (3,)
    c : np.ndarray
        Parameter vector (3,)
    n_steps : int
        Number of iterations
    A, B, W : np.ndarray
        System matrices (defaults to module constants)

    Returns
    -------
    np.ndarray
        Trajectory of shape (n_steps, 3)
    """
    if A is None:
        A = DEFAULT_A_3D
    if B is None:
        B = DEFAULT_B_3D
    if W is None:
        W = DEFAULT_W_3D

    trajectory = np.zeros((n_steps, 3))
    x = np.array(x0)
    c = np.array(c)

    for i in range(n_steps):
        trajectory[i] = x
        x = A @ x + B @ np.tanh(W @ x) + c

        if np.linalg.norm(x) > 100:
            trajectory[i+1:] = np.nan
            break

    return trajectory


def compute_lyapunov_3d(
    x0: np.ndarray,
    c: np.ndarray,
    n_steps: int = 1000,
    n_transient: int = 200,
    A: np.ndarray = None,
    B: np.ndarray = None,
    W: np.ndarray = None
) -> np.ndarray:
    """
    Compute 3D Lyapunov spectrum (3 exponents).

    Returns
    -------
    np.ndarray
        Array of 3 Lyapunov exponents (largest first)
    """
    if A is None:
        A = DEFAULT_A_3D
    if B is None:
        B = DEFAULT_B_3D
    if W is None:
        W = DEFAULT_W_3D

    x = np.array(x0)
    c = np.array(c)

    # QR decomposition for orthonormalization
    Q = np.eye(3)
    lyap_sums = np.zeros(3)

    # Transient
    for _ in range(n_transient):
        x = A @ x + B @ np.tanh(W @ x) + c
        if np.linalg.norm(x) > 100:
            return np.array([np.inf, np.nan, np.nan])

    # Accumulate Lyapunov exponents
    for _ in range(n_steps):
        # Jacobian
        Wx = W @ x
        sech2 = 1 - np.tanh(Wx)**2
        J = A + B @ np.diag(sech2) @ W

        # Evolve Q
        Q = J @ Q
        Q, R = np.linalg.qr(Q)

        # Accumulate log of diagonal
        lyap_sums += np.log(np.abs(np.diag(R)) + 1e-12)

        # Iterate
        x = A @ x + B @ np.tanh(W @ x) + c
        if np.linalg.norm(x) > 100:
            return np.array([np.inf, np.nan, np.nan])

    return lyap_sums / n_steps


def classify_attractor_3d(
    x0: np.ndarray,
    c: np.ndarray,
    threshold: float = 0.01
) -> str:
    """
    Classify 3D attractor type based on Lyapunov spectrum.

    Returns
    -------
    str
        One of: 'fixed_point', 'limit_cycle', 'torus', 'strange', 'divergent'
    """
    spectrum = compute_lyapunov_3d(x0, c, n_steps=500, n_transient=100)

    if np.any(np.isinf(spectrum)):
        return 'divergent'

    # Sort descending
    spectrum = np.sort(spectrum)[::-1]
    l1, l2, l3 = spectrum

    # Classification based on Lyapunov signature
    if l1 < -threshold and l2 < -threshold and l3 < -threshold:
        return 'fixed_point'
    elif np.abs(l1) < threshold and l2 < -threshold and l3 < -threshold:
        return 'limit_cycle'
    elif np.abs(l1) < threshold and np.abs(l2) < threshold and l3 < -threshold:
        return 'torus'
    elif l1 > threshold:
        return 'strange'
    else:
        return 'limit_cycle'  # Default


def compute_poincare_section(
    x0: np.ndarray,
    c: np.ndarray,
    n_steps: int = 5000,
    plane_coord: int = 2,
    plane_value: float = 0.0,
    direction: int = 1
) -> np.ndarray:
    """
    Compute Poincare section crossings.

    Parameters
    ----------
    plane_coord : int
        Coordinate index for the plane (0, 1, or 2)
    plane_value : float
        Value at which plane is located
    direction : int
        +1 for upward crossings, -1 for downward

    Returns
    -------
    np.ndarray
        Array of crossing points (N, 2) in the remaining 2 coordinates
    """
    trajectory = compute_orbit_3d(x0, c, n_steps)
    crossings = []

    other_coords = [i for i in range(3) if i != plane_coord]

    for i in range(1, len(trajectory)):
        if np.isnan(trajectory[i, 0]):
            break

        prev = trajectory[i-1, plane_coord]
        curr = trajectory[i, plane_coord]

        # Check for crossing
        if direction > 0:
            crossed = prev < plane_value <= curr
        else:
            crossed = prev > plane_value >= curr

        if crossed:
            # Linear interpolation
            t = (plane_value - prev) / (curr - prev + 1e-12)
            point = trajectory[i-1] + t * (trajectory[i] - trajectory[i-1])
            crossings.append(point[other_coords])

    return np.array(crossings) if crossings else np.zeros((0, 2))


def compute_correlation_dimension(
    trajectory: np.ndarray,
    r_values: np.ndarray = None,
    max_points: int = 1000
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate correlation dimension using Grassberger-Procaccia algorithm.

    Returns
    -------
    float
        Estimated correlation dimension
    np.ndarray
        log(r) values
    np.ndarray
        log(C(r)) values
    """
    # Use subset for efficiency
    valid = ~np.isnan(trajectory[:, 0])
    traj = trajectory[valid]

    if len(traj) > max_points:
        indices = np.random.choice(len(traj), max_points, replace=False)
        traj = traj[indices]

    n = len(traj)
    if n < 10:
        return np.nan, np.array([]), np.array([])

    # Compute pairwise distances
    dists = []
    for i in range(n):
        for j in range(i+1, n):
            dists.append(np.linalg.norm(traj[i] - traj[j]))
    dists = np.array(dists)

    if len(dists) == 0:
        return np.nan, np.array([]), np.array([])

    # r values
    if r_values is None:
        r_min = np.percentile(dists, 5)
        r_max = np.percentile(dists, 95)
        if r_min <= 0:
            r_min = 1e-6
        r_values = np.logspace(np.log10(r_min), np.log10(r_max), 20)

    # Correlation sum
    C_r = []
    for r in r_values:
        count = np.sum(dists < r)
        C_r.append(2 * count / (n * (n - 1)))

    C_r = np.array(C_r)
    valid_idx = C_r > 0

    if np.sum(valid_idx) < 3:
        return np.nan, np.array([]), np.array([])

    log_r = np.log(r_values[valid_idx])
    log_C = np.log(C_r[valid_idx])

    # Linear regression for dimension estimate
    coeffs = np.polyfit(log_r, log_C, 1)
    dimension = coeffs[0]

    return dimension, log_r, log_C


def render_attractor_3d_to_base64(
    x0: np.ndarray,
    c: np.ndarray,
    n_steps: int = 2000,
    elev: float = 30,
    azim: float = 45,
    figsize: Tuple[int, int] = (8, 8)
) -> str:
    """
    Render 3D attractor to base64 PNG image.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    trajectory = compute_orbit_3d(x0, c, n_steps)
    valid = ~np.isnan(trajectory[:, 0])
    traj = trajectory[valid]

    # Color by time
    colors = np.linspace(0, 1, len(traj))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2],
               c=colors, cmap='plasma', s=0.5, alpha=0.6)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=elev, azim=azim)

    # Dark theme
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.tick_params(colors='#cccccc')
    ax.xaxis.label.set_color('#cccccc')
    ax.yaxis.label.set_color('#cccccc')
    ax.zaxis.label.set_color('#cccccc')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100,
                facecolor='#1a1a2e', edgecolor='none')
    plt.close(fig)

    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def render_poincare_to_base64(
    x0: np.ndarray,
    c: np.ndarray,
    n_steps: int = 10000,
    plane_coord: int = 2,
    plane_value: float = 0.0
) -> str:
    """
    Render Poincare section to base64 PNG image.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    crossings = compute_poincare_section(
        x0, c, n_steps, plane_coord, plane_value
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    if len(crossings) > 0:
        ax.scatter(crossings[:, 0], crossings[:, 1],
                   c=np.arange(len(crossings)), cmap='viridis',
                   s=2, alpha=0.7)
    else:
        ax.text(0.5, 0.5, 'No crossings found',
                ha='center', va='center', transform=ax.transAxes,
                color='#cccccc')

    coord_names = ['x', 'y', 'z']
    other_coords = [i for i in range(3) if i != plane_coord]
    ax.set_xlabel(coord_names[other_coords[0]], color='#cccccc')
    ax.set_ylabel(coord_names[other_coords[1]], color='#cccccc')

    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    ax.tick_params(colors='#cccccc')
    ax.spines['bottom'].set_color('#444444')
    ax.spines['left'].set_color('#444444')
    ax.spines['top'].set_color('#444444')
    ax.spines['right'].set_color('#444444')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100,
                facecolor='#1a1a2e')
    plt.close(fig)

    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def scan_attractor_types(
    resolution: int = 20,
    c1_range: Tuple[float, float] = (-1, 1),
    c2_range: Tuple[float, float] = (-1, 1),
    c3: float = 0.0,
    x0: np.ndarray = None
) -> Dict[str, np.ndarray]:
    """
    Scan parameter space for attractor types.

    Returns
    -------
    dict
        Contains 'types' array (0=fixed, 1=cycle, 2=torus, 3=strange, 4=divergent)
        and 'lyapunov_max' array
    """
    if x0 is None:
        x0 = np.array([0.1, 0.1, 0.1])

    c1_vals = np.linspace(c1_range[0], c1_range[1], resolution)
    c2_vals = np.linspace(c2_range[0], c2_range[1], resolution)

    type_map = np.zeros((resolution, resolution))
    lyap_map = np.zeros((resolution, resolution))

    type_codes = {
        'fixed_point': 0,
        'limit_cycle': 1,
        'torus': 2,
        'strange': 3,
        'divergent': 4
    }

    for i, c2 in enumerate(c2_vals):
        for j, c1 in enumerate(c1_vals):
            c = np.array([c1, c2, c3])
            atype = classify_attractor_3d(x0.copy(), c)
            type_map[i, j] = type_codes.get(atype, 4)

            spectrum = compute_lyapunov_3d(x0.copy(), c, n_steps=200, n_transient=50)
            lyap_map[i, j] = np.clip(spectrum[0] if not np.isinf(spectrum[0]) else 1.0, -1, 1)

    return {
        'types': type_map,
        'lyapunov_max': lyap_map
    }


def render_attractor_scan_to_base64(
    resolution: int = 30,
    c1_range: Tuple[float, float] = (-1, 1),
    c2_range: Tuple[float, float] = (-1, 1)
) -> str:
    """
    Render attractor type scan to base64 PNG image.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    result = scan_attractor_types(resolution, c1_range, c2_range)

    # Custom colormap: fixed=blue, cycle=green, torus=yellow, strange=red, divergent=black
    colors = ['#3498db', '#2ecc71', '#f1c40f', '#e74c3c', '#1a1a2e']
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(
        result['types'],
        extent=[c1_range[0], c1_range[1], c2_range[0], c2_range[1]],
        origin='lower',
        cmap=cmap,
        vmin=0,
        vmax=4
    )

    ax.set_xlabel('c₁', color='#cccccc', fontsize=12)
    ax.set_ylabel('c₂', color='#cccccc', fontsize=12)

    # Legend
    labels = ['Fixed Point', 'Limit Cycle', 'Torus', 'Strange', 'Divergent']
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in colors]
    ax.legend(handles, labels, loc='upper right', framealpha=0.9)

    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    ax.tick_params(colors='#cccccc')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100,
                facecolor='#1a1a2e')
    plt.close(fig)

    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# Export for Pyodide
__all__ = [
    'compute_orbit_3d',
    'compute_lyapunov_3d',
    'classify_attractor_3d',
    'compute_poincare_section',
    'compute_correlation_dimension',
    'render_attractor_3d_to_base64',
    'render_poincare_to_base64',
    'scan_attractor_types',
    'render_attractor_scan_to_base64'
]
