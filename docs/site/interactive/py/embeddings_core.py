"""
Embeddings Core — Latent Space Exploration for Mind Dynamics
MindFractal Lab

Pyodide-compatible module for exploring embeddings and dimensionality reduction
of dynamical system trajectories.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
import base64
from io import BytesIO


# Default system matrices
DEFAULT_A = np.array([[0.9, 0.0], [0.0, 0.9]])
DEFAULT_B = np.array([[0.2, 0.3], [0.3, 0.2]])
DEFAULT_W = np.array([[1.0, 0.1], [0.1, 1.0]])


def compute_trajectory_features(
    trajectory: np.ndarray,
    n_features: int = 10
) -> np.ndarray:
    """
    Extract features from a trajectory for embedding.

    Parameters
    ----------
    trajectory : np.ndarray
        Trajectory of shape (n_steps, dim)
    n_features : int
        Number of statistical features to compute

    Returns
    -------
    np.ndarray
        Feature vector
    """
    valid = ~np.isnan(trajectory[:, 0])
    traj = trajectory[valid]

    if len(traj) < 10:
        return np.zeros(n_features)

    features = []

    # Basic statistics per dimension
    for d in range(trajectory.shape[1]):
        features.append(np.mean(traj[:, d]))
        features.append(np.std(traj[:, d]))

    # Range
    for d in range(trajectory.shape[1]):
        features.append(np.max(traj[:, d]) - np.min(traj[:, d]))

    # Path length
    deltas = np.diff(traj, axis=0)
    path_length = np.sum(np.linalg.norm(deltas, axis=1))
    features.append(path_length / len(traj))

    # Curvature estimate (mean angle change)
    if len(deltas) > 1:
        angles = []
        for i in range(1, len(deltas)):
            v1 = deltas[i-1]
            v2 = deltas[i]
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 > 1e-10 and n2 > 1e-10:
                cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
                angles.append(np.arccos(cos_angle))
        features.append(np.mean(angles) if angles else 0)
    else:
        features.append(0)

    # Return first n_features
    features = np.array(features)
    if len(features) < n_features:
        features = np.pad(features, (0, n_features - len(features)))
    return features[:n_features]


def pca_reduce(
    data: np.ndarray,
    n_components: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple PCA implementation for dimensionality reduction.

    Parameters
    ----------
    data : np.ndarray
        Data matrix of shape (n_samples, n_features)
    n_components : int
        Number of principal components

    Returns
    -------
    np.ndarray
        Reduced data (n_samples, n_components)
    np.ndarray
        Principal components
    np.ndarray
        Explained variance ratio
    """
    # Center data
    mean = np.mean(data, axis=0)
    centered = data - mean

    # Covariance matrix
    cov = np.cov(centered.T)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select components
    components = eigenvectors[:, :n_components]
    reduced = centered @ components

    # Explained variance
    total_var = np.sum(eigenvalues)
    explained = eigenvalues[:n_components] / total_var if total_var > 0 else np.zeros(n_components)

    return reduced, components, explained


def tsne_reduce(
    data: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    n_iter: int = 300,
    learning_rate: float = 200.0
) -> np.ndarray:
    """
    Simple t-SNE implementation for visualization.

    Parameters
    ----------
    data : np.ndarray
        Data matrix (n_samples, n_features)
    n_components : int
        Output dimensions
    perplexity : float
        Perplexity parameter
    n_iter : int
        Number of iterations
    learning_rate : float
        Learning rate

    Returns
    -------
    np.ndarray
        Reduced data (n_samples, n_components)
    """
    n_samples = data.shape[0]

    if n_samples < 4:
        return data[:, :n_components] if data.shape[1] >= n_components else np.zeros((n_samples, n_components))

    # Compute pairwise distances
    sq_dist = np.sum(data**2, axis=1, keepdims=True) + \
              np.sum(data**2, axis=1) - 2 * data @ data.T
    sq_dist = np.maximum(sq_dist, 0)

    # Compute P (high-dimensional affinities)
    P = np.exp(-sq_dist / (2 * perplexity**2))
    np.fill_diagonal(P, 0)
    P = P / (np.sum(P, axis=1, keepdims=True) + 1e-12)
    P = (P + P.T) / (2 * n_samples)
    P = np.maximum(P, 1e-12)

    # Initialize Y randomly
    Y = np.random.randn(n_samples, n_components) * 0.01

    # Gradient descent
    momentum = 0.5
    velocity = np.zeros_like(Y)

    for iteration in range(n_iter):
        # Compute Q (low-dimensional affinities)
        sq_dist_Y = np.sum(Y**2, axis=1, keepdims=True) + \
                    np.sum(Y**2, axis=1) - 2 * Y @ Y.T
        sq_dist_Y = np.maximum(sq_dist_Y, 0)
        inv_dist = 1 / (1 + sq_dist_Y)
        np.fill_diagonal(inv_dist, 0)
        Q = inv_dist / (np.sum(inv_dist) + 1e-12)
        Q = np.maximum(Q, 1e-12)

        # Gradient
        PQ_diff = P - Q
        grad = np.zeros_like(Y)
        for i in range(n_samples):
            diff = Y[i] - Y
            grad[i] = 4 * np.sum((PQ_diff[i] * inv_dist[i])[:, np.newaxis] * diff, axis=0)

        # Update
        if iteration > 100:
            momentum = 0.8
        velocity = momentum * velocity - learning_rate * grad
        Y = Y + velocity

    return Y


def sample_trajectory_manifold(
    n_samples: int = 100,
    c_range: Tuple[float, float] = (-1.5, 1.5),
    n_steps: int = 500,
    A: np.ndarray = None,
    B: np.ndarray = None,
    W: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Sample trajectories and extract features for embedding.

    Returns
    -------
    np.ndarray
        Feature matrix (n_samples, n_features)
    np.ndarray
        Parameter matrix (n_samples, 2) containing c values
    list
        Classification labels for each trajectory
    """
    if A is None:
        A = DEFAULT_A
    if B is None:
        B = DEFAULT_B
    if W is None:
        W = DEFAULT_W

    features_list = []
    params_list = []
    labels_list = []

    for _ in range(n_samples):
        c = np.random.uniform(c_range[0], c_range[1], 2)
        x0 = np.random.uniform(-0.5, 0.5, 2)

        # Compute trajectory
        trajectory = np.zeros((n_steps, 2))
        x = x0.copy()
        diverged = False

        for i in range(n_steps):
            trajectory[i] = x
            x = A @ x + B @ np.tanh(W @ x) + c
            if np.linalg.norm(x) > 100:
                trajectory[i+1:] = np.nan
                diverged = True
                break

        # Extract features
        feats = compute_trajectory_features(trajectory, n_features=12)
        features_list.append(feats)
        params_list.append(c)

        # Simple classification
        if diverged:
            labels_list.append('divergent')
        else:
            variance = np.var(trajectory[~np.isnan(trajectory[:, 0])])
            if variance < 0.01:
                labels_list.append('stable')
            elif variance < 1.0:
                labels_list.append('periodic')
            else:
                labels_list.append('chaotic')

    return np.array(features_list), np.array(params_list), labels_list


def render_embedding_to_base64(
    features: np.ndarray,
    labels: List[str],
    method: str = 'pca',
    figsize: Tuple[int, int] = (8, 8)
) -> str:
    """
    Render 2D embedding visualization to base64 PNG.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Reduce to 2D
    if method == 'pca':
        reduced, _, explained = pca_reduce(features, n_components=2)
        title = f'PCA Embedding (explained variance: {100*sum(explained):.1f}%)'
    elif method == 'tsne':
        reduced = tsne_reduce(features, n_components=2, n_iter=200)
        title = 't-SNE Embedding'
    else:
        reduced, _, _ = pca_reduce(features, n_components=2)
        title = 'Embedding'

    # Color by label
    label_colors = {
        'stable': '#3498db',
        'periodic': '#2ecc71',
        'chaotic': '#e74c3c',
        'divergent': '#7f8c8d'
    }
    colors = [label_colors.get(l, '#888888') for l in labels]

    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=colors, s=30, alpha=0.7)

    ax.set_xlabel('Component 1', color='#cccccc', fontsize=10)
    ax.set_ylabel('Component 2', color='#cccccc', fontsize=10)
    ax.set_title(title, color='#ffffff', fontsize=12)

    # Legend
    handles = [plt.scatter([], [], c=c, s=50, label=l) for l, c in label_colors.items()]
    ax.legend(handles=handles, labels=label_colors.keys(), loc='upper right', framealpha=0.9)

    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    ax.tick_params(colors='#cccccc')
    ax.spines['bottom'].set_color('#444444')
    ax.spines['left'].set_color('#444444')
    ax.spines['top'].set_color('#444444')
    ax.spines['right'].set_color('#444444')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='#1a1a2e')
    plt.close(fig)

    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def render_param_space_to_base64(
    params: np.ndarray,
    labels: List[str],
    figsize: Tuple[int, int] = (8, 8)
) -> str:
    """
    Render parameter space colored by classification.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    label_colors = {
        'stable': '#3498db',
        'periodic': '#2ecc71',
        'chaotic': '#e74c3c',
        'divergent': '#7f8c8d'
    }
    colors = [label_colors.get(l, '#888888') for l in labels]

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(params[:, 0], params[:, 1], c=colors, s=50, alpha=0.7)

    ax.set_xlabel('c₁', color='#cccccc', fontsize=12)
    ax.set_ylabel('c₂', color='#cccccc', fontsize=12)
    ax.set_title('Parameter Space Classification', color='#ffffff', fontsize=12)

    # Legend
    handles = [plt.scatter([], [], c=c, s=50, label=l) for l, c in label_colors.items()]
    ax.legend(handles=handles, labels=label_colors.keys(), loc='upper right', framealpha=0.9)

    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    ax.tick_params(colors='#cccccc')
    ax.spines['bottom'].set_color('#444444')
    ax.spines['left'].set_color('#444444')
    ax.spines['top'].set_color('#444444')
    ax.spines['right'].set_color('#444444')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='#1a1a2e')
    plt.close(fig)

    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def compute_and_render_embedding(
    n_samples: int = 100,
    method: str = 'pca'
) -> str:
    """
    Full pipeline: sample, extract features, reduce, render.

    Parameters
    ----------
    n_samples : int
        Number of trajectories to sample
    method : str
        'pca' or 'tsne'

    Returns
    -------
    str
        Base64 encoded PNG image
    """
    features, params, labels = sample_trajectory_manifold(n_samples)
    return render_embedding_to_base64(features, labels, method)


def compute_and_render_param_space(
    n_samples: int = 200
) -> str:
    """
    Sample and render parameter space classification.

    Returns
    -------
    str
        Base64 encoded PNG image
    """
    features, params, labels = sample_trajectory_manifold(n_samples)
    return render_param_space_to_base64(params, labels)


# Interactive point query
def query_trajectory_embedding(
    c: np.ndarray,
    x0: np.ndarray = None,
    reference_features: np.ndarray = None,
    reference_labels: List[str] = None
) -> Dict:
    """
    Query where a single trajectory falls in embedding space.

    Returns
    -------
    dict
        Contains 'features', 'classification', 'nearest_label'
    """
    if x0 is None:
        x0 = np.array([0.1, 0.1])

    # Compute trajectory
    A, B, W = DEFAULT_A, DEFAULT_B, DEFAULT_W
    n_steps = 500

    trajectory = np.zeros((n_steps, 2))
    x = x0.copy()
    diverged = False

    for i in range(n_steps):
        trajectory[i] = x
        x = A @ x + B @ np.tanh(W @ x) + c
        if np.linalg.norm(x) > 100:
            trajectory[i+1:] = np.nan
            diverged = True
            break

    features = compute_trajectory_features(trajectory, n_features=12)

    # Classification
    if diverged:
        classification = 'divergent'
    else:
        variance = np.var(trajectory[~np.isnan(trajectory[:, 0])])
        if variance < 0.01:
            classification = 'stable'
        elif variance < 1.0:
            classification = 'periodic'
        else:
            classification = 'chaotic'

    result = {
        'features': features.tolist(),
        'classification': classification
    }

    # Find nearest in reference if provided
    if reference_features is not None and reference_labels is not None:
        dists = np.linalg.norm(reference_features - features, axis=1)
        nearest_idx = np.argmin(dists)
        result['nearest_label'] = reference_labels[nearest_idx]
        result['nearest_distance'] = float(dists[nearest_idx])

    return result


# Export for Pyodide
__all__ = [
    'compute_trajectory_features',
    'pca_reduce',
    'tsne_reduce',
    'sample_trajectory_manifold',
    'render_embedding_to_base64',
    'render_param_space_to_base64',
    'compute_and_render_embedding',
    'compute_and_render_param_space',
    'query_trajectory_embedding'
]
