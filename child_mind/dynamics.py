"""
Child Mind v1 — Dynamics Functions
MindFractal Lab

Implements the core dynamics equations:
    z_{t+1} = F_mind(z_t + Δz_t, c_t, b_t)
    b_{t+1} = G_boundary(b_t, z_{t+1}, γ_t)
    c_{t+1} = H_coherence(c_t, β_t, z_{t+1})
    m_{t+1} = U_memory(m_t, s_t, a_t)

All functions are deterministic and numpy-based for performance.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import State, Action, ChildMindConfig


def F_mind(
    z: np.ndarray,
    c: np.ndarray,
    b: np.ndarray,
    alpha: float = 0.0
) -> np.ndarray:
    """
    Core manifold dynamics function.

    Computes z_{t+1} from the input state components using a nonlinear
    tanh/softplus transform that mixes z, c, and global statistics of b.

    Args:
        z: Current manifold position after shift (d,)
        c: Coherence state (k_c,)
        b: Boundary grid (H_b, W_b)
        alpha: Branch bias scalar

    Returns:
        z_next: Updated manifold position (d,)
    """
    d = len(z)

    # Extract boundary statistics
    b_mean = np.mean(b)
    b_std = np.std(b) + 1e-8
    b_max = np.max(b)
    b_min = np.min(b)

    # Primary coherence affects stability
    coherence = c[0] if len(c) > 0 else 0.5

    # Nonlinear mixing: tanh for bounded behavior
    z_tanh = np.tanh(z)

    # Softplus for smooth positivity where needed
    z_softplus = np.log1p(np.exp(z))

    # Mix based on coherence
    z_mixed = coherence * z_tanh + (1 - coherence) * (z_softplus - 1)

    # Apply boundary influence
    # Use b statistics to modulate different dimensions
    boundary_influence = np.zeros(d)
    boundary_influence[0] = b_mean * 0.5
    boundary_influence[1] = (b_max - b_min) * 0.3
    if d > 2:
        boundary_influence[2] = b_std * 0.4

    # Apply branch bias (alpha shifts the attractor)
    branch_shift = alpha * np.sin(np.linspace(0, 2 * np.pi, d))

    # Combine all influences
    z_next = z_mixed + 0.1 * boundary_influence[:d] + 0.05 * branch_shift

    # Soft normalization to prevent unbounded growth
    norm = np.linalg.norm(z_next)
    if norm > 5.0:
        z_next = z_next * (5.0 / norm)

    return z_next


def G_boundary(
    b: np.ndarray,
    z: np.ndarray,
    gamma: float,
    config: 'ChildMindConfig' = None
) -> np.ndarray:
    """
    Boundary grid dynamics function.

    Updates the holographic boundary by applying a small convolution-style
    update around a location derived from z.

    Args:
        b: Current boundary grid (H_b, W_b)
        z: Updated manifold state (d,)
        gamma: Boundary rewrite strength (0-1 typical)
        config: Configuration object

    Returns:
        b_next: Updated boundary grid (H_b, W_b)
    """
    H_b, W_b = b.shape

    # Derive update location from z
    # Use first two components mapped to grid coordinates
    if len(z) >= 2:
        # Map from z space to grid coordinates
        loc_i = int((np.tanh(z[0]) + 1) / 2 * (H_b - 1))
        loc_j = int((np.tanh(z[1]) + 1) / 2 * (W_b - 1))
    else:
        loc_i, loc_j = H_b // 2, W_b // 2

    # Clamp to valid range
    loc_i = max(0, min(H_b - 1, loc_i))
    loc_j = max(0, min(W_b - 1, loc_j))

    # Create update kernel (3x3 Gaussian-like)
    kernel_size = 3
    kernel = np.array([
        [0.05, 0.1, 0.05],
        [0.1,  0.4, 0.1],
        [0.05, 0.1, 0.05]
    ])

    # Value to write based on z magnitude and later components
    write_value = np.tanh(np.sum(z[2:6]) if len(z) > 2 else z[0])

    # Apply diffusion to whole grid first
    diffusion = config.boundary_diffusion if config else 0.1
    b_diffused = b.copy()

    for i in range(H_b):
        for j in range(W_b):
            neighbors = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ni, nj = (i + di) % H_b, (j + dj) % W_b
                    neighbors.append(b[ni, nj])
            b_diffused[i, j] = (1 - diffusion) * b[i, j] + diffusion * np.mean(neighbors)

    # Apply local update at derived location
    b_next = b_diffused.copy()
    gamma_clamped = np.clip(gamma, 0, 1)

    for di in range(-1, 2):
        for dj in range(-1, 2):
            ni = (loc_i + di) % H_b
            nj = (loc_j + dj) % W_b
            ki = di + 1
            kj = dj + 1
            update = kernel[ki, kj] * write_value * gamma_clamped
            b_next[ni, nj] = b_next[ni, nj] * (1 - gamma_clamped * 0.3) + update

    return b_next


def H_coherence(
    c: np.ndarray,
    beta: float,
    z: np.ndarray,
    config: 'ChildMindConfig' = None
) -> np.ndarray:
    """
    Coherence state dynamics function.

    Updates c toward a coherence target depending on ||z|| and the
    modulation parameters alpha, beta.

    Coherence state components:
        c[0]: Primary coherence metric (0-1)
        c[1]: Secondary coherence (interaction strength)
        c[2]: Phase alignment (-1 to 1)
        c[3]: Stability measure (0-1)
        c[4]: Novelty accumulator (0-1)

    Args:
        c: Current coherence state (k_c,)
        beta: Coherence modulation scalar
        z: Updated manifold state (d,)
        config: Configuration object

    Returns:
        c_next: Updated coherence state (k_c,)
    """
    target = config.coherence_target if config else 0.7
    k_c = len(c)
    c_next = np.zeros(k_c)

    z_norm = np.linalg.norm(z)

    # c[0]: Primary coherence - attracted to target, modulated by beta
    coherence_rate = 0.1 + 0.2 * np.abs(beta)
    c_next[0] = c[0] + coherence_rate * (target * (1 + 0.3 * beta) - c[0])
    c_next[0] = np.clip(c_next[0], 0, 1)

    # c[1]: Secondary coherence - based on z_norm
    if z_norm < 2.0:
        c_next[1] = c[1] * 0.95 + 0.05 * (1 - z_norm / 2.0)
    else:
        c_next[1] = c[1] * 0.9

    # c[2]: Phase alignment - oscillates based on z components
    if len(z) >= 2:
        phase = np.arctan2(z[1], z[0] + 1e-8) / np.pi
        c_next[2] = c[2] * 0.8 + 0.2 * phase
    else:
        c_next[2] = c[2] * 0.9

    # c[3]: Stability - decreases with large z changes, increases otherwise
    stability_change = -0.1 * (z_norm - 1.0) if z_norm > 1.0 else 0.05
    c_next[3] = np.clip(c[3] + stability_change, 0, 1)

    # c[4]: Novelty - based on how far z is from typical magnitude
    typical_z = 0.5
    novelty = np.abs(z_norm - typical_z) / 3.0
    c_next[4] = c[4] * 0.7 + 0.3 * np.clip(novelty, 0, 1)

    return c_next


def U_memory(
    m: np.ndarray,
    state: 'State',
    action: 'Action',
    config: 'ChildMindConfig' = None
) -> np.ndarray:
    """
    Memory summary dynamics function.

    Implements exponential moving average of recent state/action features.
    The memory summarizes the trajectory history for policy decisions.

    Args:
        m: Current memory summary (d_m,)
        state: Current state s_t
        action: Current action a_t
        config: Configuration object

    Returns:
        m_next: Updated memory summary (d_m,)
    """
    decay = config.memory_decay if config else 0.9
    d_m = len(m)

    # Extract features from state and action
    features = np.zeros(d_m)

    # State features
    z = state.z
    c = state.c
    b = state.b

    # First half: state-derived features
    half = d_m // 2

    # z statistics
    features[0] = np.mean(z)
    features[1] = np.std(z)
    features[2] = np.max(z) if len(z) > 0 else 0
    features[3] = np.min(z) if len(z) > 0 else 0

    # Coherence features
    if len(c) >= 3:
        features[4] = c[0]  # primary coherence
        features[5] = c[3] if len(c) > 3 else 0  # stability
        features[6] = c[4] if len(c) > 4 else 0  # novelty

    # Boundary statistics
    features[7] = np.mean(b)

    # Second half: action-derived features
    delta_z = action.delta_z
    features[half] = np.mean(delta_z)
    features[half + 1] = np.std(delta_z)
    features[half + 2] = np.linalg.norm(delta_z)
    features[half + 3] = action.alpha
    features[half + 4] = action.beta
    features[half + 5] = action.gamma

    # Time encoding
    features[-1] = np.sin(state.t * 0.1)

    # Exponential moving average update
    m_next = decay * m + (1 - decay) * features

    return m_next


def compute_z_centroid(m: np.ndarray) -> np.ndarray:
    """
    Extract estimated z centroid from memory.

    This is used for novelty calculation in rewards.

    Args:
        m: Memory summary vector

    Returns:
        Estimated centroid (approximation from memory stats)
    """
    # The memory encodes statistics, not the full centroid
    # We approximate using the stored mean and std
    mean_z = m[0] if len(m) > 0 else 0
    std_z = m[1] if len(m) > 1 else 0.1

    # Return a simple estimate
    return np.array([mean_z, std_z])
