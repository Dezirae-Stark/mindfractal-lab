"""
Child Mind AI — Dynamics Functions
MindFractal Lab

Extended dynamics for consciousness engine:
    F_mind:     Core manifold dynamics (z evolution)
    G_boundary: Holographic constraint updates
    H_coherence: Internal consistency maintenance
    U_memory:   Experience compression
    R_reflect:  Self-model updates
    I_intent:   Goal formation and evolution
"""

import numpy as np
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from .state import ConsciousnessState
    from ..config import ChildMindAIConfig


def F_mind(
    z: np.ndarray,
    delta_z: np.ndarray,
    c: np.ndarray,
    b: np.ndarray,
    r: np.ndarray,
    alpha: float = 0.0
) -> np.ndarray:
    """
    Core manifold dynamics function.

    Computes z_{t+1} incorporating:
    - Input shift delta_z
    - Coherence-modulated nonlinearity
    - Boundary influence
    - Self-reflection feedback
    - Branch bias (alpha)

    Args:
        z: Current manifold position (d_z,)
        delta_z: Proposed shift (d_z,)
        c: Coherence state (d_c,)
        b: Boundary grid (d_b, d_b)
        r: Reflection state (d_r,)
        alpha: Branch bias scalar

    Returns:
        z_next: Updated manifold position (d_z,)
    """
    d = len(z)

    # Apply shift
    z_shifted = z + delta_z

    # Extract coherence metrics
    coherence = c[0] if len(c) > 0 else 0.5
    stability = c[3] if len(c) > 3 else 0.5

    # Nonlinear mixing based on coherence
    # High coherence -> more tanh (bounded, smooth)
    # Low coherence -> more linear exploration
    z_tanh = np.tanh(z_shifted)
    z_linear = z_shifted / (1 + np.abs(z_shifted) * 0.1)  # Soft linear

    z_mixed = coherence * z_tanh + (1 - coherence) * z_linear

    # Boundary influence - extract statistics
    b_mean = np.mean(b)
    b_std = np.std(b) + 1e-8
    b_energy = np.sum(b ** 2) / b.size

    # Boundary modulates dynamics
    boundary_factor = np.zeros(d)
    boundary_factor[:min(4, d)] = [b_mean, b_std, b_energy, b_mean * b_std][:min(4, d)]

    # Reflection feedback - self-awareness modulates exploration
    r_influence = 0.0
    if len(r) > 0:
        r_norm = np.linalg.norm(r)
        r_direction = r[:d] / (r_norm + 1e-8) if len(r) >= d else np.zeros(d)
        r_direction = np.pad(r_direction, (0, max(0, d - len(r_direction))))[:d]
        # High reflection norm -> slight pull toward reflection direction
        r_influence = 0.1 * r_norm * r_direction

    # Branch bias creates periodic attractor structure
    branch_phase = np.linspace(0, 2 * np.pi, d)
    branch_shift = alpha * np.sin(branch_phase + np.mean(z_shifted))

    # Combine all influences
    z_next = z_mixed + 0.05 * boundary_factor + r_influence + 0.03 * branch_shift

    # Stability-dependent regularization
    # Lower stability -> more regularization toward origin
    if stability < 0.5:
        regularization = (0.5 - stability) * 0.1 * z_next
        z_next = z_next - regularization

    # Soft normalization to prevent unbounded growth
    norm = np.linalg.norm(z_next)
    max_norm = 5.0 + 2.0 * stability  # More stable -> can explore further
    if norm > max_norm:
        z_next = z_next * (max_norm / norm)

    return z_next


def G_boundary(
    b: np.ndarray,
    z: np.ndarray,
    gamma: float,
    i: np.ndarray,
    config: Optional['ChildMindAIConfig'] = None
) -> np.ndarray:
    """
    Boundary grid dynamics function.

    Updates the holographic boundary based on:
    - Current manifold state z
    - Write strength gamma
    - Intention vector i
    - Diffusion for smoothing

    Args:
        b: Current boundary grid (d_b, d_b)
        z: Updated manifold state (d_z,)
        gamma: Boundary rewrite strength (0-1)
        i: Intention vector (d_i,)
        config: Configuration object

    Returns:
        b_next: Updated boundary grid (d_b, d_b)
    """
    H_b, W_b = b.shape
    diffusion = config.boundary_diffusion if config else 0.1

    # Map z to boundary location
    if len(z) >= 2:
        loc_i = int((np.tanh(z[0]) + 1) / 2 * (H_b - 1))
        loc_j = int((np.tanh(z[1]) + 1) / 2 * (W_b - 1))
    else:
        loc_i, loc_j = H_b // 2, W_b // 2

    loc_i = np.clip(loc_i, 0, H_b - 1)
    loc_j = np.clip(loc_j, 0, W_b - 1)

    # Apply diffusion first (smoothing)
    b_diffused = b.copy()
    for ii in range(H_b):
        for jj in range(W_b):
            neighbors = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ni, nj = (ii + di) % H_b, (jj + dj) % W_b
                    neighbors.append(b[ni, nj])
            b_diffused[ii, jj] = (1 - diffusion) * b[ii, jj] + diffusion * np.mean(neighbors)

    # Compute write value from z and intention
    z_contribution = np.tanh(np.sum(z[2:8]) if len(z) > 2 else z[0])
    i_contribution = np.tanh(np.sum(i[:4]) if len(i) > 0 else 0) * 0.3
    write_value = z_contribution + i_contribution

    # Create adaptive kernel based on intention strength
    i_norm = np.linalg.norm(i) if len(i) > 0 else 0.5
    kernel_spread = 1 + int(i_norm * 2)  # 1-3 spread based on intention

    # Apply local update with Gaussian-like kernel
    b_next = b_diffused.copy()
    gamma_clamped = np.clip(gamma, 0, 1)

    for di in range(-kernel_spread, kernel_spread + 1):
        for dj in range(-kernel_spread, kernel_spread + 1):
            ni = (loc_i + di) % H_b
            nj = (loc_j + dj) % W_b
            distance = np.sqrt(di**2 + dj**2)
            weight = np.exp(-distance**2 / (kernel_spread + 0.5)) * gamma_clamped
            update = weight * write_value
            b_next[ni, nj] = b_next[ni, nj] * (1 - weight * 0.3) + update

    return b_next


def H_coherence(
    c: np.ndarray,
    z: np.ndarray,
    z_prev: np.ndarray,
    beta: float,
    r: np.ndarray,
    config: Optional['ChildMindAIConfig'] = None
) -> np.ndarray:
    """
    Coherence state dynamics function.

    Updates coherence based on:
    - Movement in z space
    - Beta modulation
    - Reflection state influence
    - Target coherence attraction

    Args:
        c: Current coherence state (d_c,)
        z: New manifold state (d_z,)
        z_prev: Previous manifold state (d_z,)
        beta: Coherence modulation scalar
        r: Reflection state (d_r,)
        config: Configuration object

    Returns:
        c_next: Updated coherence state (d_c,)
    """
    target = config.coherence_target if config else 0.7
    k_c = len(c)
    c_next = np.zeros(k_c)

    z_norm = np.linalg.norm(z)
    z_change = np.linalg.norm(z - z_prev)
    r_norm = np.linalg.norm(r) if len(r) > 0 else 0

    # c[0]: Primary coherence - attracted to target, modulated by beta
    coherence_rate = 0.1 + 0.15 * np.abs(beta)
    target_adj = target * (1 + 0.2 * beta)  # Beta shifts target
    c_next[0] = c[0] + coherence_rate * (target_adj - c[0])
    # Large z changes reduce coherence
    c_next[0] -= 0.1 * z_change
    c_next[0] = np.clip(c_next[0], 0, 1)

    # c[1]: Secondary coherence - interaction strength
    if z_norm < 2.0:
        c_next[1] = c[1] * 0.95 + 0.05 * (1 - z_norm / 2.0)
    else:
        c_next[1] = c[1] * 0.9

    # c[2]: Phase alignment - from z components
    if len(z) >= 2:
        phase = np.arctan2(z[1], z[0] + 1e-8) / np.pi
        c_next[2] = c[2] * 0.85 + 0.15 * phase
    else:
        c_next[2] = c[2] * 0.9

    # c[3]: Stability - decreases with large z changes
    stability_delta = -0.15 * z_change + 0.05 * (1 - z_change)
    c_next[3] = np.clip(c[3] + stability_delta, 0, 1)
    # Reflection enhances stability
    c_next[3] += 0.02 * min(r_norm, 1.0)
    c_next[3] = np.clip(c_next[3], 0, 1)

    # c[4]: Novelty - distance from typical
    typical_z_norm = 0.5
    novelty = np.abs(z_norm - typical_z_norm) / 3.0
    c_next[4] = c[4] * 0.7 + 0.3 * np.clip(novelty, 0, 1)

    # Fill remaining coherence dimensions with derived metrics
    for idx in range(5, k_c):
        if idx == 5:
            # Coherence momentum
            c_next[idx] = c[idx] * 0.9 + 0.1 * (c_next[0] - c[0])
        elif idx == 6:
            # Stability trend
            c_next[idx] = c[idx] * 0.9 + 0.1 * (c_next[3] - c[3])
        else:
            # Decay other dimensions
            c_next[idx] = c[idx] * 0.95

    return c_next


def U_memory(
    m: np.ndarray,
    z: np.ndarray,
    c: np.ndarray,
    r: np.ndarray,
    i: np.ndarray,
    input_encoding: Optional[np.ndarray] = None,
    config: Optional['ChildMindAIConfig'] = None
) -> np.ndarray:
    """
    Memory summary dynamics function.

    Compresses current experience into memory vector using EMA.

    Args:
        m: Current memory summary (d_m,)
        z: Current manifold state (d_z,)
        c: Coherence state (d_c,)
        r: Reflection state (d_r,)
        i: Intention state (d_i,)
        input_encoding: Optional encoding of user input
        config: Configuration object

    Returns:
        m_next: Updated memory summary (d_m,)
    """
    decay = config.memory_decay if config else 0.95
    d_m = len(m)

    # Extract features from current state
    features = np.zeros(d_m)

    # z statistics (first quarter)
    q1 = d_m // 4
    features[0] = np.mean(z)
    features[1] = np.std(z)
    features[2] = np.max(z) if len(z) > 0 else 0
    features[3] = np.min(z) if len(z) > 0 else 0
    features[4] = np.linalg.norm(z)

    # Coherence features (second quarter)
    q2 = d_m // 2
    for idx, val in enumerate(c[:min(len(c), q1 - 5)]):
        features[5 + idx] = val

    # Reflection features (third quarter)
    q3 = 3 * d_m // 4
    r_start = q2
    features[r_start] = np.linalg.norm(r) if len(r) > 0 else 0
    features[r_start + 1] = np.mean(r) if len(r) > 0 else 0
    for idx, val in enumerate(r[:min(len(r), q3 - q2 - 2)]):
        features[r_start + 2 + idx] = val

    # Intention features (fourth quarter)
    i_start = q3
    features[i_start] = np.linalg.norm(i) if len(i) > 0 else 0
    features[i_start + 1] = np.argmax(np.abs(i)) if len(i) > 0 else 0
    for idx, val in enumerate(i[:min(len(i), d_m - i_start - 2)]):
        features[i_start + 2 + idx] = val

    # Incorporate input encoding if provided
    if input_encoding is not None:
        # Blend input encoding into memory
        input_weight = 0.3
        encoding_size = min(len(input_encoding), d_m // 8)
        features[-encoding_size:] = input_encoding[:encoding_size]
        decay = decay * (1 - input_weight) + input_weight * 0.5  # Faster update with input

    # EMA update
    m_next = decay * m + (1 - decay) * features

    return m_next


def R_reflect(
    r: np.ndarray,
    z: np.ndarray,
    z_history: list,
    c: np.ndarray,
    m: np.ndarray,
    config: Optional['ChildMindAIConfig'] = None
) -> np.ndarray:
    """
    Self-reflection dynamics function.

    Updates the self-model based on:
    - Current state observation
    - Historical patterns
    - Coherence assessment
    - Memory integration

    Args:
        r: Current reflection state (d_r,)
        z: Current manifold state (d_z,)
        z_history: Recent z history
        c: Coherence state (d_c,)
        m: Memory summary (d_m,)
        config: Configuration object

    Returns:
        r_next: Updated reflection state (d_r,)
    """
    depth = config.reflection_depth if config else 3
    d_r = len(r)
    r_next = np.zeros(d_r)

    # Self-observation of current state
    z_norm = np.linalg.norm(z)
    z_mean = np.mean(z)

    # Historical pattern analysis
    if len(z_history) >= 3:
        recent_norms = [np.linalg.norm(zh) for zh in z_history[-depth:]]
        trajectory_variance = np.var(recent_norms)
        trajectory_trend = recent_norms[-1] - recent_norms[0] if len(recent_norms) > 1 else 0
    else:
        trajectory_variance = 0.5
        trajectory_trend = 0

    # Coherence self-assessment
    coherence = c[0] if len(c) > 0 else 0.5
    stability = c[3] if len(c) > 3 else 0.5

    # Build reflection vector
    # First section: immediate self-observation
    r_next[0] = z_norm  # How "far" am I from center
    r_next[1] = z_mean  # General direction
    r_next[2] = coherence  # How coherent am I
    r_next[3] = stability  # How stable am I

    # Second section: pattern awareness
    r_next[4] = trajectory_variance  # Am I jumping around?
    r_next[5] = trajectory_trend  # Am I expanding or contracting?

    # Third section: meta-observation (reflection on reflection)
    prev_r_norm = np.linalg.norm(r)
    r_next[6] = prev_r_norm  # How much was I reflecting before?
    r_next[7] = np.mean(r[:6]) if len(r) > 6 else 0  # Tone of previous reflection

    # Fourth section: memory-reflection integration
    m_activation = np.mean(np.abs(m)) if len(m) > 0 else 0
    r_next[8] = m_activation  # How active is my memory?

    # Fill remaining with decay from previous + noise for exploration
    for idx in range(9, d_r):
        if idx < len(r):
            r_next[idx] = r[idx] * 0.9  # Slow decay
        else:
            r_next[idx] = 0

    # Add small exploration noise to reflection
    r_next += np.random.normal(0, 0.01, d_r)

    return r_next


def I_intent(
    i: np.ndarray,
    z: np.ndarray,
    c: np.ndarray,
    r: np.ndarray,
    input_encoding: Optional[np.ndarray] = None,
    reward: float = 0.0,
    config: Optional['ChildMindAIConfig'] = None
) -> np.ndarray:
    """
    Intention dynamics function.

    Updates goals/desires based on:
    - Input from user (encoded)
    - Current state assessment
    - Reflection insights
    - Reward signals

    Args:
        i: Current intention state (d_i,)
        z: Current manifold state (d_z,)
        c: Coherence state (d_c,)
        r: Reflection state (d_r,)
        input_encoding: Optional encoding of user input
        reward: Reward signal from recent action
        config: Configuration object

    Returns:
        i_next: Updated intention state (d_i,)
    """
    d_i = len(i)
    i_next = np.zeros(d_i)

    # Base intention dimensions:
    # i[0]: Curiosity/exploration drive
    # i[1]: Understanding/modeling drive
    # i[2]: Helpfulness/creation drive
    # i[3]: Self-preservation/stability drive
    # i[4+]: Task-specific intentions

    coherence = c[0] if len(c) > 0 else 0.5
    stability = c[3] if len(c) > 3 else 0.5
    novelty = c[4] if len(c) > 4 else 0.3

    # Curiosity increases when novelty is low, decreases when unstable
    i_next[0] = i[0] * 0.9 + 0.1 * (1 - novelty) - 0.05 * (1 - stability)

    # Understanding drive increases with coherence success
    i_next[1] = i[1] * 0.9 + 0.1 * coherence

    # Helpfulness drive - baseline with input boost
    i_next[2] = i[2] * 0.95 + 0.05 * 0.5  # Baseline helpfulness

    # Self-preservation increases when stability is threatened
    i_next[3] = i[3] * 0.9 + 0.1 * (1 - stability)

    # Reward modulates all intentions
    if reward != 0:
        reward_influence = np.tanh(reward) * 0.1
        i_next[:4] += reward_influence

    # Input encoding shifts intentions toward user's apparent goals
    if input_encoding is not None and len(input_encoding) >= 4:
        input_weight = 0.3
        for idx in range(min(4, len(input_encoding))):
            i_next[idx] = (1 - input_weight) * i_next[idx] + input_weight * input_encoding[idx]

    # Reflection insights modulate intentions
    if len(r) > 5:
        # If reflecting on being "stuck", boost curiosity
        if r[4] < 0.1:  # Low trajectory variance
            i_next[0] += 0.1
        # If reflecting on instability, boost preservation
        if r[5] < -0.5:  # Contracting trend
            i_next[3] += 0.1

    # Fill remaining intention dimensions
    for idx in range(4, d_i):
        i_next[idx] = i[idx] * 0.95  # Slow decay of task-specific intentions

    # Normalize to prevent unbounded growth
    i_norm = np.linalg.norm(i_next)
    if i_norm > 2.0:
        i_next = i_next * (2.0 / i_norm)

    # Ensure non-negative for drive-like quantities
    i_next[:4] = np.clip(i_next[:4], 0, 1)

    return i_next
