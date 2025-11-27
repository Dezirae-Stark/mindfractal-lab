"""
Child Mind v1 — Pyodide Browser Wrapper
MindFractal Lab

Thin wrapper for running Child Mind in the browser via Pyodide.
Provides simple functions for initialization and stepping.

This module is designed to be loaded and executed in Pyodide.
It does NOT use any heavyweight frameworks - only numpy.
"""

import numpy as np
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass


# ============================================================================
# Inline Implementation (for Pyodide - no import from child_mind package)
# ============================================================================

@dataclass
class ChildMindConfig:
    """Configuration for Child Mind v1."""
    d: int = 16
    H_b: int = 16
    W_b: int = 16
    k_c: int = 5
    d_m: int = 16
    coherence_target: float = 0.7
    memory_decay: float = 0.9
    boundary_diffusion: float = 0.1


class State:
    """Complete state of the child mind."""
    def __init__(self, z, b, c, m, t=0):
        self.z = z
        self.b = b
        self.c = c
        self.m = m
        self.t = t

    def copy(self):
        return State(
            z=self.z.copy(),
            b=self.b.copy(),
            c=self.c.copy(),
            m=self.m.copy(),
            t=self.t
        )


class Action:
    """Action taken by the child mind."""
    def __init__(self, delta_z, alpha=0.0, beta=0.0, gamma=0.0):
        self.delta_z = delta_z
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma


# Global state registry
_child_mind_state: Optional[State] = None
_child_mind_config: ChildMindConfig = ChildMindConfig()
_child_mind_rng: Optional[np.random.Generator] = None
_child_mind_history: list = []


def _initial_state(seed: int, config: ChildMindConfig) -> State:
    """Create initial state."""
    rng = np.random.default_rng(seed)

    z = rng.normal(0, 0.1, size=(config.d,))

    b = rng.normal(0, 0.1, size=(config.H_b, config.W_b))
    for _ in range(3):
        b_smooth = np.zeros_like(b)
        for i in range(config.H_b):
            for j in range(config.W_b):
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = (i + di) % config.H_b, (j + dj) % config.W_b
                        neighbors.append(b[ni, nj])
                b_smooth[i, j] = np.mean(neighbors)
        b = b_smooth

    c = np.array([config.coherence_target, 0.5, 0.0, 0.5, 0.0])
    m = np.zeros(config.d_m)

    return State(z=z, b=b, c=c, m=m, t=0)


def _F_mind(z, c, b, alpha):
    """Core manifold dynamics."""
    d = len(z)
    b_mean = np.mean(b)
    b_std = np.std(b) + 1e-8
    b_max = np.max(b)
    b_min = np.min(b)
    coherence = c[0] if len(c) > 0 else 0.5

    z_tanh = np.tanh(z)
    z_softplus = np.log1p(np.exp(np.clip(z, -20, 20)))
    z_mixed = coherence * z_tanh + (1 - coherence) * (z_softplus - 1)

    boundary_influence = np.zeros(d)
    boundary_influence[0] = b_mean * 0.5
    boundary_influence[1] = (b_max - b_min) * 0.3
    if d > 2:
        boundary_influence[2] = b_std * 0.4

    branch_shift = alpha * np.sin(np.linspace(0, 2 * np.pi, d))
    z_next = z_mixed + 0.1 * boundary_influence[:d] + 0.05 * branch_shift

    norm = np.linalg.norm(z_next)
    if norm > 5.0:
        z_next = z_next * (5.0 / norm)

    return z_next


def _G_boundary(b, z, gamma, config):
    """Boundary grid dynamics."""
    H_b, W_b = b.shape

    if len(z) >= 2:
        loc_i = int((np.tanh(z[0]) + 1) / 2 * (H_b - 1))
        loc_j = int((np.tanh(z[1]) + 1) / 2 * (W_b - 1))
    else:
        loc_i, loc_j = H_b // 2, W_b // 2

    loc_i = max(0, min(H_b - 1, loc_i))
    loc_j = max(0, min(W_b - 1, loc_j))

    kernel = np.array([
        [0.05, 0.1, 0.05],
        [0.1,  0.4, 0.1],
        [0.05, 0.1, 0.05]
    ])

    write_value = np.tanh(np.sum(z[2:6]) if len(z) > 2 else z[0])

    diffusion = config.boundary_diffusion
    b_diffused = b.copy()
    for i in range(H_b):
        for j in range(W_b):
            neighbors = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ni, nj = (i + di) % H_b, (j + dj) % W_b
                    neighbors.append(b[ni, nj])
            b_diffused[i, j] = (1 - diffusion) * b[i, j] + diffusion * np.mean(neighbors)

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


def _H_coherence(c, beta, z, config):
    """Coherence dynamics."""
    target = config.coherence_target
    k_c = len(c)
    c_next = np.zeros(k_c)
    z_norm = np.linalg.norm(z)

    coherence_rate = 0.1 + 0.2 * np.abs(beta)
    c_next[0] = c[0] + coherence_rate * (target * (1 + 0.3 * beta) - c[0])
    c_next[0] = np.clip(c_next[0], 0, 1)

    if z_norm < 2.0:
        c_next[1] = c[1] * 0.95 + 0.05 * (1 - z_norm / 2.0)
    else:
        c_next[1] = c[1] * 0.9

    if len(z) >= 2:
        phase = np.arctan2(z[1], z[0] + 1e-8) / np.pi
        c_next[2] = c[2] * 0.8 + 0.2 * phase
    else:
        c_next[2] = c[2] * 0.9

    stability_change = -0.1 * (z_norm - 1.0) if z_norm > 1.0 else 0.05
    c_next[3] = np.clip(c[3] + stability_change, 0, 1)

    typical_z = 0.5
    novelty = np.abs(z_norm - typical_z) / 3.0
    c_next[4] = c[4] * 0.7 + 0.3 * np.clip(novelty, 0, 1)

    return c_next


def _U_memory(m, state, action, config):
    """Memory dynamics."""
    decay = config.memory_decay
    d_m = len(m)
    features = np.zeros(d_m)

    features[0] = np.mean(state.z)
    features[1] = np.std(state.z)
    features[2] = np.max(state.z)
    features[3] = np.min(state.z)
    features[4] = state.c[0]
    features[5] = state.c[3] if len(state.c) > 3 else 0
    features[6] = state.c[4] if len(state.c) > 4 else 0
    features[7] = np.mean(state.b)

    half = d_m // 2
    features[half] = np.mean(action.delta_z)
    features[half + 1] = np.std(action.delta_z)
    features[half + 2] = np.linalg.norm(action.delta_z)
    features[half + 3] = action.alpha
    features[half + 4] = action.beta
    features[half + 5] = action.gamma
    features[-1] = np.sin(state.t * 0.1)

    return decay * m + (1 - decay) * features


def _compute_rewards(prev_state, action, next_state):
    """Compute reward."""
    w_coh, w_nov, w_stab = 1.0, 0.5, 0.3

    # Coherence reward
    coherence = next_state.c[0]
    target = 0.7
    r_coh = np.exp(-0.5 * ((coherence - target) / 0.15) ** 2)

    # Novelty reward
    z_change = np.abs(np.linalg.norm(next_state.z) - np.linalg.norm(prev_state.z))
    r_nov = np.clip(z_change / 2.0, 0, 1)

    # Stability reward
    penalties = 0.0
    delta_z_norm = np.linalg.norm(action.delta_z)
    if delta_z_norm > 1.0:
        penalties += 0.3 * (delta_z_norm - 1.0)
    coherence_change = abs(next_state.c[0] - prev_state.c[0])
    if coherence_change > 0.2:
        penalties += 0.3 * (coherence_change - 0.2)
    r_stab = max(0, 1 - penalties)

    return w_coh * r_coh + w_nov * r_nov + w_stab * r_stab


def _bounded_random_policy(state, rng, curiosity, coherence_preference, config):
    """Generate bounded random action."""
    base_std = 0.1 + 0.4 * curiosity
    delta_z = rng.normal(0, base_std, size=(config.d,))

    if curiosity > 0.7 and rng.random() < 0.2:
        jump_direction = rng.choice(config.d)
        delta_z[jump_direction] += rng.choice([-1, 1]) * curiosity

    current_coherence = state.c[0]
    alpha_center = 0.2 if current_coherence < 0.5 else 0.0
    alpha = rng.normal(alpha_center, 0.3 * (1 - coherence_preference))

    beta_center = 0.2 * (coherence_preference - 0.5)
    beta = rng.normal(beta_center, 0.2)

    gamma_max = 0.3 + 0.3 * curiosity
    gamma = rng.uniform(0, gamma_max)

    return Action(
        delta_z=delta_z,
        alpha=float(np.clip(alpha, -1.5, 1.5)),
        beta=float(np.clip(beta, -1.0, 1.0)),
        gamma=float(np.clip(gamma, 0, 1.0))
    )


def _step(state, action, config):
    """Step the simulation."""
    z_next = _F_mind(state.z + action.delta_z, state.c, state.b, action.alpha)
    b_next = _G_boundary(state.b, z_next, action.gamma, config)
    c_next = _H_coherence(state.c, action.beta, z_next, config)
    m_next = _U_memory(state.m, state, action, config)

    next_state = State(z=z_next, b=b_next, c=c_next, m=m_next, t=state.t + 1)
    reward = _compute_rewards(state, action, next_state)

    return next_state, reward


def _get_state_summary(state):
    """Get human-readable state summary."""
    coherence = state.c[0]
    stability = state.c[3]
    novelty = state.c[4]
    z_magnitude = np.linalg.norm(state.z)

    coh_level = 'high' if coherence > 0.7 else ('medium' if coherence > 0.4 else 'low')
    stab_level = 'stable' if stability > 0.6 else ('moderate' if stability > 0.3 else 'chaotic')
    nov_level = 'discovering' if novelty > 0.5 else ('exploring' if novelty > 0.2 else 'routine')

    descriptions = {
        ('high', 'stable', 'exploring'): "The child mind is in a stable but exploratory mode.",
        ('high', 'stable', 'routine'): "The child mind is in a calm, focused state.",
        ('high', 'moderate', 'exploring'): "The child mind is actively exploring while staying coherent.",
        ('medium', 'moderate', 'exploring'): "The child mind is curious, probing its environment.",
        ('low', 'chaotic', 'exploring'): "The child mind is jumping between distant states.",
        ('low', 'chaotic', 'discovering'): "The child mind is in full creative chaos mode.",
    }

    key = (coh_level, stab_level, nov_level)
    description = descriptions.get(key, f"The child mind is {coh_level} coherence, {stab_level}, {nov_level}.")

    return {
        'coherence_level': coh_level,
        'coherence_value': float(coherence),
        'stability': stab_level,
        'stability_value': float(stability),
        'novelty': nov_level,
        'novelty_value': float(novelty),
        'z_magnitude': float(z_magnitude),
        'timestep': state.t,
        'description': description
    }


def _state_to_dict(state):
    """Convert state to dict."""
    return {
        'z': state.z[:4].tolist(),
        'z_norm': float(np.linalg.norm(state.z)),
        'b_mean': float(np.mean(state.b)),
        'b_std': float(np.std(state.b)),
        'c': state.c.tolist(),
        'm_norm': float(np.linalg.norm(state.m)),
        't': state.t,
        'summary': _get_state_summary(state)
    }


# ============================================================================
# Public API for Pyodide
# ============================================================================

def reset_child_mind(seed: int = 42) -> str:
    """
    Initialize or reset the child mind.

    Args:
        seed: Random seed for reproducibility

    Returns:
        JSON string with initial state snapshot
    """
    global _child_mind_state, _child_mind_rng, _child_mind_history

    _child_mind_rng = np.random.default_rng(seed)
    _child_mind_state = _initial_state(seed, _child_mind_config)
    _child_mind_history = [_state_to_dict(_child_mind_state)]

    return json.dumps({
        'success': True,
        'state': _state_to_dict(_child_mind_state),
        'config': {
            'd': _child_mind_config.d,
            'H_b': _child_mind_config.H_b,
            'W_b': _child_mind_config.W_b,
            'k_c': _child_mind_config.k_c,
            'd_m': _child_mind_config.d_m
        }
    })


def step_child_mind(
    n_steps: int = 1,
    curiosity: float = 0.5,
    coherence_preference: float = 0.5
) -> str:
    """
    Step the child mind simulation.

    Args:
        n_steps: Number of steps to run
        curiosity: 0-1, higher = more exploration
        coherence_preference: 0-1, higher = favor coherence

    Returns:
        JSON string with trajectory data
    """
    global _child_mind_state, _child_mind_history

    if _child_mind_state is None or _child_mind_rng is None:
        return json.dumps({
            'success': False,
            'error': 'Child mind not initialized. Call reset_child_mind first.'
        })

    states = []
    rewards = []

    curiosity = float(np.clip(curiosity, 0, 1))
    coherence_preference = float(np.clip(coherence_preference, 0, 1))

    for _ in range(n_steps):
        action = _bounded_random_policy(
            _child_mind_state,
            _child_mind_rng,
            curiosity,
            coherence_preference,
            _child_mind_config
        )

        next_state, reward = _step(_child_mind_state, action, _child_mind_config)

        _child_mind_state = next_state
        state_dict = _state_to_dict(next_state)
        states.append(state_dict)
        rewards.append(float(reward))
        _child_mind_history.append(state_dict)

        # Keep history bounded
        if len(_child_mind_history) > 200:
            _child_mind_history = _child_mind_history[-100:]

    return json.dumps({
        'success': True,
        'states': states,
        'rewards': rewards,
        'total_reward': float(sum(rewards)),
        'average_reward': float(sum(rewards) / len(rewards)) if rewards else 0.0,
        'current_state': _state_to_dict(_child_mind_state)
    })


def get_child_mind_state() -> str:
    """Get current state without stepping."""
    global _child_mind_state

    if _child_mind_state is None:
        return json.dumps({
            'success': False,
            'error': 'Child mind not initialized.'
        })

    return json.dumps({
        'success': True,
        'state': _state_to_dict(_child_mind_state)
    })


def get_child_mind_history(last_n: int = 50) -> str:
    """Get recent state history."""
    global _child_mind_history

    return json.dumps({
        'success': True,
        'history': _child_mind_history[-last_n:],
        'total_length': len(_child_mind_history)
    })


# Export for Pyodide
__all__ = [
    'reset_child_mind',
    'step_child_mind',
    'get_child_mind_state',
    'get_child_mind_history'
]
