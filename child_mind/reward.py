"""
Child Mind v1 — Reward Functions
MindFractal Lab

Implements the reward function:
    r_t = w_coh * r_coh + w_nov * r_nov + w_stab * r_stab

Components:
    r_coh:  Coherence reward (being near target coherence zone)
    r_nov:  Novelty reward (distance from rolling centroid)
    r_stab: Stability penalty (large jumps and extreme changes)

v1 weights:
    w_coh  = 1.0
    w_nov  = 0.5
    w_stab = 0.3
"""

import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import State, Action


@dataclass
class RewardWeights:
    """Weights for combining reward components."""
    w_coh: float = 1.0    # Coherence weight
    w_nov: float = 0.5    # Novelty weight
    w_stab: float = 0.3   # Stability weight

    # Target and bounds
    coherence_target: float = 0.7
    coherence_tolerance: float = 0.15
    novelty_cap: float = 2.0
    stability_threshold: float = 1.0


def compute_coherence_reward(
    state: 'State',
    weights: RewardWeights = None
) -> float:
    """
    Compute coherence reward component.

    Rewards being near the target coherence zone.
    Uses a Gaussian-like falloff from target.

    Args:
        state: Current state
        weights: Reward configuration

    Returns:
        r_coh: Coherence reward (0 to 1)
    """
    if weights is None:
        weights = RewardWeights()

    # Primary coherence metric is c[0]
    coherence = state.c[0] if len(state.c) > 0 else 0.5

    # Distance from target
    distance = abs(coherence - weights.coherence_target)

    # Gaussian-like reward
    r_coh = np.exp(-0.5 * (distance / weights.coherence_tolerance) ** 2)

    return float(r_coh)


def compute_novelty_reward(
    prev_state: 'State',
    next_state: 'State',
    weights: RewardWeights = None
) -> float:
    """
    Compute novelty reward component.

    Proportional to distance of z_{t+1} from a rolling centroid
    estimated from memory, with a cap to prevent reward hacking.

    Args:
        prev_state: Previous state (contains memory with centroid estimate)
        next_state: New state after action
        weights: Reward configuration

    Returns:
        r_nov: Novelty reward (0 to 1)
    """
    if weights is None:
        weights = RewardWeights()

    # Estimate centroid from memory
    m = prev_state.m

    # Memory stores z statistics in first few components
    centroid_mean = m[0] if len(m) > 0 else 0
    centroid_std = max(m[1], 0.1) if len(m) > 1 else 0.5

    # Distance of new z from estimated centroid
    z_mean = np.mean(next_state.z)
    distance = abs(z_mean - centroid_mean) / centroid_std

    # Also consider full z distance from typical
    z_norm = np.linalg.norm(next_state.z)
    prev_z_norm = np.linalg.norm(prev_state.z)
    z_change = abs(z_norm - prev_z_norm)

    # Combined novelty measure
    novelty = 0.5 * min(distance, weights.novelty_cap) / weights.novelty_cap
    novelty += 0.5 * min(z_change, weights.novelty_cap) / weights.novelty_cap

    # Reward is proportional but capped
    r_nov = np.clip(novelty, 0, 1)

    return float(r_nov)


def compute_stability_penalty(
    prev_state: 'State',
    action: 'Action',
    next_state: 'State',
    weights: RewardWeights = None
) -> float:
    """
    Compute stability penalty component.

    Penalizes:
    - Very large jumps ||Δz_t||
    - Extreme changes in coherence c_t
    - Large boundary changes b_t

    Args:
        prev_state: Previous state
        action: Action taken
        next_state: New state
        weights: Reward configuration

    Returns:
        r_stab: Stability reward (0 to 1, where 1 = stable)
    """
    if weights is None:
        weights = RewardWeights()

    penalties = 0.0

    # Penalty for large delta_z
    delta_z_norm = np.linalg.norm(action.delta_z)
    if delta_z_norm > weights.stability_threshold:
        penalties += 0.3 * (delta_z_norm - weights.stability_threshold)

    # Penalty for coherence drop
    coherence_change = abs(next_state.c[0] - prev_state.c[0])
    if coherence_change > 0.2:
        penalties += 0.3 * (coherence_change - 0.2)

    # Penalty for boundary turbulence
    b_change = np.mean(np.abs(next_state.b - prev_state.b))
    if b_change > 0.1:
        penalties += 0.2 * (b_change - 0.1)

    # Penalty for extreme action parameters
    if abs(action.alpha) > 1.0:
        penalties += 0.1 * (abs(action.alpha) - 1.0)
    if abs(action.beta) > 1.0:
        penalties += 0.1 * (abs(action.beta) - 1.0)
    if abs(action.gamma) > 1.0:
        penalties += 0.1 * (abs(action.gamma) - 1.0)

    # Convert penalty to reward (higher is better)
    r_stab = max(0, 1 - penalties)

    return float(r_stab)


def compute_rewards(
    prev_state: 'State',
    action: 'Action',
    next_state: 'State',
    weights: RewardWeights = None
) -> float:
    """
    Compute total reward for a transition.

    r_t = w_coh * r_coh + w_nov * r_nov + w_stab * r_stab

    Args:
        prev_state: State before action
        action: Action taken
        next_state: State after action
        weights: Reward configuration

    Returns:
        Total reward r_t
    """
    if weights is None:
        weights = RewardWeights()

    r_coh = compute_coherence_reward(next_state, weights)
    r_nov = compute_novelty_reward(prev_state, next_state, weights)
    r_stab = compute_stability_penalty(prev_state, action, next_state, weights)

    total_reward = (
        weights.w_coh * r_coh +
        weights.w_nov * r_nov +
        weights.w_stab * r_stab
    )

    return float(total_reward)


def get_reward_breakdown(
    prev_state: 'State',
    action: 'Action',
    next_state: 'State',
    weights: RewardWeights = None
) -> dict:
    """
    Get detailed breakdown of reward components.

    Returns dict with individual components and total.
    """
    if weights is None:
        weights = RewardWeights()

    r_coh = compute_coherence_reward(next_state, weights)
    r_nov = compute_novelty_reward(prev_state, next_state, weights)
    r_stab = compute_stability_penalty(prev_state, action, next_state, weights)

    return {
        'coherence': {
            'raw': r_coh,
            'weighted': weights.w_coh * r_coh
        },
        'novelty': {
            'raw': r_nov,
            'weighted': weights.w_nov * r_nov
        },
        'stability': {
            'raw': r_stab,
            'weighted': weights.w_stab * r_stab
        },
        'total': (
            weights.w_coh * r_coh +
            weights.w_nov * r_nov +
            weights.w_stab * r_stab
        )
    }
