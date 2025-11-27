"""
Child Mind v1 — Synthetic Agent Module
MindFractal Lab

A research sandbox for exploring mind-like dynamics in a synthetic agent.
This module implements a formal state-action-reward system for studying
emergent behavior in consciousness-inspired mathematical spaces.

State space: s_t = (z_t, b_t, c_t, m_t)
Action space: a_t = (Δz_t, α_t, β_t, γ_t)

See core.py for main API.
"""

from .core import (
    State,
    Action,
    initial_state,
    step,
    ChildMindConfig,
)

from .dynamics import (
    F_mind,
    G_boundary,
    H_coherence,
    U_memory,
)

from .reward import (
    compute_rewards,
    RewardWeights,
)

from .policy import (
    random_policy,
    bounded_random_policy,
)

from .serialization import (
    state_to_dict,
    dict_to_state,
    action_to_dict,
    dict_to_action,
)

__version__ = "1.0.0"
__all__ = [
    # Core
    "State",
    "Action",
    "initial_state",
    "step",
    "ChildMindConfig",
    # Dynamics
    "F_mind",
    "G_boundary",
    "H_coherence",
    "U_memory",
    # Reward
    "compute_rewards",
    "RewardWeights",
    # Policy
    "random_policy",
    "bounded_random_policy",
    # Serialization
    "state_to_dict",
    "dict_to_state",
    "action_to_dict",
    "dict_to_action",
]
