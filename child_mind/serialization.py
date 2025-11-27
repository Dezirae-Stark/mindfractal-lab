"""
Child Mind v1 — Serialization Utilities
MindFractal Lab

Helpers to convert State and Action to/from JSON-serializable dicts.
Used for browser (Pyodide) integration and trajectory storage.
"""

import numpy as np
from typing import Dict, Any, List
from .core import State, Action, ChildMindConfig, get_state_summary


def state_to_dict(state: State) -> Dict[str, Any]:
    """
    Convert a State object to a JSON-serializable dictionary.

    Args:
        state: State object

    Returns:
        Dictionary with all state components as lists
    """
    return {
        'z': state.z.tolist(),
        'b': state.b.tolist(),
        'c': state.c.tolist(),
        'm': state.m.tolist(),
        't': state.t,
        # Include computed summaries
        'summary': get_state_summary(state)
    }


def dict_to_state(d: Dict[str, Any]) -> State:
    """
    Convert a dictionary back to a State object.

    Args:
        d: Dictionary with state components

    Returns:
        State object
    """
    return State(
        z=np.array(d['z'], dtype=np.float64),
        b=np.array(d['b'], dtype=np.float64),
        c=np.array(d['c'], dtype=np.float64),
        m=np.array(d['m'], dtype=np.float64),
        t=d.get('t', 0)
    )


def action_to_dict(action: Action) -> Dict[str, Any]:
    """
    Convert an Action object to a JSON-serializable dictionary.

    Args:
        action: Action object

    Returns:
        Dictionary with all action components
    """
    return {
        'delta_z': action.delta_z.tolist(),
        'alpha': float(action.alpha),
        'beta': float(action.beta),
        'gamma': float(action.gamma)
    }


def dict_to_action(d: Dict[str, Any]) -> Action:
    """
    Convert a dictionary back to an Action object.

    Args:
        d: Dictionary with action components

    Returns:
        Action object
    """
    return Action(
        delta_z=np.array(d['delta_z'], dtype=np.float64),
        alpha=float(d.get('alpha', 0)),
        beta=float(d.get('beta', 0)),
        gamma=float(d.get('gamma', 0))
    )


def trajectory_to_dict(
    states: List[State],
    actions: List[Action],
    rewards: List[float]
) -> Dict[str, Any]:
    """
    Convert a full trajectory to a JSON-serializable dictionary.

    Args:
        states: List of State objects (length T+1)
        actions: List of Action objects (length T)
        rewards: List of rewards (length T)

    Returns:
        Dictionary containing full trajectory data
    """
    return {
        'states': [state_to_dict(s) for s in states],
        'actions': [action_to_dict(a) for a in actions],
        'rewards': [float(r) for r in rewards],
        'length': len(rewards),
        'total_reward': float(sum(rewards)),
        'average_reward': float(sum(rewards) / len(rewards)) if rewards else 0.0
    }


def compact_state_to_dict(state: State) -> Dict[str, Any]:
    """
    Convert a State to a compact dictionary for visualization.

    Omits full boundary grid, includes only statistics.
    Useful for sending many states to browser.

    Args:
        state: State object

    Returns:
        Compact dictionary
    """
    return {
        'z': state.z[:4].tolist(),  # First 4 components
        'z_full_norm': float(np.linalg.norm(state.z)),
        'z_mean': float(np.mean(state.z)),
        'b_mean': float(np.mean(state.b)),
        'b_std': float(np.std(state.b)),
        'c': state.c.tolist(),
        'm_norm': float(np.linalg.norm(state.m)),
        't': state.t,
        'summary': get_state_summary(state)
    }


def trajectory_to_compact_dict(
    states: List[State],
    rewards: List[float]
) -> Dict[str, Any]:
    """
    Convert trajectory to compact format for visualization.

    Args:
        states: List of State objects
        rewards: List of rewards

    Returns:
        Compact trajectory dictionary
    """
    return {
        'states': [compact_state_to_dict(s) for s in states],
        'rewards': [float(r) for r in rewards],
        'length': len(rewards),
        'total_reward': float(sum(rewards)),
        'average_reward': float(sum(rewards) / len(rewards)) if rewards else 0.0,
        # Trajectory statistics
        'coherence_trajectory': [float(s.c[0]) for s in states],
        'stability_trajectory': [float(s.c[3]) if len(s.c) > 3 else 0.5 for s in states],
        'novelty_trajectory': [float(s.c[4]) if len(s.c) > 4 else 0.0 for s in states],
    }


def config_to_dict(config: ChildMindConfig) -> Dict[str, Any]:
    """
    Convert configuration to dictionary.

    Args:
        config: ChildMindConfig object

    Returns:
        Dictionary with config values
    """
    return {
        'd': config.d,
        'H_b': config.H_b,
        'W_b': config.W_b,
        'k_c': config.k_c,
        'd_m': config.d_m,
        'coherence_target': config.coherence_target,
        'memory_decay': config.memory_decay,
        'boundary_diffusion': config.boundary_diffusion
    }
