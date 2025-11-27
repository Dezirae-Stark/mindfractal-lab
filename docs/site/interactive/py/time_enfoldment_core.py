"""
Time Enfoldment Core — Bidirectional Timeline Computation
MindFractal Lab

Pyodide-compatible module for visualizing past-future bidirectional time structures.
"""

import numpy as np
from typing import Dict, List, Tuple
import random


def compute_time_enfoldment(params: Dict) -> Dict:
    """
    Compute a bidirectional timeline structure with past and future branches.

    Parameters
    ----------
    params : dict
        depth : int - Number of levels in each direction (1-8)
        branching_factor : float - Average branches per node (1-4)
        retro_weight : float - Past influence on future (0-1)
        decoherence : float - Branch fading rate (0-1)
        seed : int - Random seed

    Returns
    -------
    dict
        present : [x, y] - Present moment position
        past_nodes : List of [x, y, level, weight] - Past timeline nodes
        future_nodes : List of [x, y, level, weight] - Future timeline nodes
        edges : List of [from_idx, to_idx, direction] - Connections
    """
    depth = min(max(int(params.get('depth', 4)), 1), 8)
    branching_factor = float(params.get('branching_factor', 2.0))
    retro_weight = float(params.get('retro_weight', 0.3))
    decoherence = float(params.get('decoherence', 0.2))
    seed = int(params.get('seed', 42))

    np.random.seed(seed)
    random.seed(seed)

    present = [0.5, 0.5]
    past_nodes = []
    future_nodes = []
    edges = []

    # Generate past branches (left side)
    past_idx_offset = 0
    current_level_nodes = [{'idx': -1, 'y': 0.5, 'weight': 1.0}]  # -1 = present

    for level in range(1, depth + 1):
        new_level_nodes = []
        x = 0.5 - (level / depth) * 0.45

        for parent in current_level_nodes:
            # Number of branches from this node
            n_branches = max(1, int(np.random.poisson(branching_factor)))
            n_branches = min(n_branches, 4)

            parent_weight = parent['weight']
            weight_per_branch = parent_weight / n_branches * (1 - decoherence * 0.3)

            for b in range(n_branches):
                # Y position spreads out
                spread = 0.15 * level / depth
                y = parent['y'] + np.random.uniform(-spread, spread)
                y = np.clip(y, 0.05, 0.95)

                weight = weight_per_branch * (1 + np.random.uniform(-0.1, 0.1))
                weight = max(0.1, weight)

                node_idx = len(past_nodes)
                past_nodes.append([float(x), float(y), level, float(weight)])
                new_level_nodes.append({'idx': node_idx, 'y': y, 'weight': weight})

                # Edge from parent
                parent_idx = parent['idx']
                edges.append([parent_idx, node_idx, 'past'])

        current_level_nodes = new_level_nodes
        if not current_level_nodes:
            break

    # Generate future branches (right side)
    # Future is influenced by past through retro_weight
    future_idx_offset = len(past_nodes)
    current_level_nodes = [{'idx': -1, 'y': 0.5, 'weight': 1.0}]

    for level in range(1, depth + 1):
        new_level_nodes = []
        x = 0.5 + (level / depth) * 0.45

        # Retrocausal influence: past structure affects future branching
        past_influence = 0
        if past_nodes and retro_weight > 0:
            past_at_level = [p for p in past_nodes if p[2] == level]
            if past_at_level:
                past_influence = np.mean([p[3] for p in past_at_level]) * retro_weight

        for parent in current_level_nodes:
            # Branching affected by past influence
            base_branches = branching_factor + past_influence
            n_branches = max(1, int(np.random.poisson(base_branches)))
            n_branches = min(n_branches, 4)

            parent_weight = parent['weight']
            weight_per_branch = parent_weight / n_branches * (1 - decoherence * 0.3)

            for b in range(n_branches):
                spread = 0.15 * level / depth
                y = parent['y'] + np.random.uniform(-spread, spread)
                y = np.clip(y, 0.05, 0.95)

                weight = weight_per_branch * (1 + np.random.uniform(-0.1, 0.1))
                weight = max(0.1, weight)

                node_idx = len(future_nodes)
                future_nodes.append([float(x), float(y), level, float(weight)])
                new_level_nodes.append({'idx': node_idx, 'y': y, 'weight': weight})

                parent_idx = parent['idx']
                edges.append([parent_idx, future_idx_offset + node_idx, 'future'])

        current_level_nodes = new_level_nodes
        if not current_level_nodes:
            break

    return {
        'present': present,
        'past_nodes': past_nodes,
        'future_nodes': future_nodes,
        'edges': edges,
        'past_count': len(past_nodes),
        'future_count': len(future_nodes),
        'retro_influence': float(retro_weight * (1 - decoherence))
    }


def compute_decision_impact(params: Dict, decision_strength: float) -> Dict:
    """
    Compute how a decision at present reshapes the timeline.

    Parameters
    ----------
    params : dict - Base timeline parameters
    decision_strength : float - Strength of decision (0-1)

    Returns
    -------
    dict - Modified timeline structure
    """
    params = dict(params)

    # Decision increases future branching, decreases past influence
    params['branching_factor'] = params.get('branching_factor', 2.0) * (1 + decision_strength * 0.5)
    params['retro_weight'] = params.get('retro_weight', 0.3) * (1 - decision_strength * 0.3)
    params['seed'] = params.get('seed', 42) + int(decision_strength * 1000)

    return compute_time_enfoldment(params)


# Export for Pyodide
__all__ = ['compute_time_enfoldment', 'compute_decision_impact']
