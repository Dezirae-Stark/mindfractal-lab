"""
Many Worlds Core — Branching Universe Computation
MindFractal Lab

Pyodide-compatible module for visualizing quantum branching and decoherence.
"""

import numpy as np
from typing import Dict, List, Tuple
import random


def compute_branching_universe(params: Dict) -> Dict:
    """
    Compute a branching universe tree structure.

    Parameters
    ----------
    params : dict
        depth : int - Tree depth (1-10)
        branching_factor : float - Average branches per node (1.5-4)
        decoherence : float - Rate of probability loss (0-1)
        prob_compression : float - How probability concentrates (0-1)
        seed : int - Random seed

    Returns
    -------
    dict
        nodes : List of [id, level, weight, x, y] - Node data
        edges : List of [from_id, to_id] - Edge connections
        total_branches : int - Total number of branches
        probability_sum : float - Total probability (should sum to ~1)
    """
    depth = min(max(int(params.get('depth', 5)), 1), 10)
    branching_factor = float(params.get('branching_factor', 2.0))
    decoherence = float(params.get('decoherence', 0.1))
    prob_compression = float(params.get('prob_compression', 0.3))
    seed = int(params.get('seed', 42))

    np.random.seed(seed)
    random.seed(seed)

    nodes = []
    edges = []
    node_counter = 0

    # Root node
    root_id = node_counter
    nodes.append({
        'id': root_id,
        'level': 0,
        'weight': 1.0,
        'parent': None,
        'children': []
    })
    node_counter += 1

    # Build tree level by level
    current_level = [0]  # Node IDs at current level

    for level in range(1, depth + 1):
        next_level = []

        for parent_id in current_level:
            parent = nodes[parent_id]
            parent_weight = parent['weight']

            # Number of branches
            n_branches = max(1, int(np.random.poisson(branching_factor)))
            n_branches = min(n_branches, 5)  # Cap

            # Distribute probability among children
            if prob_compression > 0:
                # Some branches get more probability
                raw_probs = np.random.dirichlet(np.ones(n_branches) * (1 - prob_compression + 0.1))
            else:
                raw_probs = np.ones(n_branches) / n_branches

            # Apply decoherence (probability loss)
            total_prob = parent_weight * (1 - decoherence * 0.2)
            child_probs = raw_probs * total_prob

            for i, prob in enumerate(child_probs):
                if prob < 0.001:  # Prune negligible branches
                    continue

                child_id = node_counter
                nodes.append({
                    'id': child_id,
                    'level': level,
                    'weight': float(prob),
                    'parent': parent_id,
                    'children': []
                })
                parent['children'].append(child_id)
                edges.append([parent_id, child_id])
                next_level.append(child_id)
                node_counter += 1

        current_level = next_level
        if not current_level:
            break

    # Compute positions for visualization (radial layout)
    _assign_positions(nodes, depth)

    # Convert to output format
    output_nodes = [
        [n['id'], n['level'], n['weight'], n.get('x', 0.5), n.get('y', 0.5)]
        for n in nodes
    ]

    # Total probability at leaves
    leaf_weights = [n['weight'] for n in nodes if not n['children']]
    prob_sum = sum(leaf_weights)

    return {
        'nodes': output_nodes,
        'edges': edges,
        'total_branches': len(nodes),
        'probability_sum': float(prob_sum),
        'max_depth': max(n['level'] for n in nodes)
    }


def _assign_positions(nodes: List[Dict], max_depth: int):
    """Assign x, y positions to nodes for visualization."""
    if not nodes:
        return

    # Group nodes by level
    levels = {}
    for node in nodes:
        level = node['level']
        if level not in levels:
            levels[level] = []
        levels[level].append(node)

    # Assign positions
    for level, level_nodes in levels.items():
        # X position based on level (left to right)
        x = 0.1 + 0.8 * (level / max(max_depth, 1))

        # Y positions spread evenly
        n = len(level_nodes)
        for i, node in enumerate(level_nodes):
            if n == 1:
                y = 0.5
            else:
                y = 0.1 + 0.8 * (i / (n - 1))

            # Add small random offset for visual interest
            y += np.random.uniform(-0.02, 0.02)
            y = np.clip(y, 0.05, 0.95)

            node['x'] = float(x)
            node['y'] = float(y)


def compute_branch_selection(params: Dict, selected_path: List[int]) -> Dict:
    """
    Compute branching with a specific path highlighted.

    Parameters
    ----------
    params : dict - Base parameters
    selected_path : list - List of node IDs representing selected branch

    Returns
    -------
    dict - Branching structure with highlight information
    """
    result = compute_branching_universe(params)

    # Mark selected path
    selected_set = set(selected_path)
    highlighted_edges = []

    for edge in result['edges']:
        if edge[0] in selected_set and edge[1] in selected_set:
            highlighted_edges.append(edge)

    result['highlighted_path'] = selected_path
    result['highlighted_edges'] = highlighted_edges
    result['path_probability'] = _compute_path_probability(result['nodes'], selected_path)

    return result


def _compute_path_probability(nodes: List, path: List[int]) -> float:
    """Compute total probability along a path."""
    if not path:
        return 0.0

    # Find the leaf node weight in the path
    node_dict = {n[0]: n[2] for n in nodes}  # id -> weight

    if path[-1] in node_dict:
        return node_dict[path[-1]]
    return 0.0


def compute_interference_pattern(params: Dict) -> Dict:
    """
    Compute interference between nearby branches.

    Parameters
    ----------
    params : dict
        n_branches : int - Number of interfering branches
        coherence : float - Quantum coherence level (0-1)
        resolution : int - Pattern resolution

    Returns
    -------
    dict
        pattern : 2D list of interference values
        visibility : float - Fringe visibility
    """
    n_branches = min(max(int(params.get('n_branches', 3)), 2), 8)
    coherence = float(params.get('coherence', 0.5))
    resolution = min(max(int(params.get('resolution', 50)), 20), 100)
    seed = int(params.get('seed', 42))

    np.random.seed(seed)

    pattern = np.zeros((resolution, resolution))

    # Generate random branch phases
    phases = np.random.uniform(0, 2 * np.pi, n_branches)
    amplitudes = np.random.uniform(0.5, 1.0, n_branches)

    for i in range(resolution):
        for j in range(resolution):
            x = j / (resolution - 1)
            y = i / (resolution - 1)

            # Sum of wavefunctions from each branch
            real_sum = 0.0
            imag_sum = 0.0

            for b in range(n_branches):
                # Each branch contributes a wave
                k = 2 * np.pi * (b + 1)
                phase = k * x + phases[b]

                # Coherent superposition
                real_sum += amplitudes[b] * np.cos(phase) * coherence
                imag_sum += amplitudes[b] * np.sin(phase) * coherence

                # Incoherent addition
                real_sum += amplitudes[b] * np.cos(phase + y * k) * (1 - coherence)

            # Probability = |amplitude|^2
            prob = real_sum**2 + imag_sum**2
            pattern[i, j] = prob

    # Normalize
    pattern = pattern / pattern.max() if pattern.max() > 0 else pattern

    # Compute fringe visibility
    visibility = (pattern.max() - pattern.min()) / (pattern.max() + pattern.min() + 1e-10)

    return {
        'pattern': pattern.tolist(),
        'visibility': float(visibility),
        'n_branches': n_branches,
        'coherence': coherence
    }


# Export for Pyodide
__all__ = [
    'compute_branching_universe',
    'compute_branch_selection',
    'compute_interference_pattern'
]
