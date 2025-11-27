"""
Quantum Gravity Core — Spacetime Weave Computation
MindFractal Lab

Pyodide-compatible module for visualizing spacetime foam and geometry emergence.
"""

import numpy as np
from typing import Dict, List, Tuple
import random


def compute_spacetime_weave(params: Dict) -> Dict:
    """
    Compute a spacetime weave network representing quantum geometry.

    Parameters
    ----------
    params : dict
        node_count : int - Number of spacetime nodes (10-500)
        noise_level : float - Quantum fluctuation intensity (0-1)
        coherence : float - Geometric coherence (0-1)
        curvature_bias : float - Curvature tendency (-1 to 1)
        seed : int - Random seed for reproducibility

    Returns
    -------
    dict
        nodes : List of [x, y, intensity] - Node positions and intensities
        edges : List of [i, j, weight] - Edge connections and weights
    """
    node_count = min(max(int(params.get('node_count', 50)), 10), 500)
    noise_level = float(params.get('noise_level', 0.3))
    coherence = float(params.get('coherence', 0.5))
    curvature_bias = float(params.get('curvature_bias', 0.0))
    seed = int(params.get('seed', 42))

    np.random.seed(seed)
    random.seed(seed)

    nodes = []
    edges = []

    # Generate base grid with noise
    grid_size = int(np.sqrt(node_count))
    actual_nodes = grid_size * grid_size

    for i in range(grid_size):
        for j in range(grid_size):
            # Base position on grid
            base_x = (i + 0.5) / grid_size
            base_y = (j + 0.5) / grid_size

            # Add quantum noise
            noise_x = np.random.normal(0, noise_level * 0.1)
            noise_y = np.random.normal(0, noise_level * 0.1)

            # Apply curvature bias (creates warping toward center or edges)
            cx, cy = 0.5, 0.5
            dx, dy = base_x - cx, base_y - cy
            dist = np.sqrt(dx*dx + dy*dy)

            if curvature_bias > 0:
                # Positive curvature: nodes pulled toward center
                warp = curvature_bias * 0.2 * dist
                base_x -= dx * warp
                base_y -= dy * warp
            elif curvature_bias < 0:
                # Negative curvature: nodes pushed outward
                warp = abs(curvature_bias) * 0.15 * (1 - dist)
                base_x += dx * warp
                base_y += dy * warp

            x = np.clip(base_x + noise_x, 0.02, 0.98)
            y = np.clip(base_y + noise_y, 0.02, 0.98)

            # Intensity based on coherence and position
            base_intensity = 0.5 + 0.5 * coherence
            intensity = base_intensity + np.random.normal(0, (1 - coherence) * 0.2)
            intensity = np.clip(intensity, 0.1, 1.0)

            nodes.append([float(x), float(y), float(intensity)])

    # Generate edges based on proximity and coherence
    max_edges = min(actual_nodes * 3, 1000)
    edge_threshold = 0.15 + 0.1 * (1 - coherence)

    edge_set = set()
    for i in range(actual_nodes):
        for j in range(i + 1, actual_nodes):
            if len(edge_set) >= max_edges:
                break

            dx = nodes[i][0] - nodes[j][0]
            dy = nodes[i][1] - nodes[j][1]
            dist = np.sqrt(dx*dx + dy*dy)

            if dist < edge_threshold:
                # Weight based on distance and coherence
                weight = (1 - dist / edge_threshold) * coherence
                weight += np.random.normal(0, noise_level * 0.1)
                weight = np.clip(weight, 0.1, 1.0)

                edge_set.add((i, j, float(weight)))

    edges = [[i, j, w] for i, j, w in edge_set]

    return {
        'nodes': nodes,
        'edges': edges,
        'coherence_score': float(coherence * (1 - noise_level * 0.5))
    }


def compute_foam_animation_frame(params: Dict, frame: int) -> Dict:
    """
    Compute a single animation frame for spacetime foam dynamics.

    Parameters
    ----------
    params : dict - Same as compute_spacetime_weave
    frame : int - Animation frame number

    Returns
    -------
    dict - Same structure as compute_spacetime_weave with time evolution
    """
    # Modify seed based on frame for animation
    params = dict(params)
    params['seed'] = int(params.get('seed', 42)) + frame

    # Add temporal fluctuation
    noise_level = params.get('noise_level', 0.3)
    params['noise_level'] = noise_level + 0.05 * np.sin(frame * 0.1)

    return compute_spacetime_weave(params)


# Export for Pyodide
__all__ = ['compute_spacetime_weave', 'compute_foam_animation_frame']
