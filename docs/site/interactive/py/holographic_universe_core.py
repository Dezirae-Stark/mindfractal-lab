"""
Holographic Universe Core — Implicate/Explicate Order Computation
MindFractal Lab

Pyodide-compatible module for visualizing holographic projection
from boundary encodings to bulk emergent patterns.
"""

import numpy as np
from typing import Dict, List, Tuple


def compute_implicate_explicate(params: Dict) -> Dict:
    """
    Compute implicate (boundary) and explicate (bulk) patterns.

    Parameters
    ----------
    params : dict
        resolution : int - Grid resolution (20-100)
        encoding_type : str - "stripes", "noise", "wave", "spiral", "checkerboard"
        projection_depth : float - How deep the projection goes (0-1)
        smoothness : float - Smoothing factor (0-1)
        seed : int - Random seed

    Returns
    -------
    dict
        boundary : 2D list of boundary encoding values
        explicate : 2D list of projected bulk values
        entropy_boundary : float - Boundary entropy
        entropy_explicate : float - Bulk entropy
    """
    resolution = min(max(int(params.get('resolution', 50)), 20), 100)
    encoding_type = str(params.get('encoding_type', 'wave'))
    projection_depth = float(params.get('projection_depth', 0.5))
    smoothness = float(params.get('smoothness', 0.3))
    seed = int(params.get('seed', 42))

    np.random.seed(seed)

    # Generate boundary encoding (1D conceptually, represented as edge of 2D)
    boundary = np.zeros((resolution, resolution))

    if encoding_type == 'stripes':
        for i in range(resolution):
            for j in range(resolution):
                boundary[i, j] = 0.5 + 0.5 * np.sin(2 * np.pi * j / 10)

    elif encoding_type == 'noise':
        boundary = np.random.rand(resolution, resolution)
        # Apply some smoothing
        if smoothness > 0:
            kernel_size = max(1, int(smoothness * 5))
            for _ in range(kernel_size):
                boundary = _smooth_grid(boundary)

    elif encoding_type == 'wave':
        for i in range(resolution):
            for j in range(resolution):
                x = (j - resolution/2) / resolution
                y = (i - resolution/2) / resolution
                r = np.sqrt(x*x + y*y)
                boundary[i, j] = 0.5 + 0.5 * np.sin(2 * np.pi * r * 5)

    elif encoding_type == 'spiral':
        for i in range(resolution):
            for j in range(resolution):
                x = (j - resolution/2) / resolution
                y = (i - resolution/2) / resolution
                r = np.sqrt(x*x + y*y)
                theta = np.arctan2(y, x)
                boundary[i, j] = 0.5 + 0.5 * np.sin(theta * 3 + r * 10)

    elif encoding_type == 'checkerboard':
        for i in range(resolution):
            for j in range(resolution):
                check = ((i // 5) + (j // 5)) % 2
                boundary[i, j] = float(check)

    else:
        # Default to wave
        for i in range(resolution):
            for j in range(resolution):
                x = (j - resolution/2) / resolution
                y = (i - resolution/2) / resolution
                boundary[i, j] = 0.5 + 0.5 * np.sin(2 * np.pi * (x + y) * 3)

    # Project boundary to explicate (bulk) using holographic-inspired transform
    explicate = _holographic_project(boundary, projection_depth, smoothness)

    # Compute entropies
    entropy_boundary = _compute_entropy(boundary)
    entropy_explicate = _compute_entropy(explicate)

    return {
        'boundary': boundary.tolist(),
        'explicate': explicate.tolist(),
        'entropy_boundary': float(entropy_boundary),
        'entropy_explicate': float(entropy_explicate),
        'resolution': resolution,
        'encoding_type': encoding_type
    }


def _smooth_grid(grid: np.ndarray) -> np.ndarray:
    """Apply simple smoothing to a 2D grid."""
    result = np.copy(grid)
    n = len(grid)

    for i in range(1, n-1):
        for j in range(1, n-1):
            result[i, j] = (
                grid[i, j] * 0.4 +
                grid[i-1, j] * 0.15 +
                grid[i+1, j] * 0.15 +
                grid[i, j-1] * 0.15 +
                grid[i, j+1] * 0.15
            )

    return result


def _holographic_project(boundary: np.ndarray, depth: float, smoothness: float) -> np.ndarray:
    """
    Project boundary encoding to bulk using holographic-inspired transform.

    This simulates how boundary information encodes bulk structure.
    """
    n = len(boundary)
    explicate = np.zeros_like(boundary)

    # Fourier-inspired projection
    # In holography, bulk emerges from boundary through integral transforms

    for i in range(n):
        for j in range(n):
            # Distance from center (bulk "depth")
            x = (j - n/2) / n
            y = (i - n/2) / n
            r = np.sqrt(x*x + y*y)

            # Sample boundary at multiple scales
            value = 0.0
            weights = 0.0

            for scale in [1, 2, 4, 8]:
                if scale > n // 4:
                    break

                # Radial sampling from boundary
                n_samples = max(4, int(8 * depth))
                for k in range(n_samples):
                    angle = 2 * np.pi * k / n_samples

                    # Sample position on boundary (edges of grid)
                    sample_r = 0.5 * (1 - r * depth)
                    si = int(n/2 + sample_r * n * np.sin(angle)) % n
                    sj = int(n/2 + sample_r * n * np.cos(angle)) % n

                    # Phase from geometry
                    phase = np.cos(angle * scale + r * depth * 10)

                    value += boundary[si, sj] * phase / scale
                    weights += 1.0 / scale

            if weights > 0:
                explicate[i, j] = value / weights

    # Normalize to 0-1
    explicate = explicate - explicate.min()
    if explicate.max() > 0:
        explicate = explicate / explicate.max()

    # Apply smoothness
    for _ in range(int(smoothness * 5)):
        explicate = _smooth_grid(explicate)

    return explicate


def _compute_entropy(grid: np.ndarray) -> float:
    """Compute approximate entropy of a 2D grid."""
    # Bin the values
    hist, _ = np.histogram(grid.flatten(), bins=20, range=(0, 1))
    hist = hist / hist.sum()

    # Shannon entropy
    entropy = 0.0
    for p in hist:
        if p > 0:
            entropy -= p * np.log2(p)

    return entropy


def compute_boundary_modification(params: Dict, brush_x: float, brush_y: float, brush_value: float) -> Dict:
    """
    Modify boundary at a specific location and recompute explicate.

    Parameters
    ----------
    params : dict - Base parameters
    brush_x, brush_y : float - Brush position (0-1)
    brush_value : float - Value to paint (0-1)

    Returns
    -------
    dict - Updated implicate/explicate structure
    """
    # First compute base
    result = compute_implicate_explicate(params)
    boundary = np.array(result['boundary'])
    resolution = len(boundary)

    # Apply brush
    bx = int(brush_x * (resolution - 1))
    by = int(brush_y * (resolution - 1))
    brush_radius = max(1, resolution // 20)

    for i in range(max(0, by - brush_radius), min(resolution, by + brush_radius + 1)):
        for j in range(max(0, bx - brush_radius), min(resolution, bx + brush_radius + 1)):
            dist = np.sqrt((i - by)**2 + (j - bx)**2)
            if dist <= brush_radius:
                weight = 1 - dist / brush_radius
                boundary[i, j] = boundary[i, j] * (1 - weight) + brush_value * weight

    # Reproject
    projection_depth = float(params.get('projection_depth', 0.5))
    smoothness = float(params.get('smoothness', 0.3))
    explicate = _holographic_project(boundary, projection_depth, smoothness)

    return {
        'boundary': boundary.tolist(),
        'explicate': explicate.tolist(),
        'entropy_boundary': float(_compute_entropy(boundary)),
        'entropy_explicate': float(_compute_entropy(explicate)),
        'resolution': resolution
    }


# Export for Pyodide
__all__ = [
    'compute_implicate_explicate',
    'compute_boundary_modification'
]
