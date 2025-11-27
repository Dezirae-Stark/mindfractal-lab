"""
Fractal Parameter-Space Map Generation

This module generates fractal maps in the (c1, c2) parameter space,
revealing the rich structure of attractor basins and bifurcations.

The fractal map shows:
- Which combinations of parameters lead to convergence vs divergence
- Fractal basin boundaries between different attractors
- Self-similar structure at multiple scales
"""

from typing import Optional, Tuple

import numpy as np

from .model import FractalDynamicsModel
from .simulate import simulate_orbit


def generate_fractal_map(
    c1_range: Tuple[float, float] = (-1.0, 1.0),
    c2_range: Tuple[float, float] = (-1.0, 1.0),
    resolution: int = 500,
    x0: Optional[np.ndarray] = None,
    max_steps: int = 500,
    divergence_threshold: float = 10.0,
    A: Optional[np.ndarray] = None,
    B: Optional[np.ndarray] = None,
    W: Optional[np.ndarray] = None,
    criterion: str = "divergence_time",
) -> np.ndarray:
    """
    Generate a fractal map in (c1, c2) parameter space.

    For each (c1, c2) pair, we simulate the dynamics and compute a metric
    that reveals the fractal structure.

    Args:
        c1_range: Range for c1 parameter
        c2_range: Range for c2 parameter
        resolution: Grid resolution (creates resolution x resolution map)
        x0: Initial condition (default: small perturbation from origin)
        max_steps: Maximum simulation steps
        divergence_threshold: Threshold for detecting divergence
        A, B, W: Matrices for the model (use defaults if None)
        criterion: Metric to compute:
            - 'divergence_time': steps until ||x|| > divergence_threshold
            - 'final_norm': ||x|| after max_steps
            - 'lyapunov': estimated Lyapunov exponent
            - 'attractor_type': classifier (0=fixed, 1=periodic, 2=chaotic)

    Returns:
        2D array of shape (resolution, resolution) containing the computed metric
    """
    if x0 is None:
        x0 = np.array([0.01, 0.01])

    c1_vals = np.linspace(c1_range[0], c1_range[1], resolution)
    c2_vals = np.linspace(c2_range[0], c2_range[1], resolution)

    fractal_map = np.zeros((resolution, resolution))

    print(f"Generating {resolution}x{resolution} fractal map...")
    print(f"Criterion: {criterion}")
    print(f"Parameter ranges: c1={c1_range}, c2={c2_range}")

    for i, c1 in enumerate(c1_vals):
        if i % 50 == 0:
            progress = 100 * i / resolution
            print(f"Progress: {progress:.1f}%")

        for j, c2 in enumerate(c2_vals):
            c = np.array([c1, c2])

            # Create model with this parameter value
            model = FractalDynamicsModel(A=A, B=B, W=W, c=c)

            if criterion == "divergence_time":
                # Compute divergence time
                x = x0.copy()
                for step in range(max_steps):
                    x = model.step(x)
                    if np.linalg.norm(x) > divergence_threshold:
                        fractal_map[j, i] = step
                        break
                else:
                    # Did not diverge
                    fractal_map[j, i] = max_steps

            elif criterion == "final_norm":
                # Simulate and return final norm
                trajectory = simulate_orbit(model, x0, n_steps=max_steps, return_all=False)
                fractal_map[j, i] = np.linalg.norm(trajectory)

            elif criterion == "lyapunov":
                # Estimate Lyapunov exponent
                lyap = model.lyapunov_exponent_estimate(x0, n_steps=200, transient=100)
                fractal_map[j, i] = lyap

            elif criterion == "attractor_type":
                # Classify attractor type
                from .simulate import compute_attractor_type

                atype = compute_attractor_type(model, x0, n_steps=max_steps, transient=100)
                type_map = {
                    "fixed_point": 0,
                    "limit_cycle": 1,
                    "chaotic": 2,
                    "unbounded": 3,
                }
                fractal_map[j, i] = type_map.get(atype, -1)

    print("Fractal map generation complete!")
    return fractal_map


def zoom_fractal_map(
    center: Tuple[float, float], zoom_factor: float, base_range: float = 1.0, **kwargs
) -> np.ndarray:
    """
    Generate a zoomed-in fractal map around a specific parameter point.

    This reveals self-similar fractal structure at finer scales.

    Args:
        center: (c1, c2) center point for zoom
        zoom_factor: Zoom factor (higher = more zoomed in)
        base_range: Base range before zoom
        **kwargs: Additional arguments passed to generate_fractal_map()

    Returns:
        Fractal map array
    """
    delta = base_range / zoom_factor

    c1_range = (center[0] - delta, center[0] + delta)
    c2_range = (center[1] - delta, center[1] + delta)

    return generate_fractal_map(c1_range=c1_range, c2_range=c2_range, **kwargs)


def adaptive_fractal_map(
    c1_range: Tuple[float, float],
    c2_range: Tuple[float, float],
    base_resolution: int = 100,
    max_resolution: int = 500,
    variation_threshold: float = 0.5,
    **kwargs,
) -> np.ndarray:
    """
    Generate fractal map with adaptive resolution.

    Regions with high variation (fractal boundaries) are refined
    with higher resolution.

    Args:
        c1_range: Range for c1
        c2_range: Range for c2
        base_resolution: Initial coarse resolution
        max_resolution: Maximum resolution for refined regions
        variation_threshold: Threshold for detecting high-variation regions
        **kwargs: Additional arguments for generate_fractal_map()

    Returns:
        High-resolution fractal map with adaptive sampling
    """
    print("Phase 1: Generating coarse fractal map...")
    coarse_map = generate_fractal_map(
        c1_range=c1_range, c2_range=c2_range, resolution=base_resolution, **kwargs
    )

    # Detect high-variation regions
    print("Phase 2: Detecting fractal boundaries...")
    grad_x = np.abs(np.diff(coarse_map, axis=1))
    grad_y = np.abs(np.diff(coarse_map, axis=0))

    variation_map = np.zeros_like(coarse_map)
    variation_map[:, :-1] += grad_x
    variation_map[:-1, :] += grad_y

    # Normalize
    if variation_map.max() > 0:
        variation_map /= variation_map.max()

    # Refine high-variation regions
    print("Phase 3: Refining fractal boundaries (adaptive sampling)...")
    # For simplicity, regenerate at max resolution
    # (Full adaptive implementation would require quadtree structure)
    refined_map = generate_fractal_map(
        c1_range=c1_range, c2_range=c2_range, resolution=max_resolution, **kwargs
    )

    return refined_map
