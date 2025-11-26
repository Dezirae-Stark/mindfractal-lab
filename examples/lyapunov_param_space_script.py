#!/usr/bin/env python3
"""
Parameter-Space Lyapunov Heatmap Generation Script

Computes Lyapunov exponent for each (c1, c2) parameter combination
and visualizes as a heatmap.

Output: docs/images/lyapunov_param_space.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Ensure output directory exists
os.makedirs('docs/images', exist_ok=True)


class FractalDynamicsModel:
    """Minimal 2D fractal dynamics model."""

    def __init__(self, A=None, B=None, W=None, c=None):
        self.A = A if A is not None else np.array([[0.9, 0.0], [0.0, 0.9]])
        self.B = B if B is not None else np.array([[0.2, 0.3], [0.3, 0.2]])
        self.W = W if W is not None else np.array([[1.0, 0.1], [0.1, 1.0]])
        self.c = c if c is not None else np.array([0.1, 0.1])

    def step(self, x):
        return self.A @ x + self.B @ np.tanh(self.W @ x) + self.c

    def jacobian(self, x):
        wx = self.W @ x
        sech_sq = 1.0 - np.tanh(wx)**2
        return self.A + self.B @ np.diag(sech_sq) @ self.W


def compute_lyapunov(model, x0, n_steps=1000, transient=200):
    """Compute largest Lyapunov exponent."""
    x = np.array(x0, dtype=np.float64)

    # Discard transient
    for _ in range(transient):
        x = model.step(x)
        if np.linalg.norm(x) > 100:
            return np.nan  # Divergent

    # Compute Lyapunov exponent
    log_sum = 0.0
    v = np.array([1.0, 0.0])

    for _ in range(n_steps):
        J = model.jacobian(x)
        v = J @ v
        norm_v = np.linalg.norm(v)

        if norm_v > 0:
            log_sum += np.log(norm_v)
            v = v / norm_v
        else:
            return -np.inf

        x = model.step(x)

        if np.linalg.norm(x) > 100:
            return np.nan  # Divergent

    return log_sum / n_steps


def compute_lyapunov_map(c1_range, c2_range, resolution, x0):
    """Compute Lyapunov exponent for grid of (c1, c2) values."""
    c1_vals = np.linspace(c1_range[0], c1_range[1], resolution)
    c2_vals = np.linspace(c2_range[0], c2_range[1], resolution)

    lyap_map = np.zeros((resolution, resolution))

    print(f"Computing Lyapunov parameter map ({resolution}x{resolution})...")

    for i, c2 in enumerate(c2_vals):
        if i % 10 == 0:
            print(f"  Progress: {100*i/resolution:.0f}%")

        for j, c1 in enumerate(c1_vals):
            c = np.array([c1, c2])
            model = FractalDynamicsModel(c=c)
            lyap = compute_lyapunov(model, x0, n_steps=500, transient=100)
            lyap_map[i, j] = lyap

    print("  Progress: 100%")
    return c1_vals, c2_vals, lyap_map


def main():
    # Parameters
    c1_range = (-2.0, 2.0)
    c2_range = (-2.0, 2.0)
    resolution = 150
    x0 = np.array([0.1, 0.1])

    # Compute Lyapunov map
    c1_vals, c2_vals, lyap_map = compute_lyapunov_map(
        c1_range, c2_range, resolution, x0
    )

    # Handle NaN values (divergent trajectories)
    lyap_map = np.nan_to_num(lyap_map, nan=1.0)

    # Clip for visualization
    lyap_map = np.clip(lyap_map, -1.0, 1.0)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    extent = [c1_range[0], c1_range[1], c2_range[0], c2_range[1]]
    im = ax.imshow(lyap_map, extent=extent, origin='lower',
                   aspect='auto', cmap='RdBu_r', interpolation='bilinear',
                   vmin=-0.5, vmax=0.5)

    # Contour at λ=0 (chaos boundary)
    X, Y = np.meshgrid(c1_vals, c2_vals)
    ax.contour(X, Y, lyap_map, levels=[0], colors='black',
               linewidths=1.5, linestyles='--')

    # Labels and title
    ax.set_xlabel('$c_1$', fontsize=14)
    ax.set_ylabel('$c_2$', fontsize=14)
    ax.set_title('Lyapunov Exponent in Parameter Space\n' +
                 r'$\lambda > 0$: Chaotic (red), $\lambda < 0$: Stable (blue)',
                 fontsize=16)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Largest Lyapunov Exponent $\\lambda$', fontsize=12)

    ax.grid(True, alpha=0.2, color='white', linewidth=0.5)

    plt.tight_layout()

    # Save figure
    output_path = 'docs/images/lyapunov_param_space.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Lyapunov parameter space map saved to {output_path}")

    plt.close()


if __name__ == '__main__':
    main()
