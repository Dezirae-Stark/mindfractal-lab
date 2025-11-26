#!/usr/bin/env python3
"""
Basin of Attraction Generation Script

Computes basin of attraction diagram by classifying initial conditions
based on which attractor they converge to.

Output: docs/images/basin.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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


def classify_final_state(final_state, threshold=10.0):
    """Classify based on final state properties."""
    norm = np.linalg.norm(final_state)

    if norm > threshold:
        return 3  # Divergent

    # Classify by quadrant/sign pattern
    if final_state[0] >= 0 and final_state[1] >= 0:
        return 0
    elif final_state[0] < 0 and final_state[1] >= 0:
        return 1
    elif final_state[0] < 0 and final_state[1] < 0:
        return 2
    else:
        return 3


def compute_basin(model, x_range, y_range, resolution, n_steps):
    """Compute basin of attraction on a grid."""
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)

    basin = np.zeros((resolution, resolution), dtype=int)

    print(f"Computing basin of attraction ({resolution}x{resolution})...")

    for i, y in enumerate(y_vals):
        if i % 20 == 0:
            print(f"  Progress: {100*i/resolution:.0f}%")

        for j, x in enumerate(x_vals):
            # Initial condition
            state = np.array([x, y], dtype=np.float64)

            # Iterate
            for _ in range(n_steps):
                state = model.step(state)

                # Check for divergence
                if np.linalg.norm(state) > 100:
                    break

            # Classify final state
            basin[i, j] = classify_final_state(state)

    print("  Progress: 100%")
    return x_vals, y_vals, basin


def main():
    # Create model
    model = FractalDynamicsModel(c=np.array([0.3, 0.2]))

    # Parameters
    x_range = (-3.0, 3.0)
    y_range = (-3.0, 3.0)
    resolution = 300
    n_steps = 500

    # Compute basin
    x_vals, y_vals, basin = compute_basin(
        model, x_range, y_range, resolution, n_steps
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Custom colormap
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    cmap = ListedColormap(colors)

    # Plot basin
    extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
    im = ax.imshow(basin, extent=extent, origin='lower',
                   aspect='auto', cmap=cmap, interpolation='nearest')

    # Labels and title
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.set_title('Basin of Attraction\nFractal Boundary Structure',
                 fontsize=16)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0.375, 1.125, 1.875, 2.625])
    cbar.ax.set_yticklabels(['Basin 1', 'Basin 2', 'Basin 3', 'Divergent'])
    cbar.set_label('Attractor', fontsize=12)

    ax.grid(True, alpha=0.2, color='white', linewidth=0.5)

    plt.tight_layout()

    # Save figure
    output_path = 'docs/images/basin.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Basin of attraction saved to {output_path}")

    plt.close()


if __name__ == '__main__':
    main()
