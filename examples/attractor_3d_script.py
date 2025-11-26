#!/usr/bin/env python3
"""
3D Attractor Generation Script

Extends the fractal dynamics model to 3D and visualizes a chaotic attractor.

Output: docs/images/attractor_3d.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Ensure output directory exists
os.makedirs('docs/images', exist_ok=True)


class FractalDynamicsModel3D:
    """3D Fractal Dynamics Model."""

    def __init__(self, A=None, B=None, W=None, c=None):
        self.dim = 3

        # Default A: weak damping
        self.A = A if A is not None else 0.9 * np.eye(3)

        # Default B: coupling matrix
        self.B = B if B is not None else np.array([
            [0.2, 0.15, 0.1],
            [0.15, 0.2, 0.15],
            [0.1, 0.15, 0.2]
        ])

        # Default W: near-identity
        self.W = W if W is not None else np.array([
            [1.0, 0.1, 0.05],
            [0.1, 1.0, 0.1],
            [0.05, 0.1, 1.0]
        ])

        # Default c: external drive
        self.c = c if c is not None else np.array([0.3, 0.2, 0.25])

    def step(self, x):
        return self.A @ x + self.B @ np.tanh(self.W @ x) + self.c


def simulate_orbit_3d(model, x0, n_steps):
    """Simulate 3D trajectory."""
    trajectory = np.zeros((n_steps, 3))
    x = np.array(x0, dtype=np.float64)
    trajectory[0] = x
    for i in range(1, n_steps):
        x = model.step(x)
        trajectory[i] = x
    return trajectory


def main():
    # Create 3D model with parameters producing interesting dynamics
    model = FractalDynamicsModel3D(c=np.array([0.4, 0.3, 0.35]))

    # Initial condition
    x0 = np.array([0.1, 0.1, 0.1])

    # Simulate trajectory
    n_steps = 10000
    trajectory = simulate_orbit_3d(model, x0, n_steps)

    # Discard transient
    trajectory = trajectory[1000:]

    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory with color gradient
    n_points = len(trajectory)
    colors = plt.cm.viridis(np.linspace(0, 1, n_points))

    # Plot in segments for color gradient
    for i in range(0, n_points - 1, 50):
        end = min(i + 51, n_points)
        ax.plot(trajectory[i:end, 0],
                trajectory[i:end, 1],
                trajectory[i:end, 2],
                color=colors[i], alpha=0.6, linewidth=0.5)

    # Mark start and end points
    ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
               c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
               c='red', s=100, marker='o', label='End', zorder=5)

    # Labels and title
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_zlabel('$x_3$', fontsize=12)
    ax.set_title('3D Fractal Attractor\n' +
                 r'$\mathbf{x}_{n+1} = A\mathbf{x}_n + B\tanh(W\mathbf{x}_n) + \mathbf{c}$',
                 fontsize=16)

    ax.legend(loc='upper left', fontsize=10)

    # Adjust view angle
    ax.view_init(elev=25, azim=45)

    plt.tight_layout()

    # Save figure
    output_path = 'docs/images/attractor_3d.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"3D attractor saved to {output_path}")

    plt.close()


if __name__ == '__main__':
    main()
