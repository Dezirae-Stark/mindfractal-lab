#!/usr/bin/env python3
"""
Phase Portrait Generation Script

Simulates a trajectory exhibiting rich dynamics and plots the 2D phase portrait.

Output: docs/images/phase_portrait.png
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


def simulate_orbit(model, x0, n_steps):
    """Simulate trajectory."""
    trajectory = np.zeros((n_steps, 2))
    x = np.array(x0, dtype=np.float64)
    trajectory[0] = x
    for i in range(1, n_steps):
        x = model.step(x)
        trajectory[i] = x
    return trajectory


def main():
    # Create model with parameters producing chaotic dynamics
    # Slightly increase c to push toward chaos
    model = FractalDynamicsModel(c=np.array([0.5, 0.3]))

    # Initial condition
    x0 = np.array([0.1, 0.1])

    # Simulate long trajectory
    n_steps = 5000
    trajectory = simulate_orbit(model, x0, n_steps)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-',
            alpha=0.5, linewidth=0.3, label='Trajectory')

    # Mark start and end
    ax.plot(trajectory[0, 0], trajectory[0, 1], 'go',
            markersize=10, label='Start', zorder=5)
    ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro',
            markersize=10, label='End', zorder=5)

    # Labels and title
    ax.set_xlabel('$x_1$ (Arousal)', fontsize=14)
    ax.set_ylabel('$x_2$ (Valence)', fontsize=14)
    ax.set_title('Phase Portrait: Fractal Dynamics Model\n' +
                 r'$\mathbf{x}_{n+1} = A\mathbf{x}_n + B\tanh(W\mathbf{x}_n) + \mathbf{c}$',
                 fontsize=16)

    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    # Save figure
    output_path = 'docs/images/phase_portrait.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Phase portrait saved to {output_path}")

    plt.close()


if __name__ == '__main__':
    main()
