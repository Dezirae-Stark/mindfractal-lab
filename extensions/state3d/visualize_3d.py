"""3D Visualization using mplot3d"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .simulate_3d import simulate_orbit_3d


def plot_orbit_3d(
    model,
    x0: np.ndarray,
    n_steps: int = 1000,
    save_path: str = None
):
    """Plot 3D orbit"""
    trajectory = simulate_orbit_3d(model, x0, n_steps)

    fig = plt.figure(figsize=(12, 10))

    # 3D trajectory
    ax = fig.add_subplot(221, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
           'b-', alpha=0.6, linewidth=0.5)
    ax.plot([trajectory[0, 0]], [trajectory[0, 1]], [trajectory[0, 2]],
           'go', markersize=8, label='Start')
    ax.plot([trajectory[-1, 0]], [trajectory[-1, 1]], [trajectory[-1, 2]],
           'ro', markersize=8, label='End')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')
    ax.set_title('3D Trajectory')
    ax.legend()

    # Time series
    ax = fig.add_subplot(222)
    ax.plot(trajectory[:, 0], label='$x_1$')
    ax.plot(trajectory[:, 1], label='$x_2$')
    ax.plot(trajectory[:, 2], label='$x_3$')
    ax.set_xlabel('Time')
    ax.set_ylabel('State')
    ax.set_title('State Components vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # XY projection
    ax = fig.add_subplot(223)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.6, linewidth=0.5)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('XY Projection')
    ax.grid(True, alpha=0.3)

    # XZ projection
    ax = fig.add_subplot(224)
    ax.plot(trajectory[:, 0], trajectory[:, 2], 'r-', alpha=0.6, linewidth=0.5)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_3$')
    ax.set_title('XZ Projection')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"3D plot saved to {save_path}")

    return fig
