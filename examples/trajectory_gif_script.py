#!/usr/bin/env python3
"""
Animated Trajectory GIF Generation Script

Loads the basin.png image, simulates a trajectory, overlays a moving point,
and encodes an animated GIF.

Output: docs/images/trajectory_on_basin.gif
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import os
import tempfile
import shutil


class FractalDynamicsModel:
    """Minimal 2D fractal dynamics model."""

    def __init__(self, A=None, B=None, W=None, c=None):
        self.A = A if A is not None else np.array([[0.9, 0.0], [0.0, 0.9]])
        self.B = B if B is not None else np.array([[0.2, 0.3], [0.3, 0.2]])
        self.W = W if W is not None else np.array([[1.0, 0.1], [0.1, 1.0]])
        self.c = c if c is not None else np.array([0.1, 0.1])

    def step(self, x):
        return self.A @ x + self.B @ np.tanh(self.W @ x) + self.c


def simulate_trajectory(model, x0, n_steps):
    """Simulate trajectory."""
    trajectory = np.zeros((n_steps, 2))
    x = np.array(x0, dtype=np.float64)
    trajectory[0] = x
    for i in range(1, n_steps):
        x = model.step(x)
        trajectory[i] = x
    return trajectory


def create_trajectory_gif(basin_path, output_path, x0, n_frames=200, fps=20):
    """Create animated GIF of trajectory on basin background."""

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create model with same parameters used for basin
    model = FractalDynamicsModel(c=np.array([0.3, 0.2]))

    # Simulate trajectory
    n_steps = n_frames + 50  # Extra steps for smooth animation
    trajectory = simulate_trajectory(model, x0, n_steps)

    # Basin extent (must match basin_script.py)
    x_range = (-3.0, 3.0)
    y_range = (-3.0, 3.0)

    # Load basin image if it exists, otherwise create placeholder
    if os.path.exists(basin_path):
        basin_img = Image.open(basin_path)
        basin_array = np.array(basin_img)
    else:
        print(f"Warning: {basin_path} not found. Creating placeholder background.")
        # Create a simple placeholder
        basin_array = np.zeros((600, 600, 3), dtype=np.uint8)
        basin_array[:, :] = [30, 30, 50]  # Dark blue-gray

    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp(prefix='trajectory_gif_')

    try:
        print(f"Generating {n_frames} frames...")
        frames = []

        for frame_idx in range(n_frames):
            if frame_idx % 20 == 0:
                print(f"  Progress: {100 * frame_idx / n_frames:.0f}%")

            # Create figure
            fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

            # Show basin background
            ax.imshow(basin_array, extent=[x_range[0], x_range[1],
                                           y_range[0], y_range[1]],
                      aspect='auto', origin='lower')

            # Plot trajectory trail (fade effect)
            trail_start = max(0, frame_idx - 50)
            trail = trajectory[trail_start:frame_idx + 1]

            if len(trail) > 1:
                # Create fading trail
                n_trail = len(trail)
                for i in range(n_trail - 1):
                    alpha = 0.1 + 0.7 * (i / n_trail)
                    ax.plot(trail[i:i + 2, 0], trail[i:i + 2, 1],
                            'w-', alpha=alpha, linewidth=1.5)

            # Plot current position
            current_pos = trajectory[frame_idx]
            ax.plot(current_pos[0], current_pos[1], 'wo',
                    markersize=12, markeredgecolor='black',
                    markeredgewidth=2, zorder=10)

            # Plot starting position
            ax.plot(x0[0], x0[1], 'g^', markersize=10,
                    markeredgecolor='white', markeredgewidth=1,
                    label='Start', zorder=9)

            # Labels and title
            ax.set_xlabel('$x_1$', fontsize=12)
            ax.set_ylabel('$x_2$', fontsize=12)
            ax.set_title(f'Trajectory on Basin of Attraction\nStep {frame_idx}',
                         fontsize=14)
            ax.set_xlim(x_range)
            ax.set_ylim(y_range)

            # Save frame
            frame_path = os.path.join(temp_dir, f'frame_{frame_idx:04d}.png')
            plt.savefig(frame_path, dpi=100, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            plt.close(fig)

            # Load frame for GIF
            frames.append(imageio.imread(frame_path))

        print("  Progress: 100%")
        print("Encoding GIF...")

        # Save as GIF
        imageio.mimsave(output_path, frames, fps=fps, loop=0)
        print(f"Animated GIF saved to {output_path}")

        # Get file size
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"File size: {size_mb:.2f} MB")

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        print("Temporary files cleaned up.")


def main():
    # Paths
    basin_path = 'docs/images/basin.png'
    output_path = 'docs/images/trajectory_on_basin.gif'

    # Initial condition (interesting starting point)
    x0 = np.array([-2.0, 1.5])

    # Create animated GIF
    create_trajectory_gif(
        basin_path=basin_path,
        output_path=output_path,
        x0=x0,
        n_frames=200,
        fps=20
    )


if __name__ == '__main__':
    main()
