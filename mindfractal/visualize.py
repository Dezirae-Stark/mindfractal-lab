"""
Visualization Tools for Fractal Dynamics

This module provides matplotlib-based visualization functions for:
- Plotting orbits (trajectories)
- Phase portraits
- Basin of attraction diagrams
- Fractal parameter-space maps
- Bifurcation diagrams

All visualizations use pure CPU matplotlib backend (Android/Termux compatible).
"""

from typing import Optional, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Use non-interactive backend for Android compatibility
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from .model import FractalDynamicsModel
from .simulate import basin_of_attraction_sample, simulate_orbit


def plot_orbit(
    model: FractalDynamicsModel,
    x0: np.ndarray,
    n_steps: int = 1000,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    show_fixed_points: bool = True,
) -> plt.Figure:
    """
    Plot an orbit in 2D phase space.

    Args:
        model: FractalDynamicsModel instance
        x0: Initial condition
        n_steps: Number of steps to simulate
        figsize: Figure size in inches
        save_path: If provided, save figure to this path
        show_fixed_points: If True, mark fixed points

    Returns:
        matplotlib Figure object
    """
    trajectory = simulate_orbit(model, x0, n_steps=n_steps)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Phase portrait
    ax = axes[0, 0]
    ax.plot(trajectory[:, 0], trajectory[:, 1], "b-", alpha=0.6, linewidth=0.5)
    ax.plot(trajectory[0, 0], trajectory[0, 1], "go", markersize=8, label="Start")
    ax.plot(trajectory[-1, 0], trajectory[-1, 1], "ro", markersize=8, label="End")

    if show_fixed_points:
        from .simulate import find_fixed_points

        fixed_points_data = find_fixed_points(model)
        for fp, stable in fixed_points_data:
            color = "blue" if stable else "red"
            marker = "o" if stable else "x"
            ax.plot(
                fp[0],
                fp[1],
                marker,
                color=color,
                markersize=10,
                markeredgewidth=2,
                label=f'FP ({"stable" if stable else "unstable"})',
            )

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Phase Portrait")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Time series for x1
    ax = axes[0, 1]
    ax.plot(trajectory[:, 0], "b-", linewidth=1)
    ax.set_xlabel("Time step")
    ax.set_ylabel("$x_1$")
    ax.set_title("$x_1$ vs Time")
    ax.grid(True, alpha=0.3)

    # Time series for x2
    ax = axes[1, 0]
    ax.plot(trajectory[:, 1], "r-", linewidth=1)
    ax.set_xlabel("Time step")
    ax.set_ylabel("$x_2$")
    ax.set_title("$x_2$ vs Time")
    ax.grid(True, alpha=0.3)

    # Trajectory norm
    ax = axes[1, 1]
    norms = np.linalg.norm(trajectory, axis=1)
    ax.plot(norms, "g-", linewidth=1)
    ax.set_xlabel("Time step")
    ax.set_ylabel("$||x||$")
    ax.set_title("State Norm vs Time")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return fig


def plot_fractal_map(
    fractal_data: np.ndarray,
    c1_range: Tuple[float, float],
    c2_range: Tuple[float, float],
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = "hot",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the fractal map in parameter space (c1, c2).

    The fractal map shows which parameter values lead to different
    dynamical regimes (fixed points, oscillations, chaos).

    Args:
        fractal_data: 2D array of attractor type labels or divergence times
        c1_range: Range of c1 parameter
        c2_range: Range of c2 parameter
        figsize: Figure size
        cmap: Colormap name
        save_path: Optional save path

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    extent = [c1_range[0], c1_range[1], c2_range[0], c2_range[1]]

    im = ax.imshow(
        fractal_data,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        interpolation="bilinear",
    )

    ax.set_xlabel("$c_1$", fontsize=14)
    ax.set_ylabel("$c_2$", fontsize=14)
    ax.set_title(
        "Fractal Parameter-Space Map\n(Attractor Type vs Control Parameters)",
        fontsize=16,
        fontweight="bold",
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Divergence Time / Attractor Type", fontsize=12)

    ax.grid(True, alpha=0.2, color="white", linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Fractal map saved to {save_path}")

    return fig


def plot_basin_of_attraction(
    model: FractalDynamicsModel,
    resolution: int = 200,
    x_range: Tuple[float, float] = (-2.0, 2.0),
    y_range: Tuple[float, float] = (-2.0, 2.0),
    figsize: Tuple[int, int] = (10, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the basin of attraction diagram.

    Different colors represent basins leading to different attractors.

    Args:
        model: FractalDynamicsModel instance
        resolution: Grid resolution
        x_range: Range for x1
        y_range: Range for x2
        figsize: Figure size
        save_path: Optional save path

    Returns:
        matplotlib Figure object
    """
    print(f"Computing basin of attraction (resolution={resolution})...")

    X, Y, basin_labels = basin_of_attraction_sample(
        model, x_range=x_range, y_range=y_range, resolution=resolution, n_steps=500
    )

    fig, ax = plt.subplots(figsize=figsize)

    # Create colormap
    n_basins = int(basin_labels.max()) + 1
    colors = plt.cm.tab10(np.linspace(0, 1, n_basins))
    cmap = ListedColormap(colors)

    im = ax.imshow(
        basin_labels,
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        origin="lower",
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
    )

    ax.set_xlabel("$x_1$", fontsize=14)
    ax.set_ylabel("$x_2$", fontsize=14)
    ax.set_title(
        "Basin of Attraction\n(Fractal Boundary Structure)",
        fontsize=16,
        fontweight="bold",
    )

    cbar = plt.colorbar(im, ax=ax, ticks=range(n_basins))
    cbar.set_label("Attractor Basin", fontsize=12)

    ax.grid(True, alpha=0.3, color="white", linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Basin of attraction plot saved to {save_path}")

    return fig


def plot_bifurcation_diagram(
    model_generator,
    param_name: str,
    param_range: Tuple[float, float],
    n_params: int = 500,
    n_transient: int = 1000,
    n_plot: int = 100,
    x0: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a bifurcation diagram by varying a parameter.

    Args:
        model_generator: Function that takes parameter value and returns FractalDynamicsModel
        param_name: Name of parameter being varied
        param_range: (min, max) range for parameter
        n_params: Number of parameter values to sample
        n_transient: Transient steps to discard
        n_plot: Number of points to plot for each parameter
        x0: Initial condition (default: origin)
        figsize: Figure size
        save_path: Optional save path

    Returns:
        matplotlib Figure object
    """
    if x0 is None:
        x0 = np.array([0.1, 0.1])

    param_vals = np.linspace(param_range[0], param_range[1], n_params)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    print(f"Computing bifurcation diagram for {param_name}...")

    for param_val in param_vals:
        model = model_generator(param_val)

        # Simulate with transient removal
        trajectory = simulate_orbit(model, x0, n_steps=n_transient + n_plot)
        trajectory_plot = trajectory[n_transient:]

        # Plot x1 component
        ax1.plot(
            [param_val] * len(trajectory_plot),
            trajectory_plot[:, 0],
            "k,",
            markersize=0.5,
            alpha=0.5,
        )

        # Plot x2 component
        ax2.plot(
            [param_val] * len(trajectory_plot),
            trajectory_plot[:, 1],
            "k,",
            markersize=0.5,
            alpha=0.5,
        )

    ax1.set_ylabel("$x_1$", fontsize=12)
    ax1.set_title(f"Bifurcation Diagram: $x_1$ vs {param_name}", fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel(param_name, fontsize=12)
    ax2.set_ylabel("$x_2$", fontsize=12)
    ax2.set_title(f"Bifurcation Diagram: $x_2$ vs {param_name}", fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Bifurcation diagram saved to {save_path}")

    return fig


def plot_lyapunov_spectrum(
    model: FractalDynamicsModel,
    x0: np.ndarray,
    n_steps: int = 5000,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the Lyapunov exponent evolution over time.

    Args:
        model: FractalDynamicsModel instance
        x0: Initial condition
        n_steps: Number of steps
        figsize: Figure size
        save_path: Optional save path

    Returns:
        matplotlib Figure object
    """
    # Compute cumulative Lyapunov exponent
    x = np.array(x0, dtype=np.float64)
    lyap_values = []
    log_sum = 0.0
    v = np.array([1.0, 0.0])

    for i in range(n_steps):
        J = model.jacobian(x)
        v = J @ v
        norm_v = np.linalg.norm(v)

        if norm_v > 0:
            log_sum += np.log(norm_v)
            v = v / norm_v

        lyap_values.append(log_sum / (i + 1))
        x = model.step(x)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(lyap_values, "b-", linewidth=1, alpha=0.8)
    ax.axhline(
        y=0, color="r", linestyle="--", linewidth=1.5, label="λ=0 (chaos threshold)"
    )
    ax.set_xlabel("Time step", fontsize=12)
    ax.set_ylabel("Largest Lyapunov Exponent", fontsize=12)
    ax.set_title("Lyapunov Exponent Evolution", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Lyapunov spectrum plot saved to {save_path}")

    return fig
