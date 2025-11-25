"""
Possibility Manifold Visualization

Tools for visualizing the high-dimensional Possibility Manifold
through projections, slices, and interactive plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Optional, List

from .possibility_manifold import PossibilityManifold, ParameterPoint, StabilityRegion
from .possibility_metrics import StabilityClassifier
from .possibility_slicer import TimelineSlicer, OrbitBranch


class PossibilityVisualizer:
    """
    Visualization tools for the Possibility Manifold

    Creates 2D/3D projections and interactive views of the
    high-dimensional manifold structure.
    """

    def __init__(self, manifold: PossibilityManifold):
        self.manifold = manifold
        self.classifier = StabilityClassifier(manifold)
        self.slicer = TimelineSlicer(manifold)

    def plot_stability_landscape(self, param_range=(-2, 2), resolution=50,
                                figsize=(12, 5)):
        """
        Plot 2D stability landscape

        Parameters:
        -----------
        param_range : tuple
            Range for parameters
        resolution : int
            Grid resolution
        figsize : tuple
            Figure size
        """
        # Compute landscape
        landscape = self.classifier.map_stability_landscape(param_range, resolution)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot stability regions
        cmap = ListedColormap(['green', 'orange', 'red', 'yellow', 'gray'])
        im1 = ax1.imshow(landscape['stability_grid'].T, origin='lower',
                        extent=[param_range[0], param_range[1]]*2,
                        cmap=cmap, aspect='auto')
        ax1.set_xlabel('Parameter c₁')
        ax1.set_ylabel('Parameter c₂')
        ax1.set_title('Stability Regions')
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_ticks([0, 1, 2, 3, 4])
        cbar1.set_ticklabels(['Stable', 'Chaotic', 'Divergent', 'Boundary', 'Unknown'])

        # Plot Lyapunov exponents
        im2 = ax2.imshow(landscape['lyapunov_grid'].T, origin='lower',
                        extent=[param_range[0], param_range[1]]*2,
                        cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
        ax2.set_xlabel('Parameter c₁')
        ax2.set_ylabel('Parameter c₂')
        ax2.set_title('Lyapunov Exponents')
        plt.colorbar(im2, ax=ax2, label='λ')

        plt.tight_layout()
        return fig

    def plot_timeline_branch(self, branch: OrbitBranch, figsize=(14, 5)):
        """
        Visualize a timeline branch through the manifold

        Parameters:
        -----------
        branch : OrbitBranch
            Timeline to visualize
        figsize : tuple
            Figure size
        """
        n_points = len(branch.points)

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Plot parameter evolution
        params = np.array([p.c for p in branch.points])
        for i in range(min(2, self.manifold.dim)):
            axes[0].plot(params[:, i].real, label=f'Re(c_{i})')
            axes[0].plot(params[:, i].imag, '--', label=f'Im(c_{i})')
        axes[0].set_xlabel('Timeline Step')
        axes[0].set_ylabel('Parameter Value')
        axes[0].set_title('Parameter Evolution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot orbit endpoints
        endpoints = np.array([orbit[-1] if not np.isnan(orbit[-1]).any()
                            else orbit[0] for orbit in branch.orbits])
        axes[1].plot(endpoints[:, 0].real, endpoints[:, 0].imag, 'o-')
        axes[1].set_xlabel('Re(z₀)')
        axes[1].set_ylabel('Im(z₀)')
        axes[1].set_title('Orbit Endpoints')
        axes[1].grid(True, alpha=0.3)

        # Plot sample orbits
        sample_indices = [0, n_points//2, n_points-1]
        colors = ['blue', 'green', 'red']
        for idx, color in zip(sample_indices, colors):
            orbit = branch.orbits[idx]
            if not np.isnan(orbit).any():
                axes[2].plot(orbit[:, 0].real, orbit[:, 0].imag,
                           alpha=0.6, color=color, label=f'Step {idx}')
        axes[2].set_xlabel('Re(z)')
        axes[2].set_ylabel('Im(z)')
        axes[2].set_title('Sample Orbits')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_manifold_slice_3d(self, points: List[ParameterPoint],
                              orbits: List[np.ndarray], figsize=(10, 8)):
        """
        3D visualization of manifold slice

        Parameters:
        -----------
        points : list
            Parameter points
        orbits : list
            Corresponding orbits
        figsize : tuple
            Figure size
        """
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Plot points colored by stability
        for point, orbit in zip(points, orbits):
            region = self.manifold.classify_stability(orbit)

            color = self._region_to_color(region)
            ax.scatter(point.c[0].real, point.c[1].real if self.manifold.dim > 1 else 0,
                      point.z0[0].real, c=color, s=50, alpha=0.6)

        ax.set_xlabel('Re(c₀)')
        ax.set_ylabel('Re(c₁)')
        ax.set_zlabel('Re(z₀)')
        ax.set_title('3D Manifold Slice')

        plt.tight_layout()
        return fig

    @staticmethod
    def _region_to_color(region: StabilityRegion) -> str:
        """Map stability region to color"""
        mapping = {
            StabilityRegion.STABLE_ATTRACTOR: 'green',
            StabilityRegion.CHAOTIC: 'orange',
            StabilityRegion.DIVERGENT: 'red',
            StabilityRegion.BOUNDARY: 'yellow',
            StabilityRegion.UNKNOWN: 'gray'
        }
        return mapping.get(region, 'gray')
