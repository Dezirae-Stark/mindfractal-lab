"""
Unit tests for mindfractal.visualize

Note: Visualization tests focus on execution without errors
rather than pixel-perfect output validation.
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
from mindfractal.model import FractalDynamicsModel
from mindfractal.visualize import (
    plot_orbit,
    plot_fractal_map,
    plot_basin_of_attraction,
    plot_bifurcation_diagram,
    plot_lyapunov_spectrum
)
import tempfile
import os


class TestPlotOrbit:
    """Test suite for plot_orbit function"""

    def test_plot_orbit_returns_figure(self):
        """Test that plot_orbit returns a Figure object"""
        model = FractalDynamicsModel()
        x0 = np.array([0.5, 0.5])

        fig = plot_orbit(model, x0, n_steps=100)

        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_orbit_saves_file(self):
        """Test that plot_orbit saves to file"""
        model = FractalDynamicsModel()
        x0 = np.array([0.5, 0.5])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test_orbit.png')
            fig = plot_orbit(model, x0, n_steps=100, save_path=save_path)

            assert os.path.exists(save_path)
            assert os.path.getsize(save_path) > 0  # File is not empty

        plt.close(fig)

    def test_plot_orbit_different_initial_conditions(self):
        """Test plotting from different initial conditions"""
        model = FractalDynamicsModel()

        initial_conditions = [
            np.array([0.1, 0.1]),
            np.array([0.9, 0.9]),
            np.array([-0.5, 0.5])
        ]

        for x0 in initial_conditions:
            fig = plot_orbit(model, x0, n_steps=100)
            assert isinstance(fig, matplotlib.figure.Figure)
            plt.close(fig)

    def test_plot_orbit_different_step_counts(self):
        """Test plotting with different step counts"""
        model = FractalDynamicsModel()
        x0 = np.array([0.5, 0.5])

        for n_steps in [50, 100, 500]:
            fig = plot_orbit(model, x0, n_steps=n_steps)
            assert isinstance(fig, matplotlib.figure.Figure)
            plt.close(fig)


class TestPlotFractalMap:
    """Test suite for plot_fractal_map function"""

    def test_plot_fractal_map_returns_figure(self):
        """Test that plot_fractal_map returns a Figure object"""
        # Create small fractal map for testing
        fractal_data = np.random.rand(50, 50)
        c1_range = (-1.0, 1.0)
        c2_range = (-1.0, 1.0)

        fig = plot_fractal_map(fractal_data, c1_range, c2_range)

        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_fractal_map_saves_file(self):
        """Test that plot_fractal_map saves to file"""
        fractal_data = np.random.rand(50, 50)
        c1_range = (-1.0, 1.0)
        c2_range = (-1.0, 1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test_fractal.png')
            fig = plot_fractal_map(fractal_data, c1_range, c2_range, save_path=save_path)

            assert os.path.exists(save_path)
            assert os.path.getsize(save_path) > 0

        plt.close(fig)

    def test_plot_fractal_map_different_sizes(self):
        """Test plotting fractal maps of different sizes"""
        c1_range = (-1.0, 1.0)
        c2_range = (-1.0, 1.0)

        for size in [20, 50, 100]:
            fractal_data = np.random.rand(size, size)
            fig = plot_fractal_map(fractal_data, c1_range, c2_range)
            assert isinstance(fig, matplotlib.figure.Figure)
            plt.close(fig)


class TestPlotBasinOfAttraction:
    """Test suite for plot_basin_of_attraction function"""

    def test_plot_basin_returns_figure(self):
        """Test that plot_basin_of_attraction returns a Figure object"""
        model = FractalDynamicsModel()

        # Use low resolution for fast testing
        fig = plot_basin_of_attraction(
            model,
            resolution=20,
            x_range=(-1, 1),
            y_range=(-1, 1)
        )

        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_basin_saves_file(self):
        """Test that basin plot saves to file"""
        model = FractalDynamicsModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test_basin.png')
            fig = plot_basin_of_attraction(
                model,
                resolution=15,
                save_path=save_path
            )

            assert os.path.exists(save_path)
            assert os.path.getsize(save_path) > 0

        plt.close(fig)


class TestPlotBifurcationDiagram:
    """Test suite for plot_bifurcation_diagram function"""

    def test_plot_bifurcation_returns_figure(self):
        """Test that plot_bifurcation_diagram returns a Figure object"""
        model = FractalDynamicsModel()

        # Use small parameters for fast testing
        fig = plot_bifurcation_diagram(
            model,
            param_name='c1',
            param_range=(-0.5, 0.5),
            n_points=20,
            n_steps=100,
            n_plot=50
        )

        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_bifurcation_different_params(self):
        """Test bifurcation diagram for different parameters"""
        model = FractalDynamicsModel()

        for param_name in ['c1', 'c2']:
            fig = plot_bifurcation_diagram(
                model,
                param_name=param_name,
                param_range=(-0.5, 0.5),
                n_points=15,
                n_steps=100,
                n_plot=50
            )
            assert isinstance(fig, matplotlib.figure.Figure)
            plt.close(fig)

    def test_plot_bifurcation_saves_file(self):
        """Test that bifurcation diagram saves to file"""
        model = FractalDynamicsModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test_bifurcation.png')
            fig = plot_bifurcation_diagram(
                model,
                param_name='c1',
                param_range=(-0.5, 0.5),
                n_points=10,
                n_steps=50,
                n_plot=30,
                save_path=save_path
            )

            assert os.path.exists(save_path)
            assert os.path.getsize(save_path) > 0

        plt.close(fig)


class TestPlotLyapunovSpectrum:
    """Test suite for plot_lyapunov_spectrum function"""

    def test_plot_lyapunov_returns_figure(self):
        """Test that plot_lyapunov_spectrum returns a Figure object"""
        model = FractalDynamicsModel()
        x0 = np.array([0.5, 0.5])

        # Use small parameters for fast testing
        fig = plot_lyapunov_spectrum(
            model,
            x0,
            n_steps=500,
            window_size=100
        )

        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_lyapunov_saves_file(self):
        """Test that Lyapunov spectrum plot saves to file"""
        model = FractalDynamicsModel()
        x0 = np.array([0.5, 0.5])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test_lyapunov.png')
            fig = plot_lyapunov_spectrum(
                model,
                x0,
                n_steps=300,
                window_size=100,
                save_path=save_path
            )

            assert os.path.exists(save_path)
            assert os.path.getsize(save_path) > 0

        plt.close(fig)

    def test_plot_lyapunov_different_initial_conditions(self):
        """Test Lyapunov plot from different initial conditions"""
        model = FractalDynamicsModel()

        initial_conditions = [
            np.array([0.1, 0.1]),
            np.array([0.5, 0.5]),
            np.array([0.9, 0.9])
        ]

        for x0 in initial_conditions:
            fig = plot_lyapunov_spectrum(
                model,
                x0,
                n_steps=300,
                window_size=100
            )
            assert isinstance(fig, matplotlib.figure.Figure)
            plt.close(fig)


class TestVisualizationIntegration:
    """Integration tests for visualization module"""

    def test_all_plots_in_sequence(self):
        """Test creating all plot types in sequence"""
        model = FractalDynamicsModel()
        x0 = np.array([0.5, 0.5])

        with tempfile.TemporaryDirectory() as tmpdir:
            # Plot orbit
            fig1 = plot_orbit(model, x0, n_steps=100,
                             save_path=os.path.join(tmpdir, 'orbit.png'))
            plt.close(fig1)

            # Plot fractal map
            fractal_data = np.random.rand(30, 30)
            fig2 = plot_fractal_map(fractal_data, (-1, 1), (-1, 1),
                                    save_path=os.path.join(tmpdir, 'fractal.png'))
            plt.close(fig2)

            # Plot basin
            fig3 = plot_basin_of_attraction(model, resolution=15,
                                           save_path=os.path.join(tmpdir, 'basin.png'))
            plt.close(fig3)

            # Plot bifurcation
            fig4 = plot_bifurcation_diagram(model, 'c1', (-0.5, 0.5), 10, 50, 30,
                                           save_path=os.path.join(tmpdir, 'bifurcation.png'))
            plt.close(fig4)

            # Plot Lyapunov
            fig5 = plot_lyapunov_spectrum(model, x0, 300, 100,
                                         save_path=os.path.join(tmpdir, 'lyapunov.png'))
            plt.close(fig5)

            # Verify all files created
            for filename in ['orbit.png', 'fractal.png', 'basin.png',
                           'bifurcation.png', 'lyapunov.png']:
                filepath = os.path.join(tmpdir, filename)
                assert os.path.exists(filepath)
                assert os.path.getsize(filepath) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
