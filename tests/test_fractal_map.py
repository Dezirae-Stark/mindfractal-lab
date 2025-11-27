"""
Unit tests for mindfractal.fractal_map
"""

import numpy as np
import pytest

from mindfractal.fractal_map import (adaptive_fractal_map,
                                     generate_fractal_map, zoom_fractal_map)


class TestGenerateFractalMap:
    """Test suite for generate_fractal_map function"""

    def test_generate_fractal_map_shape(self):
        """Test that fractal map has correct shape"""
        resolution = 20
        fractal_data = generate_fractal_map(
            resolution=resolution,
            c1_range=(-0.5, 0.5),
            c2_range=(-0.5, 0.5),
            max_steps=50,
        )

        assert fractal_data.shape == (resolution, resolution)

    def test_generate_fractal_map_finite(self):
        """Test that fractal map contains finite values"""
        fractal_data = generate_fractal_map(
            resolution=15, c1_range=(-0.5, 0.5), c2_range=(-0.5, 0.5), max_steps=50
        )

        assert np.all(np.isfinite(fractal_data))

    def test_generate_fractal_map_divergence_time(self):
        """Test fractal map with divergence_time criterion"""
        fractal_data = generate_fractal_map(
            resolution=10,
            c1_range=(-0.5, 0.5),
            c2_range=(-0.5, 0.5),
            criterion="divergence_time",
            max_steps=100,
        )

        assert fractal_data.shape == (10, 10)
        assert np.all(fractal_data >= 0)  # Divergence time should be non-negative
        assert np.all(fractal_data <= 100)  # Should not exceed max_steps

    def test_generate_fractal_map_final_norm(self):
        """Test fractal map with final_norm criterion"""
        fractal_data = generate_fractal_map(
            resolution=10,
            c1_range=(-0.5, 0.5),
            c2_range=(-0.5, 0.5),
            criterion="final_norm",
            max_steps=50,
        )

        assert fractal_data.shape == (10, 10)
        assert np.all(fractal_data >= 0)  # Norm should be non-negative

    def test_generate_fractal_map_lyapunov(self):
        """Test fractal map with lyapunov criterion"""
        fractal_data = generate_fractal_map(
            resolution=8,
            c1_range=(-0.3, 0.3),
            c2_range=(-0.3, 0.3),
            criterion="lyapunov",
            max_steps=100,
        )

        assert fractal_data.shape == (8, 8)
        assert np.all(np.isfinite(fractal_data))

    def test_generate_fractal_map_attractor_type(self):
        """Test fractal map with attractor_type criterion"""
        # Note: attractor_type criterion internally uses compute_attractor_type
        # which requires n_steps > transient (default 1000), so max_steps must be > 1000
        fractal_data = generate_fractal_map(
            resolution=5,
            c1_range=(-0.3, 0.3),
            c2_range=(-0.3, 0.3),
            criterion="attractor_type",
            max_steps=1500,
        )

        assert fractal_data.shape == (5, 5)
        # Attractor types are integers: 0, 1, 2, 3, or -1
        assert np.all((fractal_data >= -1) & (fractal_data <= 3))

    def test_generate_fractal_map_custom_initial_condition(self):
        """Test fractal map with custom initial condition"""
        x0 = np.array([0.3, 0.7])
        fractal_data = generate_fractal_map(
            resolution=10,
            x0=x0,
            c1_range=(-0.5, 0.5),
            c2_range=(-0.5, 0.5),
            max_steps=50,
        )

        assert fractal_data.shape == (10, 10)

    def test_generate_fractal_map_different_resolutions(self):
        """Test fractal map generation at different resolutions"""
        for resolution in [5, 10, 15]:
            fractal_data = generate_fractal_map(
                resolution=resolution,
                c1_range=(-0.3, 0.3),
                c2_range=(-0.3, 0.3),
                max_steps=30,
            )

            assert fractal_data.shape == (resolution, resolution)
            assert np.all(np.isfinite(fractal_data))

    def test_generate_fractal_map_asymmetric_ranges(self):
        """Test fractal map with asymmetric parameter ranges"""
        fractal_data = generate_fractal_map(
            resolution=10, c1_range=(-0.8, 0.2), c2_range=(0.1, 0.9), max_steps=50
        )

        assert fractal_data.shape == (10, 10)
        assert np.all(np.isfinite(fractal_data))


class TestZoomFractalMap:
    """Test suite for zoom_fractal_map function"""

    def test_zoom_fractal_map_shape(self):
        """Test that zoomed fractal map has correct shape"""
        center = (0.0, 0.0)
        resolution = 15

        fractal_data = zoom_fractal_map(
            center=center, zoom_factor=2.0, resolution=resolution, max_steps=30
        )

        assert fractal_data.shape == (resolution, resolution)

    def test_zoom_fractal_map_different_zoom_factors(self):
        """Test zooming with different zoom factors"""
        center = (0.0, 0.0)
        resolution = 10

        for zoom_factor in [2.0, 5.0, 10.0]:
            fractal_data = zoom_fractal_map(
                center=center,
                zoom_factor=zoom_factor,
                resolution=resolution,
                max_steps=30,
            )

            assert fractal_data.shape == (resolution, resolution)
            assert np.all(np.isfinite(fractal_data))

    def test_zoom_fractal_map_different_centers(self):
        """Test zooming at different center points"""
        resolution = 10

        for center in [(0.0, 0.0), (0.5, 0.5), (-0.3, 0.2)]:
            fractal_data = zoom_fractal_map(
                center=center, zoom_factor=3.0, resolution=resolution, max_steps=30
            )

            assert fractal_data.shape == (resolution, resolution)
            assert np.all(np.isfinite(fractal_data))

    def test_zoom_fractal_map_with_criterion(self):
        """Test zooming with different criteria"""
        center = (0.0, 0.0)
        resolution = 8

        for criterion in ["divergence_time", "final_norm"]:
            fractal_data = zoom_fractal_map(
                center=center,
                zoom_factor=2.0,
                resolution=resolution,
                criterion=criterion,
                max_steps=50,
            )

            assert fractal_data.shape == (resolution, resolution)
            assert np.all(np.isfinite(fractal_data))


class TestAdaptiveFractalMap:
    """Test suite for adaptive_fractal_map function"""

    def test_adaptive_fractal_map_shape(self):
        """Test that adaptive fractal map has correct shape"""
        fractal_data = adaptive_fractal_map(
            c1_range=(-0.3, 0.3),
            c2_range=(-0.3, 0.3),
            base_resolution=10,
            max_resolution=15,
            max_steps=30,
        )

        # Adaptive refinement returns max_resolution size
        assert fractal_data.ndim == 2
        assert np.all(np.isfinite(fractal_data))

    def test_adaptive_fractal_map_finite(self):
        """Test adaptive fractal map produces finite values"""
        fractal_data = adaptive_fractal_map(
            c1_range=(-0.3, 0.3),
            c2_range=(-0.3, 0.3),
            base_resolution=8,
            max_resolution=12,
            max_steps=30,
        )

        assert fractal_data.ndim == 2
        assert np.all(np.isfinite(fractal_data))

    def test_adaptive_fractal_map_with_criterion(self):
        """Test adaptive fractal map with different criteria"""
        for criterion in ["divergence_time", "final_norm"]:
            fractal_data = adaptive_fractal_map(
                c1_range=(-0.3, 0.3),
                c2_range=(-0.3, 0.3),
                base_resolution=8,
                max_resolution=10,
                criterion=criterion,
                max_steps=30,
            )

            assert fractal_data.ndim == 2
            assert np.all(np.isfinite(fractal_data))


class TestFractalMapIntegration:
    """Integration tests for fractal map module"""

    def test_generate_and_zoom_workflow(self):
        """Test workflow of generating base map then zooming"""
        # Generate base map
        base_map = generate_fractal_map(
            resolution=15, c1_range=(-0.5, 0.5), c2_range=(-0.5, 0.5), max_steps=50
        )

        # Zoom into a region
        zoomed_map = zoom_fractal_map(
            center=(0.2, 0.2), zoom_factor=3.0, resolution=15, max_steps=50
        )

        assert base_map.shape == zoomed_map.shape
        assert np.all(np.isfinite(base_map))
        assert np.all(np.isfinite(zoomed_map))

    def test_divergence_and_final_norm_criteria(self):
        """Test that divergence_time and final_norm criteria produce valid maps"""
        resolution = 10

        div_map = generate_fractal_map(
            resolution=resolution,
            c1_range=(-0.3, 0.3),
            c2_range=(-0.3, 0.3),
            criterion="divergence_time",
            max_steps=50,
        )

        norm_map = generate_fractal_map(
            resolution=resolution,
            c1_range=(-0.3, 0.3),
            c2_range=(-0.3, 0.3),
            criterion="final_norm",
            max_steps=50,
        )

        assert div_map.shape == (resolution, resolution)
        assert norm_map.shape == (resolution, resolution)
        assert np.all(np.isfinite(div_map))
        assert np.all(np.isfinite(norm_map))

    def test_progressive_zoom_sequence(self):
        """Test progressive zooming into fractal boundary"""
        center = (0.0, 0.0)
        resolution = 10

        zoom_factors = [1.0, 2.0, 5.0]
        previous_map = None

        for zoom_factor in zoom_factors:
            current_map = zoom_fractal_map(
                center=center,
                zoom_factor=zoom_factor,
                resolution=resolution,
                max_steps=30,
            )

            assert current_map.shape == (resolution, resolution)
            assert np.all(np.isfinite(current_map))

            previous_map = current_map


class TestFractalMapEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_single_point_map(self):
        """Test fractal map with resolution=1"""
        fractal_data = generate_fractal_map(
            resolution=1, c1_range=(-0.5, 0.5), c2_range=(-0.5, 0.5), max_steps=30
        )

        assert fractal_data.shape == (1, 1)
        assert np.isfinite(fractal_data[0, 0])

    def test_very_small_range(self):
        """Test fractal map with very small parameter range"""
        fractal_data = generate_fractal_map(
            resolution=5, c1_range=(0.0, 0.01), c2_range=(0.0, 0.01), max_steps=30
        )

        assert fractal_data.shape == (5, 5)
        assert np.all(np.isfinite(fractal_data))

    def test_extreme_zoom(self):
        """Test zooming with very high zoom factor"""
        fractal_data = zoom_fractal_map(
            center=(0.0, 0.0), zoom_factor=100.0, resolution=8, max_steps=30
        )

        assert fractal_data.shape == (8, 8)
        assert np.all(np.isfinite(fractal_data))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
