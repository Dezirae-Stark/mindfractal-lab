"""
Unit tests for mindfractal.fractal_map
"""

import numpy as np
import pytest
from mindfractal.model import FractalDynamicsModel
from mindfractal.fractal_map import (
    generate_fractal_map,
    zoom_fractal_map,
    adaptive_fractal_map
)


class TestGenerateFractalMap:
    """Test suite for generate_fractal_map function"""

    def test_generate_fractal_map_shape(self):
        """Test that fractal map has correct shape"""
        resolution = 50
        fractal_data = generate_fractal_map(
            resolution=resolution,
            c1_range=(-1.0, 1.0),
            c2_range=(-1.0, 1.0),
            n_steps=100
        )

        assert fractal_data.shape == (resolution, resolution)

    def test_generate_fractal_map_finite(self):
        """Test that fractal map contains finite values"""
        fractal_data = generate_fractal_map(
            resolution=30,
            c1_range=(-0.5, 0.5),
            c2_range=(-0.5, 0.5),
            n_steps=100
        )

        assert np.all(np.isfinite(fractal_data))

    def test_generate_fractal_map_divergence_time(self):
        """Test fractal map with divergence_time criterion"""
        fractal_data = generate_fractal_map(
            resolution=20,
            c1_range=(-1.0, 1.0),
            c2_range=(-1.0, 1.0),
            criterion='divergence_time',
            n_steps=100,
            max_steps=200
        )

        assert fractal_data.shape == (20, 20)
        assert np.all(fractal_data >= 0)  # Divergence time should be non-negative
        assert np.all(fractal_data <= 200)  # Should not exceed max_steps

    def test_generate_fractal_map_final_norm(self):
        """Test fractal map with final_norm criterion"""
        fractal_data = generate_fractal_map(
            resolution=20,
            c1_range=(-1.0, 1.0),
            c2_range=(-1.0, 1.0),
            criterion='final_norm',
            n_steps=100
        )

        assert fractal_data.shape == (20, 20)
        assert np.all(fractal_data >= 0)  # Norm should be non-negative

    def test_generate_fractal_map_lyapunov(self):
        """Test fractal map with lyapunov criterion"""
        fractal_data = generate_fractal_map(
            resolution=15,
            c1_range=(-0.5, 0.5),
            c2_range=(-0.5, 0.5),
            criterion='lyapunov',
            n_steps=200
        )

        assert fractal_data.shape == (15, 15)
        assert np.all(np.isfinite(fractal_data))

    def test_generate_fractal_map_attractor_type(self):
        """Test fractal map with attractor_type criterion"""
        fractal_data = generate_fractal_map(
            resolution=15,
            c1_range=(-0.5, 0.5),
            c2_range=(-0.5, 0.5),
            criterion='attractor_type',
            n_steps=200
        )

        assert fractal_data.shape == (15, 15)
        # Attractor types are integers: 0, 1, 2, 3
        assert np.all((fractal_data >= 0) & (fractal_data <= 3))

    def test_generate_fractal_map_custom_initial_condition(self):
        """Test fractal map with custom initial condition"""
        x0 = np.array([0.3, 0.7])
        fractal_data = generate_fractal_map(
            resolution=20,
            x0=x0,
            c1_range=(-0.5, 0.5),
            c2_range=(-0.5, 0.5),
            n_steps=100
        )

        assert fractal_data.shape == (20, 20)

    def test_generate_fractal_map_different_resolutions(self):
        """Test fractal map generation at different resolutions"""
        for resolution in [10, 20, 50]:
            fractal_data = generate_fractal_map(
                resolution=resolution,
                c1_range=(-0.5, 0.5),
                c2_range=(-0.5, 0.5),
                n_steps=50
            )

            assert fractal_data.shape == (resolution, resolution)
            assert np.all(np.isfinite(fractal_data))

    def test_generate_fractal_map_asymmetric_ranges(self):
        """Test fractal map with asymmetric parameter ranges"""
        fractal_data = generate_fractal_map(
            resolution=20,
            c1_range=(-0.8, 0.2),
            c2_range=(0.1, 0.9),
            n_steps=100
        )

        assert fractal_data.shape == (20, 20)
        assert np.all(np.isfinite(fractal_data))


class TestZoomFractalMap:
    """Test suite for zoom_fractal_map function"""

    def test_zoom_fractal_map_shape(self):
        """Test that zoomed fractal map has correct shape"""
        center = (0.0, 0.0)
        resolution = 30

        fractal_data = zoom_fractal_map(
            center=center,
            zoom_factor=2.0,
            resolution=resolution,
            n_steps=50
        )

        assert fractal_data.shape == (resolution, resolution)

    def test_zoom_fractal_map_different_zoom_factors(self):
        """Test zooming with different zoom factors"""
        center = (0.0, 0.0)
        resolution = 20

        for zoom_factor in [2.0, 5.0, 10.0]:
            fractal_data = zoom_fractal_map(
                center=center,
                zoom_factor=zoom_factor,
                resolution=resolution,
                n_steps=50
            )

            assert fractal_data.shape == (resolution, resolution)
            assert np.all(np.isfinite(fractal_data))

    def test_zoom_fractal_map_different_centers(self):
        """Test zooming at different center points"""
        resolution = 20

        for center in [(0.0, 0.0), (0.5, 0.5), (-0.3, 0.2)]:
            fractal_data = zoom_fractal_map(
                center=center,
                zoom_factor=3.0,
                resolution=resolution,
                n_steps=50
            )

            assert fractal_data.shape == (resolution, resolution)
            assert np.all(np.isfinite(fractal_data))

    def test_zoom_fractal_map_with_criterion(self):
        """Test zooming with different criteria"""
        center = (0.0, 0.0)
        resolution = 15

        for criterion in ['divergence_time', 'final_norm', 'lyapunov']:
            fractal_data = zoom_fractal_map(
                center=center,
                zoom_factor=2.0,
                resolution=resolution,
                criterion=criterion,
                n_steps=100
            )

            assert fractal_data.shape == (resolution, resolution)
            assert np.all(np.isfinite(fractal_data))


class TestAdaptiveFractalMap:
    """Test suite for adaptive_fractal_map function"""

    def test_adaptive_fractal_map_shape(self):
        """Test that adaptive fractal map has correct shape"""
        resolution = 20
        fractal_data = adaptive_fractal_map(
            c1_range=(-0.5, 0.5),
            c2_range=(-0.5, 0.5),
            resolution=resolution,
            refinement_levels=1,
            n_steps=50
        )

        # Adaptive refinement may change resolution
        assert fractal_data.ndim == 2
        assert np.all(np.isfinite(fractal_data))

    def test_adaptive_fractal_map_refinement_levels(self):
        """Test adaptive fractal map with different refinement levels"""
        resolution = 20

        for refinement_levels in [0, 1, 2]:
            fractal_data = adaptive_fractal_map(
                c1_range=(-0.5, 0.5),
                c2_range=(-0.5, 0.5),
                resolution=resolution,
                refinement_levels=refinement_levels,
                n_steps=50
            )

            assert fractal_data.ndim == 2
            assert np.all(np.isfinite(fractal_data))

    def test_adaptive_fractal_map_with_criterion(self):
        """Test adaptive fractal map with different criteria"""
        resolution = 15

        for criterion in ['divergence_time', 'final_norm']:
            fractal_data = adaptive_fractal_map(
                c1_range=(-0.5, 0.5),
                c2_range=(-0.5, 0.5),
                resolution=resolution,
                refinement_levels=1,
                criterion=criterion,
                n_steps=50
            )

            assert fractal_data.ndim == 2
            assert np.all(np.isfinite(fractal_data))


class TestFractalMapIntegration:
    """Integration tests for fractal map module"""

    def test_generate_and_zoom_workflow(self):
        """Test workflow of generating base map then zooming"""
        # Generate base map
        base_map = generate_fractal_map(
            resolution=30,
            c1_range=(-1.0, 1.0),
            c2_range=(-1.0, 1.0),
            n_steps=100
        )

        # Zoom into a region
        zoomed_map = zoom_fractal_map(
            center=(0.5, 0.5),
            zoom_factor=3.0,
            resolution=30,
            n_steps=100
        )

        assert base_map.shape == zoomed_map.shape
        assert np.all(np.isfinite(base_map))
        assert np.all(np.isfinite(zoomed_map))

    def test_all_criteria_produce_valid_maps(self):
        """Test that all criteria produce valid fractal maps"""
        criteria = ['divergence_time', 'final_norm', 'lyapunov', 'attractor_type']
        resolution = 20

        maps = {}
        for criterion in criteria:
            maps[criterion] = generate_fractal_map(
                resolution=resolution,
                c1_range=(-0.5, 0.5),
                c2_range=(-0.5, 0.5),
                criterion=criterion,
                n_steps=100
            )

            assert maps[criterion].shape == (resolution, resolution)
            assert np.all(np.isfinite(maps[criterion]))

        # Maps should generally be different (unless dynamics identical everywhere)
        # This is a weak test, just checking they're not all zeros
        for criterion, fmap in maps.items():
            assert not np.all(fmap == 0)

    def test_progressive_zoom_sequence(self):
        """Test progressive zooming into fractal boundary"""
        center = (0.0, 0.0)
        resolution = 20

        zoom_factors = [1.0, 2.0, 5.0, 10.0]
        previous_map = None

        for zoom_factor in zoom_factors:
            current_map = zoom_fractal_map(
                center=center,
                zoom_factor=zoom_factor,
                resolution=resolution,
                n_steps=50
            )

            assert current_map.shape == (resolution, resolution)
            assert np.all(np.isfinite(current_map))

            # Maps at different zoom levels should generally differ
            # (showing self-similarity rather than uniformity)
            if previous_map is not None:
                # Just check they're not identical
                assert not np.array_equal(current_map, previous_map)

            previous_map = current_map


class TestFractalMapEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_single_point_map(self):
        """Test fractal map with resolution=1"""
        fractal_data = generate_fractal_map(
            resolution=1,
            c1_range=(-0.5, 0.5),
            c2_range=(-0.5, 0.5),
            n_steps=50
        )

        assert fractal_data.shape == (1, 1)
        assert np.isfinite(fractal_data[0, 0])

    def test_very_small_range(self):
        """Test fractal map with very small parameter range"""
        fractal_data = generate_fractal_map(
            resolution=10,
            c1_range=(0.0, 0.01),
            c2_range=(0.0, 0.01),
            n_steps=50
        )

        assert fractal_data.shape == (10, 10)
        assert np.all(np.isfinite(fractal_data))

    def test_extreme_zoom(self):
        """Test zooming with very high zoom factor"""
        fractal_data = zoom_fractal_map(
            center=(0.0, 0.0),
            zoom_factor=100.0,
            resolution=15,
            n_steps=50
        )

        assert fractal_data.shape == (15, 15)
        assert np.all(np.isfinite(fractal_data))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
