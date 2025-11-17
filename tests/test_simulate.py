"""
Unit tests for mindfractal.simulate
"""

import numpy as np
import pytest
from mindfractal.model import FractalDynamicsModel
from mindfractal.simulate import (
    simulate_orbit,
    find_fixed_points,
    compute_attractor_type,
    basin_of_attraction_sample,
    poincare_section
)


class TestSimulateOrbit:
    """Test suite for simulate_orbit function"""

    def test_simulate_orbit_shape(self):
        """Test that simulate_orbit returns correct shape"""
        model = FractalDynamicsModel()
        x0 = np.array([0.5, 0.5])
        n_steps = 100

        trajectory = simulate_orbit(model, x0, n_steps=n_steps)

        assert trajectory.shape == (n_steps, 2)

    def test_simulate_orbit_initial_condition(self):
        """Test that trajectory starts at x0"""
        model = FractalDynamicsModel()
        x0 = np.array([0.5, 0.5])

        trajectory = simulate_orbit(model, x0, n_steps=100)

        np.testing.assert_array_almost_equal(trajectory[0], x0)

    def test_simulate_orbit_deterministic(self):
        """Test that simulation is deterministic"""
        model = FractalDynamicsModel()
        x0 = np.array([0.5, 0.5])

        traj1 = simulate_orbit(model, x0, n_steps=100)
        traj2 = simulate_orbit(model, x0, n_steps=100)

        np.testing.assert_array_equal(traj1, traj2)

    def test_simulate_orbit_return_final_only(self):
        """Test return_all=False returns only final state"""
        model = FractalDynamicsModel()
        x0 = np.array([0.5, 0.5])

        final_state = simulate_orbit(model, x0, n_steps=100, return_all=False)

        assert final_state.shape == (2,)

    def test_simulate_orbit_consistency(self):
        """Test that return_all=False gives same final state as full trajectory"""
        model = FractalDynamicsModel()
        x0 = np.array([0.5, 0.5])

        full_traj = simulate_orbit(model, x0, n_steps=100, return_all=True)
        final_only = simulate_orbit(model, x0, n_steps=100, return_all=False)

        np.testing.assert_array_almost_equal(full_traj[-1], final_only)

    def test_simulate_orbit_finite(self):
        """Test that all trajectory values are finite"""
        model = FractalDynamicsModel()
        x0 = np.array([0.5, 0.5])

        trajectory = simulate_orbit(model, x0, n_steps=500)

        assert np.all(np.isfinite(trajectory))


class TestFindFixedPoints:
    """Test suite for find_fixed_points function"""

    def test_find_fixed_points_returns_list(self):
        """Test that find_fixed_points returns a list"""
        model = FractalDynamicsModel()

        fixed_points = find_fixed_points(model, n_trials=5)

        assert isinstance(fixed_points, list)

    def test_find_fixed_points_simple_case(self):
        """Test finding fixed point in simple system"""
        # Create model where c is a fixed point
        A = np.zeros((2, 2))
        B = np.zeros((2, 2))
        W = np.eye(2)
        c = np.array([0.5, 0.5])

        model = FractalDynamicsModel(A=A, B=B, W=W, c=c)

        fixed_points = find_fixed_points(model, n_trials=3, tolerance=1e-6)

        # Should find at least one fixed point near c
        assert len(fixed_points) > 0

        # Check that found points are actually fixed
        for fp, is_stable in fixed_points:
            x_next = model.step(fp)
            assert np.allclose(fp, x_next, atol=1e-5)

    def test_find_fixed_points_stability_flag(self):
        """Test that stability flag is boolean"""
        model = FractalDynamicsModel()

        fixed_points = find_fixed_points(model, n_trials=5)

        for fp, is_stable in fixed_points:
            assert isinstance(is_stable, (bool, np.bool_))


class TestComputeAttractorType:
    """Test suite for compute_attractor_type function"""

    def test_compute_attractor_type_fixed_point(self):
        """Test classification of fixed point attractor"""
        # Create trajectory that converges to fixed point
        trajectory = np.zeros((1000, 2))
        trajectory[:] = [0.5, 0.5]  # Constant trajectory

        attractor_type = compute_attractor_type(trajectory, tolerance=1e-3)

        assert attractor_type == 'fixed_point'

    def test_compute_attractor_type_limit_cycle(self):
        """Test classification of limit cycle"""
        # Create simple periodic trajectory
        t = np.linspace(0, 10 * np.pi, 1000)
        trajectory = np.column_stack([np.cos(t), np.sin(t)])

        attractor_type = compute_attractor_type(trajectory, tolerance=0.1)

        # Should detect as limit_cycle or chaotic (depending on tolerance)
        assert attractor_type in ['limit_cycle', 'chaotic']

    def test_compute_attractor_type_unbounded(self):
        """Test classification of unbounded trajectory"""
        # Create diverging trajectory
        trajectory = np.zeros((1000, 2))
        for i in range(1000):
            trajectory[i] = [i * 0.1, i * 0.1]

        attractor_type = compute_attractor_type(trajectory, tolerance=1e-3)

        assert attractor_type == 'unbounded'

    def test_compute_attractor_type_returns_string(self):
        """Test that function returns a string"""
        model = FractalDynamicsModel()
        x0 = np.array([0.5, 0.5])
        trajectory = simulate_orbit(model, x0, n_steps=500)

        attractor_type = compute_attractor_type(trajectory)

        assert isinstance(attractor_type, str)
        assert attractor_type in ['fixed_point', 'limit_cycle', 'chaotic', 'unbounded']


class TestBasinOfAttraction:
    """Test suite for basin_of_attraction_sample function"""

    def test_basin_of_attraction_shape(self):
        """Test that basin map has correct shape"""
        model = FractalDynamicsModel()
        resolution = 50

        basin_map = basin_of_attraction_sample(
            model,
            x_range=(-1, 1),
            y_range=(-1, 1),
            resolution=resolution,
            criterion='divergence_time'
        )

        assert basin_map.shape == (resolution, resolution)

    def test_basin_of_attraction_finite(self):
        """Test that basin map contains finite values"""
        model = FractalDynamicsModel()

        basin_map = basin_of_attraction_sample(
            model,
            x_range=(-1, 1),
            y_range=(-1, 1),
            resolution=20,
            criterion='final_norm'
        )

        assert np.all(np.isfinite(basin_map))

    def test_basin_of_attraction_criteria(self):
        """Test different criteria produce different results"""
        model = FractalDynamicsModel()
        resolution = 30

        basin_div = basin_of_attraction_sample(
            model, (-1, 1), (-1, 1), resolution, criterion='divergence_time'
        )
        basin_norm = basin_of_attraction_sample(
            model, (-1, 1), (-1, 1), resolution, criterion='final_norm'
        )

        # Different criteria should generally give different maps
        # (unless all trajectories behave identically)
        assert basin_div.shape == basin_norm.shape


class TestPoincareSection:
    """Test suite for poincare_section function"""

    def test_poincare_section_circular_orbit(self):
        """Test Poincare section for circular orbit"""
        # Create circular trajectory
        t = np.linspace(0, 4 * np.pi, 1000)
        trajectory = np.column_stack([np.cos(t), np.sin(t)])

        # Section at x=0
        crossings = poincare_section(trajectory, axis=0, value=0.0, tolerance=0.05)

        # Should find multiple crossings
        assert len(crossings) > 0
        assert crossings.shape[1] == 2  # 2D points

    def test_poincare_section_constant_trajectory(self):
        """Test Poincare section for constant trajectory"""
        # Constant trajectory at [0.5, 0.5]
        trajectory = np.ones((1000, 2)) * 0.5

        # Section at x=0 (trajectory never crosses)
        crossings = poincare_section(trajectory, axis=0, value=0.0, tolerance=0.01)

        # Should find no crossings
        assert len(crossings) == 0

    def test_poincare_section_axes(self):
        """Test Poincare section on different axes"""
        t = np.linspace(0, 2 * np.pi, 1000)
        trajectory = np.column_stack([np.cos(t), np.sin(t)])

        crossings_x = poincare_section(trajectory, axis=0, value=0.0, tolerance=0.05)
        crossings_y = poincare_section(trajectory, axis=1, value=0.0, tolerance=0.05)

        # Both should find crossings for circular orbit
        assert len(crossings_x) > 0
        assert len(crossings_y) > 0


class TestIntegration:
    """Integration tests combining multiple functions"""

    def test_full_workflow(self):
        """Test complete simulation and analysis workflow"""
        # Create model
        model = FractalDynamicsModel()

        # Simulate orbit
        x0 = np.array([0.5, 0.5])
        trajectory = simulate_orbit(model, x0, n_steps=500)

        # Classify attractor
        attractor_type = compute_attractor_type(trajectory)

        # All steps should complete without error
        assert trajectory.shape == (500, 2)
        assert attractor_type in ['fixed_point', 'limit_cycle', 'chaotic', 'unbounded']

    def test_multiple_initial_conditions(self):
        """Test simulations from multiple initial conditions"""
        model = FractalDynamicsModel()

        initial_conditions = [
            np.array([0.1, 0.1]),
            np.array([0.5, 0.5]),
            np.array([0.9, 0.9]),
            np.array([-0.5, 0.5])
        ]

        for x0 in initial_conditions:
            trajectory = simulate_orbit(model, x0, n_steps=200)
            assert trajectory.shape == (200, 2)
            assert np.all(np.isfinite(trajectory))

            attractor_type = compute_attractor_type(trajectory)
            assert isinstance(attractor_type, str)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
