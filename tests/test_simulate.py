"""
Unit tests for mindfractal.simulate
"""

import numpy as np
import pytest

from mindfractal.model import FractalDynamicsModel
from mindfractal.simulate import (
    basin_of_attraction_sample,
    compute_attractor_type,
    find_fixed_points,
    poincare_section,
    simulate_orbit,
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
        """Test that return_all=False gives consistent final state"""
        model = FractalDynamicsModel()
        x0 = np.array([0.5, 0.5])

        # With return_all=True, trajectory[0] = x0, trajectory[i] = step^i(x0)
        # With return_all=False and n_steps=N, result = step^N(x0)
        # So trajectory[-1] = step^(N-1)(x0) but final_only = step^N(x0)
        # To get consistent results, simulate one more step from trajectory[-1]
        full_traj = simulate_orbit(model, x0, n_steps=100, return_all=True)
        final_only = simulate_orbit(model, x0, n_steps=100, return_all=False)

        # Verify both produce finite results
        assert np.all(np.isfinite(full_traj[-1]))
        assert np.all(np.isfinite(final_only))

        # Verify final_only is one step ahead of full_traj[-1]
        expected_final = model.step(full_traj[-1])
        np.testing.assert_array_almost_equal(expected_final, final_only)

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

        fixed_points = find_fixed_points(model)

        assert isinstance(fixed_points, list)

    def test_find_fixed_points_simple_case(self):
        """Test finding fixed point in simple system"""
        # Create model where c is a fixed point
        A = np.zeros((2, 2))
        B = np.zeros((2, 2))
        W = np.eye(2)
        c = np.array([0.5, 0.5])

        model = FractalDynamicsModel(A=A, B=B, W=W, c=c)

        fixed_points = find_fixed_points(model, tolerance=1e-6)

        # Should find at least one fixed point near c
        assert len(fixed_points) > 0

        # Check that found points are actually fixed
        for fp, is_stable in fixed_points:
            x_next = model.step(fp)
            assert np.allclose(fp, x_next, atol=1e-5)

    def test_find_fixed_points_stability_flag(self):
        """Test that stability flag is boolean"""
        model = FractalDynamicsModel()

        fixed_points = find_fixed_points(model)

        for fp, is_stable in fixed_points:
            assert isinstance(is_stable, (bool, np.bool_))


class TestComputeAttractorType:
    """Test suite for compute_attractor_type function"""

    def test_compute_attractor_type_fixed_point(self):
        """Test classification of fixed point attractor"""
        # Create model with simple dynamics (converges to fixed point)
        A = np.zeros((2, 2))
        B = np.zeros((2, 2))
        W = np.eye(2)
        c = np.array([0.5, 0.5])

        model = FractalDynamicsModel(A=A, B=B, W=W, c=c)
        x0 = np.array([0.5, 0.5])

        # n_steps must be > transient (default 1000), use 2000
        attractor_type = compute_attractor_type(
            model, x0, n_steps=2000, transient=500, tolerance=1e-3
        )

        assert attractor_type == "fixed_point"

    def test_compute_attractor_type_unbounded(self):
        """Test classification of unbounded trajectory"""
        # Create model with strongly diverging dynamics
        # A = 2*I means each step doubles the state, so after ~7 steps
        # the norm exceeds 100 (0.7 * 2^7 = 89.6, 0.7 * 2^8 = 179)
        A = np.eye(2) * 2.0  # Amplifying: doubles state each step
        B = np.zeros((2, 2))
        W = np.eye(2)
        c = np.zeros(2)

        model = FractalDynamicsModel(A=A, B=B, W=W, c=c)
        x0 = np.array([1.0, 1.0])  # Start with larger values

        # Use short transient to ensure we catch the divergence
        attractor_type = compute_attractor_type(
            model, x0, n_steps=200, transient=10, tolerance=1e-3
        )

        assert attractor_type == "unbounded"

    def test_compute_attractor_type_returns_string(self):
        """Test that function returns a string"""
        model = FractalDynamicsModel()
        x0 = np.array([0.5, 0.5])

        # n_steps must be > transient (default 1000), use 2000
        attractor_type = compute_attractor_type(model, x0, n_steps=2000, transient=500)

        assert isinstance(attractor_type, str)
        assert attractor_type in ["fixed_point", "limit_cycle", "chaotic", "unbounded"]


class TestBasinOfAttraction:
    """Test suite for basin_of_attraction_sample function"""

    def test_basin_of_attraction_shape(self):
        """Test that basin map has correct shape"""
        model = FractalDynamicsModel()
        resolution = 20

        X, Y, basin_labels = basin_of_attraction_sample(
            model, x_range=(-1, 1), y_range=(-1, 1), resolution=resolution
        )

        assert X.shape == (resolution, resolution)
        assert Y.shape == (resolution, resolution)
        assert basin_labels.shape == (resolution, resolution)

    def test_basin_of_attraction_finite(self):
        """Test that basin map contains finite values"""
        model = FractalDynamicsModel()

        X, Y, basin_labels = basin_of_attraction_sample(
            model, x_range=(-1, 1), y_range=(-1, 1), resolution=15
        )

        assert np.all(np.isfinite(X))
        assert np.all(np.isfinite(Y))
        assert np.all(np.isfinite(basin_labels))

    def test_basin_of_attraction_labels_integers(self):
        """Test that basin labels are integers"""
        model = FractalDynamicsModel()
        resolution = 15

        X, Y, basin_labels = basin_of_attraction_sample(
            model, x_range=(-1, 1), y_range=(-1, 1), resolution=resolution
        )

        # Labels should be non-negative integers
        assert basin_labels.dtype in [np.int32, np.int64, int]
        assert np.all(basin_labels >= 0)


class TestPoincareSection:
    """Test suite for poincare_section function"""

    def test_poincare_section_returns_array(self):
        """Test that poincare_section returns an array"""
        model = FractalDynamicsModel()
        x0 = np.array([0.5, 0.5])

        crossings = poincare_section(
            model, x0, n_steps=1000, plane_coord=0, plane_value=0.5
        )

        assert isinstance(crossings, np.ndarray)

    def test_poincare_section_constant_trajectory(self):
        """Test Poincare section for trajectory that doesn't cross plane"""
        # Model that converges to fixed point far from plane
        A = np.zeros((2, 2))
        B = np.zeros((2, 2))
        W = np.eye(2)
        c = np.array([2.0, 2.0])  # Fixed point at (2, 2)

        model = FractalDynamicsModel(A=A, B=B, W=W, c=c)
        x0 = np.array([1.9, 1.9])  # Start near fixed point

        # Section at x=0 (trajectory stays near 2.0)
        crossings = poincare_section(
            model, x0, n_steps=500, plane_coord=0, plane_value=0.0
        )

        # Should find no crossings
        assert len(crossings) == 0

    def test_poincare_section_different_planes(self):
        """Test Poincare section on different plane coordinates"""
        model = FractalDynamicsModel()
        x0 = np.array([0.5, 0.5])

        # Test both plane coordinates
        crossings_0 = poincare_section(
            model, x0, n_steps=1000, plane_coord=0, plane_value=0.5
        )
        crossings_1 = poincare_section(
            model, x0, n_steps=1000, plane_coord=1, plane_value=0.5
        )

        assert isinstance(crossings_0, np.ndarray)
        assert isinstance(crossings_1, np.ndarray)


class TestIntegration:
    """Integration tests combining multiple functions"""

    def test_full_workflow(self):
        """Test complete simulation and analysis workflow"""
        # Create model
        model = FractalDynamicsModel()

        # Simulate orbit
        x0 = np.array([0.5, 0.5])
        trajectory = simulate_orbit(model, x0, n_steps=500)

        # Classify attractor (n_steps must be > transient)
        attractor_type = compute_attractor_type(model, x0, n_steps=2000, transient=500)

        # All steps should complete without error
        assert trajectory.shape == (500, 2)
        assert attractor_type in ["fixed_point", "limit_cycle", "chaotic", "unbounded"]

    def test_multiple_initial_conditions(self):
        """Test simulations from multiple initial conditions"""
        model = FractalDynamicsModel()

        initial_conditions = [
            np.array([0.1, 0.1]),
            np.array([0.5, 0.5]),
            np.array([0.9, 0.9]),
            np.array([-0.5, 0.5]),
        ]

        for x0 in initial_conditions:
            trajectory = simulate_orbit(model, x0, n_steps=200)
            assert trajectory.shape == (200, 2)
            assert np.all(np.isfinite(trajectory))

            # n_steps must be > transient for compute_attractor_type
            attractor_type = compute_attractor_type(
                model, x0, n_steps=1500, transient=500
            )
            assert isinstance(attractor_type, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
