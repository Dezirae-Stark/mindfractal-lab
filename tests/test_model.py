"""
Unit tests for mindfractal.model
"""

import numpy as np
import pytest

from mindfractal.model import FractalDynamicsModel


class TestFractalDynamicsModel:
    """Test suite for FractalDynamicsModel class"""

    def test_initialization_default(self):
        """Test model initialization with default parameters"""
        model = FractalDynamicsModel()

        assert model.A.shape == (2, 2)
        assert model.B.shape == (2, 2)
        assert model.W.shape == (2, 2)
        assert model.c.shape == (2,)

        # Check default A is diagonal
        assert model.A[0, 0] == pytest.approx(0.9)
        assert model.A[1, 1] == pytest.approx(0.9)

    def test_initialization_custom(self):
        """Test model initialization with custom parameters"""
        A = np.array([[0.8, 0.1], [0.1, 0.8]])
        B = np.array([[0.3, 0.2], [0.2, 0.3]])
        W = np.eye(2)
        c = np.array([0.2, 0.3])

        model = FractalDynamicsModel(A=A, B=B, W=W, c=c)

        np.testing.assert_array_equal(model.A, A)
        np.testing.assert_array_equal(model.B, B)
        np.testing.assert_array_equal(model.W, W)
        np.testing.assert_array_equal(model.c, c)

    def test_step_shape(self):
        """Test that step() returns correct shape"""
        model = FractalDynamicsModel()
        x = np.array([0.5, 0.5])
        x_next = model.step(x)

        assert x_next.shape == (2,)
        assert isinstance(x_next, np.ndarray)

    def test_step_deterministic(self):
        """Test that step() is deterministic"""
        model = FractalDynamicsModel()
        x = np.array([0.5, 0.5])

        x_next1 = model.step(x)
        x_next2 = model.step(x)

        np.testing.assert_array_equal(x_next1, x_next2)

    def test_step_fixed_point(self):
        """Test that fixed point satisfies x = f(x)"""
        # Create model with simple dynamics
        A = np.zeros((2, 2))
        B = np.zeros((2, 2))
        W = np.eye(2)
        c = np.array([0.5, 0.5])

        model = FractalDynamicsModel(A=A, B=B, W=W, c=c)

        # c should be a fixed point
        x_next = model.step(c)
        np.testing.assert_array_almost_equal(x_next, c, decimal=6)

    def test_jacobian_shape(self):
        """Test that jacobian() returns correct shape"""
        model = FractalDynamicsModel()
        x = np.array([0.5, 0.5])
        J = model.jacobian(x)

        assert J.shape == (2, 2)
        assert isinstance(J, np.ndarray)

    def test_jacobian_at_origin(self):
        """Test Jacobian at origin for simple case"""
        A = np.eye(2) * 0.9
        B = np.array([[0.2, 0.3], [0.3, 0.2]])
        W = np.eye(2)
        c = np.zeros(2)

        model = FractalDynamicsModel(A=A, B=B, W=W, c=c)
        x = np.zeros(2)
        J = model.jacobian(x)

        # At origin, sech²(0) = 1, so J = A + B * W = A + B
        J_expected = A + B
        np.testing.assert_array_almost_equal(J, J_expected, decimal=6)

    def test_lyapunov_exponent_stable(self):
        """Test Lyapunov exponent for stable system"""
        # Create strongly damped system
        A = np.eye(2) * 0.5
        B = np.zeros((2, 2))
        W = np.eye(2)
        c = np.zeros(2)

        model = FractalDynamicsModel(A=A, B=B, W=W, c=c)
        x0 = np.array([0.5, 0.5])

        lyap = model.lyapunov_exponent_estimate(x0, n_steps=1000, transient=100)

        # Should be negative for stable system
        assert lyap < 0

    def test_lyapunov_exponent_deterministic(self):
        """Test that Lyapunov exponent is deterministic"""
        model = FractalDynamicsModel()
        x0 = np.array([0.5, 0.5])

        lyap1 = model.lyapunov_exponent_estimate(x0, n_steps=500, transient=100)
        lyap2 = model.lyapunov_exponent_estimate(x0, n_steps=500, transient=100)

        assert lyap1 == pytest.approx(lyap2, rel=1e-6)

    def test_energy_shape(self):
        """Test that energy() returns scalar"""
        model = FractalDynamicsModel()
        x = np.array([0.5, 0.5])
        E = model.energy(x)

        assert isinstance(E, (float, np.floating))

    def test_energy_finite(self):
        """Test that energy is finite"""
        model = FractalDynamicsModel()
        x = np.array([0.5, 0.5])
        E = model.energy(x)

        assert np.isfinite(E)

    def test_energy_at_origin(self):
        """Test energy at origin"""
        model = FractalDynamicsModel()
        x = np.zeros(2)
        E = model.energy(x)

        # Energy should be small (but not exactly zero due to c term)
        assert E >= 0
        assert E < 1.0

    def test_step_finite_values(self):
        """Test that step() produces finite values"""
        model = FractalDynamicsModel()
        x = np.array([1.0, 1.0])

        for _ in range(100):
            x = model.step(x)
            assert np.all(np.isfinite(x))

    def test_different_initial_conditions(self):
        """Test that different ICs lead to different trajectories"""
        model = FractalDynamicsModel()

        x1 = np.array([0.1, 0.1])
        x2 = np.array([0.9, 0.9])

        # Evolve both
        for _ in range(10):
            x1 = model.step(x1)
            x2 = model.step(x2)

        # Should be different (unless both converged to same attractor)
        # For default params, trajectories should differ
        assert not np.allclose(x1, x2, rtol=1e-3)


class TestModelEdgeCases:
    """Test edge cases and error handling"""

    def test_step_large_values(self):
        """Test step with large input values"""
        model = FractalDynamicsModel()
        x = np.array([100.0, 100.0])
        x_next = model.step(x)

        # tanh should saturate, preventing explosion
        assert np.all(np.abs(x_next) < 200)

    def test_step_negative_values(self):
        """Test step with negative values"""
        model = FractalDynamicsModel()
        x = np.array([-1.0, -1.0])
        x_next = model.step(x)

        assert np.all(np.isfinite(x_next))

    def test_jacobian_large_values(self):
        """Test Jacobian at large state values"""
        model = FractalDynamicsModel()
        x = np.array([10.0, 10.0])
        J = model.jacobian(x)

        assert np.all(np.isfinite(J))
        # sech² should be near zero for large x
        assert np.all(np.abs(J - model.A) < 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
