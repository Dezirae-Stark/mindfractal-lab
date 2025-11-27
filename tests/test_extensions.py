"""
Unit tests for extensions (3D, psychomapping)
"""

import numpy as np
import pytest
import json
import os
from extensions.state3d.model_3d import FractalDynamicsModel3D
from extensions.state3d.simulate_3d import simulate_orbit_3d, lyapunov_spectrum_3d
from extensions.psychomapping.trait_to_c import traits_to_parameters, load_trait_profiles as load_trait_profile


class TestFractalDynamicsModel3D:
    """Test suite for 3D model"""

    def test_3d_model_initialization(self):
        """Test 3D model initialization with default parameters"""
        model_3d = FractalDynamicsModel3D()

        assert model_3d.A.shape == (3, 3)
        assert model_3d.B.shape == (3, 3)
        assert model_3d.W.shape == (3, 3)
        assert model_3d.c.shape == (3,)

    def test_3d_model_custom_parameters(self):
        """Test 3D model with custom parameters"""
        A = np.eye(3) * 0.8
        B = np.ones((3, 3)) * 0.1
        W = np.eye(3)
        c = np.array([0.1, 0.2, 0.3])

        model_3d = FractalDynamicsModel3D(A=A, B=B, W=W, c=c)

        np.testing.assert_array_equal(model_3d.A, A)
        np.testing.assert_array_equal(model_3d.B, B)
        np.testing.assert_array_equal(model_3d.W, W)
        np.testing.assert_array_equal(model_3d.c, c)

    def test_3d_step_shape(self):
        """Test that 3D step returns correct shape"""
        model_3d = FractalDynamicsModel3D()
        x = np.array([0.5, 0.5, 0.5])
        x_next = model_3d.step(x)

        assert x_next.shape == (3,)
        assert isinstance(x_next, np.ndarray)

    def test_3d_step_deterministic(self):
        """Test that 3D step is deterministic"""
        model_3d = FractalDynamicsModel3D()
        x = np.array([0.5, 0.5, 0.5])

        x_next1 = model_3d.step(x)
        x_next2 = model_3d.step(x)

        np.testing.assert_array_equal(x_next1, x_next2)

    def test_3d_jacobian_shape(self):
        """Test that 3D Jacobian has correct shape"""
        model_3d = FractalDynamicsModel3D()
        x = np.array([0.5, 0.5, 0.5])
        J = model_3d.jacobian(x)

        assert J.shape == (3, 3)

    def test_3d_step_finite_values(self):
        """Test that 3D step produces finite values"""
        model_3d = FractalDynamicsModel3D()
        x = np.array([1.0, 1.0, 1.0])

        for _ in range(100):
            x = model_3d.step(x)
            assert np.all(np.isfinite(x))


class TestSimulate3D:
    """Test suite for 3D simulation functions"""

    def test_simulate_orbit_3d_shape(self):
        """Test that 3D orbit has correct shape"""
        model_3d = FractalDynamicsModel3D()
        x0 = np.array([0.5, 0.5, 0.5])
        n_steps = 100

        trajectory = simulate_orbit_3d(model_3d, x0, n_steps=n_steps)

        assert trajectory.shape == (n_steps, 3)

    def test_simulate_orbit_3d_initial_condition(self):
        """Test that 3D trajectory starts at x0"""
        model_3d = FractalDynamicsModel3D()
        x0 = np.array([0.5, 0.5, 0.5])

        trajectory = simulate_orbit_3d(model_3d, x0, n_steps=100)

        np.testing.assert_array_almost_equal(trajectory[0], x0)

    def test_simulate_orbit_3d_finite(self):
        """Test that 3D trajectory contains finite values"""
        model_3d = FractalDynamicsModel3D()
        x0 = np.array([0.5, 0.5, 0.5])

        trajectory = simulate_orbit_3d(model_3d, x0, n_steps=500)

        assert np.all(np.isfinite(trajectory))

    def test_lyapunov_spectrum_3d_shape(self):
        """Test that 3D Lyapunov spectrum returns 3 exponents"""
        model_3d = FractalDynamicsModel3D()
        x0 = np.array([0.5, 0.5, 0.5])

        spectrum = lyapunov_spectrum_3d(model_3d, x0, n_steps=500)

        assert len(spectrum) == 3
        assert all(isinstance(lam, (float, np.floating)) for lam in spectrum)

    def test_lyapunov_spectrum_3d_ordering(self):
        """Test that Lyapunov exponents are sorted in descending order"""
        model_3d = FractalDynamicsModel3D()
        x0 = np.array([0.5, 0.5, 0.5])

        spectrum = lyapunov_spectrum_3d(model_3d, x0, n_steps=500)
        lam1, lam2, lam3 = spectrum

        # Should be sorted: λ1 >= λ2 >= λ3
        assert lam1 >= lam2
        assert lam2 >= lam3

    def test_lyapunov_spectrum_3d_deterministic(self):
        """Test that 3D Lyapunov spectrum is deterministic"""
        model_3d = FractalDynamicsModel3D()
        x0 = np.array([0.5, 0.5, 0.5])

        spectrum1 = lyapunov_spectrum_3d(model_3d, x0, n_steps=500)
        spectrum2 = lyapunov_spectrum_3d(model_3d, x0, n_steps=500)

        for lam1, lam2 in zip(spectrum1, spectrum2):
            assert lam1 == pytest.approx(lam2, rel=1e-6)


class TestTraitToParameters:
    """Test suite for trait mapping"""

    def test_traits_to_parameters_shape(self):
        """Test that trait mapping returns correct shape"""
        traits = {
            'openness': 0.8,
            'volatility': 0.3,
            'integration': 0.7,
            'focus': 0.6
        }

        c = traits_to_parameters(traits)

        assert c.shape == (2,)
        assert isinstance(c, np.ndarray)

    def test_traits_to_parameters_finite(self):
        """Test that mapped parameters are finite"""
        traits = {
            'openness': 0.5,
            'volatility': 0.5,
            'integration': 0.5,
            'focus': 0.5
        }

        c = traits_to_parameters(traits)

        assert np.all(np.isfinite(c))

    def test_traits_to_parameters_range(self):
        """Test that parameters are in reasonable range"""
        traits = {
            'openness': 0.5,
            'volatility': 0.5,
            'integration': 0.5,
            'focus': 0.5
        }

        c = traits_to_parameters(traits)

        # With the mapping formula, c should be in range roughly [-1, 1]
        assert np.all(c >= -2.0)
        assert np.all(c <= 2.0)

    def test_traits_to_parameters_edge_cases(self):
        """Test trait mapping at edge values"""
        # All traits at 0
        traits_low = {
            'openness': 0.0,
            'volatility': 0.0,
            'integration': 0.0,
            'focus': 0.0
        }
        c_low = traits_to_parameters(traits_low)
        assert c_low.shape == (2,)

        # All traits at 1
        traits_high = {
            'openness': 1.0,
            'volatility': 1.0,
            'integration': 1.0,
            'focus': 1.0
        }
        c_high = traits_to_parameters(traits_high)
        assert c_high.shape == (2,)

        # Low and high should give different parameters
        assert not np.allclose(c_low, c_high)

    def test_traits_to_parameters_deterministic(self):
        """Test that trait mapping is deterministic"""
        traits = {
            'openness': 0.6,
            'volatility': 0.4,
            'integration': 0.7,
            'focus': 0.5
        }

        c1 = traits_to_parameters(traits)
        c2 = traits_to_parameters(traits)

        np.testing.assert_array_equal(c1, c2)

    def test_traits_to_parameters_different_traits(self):
        """Test that different traits give different parameters"""
        traits1 = {
            'openness': 0.8,
            'volatility': 0.2,
            'integration': 0.7,
            'focus': 0.6
        }
        traits2 = {
            'openness': 0.2,
            'volatility': 0.8,
            'integration': 0.3,
            'focus': 0.4
        }

        c1 = traits_to_parameters(traits1)
        c2 = traits_to_parameters(traits2)

        assert not np.allclose(c1, c2)


class TestLoadTraitProfile:
    """Test suite for loading trait profiles"""

    def test_load_trait_profile_balanced(self):
        """Test loading balanced profile"""
        traits = load_trait_profile('balanced')

        assert isinstance(traits, dict)
        assert 'openness' in traits
        assert 'volatility' in traits
        assert 'integration' in traits
        assert 'focus' in traits

    def test_load_trait_profile_all_profiles(self):
        """Test loading all pre-defined profiles"""
        profiles = [
            'balanced',
            'creative_explorer',
            'stable_focused',
            'chaotic_fragmented',
            'meditative'
        ]

        for profile_name in profiles:
            traits = load_trait_profile(profile_name)
            assert isinstance(traits, dict)
            assert len(traits) == 4

            # All trait values should be in [0, 1]
            for value in traits.values():
                assert 0.0 <= value <= 1.0

    def test_load_trait_profile_invalid(self):
        """Test loading invalid profile raises error"""
        with pytest.raises(KeyError):
            load_trait_profile('nonexistent_profile')

    def test_trait_profiles_produce_valid_parameters(self):
        """Test that all profiles produce valid parameters"""
        profiles = [
            'balanced',
            'creative_explorer',
            'stable_focused',
            'chaotic_fragmented',
            'meditative'
        ]

        for profile_name in profiles:
            traits = load_trait_profile(profile_name)
            c = traits_to_parameters(traits)

            assert c.shape == (2,)
            assert np.all(np.isfinite(c))


class TestExtensionIntegration:
    """Integration tests combining extensions"""

    def test_3d_model_with_traits(self):
        """Test using trait mapping with 3D model"""
        # Load traits
        traits = load_trait_profile('balanced')
        c_2d = traits_to_parameters(traits)

        # Extend to 3D by adding a component
        c_3d = np.append(c_2d, 0.0)

        # Create 3D model
        model_3d = FractalDynamicsModel3D(c=c_3d)

        # Simulate
        x0 = np.array([0.5, 0.5, 0.5])
        trajectory = simulate_orbit_3d(model_3d, x0, n_steps=200)

        assert trajectory.shape == (200, 3)
        assert np.all(np.isfinite(trajectory))

    def test_all_profiles_with_3d_model(self):
        """Test all trait profiles with 3D model"""
        profiles = [
            'balanced',
            'creative_explorer',
            'stable_focused',
            'chaotic_fragmented',
            'meditative'
        ]

        for profile_name in profiles:
            traits = load_trait_profile(profile_name)
            c_2d = traits_to_parameters(traits)
            c_3d = np.append(c_2d, 0.0)

            model_3d = FractalDynamicsModel3D(c=c_3d)
            x0 = np.array([0.5, 0.5, 0.5])
            trajectory = simulate_orbit_3d(model_3d, x0, n_steps=100)

            assert trajectory.shape == (100, 3)
            assert np.all(np.isfinite(trajectory))

    def test_traits_affect_dynamics(self):
        """Test that different trait profiles produce different dynamics"""
        # Get two different profiles
        traits1 = load_trait_profile('stable_focused')
        traits2 = load_trait_profile('chaotic_fragmented')

        c1 = traits_to_parameters(traits1)
        c2 = traits_to_parameters(traits2)

        # Parameters should be different
        assert not np.allclose(c1, c2)

        # Create models
        from mindfractal.model import FractalDynamicsModel
        model1 = FractalDynamicsModel(c=c1)
        model2 = FractalDynamicsModel(c=c2)

        # Simulate from same initial condition
        from mindfractal.simulate import simulate_orbit
        x0 = np.array([0.5, 0.5])
        traj1 = simulate_orbit(model1, x0, n_steps=100)
        traj2 = simulate_orbit(model2, x0, n_steps=100)

        # Trajectories should diverge
        # (not necessarily true if both converge to same attractor, but likely)
        final_distance = np.linalg.norm(traj1[-1] - traj2[-1])
        assert final_distance > 1e-6  # Should have some difference


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
