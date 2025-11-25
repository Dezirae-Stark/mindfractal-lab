"""
Test Suite for Tenth Dimension Possibility Module
"""

import pytest
import numpy as np
from ..possibility_manifold import (
    PossibilityManifold, ParameterPoint, UpdateRuleFamily, StabilityRegion
)
from ..possibility_metrics import ManifoldMetrics, StabilityClassifier
from ..possibility_slicer import TimelineSlicer
from ..possibility_viewer import PossibilityVisualizer


class TestPossibilityManifold:
    def test_manifold_creation(self):
        manifold = PossibilityManifold(dim=2)
        assert manifold.dim == 2
        assert manifold.bounds == (-2.0, 2.0)

    def test_sample_point(self):
        manifold = PossibilityManifold(dim=2)
        point = manifold.sample_point(rule_family=UpdateRuleFamily.TANH_2D)
        assert point.dimension == 2
        assert point.rule_family == UpdateRuleFamily.TANH_2D
        assert point.A is not None
        assert point.B is not None
        assert point.W is not None

    def test_compute_orbit(self):
        manifold = PossibilityManifold(dim=2)
        point = manifold.sample_point()
        orbit = manifold.compute_orbit(point, steps=100)
        assert orbit.shape == (100, 2)

    def test_classify_stability(self):
        manifold = PossibilityManifold(dim=2)
        point = manifold.sample_point()
        orbit = manifold.compute_orbit(point, steps=100)
        region = manifold.classify_stability(orbit)
        assert isinstance(region, StabilityRegion)

    def test_distance(self):
        manifold = PossibilityManifold(dim=2)
        p1 = manifold.sample_point()
        p2 = manifold.sample_point()
        d = manifold.distance(p1, p2)
        assert d >= 0


class TestManifoldMetrics:
    def test_metrics_creation(self):
        manifold = PossibilityManifold(dim=2)
        metrics = ManifoldMetrics(manifold)
        assert metrics.manifold == manifold

    def test_lyapunov_exponent(self):
        manifold = PossibilityManifold(dim=2)
        metrics = ManifoldMetrics(manifold)
        point = manifold.sample_point()
        orbit = manifold.compute_orbit(point, steps=200)
        lyap = metrics.lyapunov_exponent(orbit)
        assert isinstance(lyap, (int, float))


class TestTimelineSlicer:
    def test_slicer_creation(self):
        manifold = PossibilityManifold(dim=2)
        slicer = TimelineSlicer(manifold)
        assert slicer.manifold == manifold

    def test_slice_parameter_line(self):
        manifold = PossibilityManifold(dim=2)
        slicer = TimelineSlicer(manifold)
        start = manifold.sample_point()
        end = manifold.sample_point()
        branch = slicer.slice_parameter_line(start, end, n_steps=10)
        assert len(branch.points) == 10
        assert len(branch.orbits) == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
