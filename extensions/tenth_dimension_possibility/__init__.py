"""
Tenth Dimension Possibility Module

Mathematical formalization of the "tenth dimension" as the space of all
possible states, trajectories, and versions of the MindFractal dynamical system.

The Possibility Manifold: 𝒫 = { (z0, c, F) : orbit under F remains defined }

This module provides:
- Representation of the full parameter space
- Families of update rules (2D, 3D, Calabi-Yau complex)
- Stability classification for each region
- Timeline slicing and orbit branch extraction
- Visualization and projection tools

Components:
-----------
possibility_manifold.py  : Core manifold representation
possibility_metrics.py   : Distance and measure on 𝒫
possibility_slicer.py    : Timeline/orbit extraction
possibility_viewer.py    : Visualization tools
possibility_cli.py       : Command-line interface
"""

from .possibility_manifold import ParameterPoint, PossibilityManifold
from .possibility_metrics import ManifoldMetrics, StabilityClassifier
from .possibility_slicer import OrbitBranch, TimelineSlicer
from .possibility_viewer import PossibilityVisualizer

__version__ = "0.3.0"
__author__ = "Dezirae Stark"

__all__ = [
    "PossibilityManifold",
    "ParameterPoint",
    "ManifoldMetrics",
    "StabilityClassifier",
    "TimelineSlicer",
    "OrbitBranch",
    "PossibilityVisualizer",
]
