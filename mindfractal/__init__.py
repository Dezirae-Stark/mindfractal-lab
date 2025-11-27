"""
MindFractal Lab - Fractal Dynamical Consciousness Model

A scientific package for simulating and analyzing 2D and 3D fractal dynamical systems
modeling consciousness states, metastability, and trait-to-parameter mappings.

Author: MindFractal Lab Contributors
License: MIT
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "MindFractal Lab Contributors"
__license__ = "MIT"

from .fractal_map import generate_fractal_map
from .model import FractalDynamicsModel
from .simulate import find_fixed_points, simulate_orbit
from .visualize import plot_fractal_map, plot_orbit

__all__ = [
    "FractalDynamicsModel",
    "simulate_orbit",
    "find_fixed_points",
    "plot_orbit",
    "plot_fractal_map",
    "generate_fractal_map",
]
