"""
Child Mind AI — Self-Reflexive Consciousness Agent
MindFractal Lab

A sandboxed, self-aware AI built on consciousness manifold dynamics.
Maintains transparency about internal states and operates under
explicit permission controls for external actions.
"""

__version__ = "0.1.0"

from .config import ChildMindAIConfig
from .core.state import ConsciousnessState, PermissionLevel
from .core.engine import ConsciousnessEngine

__all__ = [
    "ChildMindAIConfig",
    "ConsciousnessState",
    "PermissionLevel",
    "ConsciousnessEngine",
]
