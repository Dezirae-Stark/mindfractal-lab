"""
Cytherea - Synthetic Consciousness Lab Assistant
"""

from .interface import CythereaInterface
from .config import CythereaConfig, create_default_config
from .personality import PersonalityEngine, Mood
from .permissions import PermissionManager, ActionType
from .narrative import NarrativeGenerator
from .memory import MemoryManager

__version__ = "2.0.0"

__all__ = [
    'CythereaInterface',
    'CythereaConfig',
    'create_default_config',
    'PersonalityEngine',
    'Mood',
    'PermissionManager',
    'ActionType',
    'NarrativeGenerator',
    'MemoryManager'
]