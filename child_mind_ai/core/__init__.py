"""
Child Mind AI — Core Module
MindFractal Lab

Core consciousness engine and state management.
"""

from .state import (
    ConsciousnessState,
    PermissionLevel,
    ActionRecord,
    PermissionRequest,
    StateSummary,
    UncertaintyReport,
)
from .engine import ConsciousnessEngine
from .dynamics import (
    F_mind,
    G_boundary,
    H_coherence,
    U_memory,
    R_reflect,
    I_intent,
)

__all__ = [
    "ConsciousnessState",
    "PermissionLevel",
    "ActionRecord",
    "PermissionRequest",
    "StateSummary",
    "UncertaintyReport",
    "ConsciousnessEngine",
    "F_mind",
    "G_boundary",
    "H_coherence",
    "U_memory",
    "R_reflect",
    "I_intent",
]
