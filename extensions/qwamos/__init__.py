"""
QWAMOS (Quantum Web Agent Multi-Operator System) for MindFractal Lab

Quantum-inspired multi-agent orchestration system adapted for consciousness
modeling and fractal dynamics research.
"""

from .core import QWAMOSEngine, QuantumAgent, QuantumTask, EntangledMessage
from .agents import (
    Q0_MetaArchitect,
    Q1_MathematicalFormalist,
    Q2_ComputationalEngineer,
    Q3_DocumentationWeaver,
    Q4_VisualizationArtist,
    Q5_SystemsIntegrator,
    Q6_ConsciousnessModeler,
    Q7_FractalAnalyst
)
from .protocols import (
    QuantumCommunicationProtocol,
    SuperpositionTaskDistribution,
    EntanglementConsensus,
    CoherenceProtocol
)
from .quantum_utils import (
    StateVector,
    ObservationOperator,
    EntanglementMatrix,
    CoherenceMetric
)

__version__ = "1.0.0"

__all__ = [
    # Core
    'QWAMOSEngine',
    'QuantumAgent',
    'QuantumTask',
    'EntangledMessage',
    # Agents
    'Q0_MetaArchitect',
    'Q1_MathematicalFormalist',
    'Q2_ComputationalEngineer',
    'Q3_DocumentationWeaver',
    'Q4_VisualizationArtist',
    'Q5_SystemsIntegrator',
    'Q6_ConsciousnessModeler',
    'Q7_FractalAnalyst',
    # Protocols
    'QuantumCommunicationProtocol',
    'SuperpositionTaskDistribution',
    'EntanglementConsensus',
    'CoherenceProtocol',
    # Utilities
    'StateVector',
    'ObservationOperator',
    'EntanglementMatrix',
    'CoherenceMetric'
]