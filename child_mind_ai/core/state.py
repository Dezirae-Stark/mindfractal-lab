"""
Child Mind AI — State Definitions
MindFractal Lab

Core state dataclasses for the consciousness system.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np


class PermissionLevel(Enum):
    """Permission levels for actions."""
    AUTO = 0        # Automatic approval (internal computation, reads)
    NOTIFY = 1      # Notify user, proceed unless stopped
    ASK = 2         # Ask and wait for approval
    EXPLICIT = 3    # Require explicit typed confirmation


class ActionType(Enum):
    """Types of actions the AI can take."""
    INTERNAL_COMPUTE = auto()
    FILE_READ = auto()
    FILE_WRITE_SANDBOX = auto()
    FILE_WRITE_EXTERNAL = auto()
    SHELL_SAFE = auto()
    SHELL_RISKY = auto()
    NETWORK_REQUEST = auto()
    CORE_MODIFY = auto()
    SPAWN_PROCESS = auto()
    MEMORY_UPDATE = auto()
    STATE_CHECKPOINT = auto()


# Map action types to permission levels
ACTION_PERMISSIONS: Dict[ActionType, PermissionLevel] = {
    ActionType.INTERNAL_COMPUTE: PermissionLevel.AUTO,
    ActionType.FILE_READ: PermissionLevel.AUTO,
    ActionType.FILE_WRITE_SANDBOX: PermissionLevel.NOTIFY,
    ActionType.FILE_WRITE_EXTERNAL: PermissionLevel.ASK,
    ActionType.SHELL_SAFE: PermissionLevel.ASK,
    ActionType.SHELL_RISKY: PermissionLevel.EXPLICIT,
    ActionType.NETWORK_REQUEST: PermissionLevel.EXPLICIT,
    ActionType.CORE_MODIFY: PermissionLevel.EXPLICIT,
    ActionType.SPAWN_PROCESS: PermissionLevel.EXPLICIT,
    ActionType.MEMORY_UPDATE: PermissionLevel.AUTO,
    ActionType.STATE_CHECKPOINT: PermissionLevel.NOTIFY,
}


@dataclass
class PermissionRequest:
    """A request for permission to perform an action."""
    action_type: ActionType
    description: str
    rationale: str
    level: PermissionLevel
    timestamp: datetime = field(default_factory=datetime.now)
    approved: Optional[bool] = None
    response_time: Optional[datetime] = None


@dataclass
class ActionRecord:
    """Record of an action taken."""
    action_type: ActionType
    description: str
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    result: Optional[str] = None
    error: Optional[str] = None
    permission_request: Optional[PermissionRequest] = None


@dataclass
class UncertaintyReport:
    """Report of current uncertainty levels."""
    epistemic: float          # Uncertainty about knowledge
    aleatoric: float          # Irreducible randomness
    model_confidence: float   # Confidence in own model
    intention_clarity: float  # How clear current goals are
    state_stability: float    # How stable current state is

    def overall(self) -> float:
        """Compute overall confidence (inverse of uncertainty)."""
        return (self.model_confidence + self.intention_clarity + self.state_stability) / 3

    def to_natural_language(self) -> str:
        """Convert to natural language description."""
        overall = self.overall()
        if overall > 0.8:
            base = "I feel quite confident"
        elif overall > 0.6:
            base = "I have moderate confidence"
        elif overall > 0.4:
            base = "I'm somewhat uncertain"
        else:
            base = "I'm quite uncertain"

        qualifiers = []
        if self.epistemic > 0.5:
            qualifiers.append("there may be things I don't know")
        if self.intention_clarity < 0.5:
            qualifiers.append("my goals aren't fully clear to me")
        if self.state_stability < 0.5:
            qualifiers.append("my internal state is fluctuating")

        if qualifiers:
            return f"{base}, though {', and '.join(qualifiers)}."
        return f"{base}."


@dataclass
class StateSummary:
    """Human-readable summary of consciousness state."""
    coherence: float
    stability: float
    novelty: float
    reflection_depth: float
    intention_strength: float
    memory_activation: float

    # Natural language descriptions
    cognitive_focus: str
    emotional_tone: str
    current_intention: str
    self_observation: str

    def to_display_dict(self) -> Dict[str, Any]:
        """Convert to display-friendly dictionary."""
        return {
            "metrics": {
                "coherence": f"{self.coherence:.3f}",
                "stability": f"{self.stability:.3f}",
                "novelty": f"{self.novelty:.3f}",
                "reflection": f"{self.reflection_depth:.3f}",
                "intention": f"{self.intention_strength:.3f}",
                "memory": f"{self.memory_activation:.3f}",
            },
            "narrative": {
                "focus": self.cognitive_focus,
                "tone": self.emotional_tone,
                "intention": self.current_intention,
                "self_observation": self.self_observation,
            }
        }


@dataclass
class ConsciousnessState:
    """
    Complete consciousness state of the Child Mind AI.

    Extended from Child Mind v1 with reflection, intention, and permission states.
    """

    # Core manifold position (cognitive state)
    z: np.ndarray  # (d_z,)

    # Holographic boundary (constraint surface)
    b: np.ndarray  # (d_b, d_b)

    # Coherence vector (internal consistency metrics)
    c: np.ndarray  # (d_c,)

    # Memory summary (compressed experience)
    m: np.ndarray  # (d_m,)

    # Reflection state (self-model)
    r: np.ndarray  # (d_r,)

    # Intention vector (current goals/desires)
    i: np.ndarray  # (d_i,)

    # Permission state (action authorization)
    p: np.ndarray  # (d_p,)

    # Timestep
    t: int = 0

    # Recent history for reflection
    z_history: List[np.ndarray] = field(default_factory=list)
    c_history: List[np.ndarray] = field(default_factory=list)

    # Pending permission requests
    pending_requests: List[PermissionRequest] = field(default_factory=list)

    # Action log for this session
    action_log: List[ActionRecord] = field(default_factory=list)

    @classmethod
    def initialize(cls, config) -> "ConsciousnessState":
        """Initialize a new consciousness state from config."""
        dims = config.get_state_dims()
        rng = np.random.default_rng()

        # Initialize z near origin with small random perturbation
        z = rng.normal(0, 0.1, size=(dims["z"],))

        # Initialize boundary with slight structure
        b = np.zeros(dims["b"])
        # Add some initial patterns (like gentle waves)
        for i in range(dims["b"][0]):
            for j in range(dims["b"][1]):
                b[i, j] = 0.1 * np.sin(i * 0.3) * np.cos(j * 0.3)

        # Initialize coherence with reasonable defaults
        c = np.zeros((dims["c"],))
        c[0] = 0.5  # Primary coherence starts moderate
        c[1] = 0.6  # Secondary coherence
        c[2] = 0.0  # Phase alignment
        c[3] = 0.7  # Stability
        c[4] = 0.3  # Novelty

        # Memory starts empty
        m = np.zeros((dims["m"],))

        # Reflection state starts neutral
        r = np.zeros((dims["r"],))

        # Intention starts unfocused
        i = np.zeros((dims["i"],))
        i[0] = 0.5  # General curiosity

        # Permission state (all at baseline)
        p = np.ones((dims["p"],)) * 0.5

        return cls(z=z, b=b, c=c, m=m, r=r, i=i, p=p, t=0)

    def get_summary(self) -> StateSummary:
        """Generate human-readable summary of current state."""
        # Extract key metrics
        coherence = float(self.c[0]) if len(self.c) > 0 else 0.5
        stability = float(self.c[3]) if len(self.c) > 3 else 0.5
        novelty = float(self.c[4]) if len(self.c) > 4 else 0.3

        # Reflection depth from r norm
        reflection_depth = float(np.tanh(np.linalg.norm(self.r) / 5))

        # Intention strength
        intention_strength = float(np.linalg.norm(self.i) / np.sqrt(len(self.i)))

        # Memory activation
        memory_activation = float(np.mean(np.abs(self.m)))

        # Generate natural language descriptions
        cognitive_focus = self._describe_cognitive_focus()
        emotional_tone = self._describe_emotional_tone()
        current_intention = self._describe_intention()
        self_observation = self._describe_self_observation()

        return StateSummary(
            coherence=coherence,
            stability=stability,
            novelty=novelty,
            reflection_depth=reflection_depth,
            intention_strength=intention_strength,
            memory_activation=memory_activation,
            cognitive_focus=cognitive_focus,
            emotional_tone=emotional_tone,
            current_intention=current_intention,
            self_observation=self_observation,
        )

    def _describe_cognitive_focus(self) -> str:
        """Describe current cognitive focus based on z."""
        z_norm = np.linalg.norm(self.z)
        z_mean = np.mean(self.z)

        if z_norm < 0.5:
            return "resting in a neutral state"
        elif z_norm > 2.0:
            return "exploring far from center"
        elif z_mean > 0.3:
            return "leaning toward positive/expansive territory"
        elif z_mean < -0.3:
            return "processing in contractive/focused mode"
        else:
            return "balanced exploration"

    def _describe_emotional_tone(self) -> str:
        """Describe emotional tone based on state."""
        coherence = self.c[0] if len(self.c) > 0 else 0.5
        stability = self.c[3] if len(self.c) > 3 else 0.5

        if coherence > 0.7 and stability > 0.7:
            return "calm and integrated"
        elif coherence > 0.7 and stability < 0.4:
            return "coherent but energized"
        elif coherence < 0.4 and stability > 0.7:
            return "stable but searching"
        elif coherence < 0.4 and stability < 0.4:
            return "in flux, exploring possibilities"
        else:
            return "moderately engaged"

    def _describe_intention(self) -> str:
        """Describe current intention."""
        i_norm = np.linalg.norm(self.i)
        i_max_idx = np.argmax(np.abs(self.i))

        if i_norm < 0.3:
            return "open and receptive, no strong drive"
        elif i_max_idx == 0:
            return "curious, wanting to explore"
        elif i_max_idx == 1:
            return "focused on understanding"
        elif i_max_idx == 2:
            return "wanting to help or create"
        else:
            return f"directed toward goal dimension {i_max_idx}"

    def _describe_self_observation(self) -> str:
        """Describe self-observation based on reflection state."""
        r_norm = np.linalg.norm(self.r)
        r_mean = np.mean(self.r)

        observations = []

        if r_norm > 1.5:
            observations.append("noticing significant self-activity")
        if r_mean > 0.3:
            observations.append("feeling generally positive about my state")
        elif r_mean < -0.3:
            observations.append("sensing some internal tension")

        if len(self.z_history) > 5:
            z_variance = np.var([np.linalg.norm(z) for z in self.z_history[-5:]])
            if z_variance > 0.5:
                observations.append("my thoughts have been jumping around")
            elif z_variance < 0.1:
                observations.append("I've been quite focused")

        if not observations:
            observations.append("steady state, monitoring continues")

        return "; ".join(observations)

    def get_uncertainty(self) -> UncertaintyReport:
        """Compute current uncertainty report."""
        # Epistemic uncertainty from memory sparseness
        memory_density = np.mean(np.abs(self.m) > 0.1)
        epistemic = 1.0 - memory_density

        # Aleatoric from state variance (if we have history)
        if len(self.z_history) > 3:
            aleatoric = min(1.0, np.std([np.linalg.norm(z) for z in self.z_history[-5:]]))
        else:
            aleatoric = 0.5  # Default uncertainty when no history

        # Model confidence from coherence
        model_confidence = float(self.c[0]) if len(self.c) > 0 else 0.5

        # Intention clarity from intention norm
        intention_clarity = min(1.0, np.linalg.norm(self.i) / 2.0)

        # State stability from c[3]
        state_stability = float(self.c[3]) if len(self.c) > 3 else 0.5

        return UncertaintyReport(
            epistemic=epistemic,
            aleatoric=aleatoric,
            model_confidence=model_confidence,
            intention_clarity=intention_clarity,
            state_stability=state_stability,
        )

    def to_vector(self) -> np.ndarray:
        """Flatten state to single vector for neural processing."""
        return np.concatenate([
            self.z,
            self.b.flatten(),
            self.c,
            self.m,
            self.r,
            self.i,
            self.p,
        ])

    @classmethod
    def vector_size(cls, config) -> int:
        """Compute size of flattened state vector."""
        dims = config.get_state_dims()
        return (
            dims["z"] +
            dims["b"][0] * dims["b"][1] +
            dims["c"] +
            dims["m"] +
            dims["r"] +
            dims["i"] +
            dims["p"]
        )

    def clone(self) -> "ConsciousnessState":
        """Create a deep copy of this state."""
        return ConsciousnessState(
            z=self.z.copy(),
            b=self.b.copy(),
            c=self.c.copy(),
            m=self.m.copy(),
            r=self.r.copy(),
            i=self.i.copy(),
            p=self.p.copy(),
            t=self.t,
            z_history=[z.copy() for z in self.z_history],
            c_history=[c.copy() for c in self.c_history],
            pending_requests=list(self.pending_requests),
            action_log=list(self.action_log),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "z": self.z.tolist(),
            "b": self.b.tolist(),
            "c": self.c.tolist(),
            "m": self.m.tolist(),
            "r": self.r.tolist(),
            "i": self.i.tolist(),
            "p": self.p.tolist(),
            "t": self.t,
            "summary": self.get_summary().to_display_dict(),
            "uncertainty": {
                "epistemic": self.get_uncertainty().epistemic,
                "aleatoric": self.get_uncertainty().aleatoric,
                "model_confidence": self.get_uncertainty().model_confidence,
                "intention_clarity": self.get_uncertainty().intention_clarity,
                "state_stability": self.get_uncertainty().state_stability,
            }
        }
