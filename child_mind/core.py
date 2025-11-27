"""
Child Mind v1 — Core State and Step Functions
MindFractal Lab

Defines the core State and Action dataclasses, configuration,
and the main step() function that advances the child mind simulation.

State space:
    s_t = (z_t, b_t, c_t, m_t)

    z_t ∈ R^d           - core manifold state (d=16)
    b_t ∈ R^{H_b×W_b}   - boundary holographic grid (16×16)
    c_t ∈ R^{k_c}       - coherence state (k_c=5)
    m_t ∈ R^{d_m}       - memory summary (d_m=16)

Action space:
    a_t = (Δz_t, α_t, β_t, γ_t)

    Δz_t ∈ R^d  - focus shift
    α_t ∈ R     - branch bias
    β_t ∈ R     - coherence modulation
    γ_t ∈ R     - boundary rewrite strength
"""

from dataclasses import dataclass, field
from typing import Tuple
import numpy as np

from .dynamics import F_mind, G_boundary, H_coherence, U_memory
from .reward import compute_rewards


@dataclass
class ChildMindConfig:
    """Configuration for Child Mind v1 dimensions."""
    d: int = 16          # Core manifold dimension
    H_b: int = 16        # Boundary grid height
    W_b: int = 16        # Boundary grid width
    k_c: int = 5         # Coherence state dimension
    d_m: int = 16        # Memory summary dimension

    # Dynamics parameters
    coherence_target: float = 0.7
    memory_decay: float = 0.9
    boundary_diffusion: float = 0.1


@dataclass
class State:
    """
    Complete state of the child mind at time t.

    Attributes:
        z: Core manifold state vector (d,)
        b: Boundary holographic grid (H_b, W_b)
        c: Coherence state vector (k_c,)
        m: Memory summary vector (d_m,)
        t: Current timestep
    """
    z: np.ndarray  # (d,)
    b: np.ndarray  # (H_b, W_b)
    c: np.ndarray  # (k_c,)
    m: np.ndarray  # (d_m,)
    t: int = 0

    def copy(self) -> 'State':
        """Create a deep copy of this state."""
        return State(
            z=self.z.copy(),
            b=self.b.copy(),
            c=self.c.copy(),
            m=self.m.copy(),
            t=self.t
        )


@dataclass
class Action:
    """
    Action taken by the child mind agent.

    Attributes:
        delta_z: Focus shift vector (d,)
        alpha: Branch bias scalar
        beta: Coherence modulation scalar
        gamma: Boundary rewrite strength scalar
    """
    delta_z: np.ndarray  # (d,)
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0

    def copy(self) -> 'Action':
        """Create a deep copy of this action."""
        return Action(
            delta_z=self.delta_z.copy(),
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma
        )


def initial_state(seed: int = 42, config: ChildMindConfig = None) -> State:
    """
    Create an initial state for the child mind.

    Args:
        seed: Random seed for reproducibility
        config: Configuration object (uses defaults if None)

    Returns:
        Initial State object
    """
    if config is None:
        config = ChildMindConfig()

    rng = np.random.default_rng(seed)

    # Initialize z: small random values centered at origin
    z = rng.normal(0, 0.1, size=(config.d,))

    # Initialize b: smooth random boundary pattern
    b = rng.normal(0, 0.1, size=(config.H_b, config.W_b))
    # Apply simple smoothing
    for _ in range(3):
        b_smooth = np.zeros_like(b)
        for i in range(config.H_b):
            for j in range(config.W_b):
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = (i + di) % config.H_b, (j + dj) % config.W_b
                        neighbors.append(b[ni, nj])
                b_smooth[i, j] = np.mean(neighbors)
        b = b_smooth

    # Initialize c: coherence starting near target
    c = np.array([
        config.coherence_target,  # primary coherence metric
        0.5,                       # secondary coherence
        0.0,                       # phase alignment
        0.5,                       # stability measure
        0.0                        # novelty accumulator
    ])

    # Initialize m: zero memory (no history yet)
    m = np.zeros(config.d_m)

    return State(z=z, b=b, c=c, m=m, t=0)


def step(
    state: State,
    action: Action,
    config: ChildMindConfig = None
) -> Tuple[State, float]:
    """
    Advance the child mind by one timestep.

    Implements the dynamics:
        z_{t+1} = F_mind(z_t + Δz_t, c_t, b_t)
        b_{t+1} = G_boundary(b_t, z_{t+1}, γ_t)
        c_{t+1} = H_coherence(c_t, β_t, z_{t+1})
        m_{t+1} = U_memory(m_t, s_t, a_t)

    Args:
        state: Current state s_t
        action: Action a_t to take
        config: Configuration object

    Returns:
        (next_state, reward): The new state s_{t+1} and reward r_t
    """
    if config is None:
        config = ChildMindConfig()

    # Apply dynamics
    z_next = F_mind(
        z=state.z + action.delta_z,
        c=state.c,
        b=state.b,
        alpha=action.alpha
    )

    b_next = G_boundary(
        b=state.b,
        z=z_next,
        gamma=action.gamma,
        config=config
    )

    c_next = H_coherence(
        c=state.c,
        beta=action.beta,
        z=z_next,
        config=config
    )

    m_next = U_memory(
        m=state.m,
        state=state,
        action=action,
        config=config
    )

    # Create next state
    next_state = State(
        z=z_next,
        b=b_next,
        c=c_next,
        m=m_next,
        t=state.t + 1
    )

    # Compute reward
    reward = compute_rewards(state, action, next_state)

    return next_state, reward


def get_state_summary(state: State) -> dict:
    """
    Get a human-readable summary of the current state.

    Returns dict with:
        - coherence_level: 'low', 'medium', 'high'
        - stability: 'stable', 'moderate', 'chaotic'
        - novelty: 'routine', 'exploring', 'discovering'
        - description: Short text description
    """
    coherence = state.c[0]
    stability = state.c[3]
    novelty = state.c[4]
    z_magnitude = np.linalg.norm(state.z)

    # Classify coherence
    if coherence > 0.7:
        coh_level = 'high'
    elif coherence > 0.4:
        coh_level = 'medium'
    else:
        coh_level = 'low'

    # Classify stability
    if stability > 0.6:
        stab_level = 'stable'
    elif stability > 0.3:
        stab_level = 'moderate'
    else:
        stab_level = 'chaotic'

    # Classify novelty
    if novelty > 0.5:
        nov_level = 'discovering'
    elif novelty > 0.2:
        nov_level = 'exploring'
    else:
        nov_level = 'routine'

    # Generate description
    descriptions = {
        ('high', 'stable', 'routine'):
            "The child mind is in a calm, focused state.",
        ('high', 'stable', 'exploring'):
            "The child mind is in a stable but exploratory mode.",
        ('high', 'stable', 'discovering'):
            "The child mind is making discoveries while maintaining coherence.",
        ('high', 'moderate', 'routine'):
            "The child mind is coherent with some internal movement.",
        ('high', 'moderate', 'exploring'):
            "The child mind is actively exploring while staying coherent.",
        ('high', 'moderate', 'discovering'):
            "The child mind is in a creative, discovery-rich state.",
        ('high', 'chaotic', 'routine'):
            "The child mind is coherent despite internal turbulence.",
        ('high', 'chaotic', 'exploring'):
            "The child mind is navigating chaos with high coherence.",
        ('high', 'chaotic', 'discovering'):
            "The child mind is making breakthroughs amid turbulence.",
        ('medium', 'stable', 'routine'):
            "The child mind is in a quiet, contemplative phase.",
        ('medium', 'stable', 'exploring'):
            "The child mind is gently exploring nearby states.",
        ('medium', 'stable', 'discovering'):
            "The child mind is finding new patterns at a steady pace.",
        ('medium', 'moderate', 'routine'):
            "The child mind is in a balanced, ordinary mode.",
        ('medium', 'moderate', 'exploring'):
            "The child mind is curious, probing its environment.",
        ('medium', 'moderate', 'discovering'):
            "The child mind is actively learning and adapting.",
        ('medium', 'chaotic', 'routine'):
            "The child mind is somewhat unsettled.",
        ('medium', 'chaotic', 'exploring'):
            "The child mind is searching amid uncertainty.",
        ('medium', 'chaotic', 'discovering'):
            "The child mind is finding novelty in chaos.",
        ('low', 'stable', 'routine'):
            "The child mind is in a diffuse, unfocused state.",
        ('low', 'stable', 'exploring'):
            "The child mind is wandering without clear direction.",
        ('low', 'stable', 'discovering'):
            "The child mind stumbles upon novelty despite low coherence.",
        ('low', 'moderate', 'routine'):
            "The child mind is fragmented but persisting.",
        ('low', 'moderate', 'exploring'):
            "The child mind is scattered but still seeking.",
        ('low', 'moderate', 'discovering'):
            "The child mind finds fragments of novelty.",
        ('low', 'chaotic', 'routine'):
            "The child mind is in a chaotic, disorganized phase.",
        ('low', 'chaotic', 'exploring'):
            "The child mind is jumping between distant states.",
        ('low', 'chaotic', 'discovering'):
            "The child mind is in full creative chaos mode.",
    }

    key = (coh_level, stab_level, nov_level)
    description = descriptions.get(key, "The child mind is in an undefined state.")

    return {
        'coherence_level': coh_level,
        'coherence_value': float(coherence),
        'stability': stab_level,
        'stability_value': float(stability),
        'novelty': nov_level,
        'novelty_value': float(novelty),
        'z_magnitude': float(z_magnitude),
        'timestep': state.t,
        'description': description
    }
