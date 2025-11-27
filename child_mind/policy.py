"""
Child Mind v1 — Policy Functions
MindFractal Lab

Provides policy functions for action selection.
For v1, these are simple random or bounded-random policies.

The purpose is visualization, not trained performance.
Future versions may include neural network policies.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import State, Action, ChildMindConfig


def random_policy(
    state: 'State',
    rng: np.random.Generator,
    config: 'ChildMindConfig' = None
) -> 'Action':
    """
    Fully random policy.

    Generates random actions within reasonable bounds.

    Args:
        state: Current state (unused in random policy)
        rng: NumPy random generator
        config: Configuration object

    Returns:
        Random Action
    """
    from .core import Action, ChildMindConfig

    if config is None:
        config = ChildMindConfig()

    # Random delta_z: small Gaussian shifts
    delta_z = rng.normal(0, 0.3, size=(config.d,))

    # Random scalars in bounded ranges
    alpha = rng.uniform(-1, 1)
    beta = rng.uniform(-0.5, 0.5)
    gamma = rng.uniform(0, 0.5)

    return Action(
        delta_z=delta_z,
        alpha=alpha,
        beta=beta,
        gamma=gamma
    )


def bounded_random_policy(
    state: 'State',
    rng: np.random.Generator,
    curiosity: float = 0.5,
    coherence_preference: float = 0.5,
    config: 'ChildMindConfig' = None
) -> 'Action':
    """
    Bounded random policy with controllable parameters.

    Allows external control of exploration vs exploitation trade-off
    through curiosity and coherence_preference parameters.

    Args:
        state: Current state (influences action bounds)
        rng: NumPy random generator
        curiosity: 0-1, higher = larger delta_z magnitude
        coherence_preference: 0-1, higher = actions favor coherence
        config: Configuration object

    Returns:
        Bounded random Action
    """
    from .core import Action, ChildMindConfig

    if config is None:
        config = ChildMindConfig()

    # Scale delta_z magnitude by curiosity
    base_std = 0.1 + 0.4 * curiosity  # 0.1 to 0.5
    delta_z = rng.normal(0, base_std, size=(config.d,))

    # High curiosity allows larger jumps
    if curiosity > 0.7:
        # Occasionally add a big jump
        if rng.random() < 0.2:
            jump_direction = rng.choice(config.d)
            delta_z[jump_direction] += rng.choice([-1, 1]) * curiosity

    # Alpha (branch bias): influenced by state's current position
    # When coherence is low, might want to branch more
    current_coherence = state.c[0] if len(state.c) > 0 else 0.5
    alpha_center = 0.0
    if current_coherence < 0.5:
        alpha_center = 0.2  # Slight bias to explore branches
    alpha = rng.normal(alpha_center, 0.3 * (1 - coherence_preference))

    # Beta (coherence modulation): favor positive when coherence_preference high
    beta_center = 0.2 * (coherence_preference - 0.5)  # -0.1 to 0.1
    beta = rng.normal(beta_center, 0.2)

    # Gamma (boundary rewrite): lower when we want stability
    gamma_max = 0.3 + 0.3 * curiosity  # 0.3 to 0.6
    gamma = rng.uniform(0, gamma_max)

    # Clip all values to reasonable ranges
    alpha = np.clip(alpha, -1.5, 1.5)
    beta = np.clip(beta, -1.0, 1.0)
    gamma = np.clip(gamma, 0, 1.0)

    return Action(
        delta_z=delta_z,
        alpha=float(alpha),
        beta=float(beta),
        gamma=float(gamma)
    )


def state_reactive_policy(
    state: 'State',
    rng: np.random.Generator,
    config: 'ChildMindConfig' = None
) -> 'Action':
    """
    Simple reactive policy that responds to state conditions.

    Not trained, but uses heuristics:
    - If coherence is low, try to increase it
    - If stuck in one region, try to explore
    - If unstable, reduce action magnitude

    Args:
        state: Current state
        rng: NumPy random generator
        config: Configuration object

    Returns:
        Reactive Action
    """
    from .core import Action, ChildMindConfig

    if config is None:
        config = ChildMindConfig()

    coherence = state.c[0] if len(state.c) > 0 else 0.5
    stability = state.c[3] if len(state.c) > 3 else 0.5
    novelty = state.c[4] if len(state.c) > 4 else 0.3
    z_norm = np.linalg.norm(state.z)

    # Base action magnitude inversely proportional to instability
    base_scale = 0.3 * stability + 0.1

    # Delta_z: move toward origin if far, explore if near
    if z_norm > 2.0:
        # Pull back toward origin
        direction = -state.z / (z_norm + 1e-8)
        delta_z = direction * base_scale + rng.normal(0, 0.1, size=(config.d,))
    elif z_norm < 0.5 and novelty < 0.2:
        # Stuck near origin, explore outward
        delta_z = rng.normal(0, 0.4, size=(config.d,))
    else:
        # Normal random exploration
        delta_z = rng.normal(0, base_scale, size=(config.d,))

    # Alpha: try to maintain moderate branching
    alpha = rng.normal(0, 0.2)

    # Beta: push toward coherence if low
    if coherence < 0.5:
        beta = rng.uniform(0.1, 0.4)  # Positive to increase coherence
    elif coherence > 0.8:
        beta = rng.uniform(-0.2, 0.1)  # Might reduce if too high
    else:
        beta = rng.normal(0, 0.1)

    # Gamma: reduce boundary writes if unstable
    if stability < 0.3:
        gamma = rng.uniform(0, 0.1)
    else:
        gamma = rng.uniform(0, 0.3)

    return Action(
        delta_z=delta_z,
        alpha=float(alpha),
        beta=float(beta),
        gamma=float(gamma)
    )


class SimpleNNPolicy:
    """
    Simple neural network policy (not trained).

    Uses random weights for demonstration purposes.
    Future versions could load trained weights.

    Architecture:
        Input: flattened state features
        Hidden: single layer with tanh activation
        Output: action parameters
    """

    def __init__(self, config: 'ChildMindConfig' = None, seed: int = 42):
        from .core import ChildMindConfig

        if config is None:
            config = ChildMindConfig()

        self.config = config
        self.rng = np.random.default_rng(seed)

        # Input size: z + c + b_stats + m_stats
        self.input_size = config.d + config.k_c + 4 + 4

        # Hidden size
        self.hidden_size = 32

        # Output size: d (for delta_z) + 3 (alpha, beta, gamma)
        self.output_size = config.d + 3

        # Initialize random weights
        self.W1 = self.rng.normal(0, 0.5, (self.input_size, self.hidden_size))
        self.b1 = self.rng.normal(0, 0.1, (self.hidden_size,))
        self.W2 = self.rng.normal(0, 0.3, (self.hidden_size, self.output_size))
        self.b2 = self.rng.normal(0, 0.1, (self.output_size,))

    def _extract_features(self, state: 'State') -> np.ndarray:
        """Extract feature vector from state."""
        features = []

        # z components
        features.extend(state.z)

        # c components
        features.extend(state.c)

        # b statistics
        features.append(np.mean(state.b))
        features.append(np.std(state.b))
        features.append(np.max(state.b))
        features.append(np.min(state.b))

        # m statistics
        features.append(np.mean(state.m))
        features.append(np.std(state.m))
        features.append(np.max(state.m))
        features.append(np.min(state.m))

        return np.array(features)

    def __call__(self, state: 'State') -> 'Action':
        """Generate action from state using the network."""
        from .core import Action

        # Extract features
        x = self._extract_features(state)

        # Forward pass
        h = np.tanh(x @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2

        # Parse output
        delta_z = np.tanh(out[:self.config.d]) * 0.5  # Bounded
        alpha = float(np.tanh(out[self.config.d]) * 1.0)
        beta = float(np.tanh(out[self.config.d + 1]) * 0.5)
        gamma = float((np.tanh(out[self.config.d + 2]) + 1) / 2 * 0.5)  # 0 to 0.5

        return Action(
            delta_z=delta_z,
            alpha=alpha,
            beta=beta,
            gamma=gamma
        )
