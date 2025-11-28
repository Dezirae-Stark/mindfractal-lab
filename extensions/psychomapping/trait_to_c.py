"""
Psychological Trait → Parameter Mapping

Maps personality/consciousness traits (0-1 scale) to system parameters.

Traits:
- openness: exploration vs stability
- volatility: emotional reactivity
- integration: coherence vs fragmentation
- focus: attention stability
"""

import json
from pathlib import Path

import numpy as np


def traits_to_parameters(traits: dict) -> np.ndarray:
    """
    Map psychological traits to parameter vector c.

    Args:
        traits: Dict with keys like {'openness': 0.7, 'volatility': 0.3, ...}

    Returns:
        c parameter vector (2D)
    """
    openness = traits.get("openness", 0.5)
    volatility = traits.get("volatility", 0.5)
    integration = traits.get("integration", 0.5)
    focus = traits.get("focus", 0.5)

    # Map to c1, c2
    # c1: driven by openness and volatility
    # c2: driven by integration and focus
    c1 = -1.0 + 2.0 * openness + 0.5 * (volatility - 0.5)
    c2 = -1.0 + 2.0 * integration + 0.5 * (focus - 0.5)

    # Clip to reasonable range
    c1 = np.clip(c1, -2.0, 2.0)
    c2 = np.clip(c2, -2.0, 2.0)

    return np.array([c1, c2])


def load_trait_profiles(json_path: str = None) -> dict:
    """Load pre-defined trait profiles from JSON"""
    if json_path is None:
        json_path = Path(__file__).parent / "traits.json"

    with open(json_path, "r") as f:
        return json.load(f)


def save_trait_profile(profile_name: str, traits: dict, json_path: str = None):
    """Save a trait profile"""
    if json_path is None:
        json_path = Path(__file__).parent / "traits.json"

    try:
        profiles = load_trait_profiles(json_path)
    except FileNotFoundError:
        profiles = {}

    profiles[profile_name] = traits

    with open(json_path, "w") as f:
        json.dump(profiles, f, indent=2)

    print(f"Saved profile '{profile_name}' to {json_path}")
