"""
Gateway Coherence Core — Hemispheric Synchronization Patterns
MindFractal Lab

Pyodide-compatible module for visualizing coherence and resonance patterns
inspired by hemispheric synchronization concepts.
"""

import numpy as np
from typing import Dict, List, Tuple


def compute_coherence_pattern(params: Dict) -> Dict:
    """
    Compute a coherence pattern from two coupled oscillators.

    Parameters
    ----------
    params : dict
        freq_left : float - Left oscillator frequency (0.5-5.0 Hz)
        freq_right : float - Right oscillator frequency (0.5-5.0 Hz)
        phase_offset : float - Initial phase difference (0-2π)
        coupling : float - Coupling strength between oscillators (0-1)
        noise : float - Noise level (0-1)
        steps : int - Number of time steps (100-1000)

    Returns
    -------
    dict
        points : List of [x, y, intensity] - Pattern points
        coherence_metric : float - Overall coherence (0-1)
        phase_lock : bool - Whether oscillators are phase-locked
    """
    freq_left = float(params.get('freq_left', 1.0))
    freq_right = float(params.get('freq_right', 1.0))
    phase_offset = float(params.get('phase_offset', 0.0))
    coupling = float(params.get('coupling', 0.5))
    noise = float(params.get('noise', 0.1))
    steps = min(max(int(params.get('steps', 500)), 100), 1000)

    # Clamp frequencies
    freq_left = np.clip(freq_left, 0.5, 5.0)
    freq_right = np.clip(freq_right, 0.5, 5.0)

    # Time array
    t = np.linspace(0, 4 * np.pi, steps)
    dt = t[1] - t[0]

    # Initialize phases
    phase_l = 0.0
    phase_r = phase_offset

    points = []
    phases_l = []
    phases_r = []

    for i, time in enumerate(t):
        # Coupled oscillator dynamics (Kuramoto-like)
        # Phase evolution with coupling
        dphi_l = freq_left + coupling * np.sin(phase_r - phase_l)
        dphi_r = freq_right + coupling * np.sin(phase_l - phase_r)

        phase_l += dphi_l * dt
        phase_r += dphi_r * dt

        # Add noise
        phase_l += np.random.normal(0, noise * 0.1)
        phase_r += np.random.normal(0, noise * 0.1)

        phases_l.append(phase_l)
        phases_r.append(phase_r)

        # Lissajous-like pattern generation
        x = 0.5 + 0.4 * np.cos(phase_l) * (1 + 0.1 * np.sin(phase_r))
        y = 0.5 + 0.4 * np.sin(phase_r) * (1 + 0.1 * np.cos(phase_l))

        # Intensity based on phase difference
        phase_diff = abs(np.sin((phase_l - phase_r) / 2))
        intensity = 1.0 - phase_diff * (1 - coupling)
        intensity = np.clip(intensity + np.random.normal(0, noise * 0.05), 0.2, 1.0)

        points.append([float(x), float(y), float(intensity)])

    # Compute coherence metric
    phases_l = np.array(phases_l)
    phases_r = np.array(phases_r)
    phase_diffs = phases_l - phases_r

    # Circular variance of phase difference
    mean_vec = np.mean(np.exp(1j * phase_diffs))
    coherence_metric = float(np.abs(mean_vec))

    # Check for phase locking
    phase_lock = coherence_metric > 0.7 and abs(freq_left - freq_right) < 0.5

    return {
        'points': points,
        'coherence_metric': coherence_metric,
        'phase_lock': phase_lock,
        'effective_freq': float((freq_left + freq_right) / 2)
    }


def compute_resonance_field(params: Dict) -> Dict:
    """
    Compute a 2D resonance field showing coherence intensity.

    Parameters
    ----------
    params : dict
        resolution : int - Grid resolution (20-100)
        freq_left : float - Left frequency
        freq_right : float - Right frequency
        coupling : float - Coupling strength

    Returns
    -------
    dict
        field : 2D list of coherence values
        peak_locations : List of [x, y] peak positions
    """
    resolution = min(max(int(params.get('resolution', 40)), 20), 100)
    freq_left = float(params.get('freq_left', 1.0))
    freq_right = float(params.get('freq_right', 1.0))
    coupling = float(params.get('coupling', 0.5))

    field = []
    peaks = []

    for i in range(resolution):
        row = []
        y = i / (resolution - 1)
        for j in range(resolution):
            x = j / (resolution - 1)

            # Distance from center
            dx = x - 0.5
            dy = y - 0.5
            r = np.sqrt(dx*dx + dy*dy)

            # Angular position
            theta = np.arctan2(dy, dx)

            # Interference pattern
            wave1 = np.sin(2 * np.pi * freq_left * r + theta)
            wave2 = np.sin(2 * np.pi * freq_right * r - theta)

            # Coherence increases with coupling
            interference = coupling * wave1 * wave2 + (1 - coupling) * (wave1 + wave2) / 2
            value = 0.5 + 0.5 * interference

            row.append(float(value))

            # Track peaks
            if value > 0.85:
                peaks.append([float(x), float(y)])

        field.append(row)

    return {
        'field': field,
        'peak_locations': peaks[:20],  # Limit peaks
        'resolution': resolution
    }


def generate_binaural_pattern(params: Dict) -> Dict:
    """
    Generate a pattern representing binaural beat entrainment.

    Parameters
    ----------
    params : dict
        base_freq : float - Base frequency (100-400 Hz metaphor)
        beat_freq : float - Beat frequency (1-40 Hz metaphor)
        depth : float - Modulation depth (0-1)
        duration : float - Pattern duration (1-10 cycles)

    Returns
    -------
    dict
        pattern : List of amplitude values over time
        entrainment_state : str - Current entrainment state
    """
    base_freq = float(params.get('base_freq', 200))
    beat_freq = float(params.get('beat_freq', 10))
    depth = float(params.get('depth', 0.5))
    duration = float(params.get('duration', 3))

    # Scale for visualization (not actual audio frequencies)
    n_points = int(duration * 100)
    t = np.linspace(0, duration, n_points)

    # Beat envelope
    envelope = 0.5 + 0.5 * depth * np.sin(2 * np.pi * beat_freq / 10 * t)

    # Carrier modulated by envelope
    carrier = np.sin(2 * np.pi * base_freq / 100 * t)
    pattern = carrier * envelope

    # Normalize to 0-1
    pattern = 0.5 + 0.5 * pattern

    # Determine entrainment state based on beat frequency
    if beat_freq < 4:
        state = "delta (deep rest)"
    elif beat_freq < 8:
        state = "theta (meditative)"
    elif beat_freq < 13:
        state = "alpha (relaxed focus)"
    elif beat_freq < 30:
        state = "beta (active thinking)"
    else:
        state = "gamma (heightened awareness)"

    return {
        'pattern': [float(p) for p in pattern],
        'entrainment_state': state,
        'beat_freq': float(beat_freq),
        'depth': float(depth)
    }


# Export for Pyodide
__all__ = [
    'compute_coherence_pattern',
    'compute_resonance_field',
    'generate_binaural_pattern'
]
