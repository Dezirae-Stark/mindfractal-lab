# MindFractal Lab

<div class="hero" markdown>

# Fractal Consciousness Model

**A Scientific Python Framework for Nonlinear Dynamical Systems**

Simulate, analyze, and visualize fractal dynamics modeling consciousness states, metastability, and personality traits.

[Get Started](getting-started/installation.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/Dezirae-Stark/mindfractal-lab){ .md-button }

</div>

---

## Overview

MindFractal Lab provides a complete mathematical framework for modeling consciousness states using nonlinear dynamical systems. The core model is defined by:

$$
\mathbf{x}_{n+1} = \mathbf{A}\mathbf{x}_n + \mathbf{B}\tanh(\mathbf{W}\mathbf{x}_n) + \mathbf{c}
$$

This discrete-time map exhibits:

- **Fixed points** — Stable equilibria (persistent mental states)
- **Limit cycles** — Periodic oscillations (rhythmic patterns)
- **Strange attractors** — Chaotic dynamics (creative flow)
- **Fractal basin boundaries** — Metastable transitions

## Key Features

<div class="grid cards" markdown>

-   :material-math-integral-box:{ .lg .middle } **Mathematical Rigor**

    ---

    Complete analytical framework with Jacobian derivation, Lyapunov exponents, and bifurcation analysis.

    [:octicons-arrow-right-24: Mathematical Foundations](math/overview.md)

-   :material-cube-outline:{ .lg .middle } **Multi-Dimensional**

    ---

    2D, 3D, and complex high-dimensional dynamics including Calabi-Yau inspired extensions.

    [:octicons-arrow-right-24: CY Dynamics](math/cy-dynamics.md)

-   :material-chart-scatter-plot:{ .lg .middle } **Rich Visualization**

    ---

    Phase portraits, basin maps, fractal slices, and interactive 3D exploration.

    [:octicons-arrow-right-24: Visualization Guide](playground/explorer.md)

-   :material-brain:{ .lg .middle } **Trait Mapping**

    ---

    Map psychological traits to dynamical parameters for personalized models.

    [:octicons-arrow-right-24: API Reference](api/extensions.md)

</div>

## The Possibility Manifold

The **Possibility Manifold** $\mathcal{P}$ formalizes the "tenth dimension" concept:

$$
\mathcal{P} = \left\{ (\mathbf{z}_0, \mathbf{c}, F) : \text{orbit}(\mathbf{z}_0, \mathbf{c}, F) \text{ is bounded} \right\}
$$

This provides a rigorous mathematical framework for:

| Metaphor | Mathematical Object |
|----------|-------------------|
| "All possible realities" | Complete manifold $\mathcal{P}$ |
| "Single timeline" | Point $p \in \mathcal{P}$ and orbit |
| "Branching realities" | Bifurcation points |
| "Adjacent realities" | Nearby points in metric |

[:octicons-arrow-right-24: Learn more about the Possibility Manifold](math/possibility-manifold.md)

## Quick Start

=== "Python API"

    ```python
    import numpy as np
    from mindfractal import FractalDynamicsModel, simulate_orbit, plot_orbit

    # Create model
    model = FractalDynamicsModel()

    # Simulate
    x0 = np.array([0.5, 0.5])
    trajectory = simulate_orbit(model, x0, n_steps=1000)

    # Visualize
    plot_orbit(model, x0, save_path='orbit.png')
    ```

=== "CLI"

    ```bash
    # Simulate trajectory
    mindfractal simulate --x0 0.5 0.5 --steps 1000

    # Generate visualization
    mindfractal visualize --mode orbit --output orbit.png

    # Compute fractal map
    mindfractal fractal --resolution 500 --output fractal.png
    ```

=== "Trait Mapping"

    ```python
    from extensions.psychomapping.trait_to_c import traits_to_parameters

    traits = {
        'openness': 0.8,
        'volatility': 0.3,
        'integration': 0.7,
        'focus': 0.6
    }
    c = traits_to_parameters(traits)
    model = FractalDynamicsModel(c=c)
    ```

## Visualizations

<figure markdown>
  ![Phase Portrait](images/phase_portrait.png){ width="400" }
  <figcaption>Phase portrait showing trajectory evolution</figcaption>
</figure>

<figure markdown>
  ![Basin of Attraction](images/basin.png){ width="400" }
  <figcaption>Basin of attraction with fractal boundaries</figcaption>
</figure>

<figure markdown>
  ![Lyapunov Parameter Space](images/lyapunov_param_space.png){ width="400" }
  <figcaption>Parameter-space Lyapunov exponent map</figcaption>
</figure>

## Documentation

| Document | Description |
|----------|-------------|
| [Scientific Paper](research/paper.md) | Full mathematical framework |
| [LaTeX Book](research/book.md) | Comprehensive textbook |
| [API Reference](api/index.md) | Complete API documentation |
| [Developer Guide](developer/architecture.md) | Architecture and contributing |

## Installation

```bash
# Clone repository
git clone https://github.com/Dezirae-Stark/mindfractal-lab.git
cd mindfractal-lab

# Install in development mode
pip install -e .

# Install with all extras
pip install -e ".[dev,gui,web]"
```

See the [Installation Guide](getting-started/installation.md) for detailed instructions including Android (Termux/PyDroid3) setup.

## Extensions

| Extension | Description |
|-----------|-------------|
| **3D State Space** | Extended model with richer dynamics |
| **Trait Mapping** | Psychological traits → parameter conversion |
| **Kivy GUI** | Android/desktop interface |
| **FastAPI Web** | Browser-based visualization |
| **C++ Backend** | 10-100x speedup |

## Contributing

Contributions are welcome! See the [Contributing Guide](developer/contributing.md) for:

- Code style and conventions
- Testing requirements
- Pull request process
- Issue reporting

## License

MindFractal Lab is released under the **MIT License**.

---

<div style="text-align: center; color: rgba(255,255,255,0.6); margin-top: 2rem;">
  Built with love for consciousness research and fractal mathematics.
</div>
