# MindFractal Lab Wiki

Welcome to the **MindFractal Lab** wiki — the documentation hub for the Fractal Dynamical Consciousness Model.

---

## Quick Navigation

| Page | Description |
|------|-------------|
| [Getting Started](Getting-Started) | Installation and first steps |
| [Mathematical Framework](Mathematical-Framework) | Core equations and theory |
| [API Reference](API-Reference) | Python module documentation |
| [Figure Gallery](Figure-Gallery) | Visualization examples |
| [Extensions](Extensions) | 3D model, trait mapping, GUI, web app |
| [FAQ](FAQ) | Frequently asked questions |

---

## What is MindFractal Lab?

MindFractal Lab is a scientific Python package for simulating and analyzing fractal dynamical systems. The core model:

$$\mathbf{x}_{n+1} = A\mathbf{x}_n + B\tanh(W\mathbf{x}_n) + \mathbf{c}$$

exhibits rich dynamics including:
- Stable fixed points
- Limit cycles (periodic orbits)
- Chaotic attractors
- Fractal basin boundaries

---

## Documentation Structure

```
docs/
├── paper.md           # Full scientific paper
├── supplement.md      # Mathematical derivations
├── user_guide.md      # Usage instructions
├── developer.md       # API and architecture
└── images/            # Generated figures
    └── README.md      # Image embedding snippets
```

---

## Key Features

### Core Package (`mindfractal/`)
- `model.py` — 2D fractal dynamics model with Jacobian and Lyapunov exponent
- `simulate.py` — Orbit simulation, fixed point finding, attractor classification
- `visualize.py` — Phase portraits, basin of attraction, bifurcation diagrams
- `fractal_map.py` — Parameter-space fractal generation

### Extensions (`extensions/`)
- `state3d/` — 3D model with full Lyapunov spectrum
- `psychomapping/` — Trait-to-parameter conversion
- `gui_kivy/` — Android/desktop GUI
- `webapp/` — FastAPI browser interface
- `cpp_backend/` — C++ accelerated backend

---

## Resources

- **GitHub Repository**: https://github.com/Dezirae-Stark/mindfractal-lab
- **Scientific Paper**: [docs/paper.md](../docs/paper.md)
- **Issue Tracker**: https://github.com/Dezirae-Stark/mindfractal-lab/issues

---

## Version History

| Version | Date | Notes |
|---------|------|-------|
| 1.0.0 | 2025-11-26 | Production release |
| 0.1.0 | 2025-11-25 | Initial release |

---

*Last updated: 2025-11-26*
