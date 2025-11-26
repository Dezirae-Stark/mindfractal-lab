# MindFractal Lab

**Fractal Dynamical Consciousness Model — A Scientific Python Package**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Android Compatible](https://img.shields.io/badge/Android-Compatible-green.svg)](#installation)

A complete scientific software system for simulating and analyzing 2D and 3D fractal dynamical systems modeling consciousness states, metastability, and personality traits.

---

## Overview

MindFractal Lab implements the discrete-time nonlinear dynamical system:

$$\mathbf{x}_{n+1} = A\mathbf{x}_n + B\tanh(W\mathbf{x}_n) + \mathbf{c}$$

where:
- **x ∈ ℝ²** (or ℝ³): consciousness state vector
- **A, B, W**: system matrices encoding feedback, coupling, and weights
- **c ∈ ℝ²**: external drive / personality parameter vector

This model exhibits:
- Fixed points, limit cycles, and chaotic attractors
- Fractal basin boundaries (metastable regions)
- Rich bifurcation structure in parameter space
- Trait-to-parameter mappings for personalized modeling

---

## Visualizations

<p align="center">
  <img src="docs/images/phase_portrait.png" alt="Phase Portrait" width="400">
  <img src="docs/images/basin.png" alt="Basin of Attraction" width="400">
</p>

<p align="center">
  <img src="docs/images/lyapunov_param_space.png" alt="Lyapunov Parameter Space" width="400">
  <img src="docs/images/attractor_3d.png" alt="3D Attractor" width="400">
</p>

<p align="center">
  <img src="docs/images/trajectory_on_basin.gif" alt="Trajectory Animation" width="600">
</p>

*Run scripts in `examples/` to generate these figures.*

---

## Features

### Core Capabilities
- **2D & 3D Models**: Complete dynamics engine with Jacobian, Lyapunov exponents
- **Visualization**: Phase portraits, basin of attraction, fractal maps
- **Analysis Tools**: Fixed point finder, attractor classifier, bifurcation diagrams
- **CLI Interface**: Full command-line control

### Extensions
| Extension | Description |
|-----------|-------------|
| **3D State Space** | Extended model with richer dynamics |
| **Trait Mapping** | Psychological traits → parameter conversion |
| **Kivy GUI** | Android/desktop interface with sliders |
| **FastAPI Web App** | Browser-based visualization |
| **C++ Backend** | 10-100x speedup via pybind11 |

---

## Installation

### PyDroid 3 (Android)
```python
import os
os.system('pip install numpy matplotlib')
os.system('pip install git+https://github.com/Dezirae-Stark/mindfractal-lab.git')
```

### Termux (Android)
```bash
pkg install python numpy matplotlib git
git clone https://github.com/Dezirae-Stark/mindfractal-lab.git
cd mindfractal-lab
pip install -e .
```

### Linux/macOS/Windows
```bash
git clone https://github.com/Dezirae-Stark/mindfractal-lab.git
cd mindfractal-lab
pip install -e .
```

---

## Quick Start

### Python API
```python
import numpy as np
from mindfractal import FractalDynamicsModel, simulate_orbit, plot_orbit

# Create model
model = FractalDynamicsModel()

# Simulate orbit
x0 = np.array([0.5, 0.5])
trajectory = simulate_orbit(model, x0, n_steps=1000)

# Visualize
plot_orbit(model, x0, save_path='orbit.png')
```

### Command Line
```bash
# Simulate
python -m mindfractal.mindfractal_cli simulate --x0 0.5 0.5 --steps 1000

# Visualize
python -m mindfractal.mindfractal_cli visualize --mode orbit --output orbit.png

# Generate fractal map
python -m mindfractal.mindfractal_cli fractal --resolution 500 --output fractal.png
```

### Trait Mapping
```python
from extensions.psychomapping.trait_to_c import traits_to_parameters

traits = {'openness': 0.8, 'volatility': 0.3, 'integration': 0.7, 'focus': 0.6}
c = traits_to_parameters(traits)

model = FractalDynamicsModel(c=c)
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Scientific Paper](docs/paper.md) | Mathematical framework and theory |
| [Mathematical Supplement](docs/supplement.md) | Detailed derivations and algorithms |
| [User Guide](docs/user_guide.md) | Installation and usage instructions |
| [Developer Guide](docs/developer.md) | API reference and architecture |
| [Image Embedding](docs/images/README.md) | Markdown snippets for visualizations |

---

## Project Structure

```
mindfractal-lab/
├── mindfractal/              # Core package
│   ├── model.py              # 2D fractal dynamics model
│   ├── simulate.py           # Simulation engine
│   ├── visualize.py          # Matplotlib plotting
│   ├── fractal_map.py        # Parameter-space fractals
│   └── mindfractal_cli.py    # Command-line interface
├── extensions/               # Optional extensions
│   ├── state3d/              # 3D model
│   ├── psychomapping/        # Trait → parameter mapping
│   ├── gui_kivy/             # Android/desktop GUI
│   ├── webapp/               # FastAPI web interface
│   └── cpp_backend/          # C++ accelerated backend
├── examples/                 # Figure generation scripts
│   ├── phase_portrait_script.py
│   ├── basin_script.py
│   ├── lyapunov_param_space_script.py
│   ├── attractor_3d_script.py
│   └── trajectory_gif_script.py
├── docs/                     # Documentation
│   ├── paper.md              # Scientific paper
│   ├── supplement.md         # Mathematical supplement
│   └── images/               # Generated figures
├── tests/                    # Unit tests
└── notebooks/                # Jupyter notebooks
```

---

## Scientific Background

This model is grounded in:
- **Dynamical systems theory**: chaos, bifurcations, fractals
- **Computational neuroscience**: metastability, attractor dynamics
- **Complexity science**: self-similarity, criticality

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Metastability** | Systems near fractal basin boundaries exhibit prolonged transients |
| **Fractal dimension** | Parameter-space boundaries have D ≈ 1.3–1.8 |
| **Lyapunov exponent** | λ > 0: chaotic, λ = 0: periodic, λ < 0: stable |

See [docs/paper.md](docs/paper.md) for the full mathematical treatment.

---

## Figure Generation

Generate all figures for the paper:

```bash
cd mindfractal-lab

# Generate static figures
python examples/phase_portrait_script.py
python examples/basin_script.py
python examples/lyapunov_param_space_script.py
python examples/attractor_3d_script.py

# Generate animated GIF
python examples/trajectory_gif_script.py
```

Output files are saved to `docs/images/`.

---

## Examples

### Basin of Attraction
```python
from mindfractal.visualize import plot_basin_of_attraction
plot_basin_of_attraction(model, resolution=200, save_path='basin.png')
```

### Lyapunov Exponent
```python
lyap = model.lyapunov_exponent_estimate(x0, n_steps=5000)
print(f"λ = {lyap:.4f} ({'chaotic' if lyap > 0 else 'stable'})")
```

### Fractal Map
```python
from mindfractal.fractal_map import generate_fractal_map
fractal_data = generate_fractal_map(resolution=500)
```

### 3D Model
```python
from extensions.state3d.model_3d import FractalDynamicsModel3D
model_3d = FractalDynamicsModel3D()
```

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Inspired by dynamical systems research in neuroscience and consciousness studies
- Built for compatibility with Android (PyDroid 3, Termux)
- Pure CPU implementation (no GPU dependencies)

---

## Contact

- **Issues**: https://github.com/Dezirae-Stark/mindfractal-lab/issues
- **Discussions**: https://github.com/Dezirae-Stark/mindfractal-lab/discussions

---

**Version**: 1.0.0
**Author**: MindFractal Lab Contributors
**Status**: Production-ready research software
