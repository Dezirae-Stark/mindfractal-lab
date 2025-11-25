# 🧠 MindFractal Lab

**Fractal Dynamical Consciousness Model - A Scientific Python Package**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Android Compatible](https://img.shields.io/badge/Android-Compatible-green.svg)](README.md#installation)

A complete scientific software system for simulating and analyzing 2D and 3D fractal dynamical systems modeling consciousness states, metastability, and personality traits.

## 🎯 Overview

MindFractal Lab implements the discrete-time nonlinear dynamical system:

```
x_{n+1} = A x_n + B tanh(W x_n) + c
```

where:
- **x ∈ ℝ²** (or ℝ³): consciousness state vector
- **A, B, W**: system matrices encoding feedback, coupling, and weights
- **c ∈ ℝ²**: external drive / personality parameter vector

This model exhibits:
- ✨ Fixed points, limit cycles, and chaotic attractors
- 🌀 Fractal basin boundaries (metastable regions)
- 🎨 Rich bifurcation structure in parameter space
- 🧬 Trait-to-parameter mappings for personalized modeling

## 🚀 Features

### Core Capabilities
- **2D & 3D Models**: Complete dynamics engine with Jacobian, Lyapunov exponents
- **Visualization**: Phase portraits, basin of attraction, fractal maps
- **Analysis Tools**: Fixed point finder, attractor classifier, bifurcation diagrams
- **CLI Interface**: Full command-line control

### Extensions
1. **3D State Space**: Extended model with richer dynamics
2. **Trait Mapping**: Psychological traits → parameter conversion
3. **Tenth Dimension**: Possibility manifold explorer (𝒫 space)
4. **Kivy GUI**: Android/desktop interface with sliders
5. **FastAPI Web App**: Browser-based visualization
6. **C++ Backend**: 10-100x speedup via pybind11

## 📦 Installation

### PyDroid 3 (Android)
```python
# In PyDroid 3
import os
os.system('pip install numpy matplotlib')
os.system('pip install git+https://github.com/YOUR_USERNAME/mindfractal-lab.git')
```

### Termux (Android)
```bash
pkg install python numpy matplotlib git
git clone https://github.com/YOUR_USERNAME/mindfractal-lab.git
cd mindfractal-lab
pip install -e .
```

### Linux/macOS/Windows
```bash
git clone https://github.com/YOUR_USERNAME/mindfractal-lab.git
cd mindfractal-lab
pip install -e .
```

## 🎯 Quick Start

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

## 📚 Documentation

- **[Scientific Paper](docs/paper.md)**: Mathematical framework and theory
- **[User Guide](docs/user_guide.md)**: Installation and usage instructions
- **[Developer Guide](docs/developer.md)**: API reference and architecture
- **[Architecture](docs/architecture.md)**: System design and diagrams

## 🏗️ Project Structure

```
mindfractal-lab/
├── mindfractal/              # Core package
│   ├── model.py             # 2D fractal dynamics model
│   ├── simulate.py          # Simulation engine
│   ├── visualize.py         # Matplotlib plotting
│   ├── fractal_map.py       # Parameter-space fractals
│   └── mindfractal_cli.py   # Command-line interface
├── extensions/              # Optional extensions
│   ├── state3d/            # 3D model
│   ├── psychomapping/      # Trait → parameter mapping
│   ├── gui_kivy/           # Android/desktop GUI
│   ├── webapp/             # FastAPI web interface
│   ├── cpp_backend/        # C++ accelerated backend
│   └── tenth_dimension_possibility/  # Possibility manifold explorer
├── docs/                   # Documentation
├── tests/                  # Unit tests
├── notebooks/              # Jupyter notebooks
└── README.md              # This file
```

## 🔬 Scientific Background

This model is grounded in:
- Dynamical systems theory (chaos, bifurcations, fractals)
- Computational neuroscience (metastability, attractor dynamics)
- Complexity science (self-similarity, criticality)

Key concepts:
- **Metastability**: Systems near fractal basin boundaries exhibit prolonged transients
- **Fractal dimensions**: Parameter-space boundaries have fractal dimension D ≈ 1.3-1.8
- **Lyapunov exponents**: Positive → chaos, zero → periodic, negative → stable

See [docs/paper.md](docs/paper.md) for full mathematical treatment.

## 🎮 Extensions

### 3D Model
```python
from extensions.state3d.model_3d import FractalDynamicsModel3D
model_3d = FractalDynamicsModel3D()
```

### Tenth Dimension: Possibility Manifold
```python
from extensions.tenth_dimension_possibility import PossibilityManifold
manifold = PossibilityManifold(dim=2)
point = manifold.sample_point()
orbit = manifold.compute_orbit(point, steps=500)
```

### Web App
```bash
python extensions/webapp/app.py
# Open http://localhost:8000
```

### C++ Backend (10-100x faster)
```bash
cd extensions/cpp_backend
# See build_instructions.md
```

## 🌌 Tenth Dimension: Possibility Manifold

The **Tenth Dimension Possibility Module** extends MindFractal Lab with a mathematical formalization of the "tenth dimension" metaphor - the space of all possible dynamical configurations and timelines.

### Mathematical Framework

The **Possibility Manifold** 𝒫 is defined as:

```
𝒫 = { (z₀, c, F) : z₀ ∈ ℂⁿ, c ∈ ℂⁿ, F: ℂⁿ → ℂⁿ, orbit(z₀, c, F) bounded }
```

where:
- **z₀**: Initial state vector
- **c**: Parameter vector (personality/drive)
- **F**: Update rule family (TANH_2D, SIGMOID_2D, STATE_3D, CALABI_YAU)
- **orbit bounded**: No divergence to infinity

### Features

- **Manifold Sampling**: Explore parameter space systematically
- **Timeline Slicing**: Extract continuous curves γ(t) through 𝒫
- **Stability Classification**: Automatic categorization (stable, chaotic, divergent, boundary)
- **Metrics**: Lyapunov exponents, correlation dimension, manifold distance
- **Visualization**: 2D stability landscapes, timeline branches, Lyapunov heatmaps

### Quick Start

```python
from extensions.tenth_dimension_possibility import PossibilityManifold, TimelineSlicer

# Create 2D possibility manifold
manifold = PossibilityManifold(dim=2)

# Sample a point
point = manifold.sample_point()

# Compute orbit
orbit = manifold.compute_orbit(point, steps=500)

# Classify stability
region = manifold.classify_stability(orbit)
print(f"Stability: {region.value}")

# Create timeline slice
slicer = TimelineSlicer(manifold)
start = manifold.sample_point()
end = manifold.sample_point()
branch = slicer.slice_parameter_line(start, end, n_steps=20)
```

### CLI Commands

```bash
# Create timeline slice
python -m extensions.tenth_dimension_possibility.possibility_cli slice --steps 20 --output timeline.png

# Visualize stability landscape
python -m extensions.tenth_dimension_possibility.possibility_cli visualize --resolution 100 --output landscape.png

# Generate random orbit
python -m extensions.tenth_dimension_possibility.possibility_cli random-orbit --steps 500 --output orbit.png

# Map stability boundaries
python -m extensions.tenth_dimension_possibility.possibility_cli boundary-map --resolution 150 --output boundaries.png
```

### Documentation

- **[Mathematical Reference](extensions/tenth_dimension_possibility/td_math_reference.md)**: Complete mathematical foundations
- **[Tests](extensions/tenth_dimension_possibility/tests/test_possibility.py)**: Comprehensive test suite

### Conceptual Mapping

| Metaphor | Mathematical Object |
|----------|---------------------|
| "All possible realities" | Complete parameter space 𝒫 |
| "Timeline" | Curve γ(t) through 𝒫 |
| "Branching realities" | Bifurcation points |
| "Choosing a reality" | Fixing (z₀, c, F) |
| "Space of possibilities" | Manifold topology |

This extension provides rigorous foundations for exploring the full space of dynamical possibilities - from stable attractors to chaotic trajectories across parameter variations.

## 🧪 Examples

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

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Inspired by dynamical systems research in neuroscience and consciousness studies
- Built for compatibility with Android (PyDroid 3, Termux)
- Pure CPU implementation (no GPU dependencies)

## 📞 Contact

- **Issues**: https://github.com/YOUR_USERNAME/mindfractal-lab/issues
- **Discussions**: https://github.com/YOUR_USERNAME/mindfractal-lab/discussions

---

**Version**: 0.1.0  
**Author**: MindFractal Lab Contributors  
**Status**: Production-ready research software
