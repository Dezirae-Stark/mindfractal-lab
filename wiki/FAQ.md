# Frequently Asked Questions

Common questions about MindFractal Lab.

---

## General

### What is MindFractal Lab?

MindFractal Lab is a scientific Python package for simulating fractal dynamical systems. It implements the discrete-time nonlinear map:

$$\mathbf{x}_{n+1} = A\mathbf{x}_n + B\tanh(W\mathbf{x}_n) + \mathbf{c}$$

The model exhibits rich dynamics including fixed points, limit cycles, chaos, and fractal basin boundaries.

### What are the applications?

- Modeling consciousness states and metastability
- Studying dynamical systems and chaos
- Visualizing fractal structures
- Educational tool for nonlinear dynamics

### Is GPU required?

No. MindFractal Lab uses pure CPU computation with NumPy. It runs on any platform including Android (PyDroid 3, Termux).

---

## Installation

### How do I install on Android?

**PyDroid 3:**
```python
import os
os.system('pip install numpy matplotlib')
os.system('pip install git+https://github.com/Dezirae-Stark/mindfractal-lab.git')
```

**Termux:**
```bash
pkg install python numpy matplotlib git
git clone https://github.com/Dezirae-Stark/mindfractal-lab.git
cd mindfractal-lab
pip install -e .
```

### What dependencies are required?

- Python 3.7+
- NumPy
- Matplotlib

Optional:
- imageio (for GIF generation)
- Kivy (for GUI)
- FastAPI (for web app)

---

## Usage

### How do I change model parameters?

```python
import numpy as np
from mindfractal import FractalDynamicsModel

# Custom c parameter
model = FractalDynamicsModel(c=np.array([0.5, 0.3]))

# Custom matrices
A = np.array([[0.85, 0.1], [0.1, 0.85]])
model = FractalDynamicsModel(A=A)
```

### How do I generate figures?

Run the example scripts:

```bash
python examples/phase_portrait_script.py
python examples/basin_script.py
python examples/lyapunov_param_space_script.py
python examples/attractor_3d_script.py
python examples/trajectory_gif_script.py
```

### How do I compute the Lyapunov exponent?

```python
lyap = model.lyapunov_exponent_estimate(x0, n_steps=5000)
print(f"λ = {lyap:.4f}")
# λ > 0: chaotic
# λ ≈ 0: periodic
# λ < 0: stable
```

---

## Mathematics

### What is a Lyapunov exponent?

The Lyapunov exponent measures the rate of separation of infinitesimally close trajectories. Positive values indicate chaos (sensitive dependence on initial conditions).

### What is a basin of attraction?

The basin of attraction is the set of all initial conditions that converge to a particular attractor. When multiple attractors coexist, basin boundaries can be fractal.

### What is a fixed point?

A fixed point is a state $\mathbf{x}^*$ where the system remains unchanged: $f(\mathbf{x}^*) = \mathbf{x}^*$.

### What causes chaos in this model?

Chaos arises from the combination of:
1. Nonlinear feedback via $\tanh$
2. Parameter values near bifurcation boundaries
3. Coupling between state dimensions

---

## Troubleshooting

### Simulation diverges to infinity

Some parameter combinations lead to unbounded growth. Try:
- Reducing $\|\mathbf{c}\|$
- Ensuring $\|A\| < 1$ (spectral radius)
- Starting from initial conditions near the origin

### Figures not saving

Ensure the output directory exists:
```python
import os
os.makedirs('docs/images', exist_ok=True)
```

### Import errors

Make sure you installed the package:
```bash
pip install -e .
```

Or run from the repository root.

---

## Contributing

### How do I report bugs?

Open an issue at: https://github.com/Dezirae-Stark/mindfractal-lab/issues

### How do I contribute code?

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Submit a pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

---

*Last updated: 2025-11-26*
