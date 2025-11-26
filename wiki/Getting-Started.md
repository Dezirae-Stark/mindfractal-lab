# Getting Started

This guide covers installation and basic usage of MindFractal Lab.

---

## Installation

### Requirements
- Python 3.7+
- NumPy
- Matplotlib

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

## First Steps

### 1. Create a Model

```python
import numpy as np
from mindfractal import FractalDynamicsModel

# Default parameters
model = FractalDynamicsModel()

# Custom parameters
model = FractalDynamicsModel(c=np.array([0.5, 0.3]))
```

### 2. Simulate an Orbit

```python
from mindfractal import simulate_orbit

x0 = np.array([0.1, 0.1])
trajectory = simulate_orbit(model, x0, n_steps=1000)
```

### 3. Visualize

```python
from mindfractal import plot_orbit

plot_orbit(model, x0, save_path='orbit.png')
```

---

## Command Line Interface

```bash
# Simulate trajectory
python -m mindfractal.mindfractal_cli simulate --x0 0.5 0.5 --steps 1000

# Generate phase portrait
python -m mindfractal.mindfractal_cli visualize --mode orbit --output orbit.png

# Generate fractal map
python -m mindfractal.mindfractal_cli fractal --resolution 500 --output fractal.png
```

---

## Generate Figures

Run the example scripts to generate all figures:

```bash
cd mindfractal-lab

python examples/phase_portrait_script.py      # → docs/images/phase_portrait.png
python examples/basin_script.py               # → docs/images/basin.png
python examples/lyapunov_param_space_script.py # → docs/images/lyapunov_param_space.png
python examples/attractor_3d_script.py        # → docs/images/attractor_3d.png
python examples/trajectory_gif_script.py      # → docs/images/trajectory_on_basin.gif
```

---

## Next Steps

- Read the [Mathematical Framework](Mathematical-Framework) for theory
- Explore the [API Reference](API-Reference) for detailed documentation
- Check out the [Extensions](Extensions) for advanced features

---

*See also: [User Guide](../docs/user_guide.md)*
