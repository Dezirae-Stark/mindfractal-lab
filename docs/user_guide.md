# MindFractal Lab - User Guide

## Installation

### PyDroid 3 (Android)

1. Install PyDroid 3 from Google Play
2. Open PyDroid 3, go to Menu → Pip
3. Install dependencies:
   ```
   numpy
   matplotlib
   ```
4. Download MindFractal Lab:
   ```python
   import os
   os.system('pip install git+https://github.com/YOUR_USERNAME/mindfractal-lab.git')
   ```

### Termux (Android)

```bash
# Install Python and dependencies
pkg install python numpy matplotlib git

# Clone repository
git clone https://github.com/YOUR_USERNAME/mindfractal-lab.git
cd mindfractal-lab

# Install
pip install -e .
```

### Linux/macOS

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/mindfractal-lab.git
cd mindfractal-lab

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install
pip install -e .
```

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
# Simulate an orbit
python -m mindfractal.mindfractal_cli simulate --x0 0.5 0.5 --steps 1000

# Visualize
python -m mindfractal.mindfractal_cli visualize --mode orbit --output orbit.png

# Generate fractal map
python -m mindfractal.mindfractal_cli fractal --resolution 500 --output fractal.png

# Analyze dynamics
python -m mindfractal.mindfractal_cli analyze --mode lyapunov
```

## Extensions

### 3D Model

```python
from extensions.state3d.model_3d import FractalDynamicsModel3D
from extensions.state3d.simulate_3d import simulate_orbit_3d
from extensions.state3d.visualize_3d import plot_orbit_3d

model_3d = FractalDynamicsModel3D()
x0_3d = np.array([0.5, 0.5, 0.5])
plot_orbit_3d(model_3d, x0_3d, save_path='orbit_3d.png')
```

### Trait Mapping

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

### GUI (Kivy)

```bash
python extensions/gui_kivy/mindfractal_app.py
```

### Web App (FastAPI)

```bash
# Install FastAPI
pip install fastapi uvicorn

# Run server
python extensions/webapp/app.py

# Open browser: http://localhost:8000
```

### C++ Backend (Advanced)

See `extensions/cpp_backend/build_instructions.md`

## Common Workflows

### Exploring Parameter Space

```python
from mindfractal.fractal_map import generate_fractal_map
from mindfractal.visualize import plot_fractal_map

# Generate fractal map
fractal_data = generate_fractal_map(
    c1_range=(-1.0, 1.0),
    c2_range=(-1.0, 1.0),
    resolution=500,
    criterion='divergence_time'
)

# Visualize
plot_fractal_map(fractal_data, (-1.0, 1.0), (-1.0, 1.0), save_path='fractal.png')
```

### Finding Fixed Points

```python
from mindfractal.simulate import find_fixed_points

fixed_points = find_fixed_points(model)
for fp, stable in fixed_points:
    print(f"Fixed point: {fp}, Stable: {stable}")
```

### Computing Lyapunov Exponent

```python
lyap = model.lyapunov_exponent_estimate(x0, n_steps=5000)
print(f"Lyapunov exponent: {lyap}")

if lyap > 0:
    print("Chaotic dynamics!")
```

## Troubleshooting

### "No module named 'matplotlib.backends.backend_tkagg'"

Solution: Use Agg backend (already set in visualize.py)

### Slow fractal map generation

- Reduce resolution: `resolution=200` instead of `500`
- Use C++ backend (if available)
- Reduce `max_steps`

### PyDroid 3 memory errors

- Reduce resolution
- Process in batches
- Use `return_all=False` in simulate_orbit

## Known Limitations

- GUI requires Kivy (not available in all Android environments)
- Web app requires network access
- C++ backend requires compilation
- Large fractal maps (>1000x1000) may be slow without C++ backend

## Support

- GitHub Issues: https://github.com/YOUR_USERNAME/mindfractal-lab/issues
- Documentation: See `docs/` directory

