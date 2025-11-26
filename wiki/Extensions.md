# Extensions

MindFractal Lab includes several optional extensions for advanced functionality.

---

## 3D State Space

**Location:** `extensions/state3d/`

Extends the model to three dimensions for richer dynamics.

### Usage

```python
from extensions.state3d.model_3d import FractalDynamicsModel3D
import numpy as np

# Create 3D model
model = FractalDynamicsModel3D()

# Or with custom parameters
model = FractalDynamicsModel3D(c=np.array([0.4, 0.3, 0.35]))

# Simulate
x0 = np.array([0.1, 0.1, 0.1])
x = x0.copy()
for _ in range(1000):
    x = model.step(x)
```

### Features

- Full 3-exponent Lyapunov spectrum
- Hyperchaos possible (two positive exponents)
- 3D phase portrait visualization

---

## Trait Mapping (Psychomapping)

**Location:** `extensions/psychomapping/`

Maps psychological traits to model parameters.

### Usage

```python
from extensions.psychomapping.trait_to_c import traits_to_parameters
from mindfractal import FractalDynamicsModel

# Define traits (all values in [0, 1])
traits = {
    'openness': 0.8,      # Creativity, curiosity
    'volatility': 0.3,    # Emotional stability (low = stable)
    'integration': 0.7,   # Coherence, unity
    'focus': 0.6          # Attention, concentration
}

# Convert to parameters
c = traits_to_parameters(traits)

# Create personalized model
model = FractalDynamicsModel(c=c)
```

### Trait Definitions

| Trait | Low Value | High Value |
|-------|-----------|------------|
| Openness | Conventional | Creative |
| Volatility | Stable | Reactive |
| Integration | Fragmented | Unified |
| Focus | Diffuse | Concentrated |

---

## Kivy GUI

**Location:** `extensions/gui_kivy/`

Cross-platform graphical interface for Android and desktop.

### Installation

```bash
pip install kivy
```

### Usage

```bash
python extensions/gui_kivy/main.py
```

### Features

- Real-time parameter adjustment via sliders
- Live trajectory visualization
- Touch-friendly interface

---

## FastAPI Web App

**Location:** `extensions/webapp/`

Browser-based interface with REST API.

### Installation

```bash
pip install fastapi uvicorn
```

### Usage

```bash
python extensions/webapp/app.py
# Open http://localhost:8000
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/api/simulate` | POST | Run simulation |
| `/api/fractal` | POST | Generate fractal map |

---

## C++ Backend

**Location:** `extensions/cpp_backend/`

High-performance C++ implementation with Python bindings.

### Build Requirements

- C++17 compiler
- CMake 3.14+
- pybind11

### Build

```bash
cd extensions/cpp_backend
mkdir build && cd build
cmake ..
make
```

### Usage

```python
from extensions.cpp_backend import fast_simulate

trajectory = fast_simulate(model_params, x0, n_steps)
```

### Performance

- 10-100x speedup for large simulations
- Particularly effective for parameter sweeps
- Maintains numerical precision

---

## Extension Development

To create a new extension:

1. Create directory under `extensions/`
2. Add `__init__.py` with exports
3. Follow existing patterns for model integration
4. Add tests under `tests/`

---

*See [docs/developer.md](../docs/developer.md) for architecture details.*
