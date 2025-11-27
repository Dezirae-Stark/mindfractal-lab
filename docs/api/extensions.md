# Extensions

Optional modules extending core functionality.

## 3D State Space

Extended three-dimensional dynamics.

```python
from extensions.state3d.model_3d import FractalDynamicsModel3D

model = FractalDynamicsModel3D(
    A=np.eye(3) * 0.9,
    B=np.eye(3) * 0.5,
    W=np.ones((3, 3)),
    c=np.array([0.1, 0.05, 0.02])
)

x0 = np.array([0.5, 0.5, 0.5])
trajectory = model.simulate(x0, n_steps=1000)
```

## Trait Mapping

Map psychological traits to model parameters.

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

## CY Dynamics

Complex Calabi-Yau inspired extension.

```python
from extensions.cy_extension.cy_model import CYDynamicsModel

model = CYDynamicsModel(dim=2)
z0 = np.array([0.5 + 0.5j, 0.3 - 0.2j])
trajectory = model.simulate(z0, n_steps=1000)
```

## Tenth Dimension / Possibility Manifold

Higher-dimensional possibility space exploration.

```python
from extensions.tenth_dimension_possibility.possibility import PossibilityManifold
from extensions.tenth_dimension_possibility.navigator import PossibilityNavigator

manifold = PossibilityManifold(base_model=model)
navigator = PossibilityNavigator(manifold)

path = navigator.find_path(start_state, target_state)
```

## Kivy GUI

Android/desktop graphical interface.

```bash
python -m extensions.gui_kivy.main
```

## FastAPI Web App

Browser-based visualization server.

```bash
cd extensions/webapp
uvicorn main:app --reload
```

## C++ Backend

High-performance computation (10-100x speedup).

```python
from extensions.cpp_backend import FractalDynamicsModelCpp

model = FractalDynamicsModelCpp()  # Same API, faster execution
```

Build with:

```bash
cd extensions/cpp_backend
mkdir build && cd build
cmake .. && make
```
