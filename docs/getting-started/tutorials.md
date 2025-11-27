# Tutorials

Step-by-step guides to master MindFractal Lab.

## Tutorial 1: Phase Portraits

Learn to visualize the flow of the dynamical system.

```python
from mindfractal.visualize import plot_phase_portrait

model = FractalDynamicsModel()
plot_phase_portrait(model, xlim=(-2, 2), ylim=(-2, 2), resolution=20)
```

## Tutorial 2: Basin of Attraction

Explore the complex boundary structure.

```python
from mindfractal.visualize import plot_basin_of_attraction

plot_basin_of_attraction(model, resolution=200, save_path='basin.png')
```

## Tutorial 3: Lyapunov Exponents

Quantify chaos vs stability.

```python
lyap = model.lyapunov_exponent_estimate(x0, n_steps=5000)
print(f"Lyapunov exponent: {lyap:.4f}")
```

## Tutorial 4: 3D State Space

Extend to three dimensions.

```python
from extensions.state3d.model_3d import FractalDynamicsModel3D
model_3d = FractalDynamicsModel3D()
```

## Tutorial 5: Trait Mapping

Map psychological traits to model parameters.

```python
from extensions.psychomapping.trait_to_c import traits_to_parameters

traits = {'openness': 0.8, 'volatility': 0.3}
c = traits_to_parameters(traits)
```
