# Calabi-Yau Dynamics

Extension to complex-valued state spaces inspired by Calabi-Yau manifold geometry.

## Complex State Space

The CY extension lifts real dynamics to $\mathbb{C}^n$:

$$
\mathbf{z}_{n+1} = \mathbf{A}_\mathbb{C} \mathbf{z}_n + \mathbf{B}_\mathbb{C} \tanh(\mathbf{W}_\mathbb{C} \mathbf{z}_n) + \mathbf{c}_\mathbb{C}
$$

Where $\mathbf{z} = \mathbf{x} + i\mathbf{y} \in \mathbb{C}^d$.

## Geometric Interpretation

### Complex Manifold Structure

The state space becomes a complex manifold with:

- **Holomorphic structure** — complex-differentiable dynamics
- **Kähler metric** — natural geometric measure
- **Calabi-Yau-like properties** — special holonomy constraints

### Visualization

Complex states are visualized as:

1. **Real/Imaginary projections** — $(Re(z_1), Im(z_1))$ planes
2. **Modulus/Phase** — $|z|$ and $\arg(z)$ representations
3. **Slice views** — 2D cuts through the complex manifold

## Julia-Mandelbrot Connection

The CY dynamics relate to classical Julia/Mandelbrot sets:

$$
z_{n+1} = z_n^2 + c
$$

Our generalization extends this with matrix structure and $\tanh$ nonlinearity.

## Implementation

```python
from extensions.cy_extension.cy_model import CYDynamicsModel

model = CYDynamicsModel(dim=2)
z0 = np.array([0.5 + 0.5j, 0.3 - 0.2j])
trajectory = model.simulate(z0, n_steps=1000)
```
