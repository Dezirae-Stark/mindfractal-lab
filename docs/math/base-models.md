# Base Models

The foundation of MindFractal Lab: 2D and 3D real-valued dynamical systems.

## 2D Model

The core two-dimensional map:

$$
\begin{pmatrix} x_{n+1} \\ y_{n+1} \end{pmatrix} =
\mathbf{A} \begin{pmatrix} x_n \\ y_n \end{pmatrix} +
\mathbf{B} \tanh\left(\mathbf{W} \begin{pmatrix} x_n \\ y_n \end{pmatrix}\right) + \mathbf{c}
$$

### Default Parameters

```python
A = [[0.9, 0.1], [-0.1, 0.9]]  # Linear damping
B = [[0.5, 0.0], [0.0, 0.5]]   # Nonlinear coupling
W = [[1.0, 0.5], [0.5, 1.0]]   # Weight matrix
c = [0.1, 0.05]                 # Drive parameters
```

### Jacobian

$$
\mathbf{J}(\mathbf{x}) = \mathbf{A} + \mathbf{B} \cdot \text{diag}(\text{sech}^2(\mathbf{W}\mathbf{x})) \cdot \mathbf{W}
$$

## 3D Model

Extended three-dimensional dynamics:

$$
\mathbf{x}_{n+1} = \mathbf{A}_3 \mathbf{x}_n + \mathbf{B}_3 \tanh(\mathbf{W}_3 \mathbf{x}_n) + \mathbf{c}_3
$$

Where all matrices are $3 \times 3$.

### Richer Dynamics

The 3D model exhibits:

- Torus attractors
- Higher-dimensional chaos
- More complex bifurcation structure

## Implementation

```python
from mindfractal import FractalDynamicsModel
from extensions.state3d.model_3d import FractalDynamicsModel3D

# 2D model
model_2d = FractalDynamicsModel()

# 3D model
model_3d = FractalDynamicsModel3D()
```
