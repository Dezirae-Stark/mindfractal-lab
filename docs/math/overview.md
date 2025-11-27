# Mathematical Foundations

MindFractal Lab is grounded in rigorous mathematical theory from dynamical systems, chaos, and fractal geometry.

## The Core Dynamical System

The fundamental equation governing state evolution:

$$
\mathbf{x}_{n+1} = \mathbf{A}\mathbf{x}_n + \mathbf{B}\tanh(\mathbf{W}\mathbf{x}_n) + \mathbf{c}
$$

Where:

- $\mathbf{x} \in \mathbb{R}^d$ — state vector (consciousness coordinates)
- $\mathbf{A} \in \mathbb{R}^{d \times d}$ — linear feedback matrix
- $\mathbf{B} \in \mathbb{R}^{d \times d}$ — nonlinear coupling matrix
- $\mathbf{W} \in \mathbb{R}^{d \times d}$ — weight matrix
- $\mathbf{c} \in \mathbb{R}^d$ — external drive / personality parameters

## Key Concepts

### Attractors

Long-term behavior converges to:

- **Fixed points** — stable equilibria
- **Limit cycles** — periodic orbits
- **Strange attractors** — chaotic, fractal sets

### Lyapunov Exponents

Quantify sensitivity to initial conditions:

$$
\lambda = \lim_{n \to \infty} \frac{1}{n} \sum_{k=0}^{n-1} \ln \|\mathbf{J}(\mathbf{x}_k)\|
$$

### Fractal Dimension

Basin boundaries exhibit fractal structure with dimension $D_B > d - 1$.

## Further Reading

- [Base Models](base-models.md) — 2D/3D real dynamics
- [CY Dynamics](cy-dynamics.md) — complex geometry extension
- [Possibility Manifold](possibility-manifold.md) — the space of all possible states
