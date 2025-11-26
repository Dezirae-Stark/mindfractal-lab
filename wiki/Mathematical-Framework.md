# Mathematical Framework

This page summarizes the mathematical foundations of the Fractal Dynamical Consciousness Model.

---

## Core Model

The discrete-time dynamical system is defined by:

$$\mathbf{x}_{n+1} = A\mathbf{x}_n + B\tanh(W\mathbf{x}_n) + \mathbf{c}$$

### Parameters

| Symbol | Dimension | Role |
|--------|-----------|------|
| $\mathbf{x}$ | $\mathbb{R}^d$ | State vector |
| $A$ | $d \times d$ | Linear feedback matrix |
| $B$ | $d \times d$ | Nonlinear coupling matrix |
| $W$ | $d \times d$ | Weight matrix (activation) |
| $\mathbf{c}$ | $\mathbb{R}^d$ | External drive / bias |

### Default Values (2D)

$$
A = \begin{pmatrix} 0.9 & 0 \\ 0 & 0.9 \end{pmatrix}, \quad
B = \begin{pmatrix} 0.2 & 0.3 \\ 0.3 & 0.2 \end{pmatrix}, \quad
W = \begin{pmatrix} 1.0 & 0.1 \\ 0.1 & 1.0 \end{pmatrix}
$$

---

## Fixed Point Analysis

### Definition

A fixed point $\mathbf{x}^*$ satisfies:

$$\mathbf{x}^* = A\mathbf{x}^* + B\tanh(W\mathbf{x}^*) + \mathbf{c}$$

### Newton's Method

Define residual $\mathbf{g}(\mathbf{x}) = f(\mathbf{x}) - \mathbf{x}$.

Iterate:
$$\mathbf{x}_{k+1} = \mathbf{x}_k - \left[\frac{\partial \mathbf{g}}{\partial \mathbf{x}}\right]^{-1} \mathbf{g}(\mathbf{x}_k)$$

### Stability

A fixed point is stable if all eigenvalues $\mu_i$ of the Jacobian satisfy $|\mu_i| < 1$.

---

## Jacobian Matrix

The Jacobian at point $\mathbf{x}$ is:

$$J(\mathbf{x}) = A + B \cdot \text{diag}(\text{sech}^2(W\mathbf{x})) \cdot W$$

where $\text{sech}^2(u) = 1 - \tanh^2(u)$.

---

## Lyapunov Exponent

The largest Lyapunov exponent measures sensitivity to initial conditions:

$$\lambda = \lim_{N \to \infty} \frac{1}{N} \sum_{n=0}^{N-1} \log \|J(\mathbf{x}_n) \mathbf{v}_n\|$$

| Value | Dynamics |
|-------|----------|
| $\lambda > 0$ | Chaotic |
| $\lambda = 0$ | Periodic (limit cycle) |
| $\lambda < 0$ | Stable (fixed point) |

---

## Attractor Types

| Type | Description | Lyapunov |
|------|-------------|----------|
| Fixed point | Single equilibrium | $\lambda < 0$ |
| Limit cycle | Periodic orbit | $\lambda = 0$ |
| Strange attractor | Chaotic, fractal structure | $\lambda > 0$ |

---

## Bifurcations

At bifurcation points, qualitative dynamics change:

| Eigenvalue Condition | Bifurcation |
|----------------------|-------------|
| Real $\mu$ crosses $+1$ | Saddle-node |
| Real $\mu$ crosses $-1$ | Period-doubling |
| Complex $|\mu|$ crosses $1$ | Neimark-Sacker (torus) |

---

## Fractal Basin Boundaries

The basin of attraction is the set of initial conditions converging to a given attractor.

When multiple attractors coexist, basin boundaries can be fractal with dimension $1 < D < 2$.

### Box-Counting Dimension

$$D_0 = \lim_{\epsilon \to 0} \frac{\log N(\epsilon)}{\log(1/\epsilon)}$$

where $N(\epsilon)$ is the number of boxes of size $\epsilon$ covering the boundary.

---

## N-Dimensional Generalization

The model extends to arbitrary dimension $d$:

- State: $\mathbf{x} \in \mathbb{R}^d$
- Matrices: $A, B, W \in \mathbb{R}^{d \times d}$
- Drive: $\mathbf{c} \in \mathbb{R}^d$
- Full Lyapunov spectrum: $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d$

For $d = 3$, hyperchaos ($\lambda_1, \lambda_2 > 0$) becomes possible.

---

## References

See [docs/paper.md](../docs/paper.md) and [docs/supplement.md](../docs/supplement.md) for complete derivations.

---

*Last updated: 2025-11-26*
