# Mathematical Supplement

Detailed derivations and proofs for the Fractal Consciousness Framework.

## Full Document

See the complete mathematical supplement: [supplement.md](../supplement.md)

## Contents

### 1. Dynamical System Definition

The core map:

$$
\mathbf{x}_{n+1} = f(\mathbf{x}_n) = \mathbf{A}\mathbf{x}_n + \mathbf{B}\tanh(\mathbf{W}\mathbf{x}_n) + \mathbf{c}
$$

### 2. Jacobian Derivation

The Jacobian matrix:

$$
\mathbf{J}(\mathbf{x}) = \frac{\partial f}{\partial \mathbf{x}} = \mathbf{A} + \mathbf{B} \cdot \text{diag}(\text{sech}^2(\mathbf{W}\mathbf{x})) \cdot \mathbf{W}
$$

**Derivation:**

Let $\mathbf{y} = \mathbf{W}\mathbf{x}$. Then:

$$
\frac{\partial}{\partial x_j} \tanh(y_i) = \text{sech}^2(y_i) \cdot W_{ij}
$$

Assembling in matrix form yields the result.

### 3. Lyapunov Exponent

**Definition:**

$$
\lambda = \lim_{n \to \infty} \frac{1}{n} \ln \|\mathbf{J}_n \cdots \mathbf{J}_1 \mathbf{v}\|
$$

for almost all initial vectors $\mathbf{v}$.

**Numerical Algorithm:**

1. Initialize: $\mathbf{x}_0$, $\mathbf{v}_0 = $ random unit vector
2. For $k = 1, \ldots, N$:
   - $\mathbf{x}_k = f(\mathbf{x}_{k-1})$
   - $\mathbf{v}_k = \mathbf{J}(\mathbf{x}_{k-1}) \mathbf{v}_{k-1}$
   - $r_k = \|\mathbf{v}_k\|$
   - $\mathbf{v}_k \leftarrow \mathbf{v}_k / r_k$
3. Return: $\lambda = \frac{1}{N} \sum_{k=1}^N \ln r_k$

### 4. Fixed Point Analysis

**Existence:** Fixed points satisfy $\mathbf{x}^* = f(\mathbf{x}^*)$.

**Stability:** Determined by eigenvalues of $\mathbf{J}(\mathbf{x}^*)$:

- $|\lambda_i| < 1$ for all $i$ → stable
- $|\lambda_i| > 1$ for some $i$ → unstable
- $|\lambda_i| = 1$ → marginal (requires higher-order analysis)

### 5. Fractal Dimension

**Box-counting dimension:**

$$
D_B = \lim_{\epsilon \to 0} \frac{\ln N(\epsilon)}{\ln(1/\epsilon)}
$$

where $N(\epsilon)$ = number of boxes of size $\epsilon$ needed to cover the set.

### 6. Metastability

**Dwell time scaling:**

$$
\tau(d) \sim d^{-\alpha}
$$

where $d$ = distance to basin boundary, $\alpha > 0$ depends on local dynamics.

## LaTeX Source

For formal typeset versions, see:

- `docs/math/base_model.tex`
- `docs/math/cy_extension.tex`
- `docs/math/possibility_manifold.tex`
- `docs/math/embeddings.tex`
- `docs/math/visualization_algorithms.tex`
