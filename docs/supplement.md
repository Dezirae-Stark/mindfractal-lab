# Mathematical Supplement: Fractal Dynamical Consciousness Model

This supplement provides detailed derivations and algorithms for the main paper.

---

## S1. Newton's Method for Fixed Point Finding

### S1.1 Problem Formulation

We seek fixed points $\mathbf{x}^*$ satisfying:

$$
\mathbf{x}^* = f(\mathbf{x}^*) = A\mathbf{x}^* + B\tanh(W\mathbf{x}^*) + \mathbf{c}
$$

Define the residual function:

$$
\mathbf{g}(\mathbf{x}) = f(\mathbf{x}) - \mathbf{x} = (A - I)\mathbf{x} + B\tanh(W\mathbf{x}) + \mathbf{c}
$$

Fixed points satisfy $\mathbf{g}(\mathbf{x}^*) = \mathbf{0}$.

### S1.2 Newton Iteration

The Jacobian of $\mathbf{g}$ is:

$$
\frac{\partial \mathbf{g}}{\partial \mathbf{x}} = J(\mathbf{x}) - I = (A - I) + B \cdot \text{diag}(\text{sech}^2(W\mathbf{x})) \cdot W
$$

Newton's method iterates:

$$
\mathbf{x}_{k+1} = \mathbf{x}_k - \left[\frac{\partial \mathbf{g}}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_k}\right]^{-1} \mathbf{g}(\mathbf{x}_k)
$$

### S1.3 Algorithm

```
ALGORITHM: Newton Fixed Point Finder
INPUT: Model (A, B, W, c), initial guess x₀, tolerance ε, max iterations N
OUTPUT: Fixed point x* or failure

1. x ← x₀
2. FOR k = 1 TO N:
   a. g ← f(x) - x
   b. IF ||g|| < ε THEN RETURN x
   c. J_g ← J(x) - I
   d. Solve: J_g · δ = g
   e. x ← x - δ
3. RETURN failure
```

### S1.4 Multiple Fixed Points

To find all fixed points:

1. Generate grid of initial guesses over $[-L, L]^d$
2. Run Newton's method from each initial guess
3. Cluster converged points (tolerance $10\epsilon$)
4. Remove duplicates

---

## S2. Jacobian Derivation Details

### S2.1 Component-wise Derivation

The map is:

$$
f_i(\mathbf{x}) = \sum_j A_{ij} x_j + \sum_j B_{ij} \tanh\left(\sum_k W_{jk} x_k\right) + c_i
$$

Partial derivative:

$$
\frac{\partial f_i}{\partial x_\ell} = A_{i\ell} + \sum_j B_{ij} \cdot \text{sech}^2\left(\sum_k W_{jk} x_k\right) \cdot W_{j\ell}
$$

### S2.2 Matrix Form

Let $\mathbf{u} = W\mathbf{x}$ and $\mathbf{s} = \text{sech}^2(\mathbf{u})$ (element-wise). Then:

$$
J(\mathbf{x}) = A + B \cdot \text{diag}(\mathbf{s}) \cdot W
$$

### S2.3 2D Explicit Form

For $d = 2$:

$$
J(\mathbf{x}) = \begin{pmatrix}
A_{11} + \sum_j B_{1j} s_j W_{j1} & A_{12} + \sum_j B_{1j} s_j W_{j2} \\
A_{21} + \sum_j B_{2j} s_j W_{j1} & A_{22} + \sum_j B_{2j} s_j W_{j2}
\end{pmatrix}
$$

where $s_j = \text{sech}^2(W_{j1} x_1 + W_{j2} x_2)$.

---

## S3. Largest Lyapunov Exponent Algorithm

### S3.1 Tangent Vector Method

The largest Lyapunov exponent measures exponential divergence of nearby trajectories.

```
ALGORITHM: Largest Lyapunov Exponent
INPUT: Model, initial condition x₀, steps N, transient T
OUTPUT: Lyapunov exponent λ

1. x ← x₀
2. FOR t = 1 TO T:                    // Discard transient
   x ← f(x)
3. v ← random unit vector
4. S ← 0
5. FOR t = 1 TO N:
   a. J ← Jacobian(x)
   b. v ← J · v
   c. norm ← ||v||
   d. IF norm > 0:
      S ← S + log(norm)
      v ← v / norm                    // Renormalize
   e. x ← f(x)
6. λ ← S / N
7. RETURN λ
```

### S3.2 Convergence

The algorithm converges as $N \to \infty$. Typical values:
- $T = 1000$ (transient)
- $N = 5000$ (estimation)
- Relative error $\approx O(1/\sqrt{N})$

### S3.3 Full Lyapunov Spectrum (QR Method)

For all $d$ exponents, use QR decomposition:

```
ALGORITHM: Full Lyapunov Spectrum
INPUT: Model, x₀, N, T
OUTPUT: Spectrum {λ₁, λ₂, ..., λ_d}

1. x ← x₀; discard transient
2. Q ← I_d (orthonormal basis)
3. S[i] ← 0 for i = 1..d
4. FOR t = 1 TO N:
   a. J ← Jacobian(x)
   b. Z ← J · Q
   c. Q, R ← QR(Z)              // QR decomposition
   d. FOR i = 1 TO d:
      S[i] ← S[i] + log|R[i,i]|
   e. x ← f(x)
5. λ[i] ← S[i] / N for i = 1..d
6. RETURN sorted {λ₁ ≥ λ₂ ≥ ... ≥ λ_d}
```

---

## S4. Box-Counting Fractal Dimension

### S4.1 Definition

The box-counting dimension of a set $S$ is:

$$
D_0 = \lim_{\epsilon \to 0} \frac{\log N(\epsilon)}{\log(1/\epsilon)}
$$

where $N(\epsilon)$ is the minimum number of boxes of side $\epsilon$ covering $S$.

### S4.2 Algorithm for Basin Boundaries

```
ALGORITHM: Box-Counting Dimension
INPUT: Basin boundary points, box sizes {ε₁, ε₂, ..., ε_k}
OUTPUT: Fractal dimension D

1. FOR each ε_i:
   a. Create grid of boxes with side ε_i
   b. Count N(ε_i) = number of boxes containing boundary points
2. Fit line: log N(ε) = D · log(1/ε) + const
3. D ← slope of fitted line
4. RETURN D
```

### S4.3 Boundary Detection

Basin boundaries are detected by:

1. Compute basin labels on fine grid
2. For each pixel, check if any neighbor has different label
3. Mark as boundary if neighbors differ

### S4.4 Expected Values

For 2D systems with fractal basins:
- Smooth boundary: $D \approx 1$
- Fractal boundary: $D \in (1, 2)$
- Space-filling: $D \approx 2$

---

## S5. Numerical Integration Considerations

### S5.1 Discrete vs. Continuous Time

The model is inherently discrete-time (map iteration). No ODE integration is required.

### S5.2 Floating Point Precision

Use `float64` (double precision) to avoid:
- Accumulation errors in Lyapunov sums
- Loss of significance near fixed points
- Overflow in exponential growth

### S5.3 Overflow Prevention

In chaotic regimes, trajectories may grow unboundedly. Safeguards:

1. **Norm check**: If $\|\mathbf{x}\| > \theta$ (e.g., $\theta = 100$), classify as divergent
2. **Tangent renormalization**: Renormalize after each Jacobian multiplication
3. **Log-domain**: Accumulate $\log\|\cdot\|$ instead of products

### S5.4 Numerical Stability of $\tanh$

The function $\tanh(u)$ is numerically stable:
- For $|u| < 20$: direct evaluation
- For $|u| \geq 20$: $\tanh(u) \approx \text{sign}(u)$

Similarly, $\text{sech}^2(u) = 1 - \tanh^2(u)$ is stable.

---

## S6. Parameter Regimes and Bifurcations

### S6.1 Linear Stability Analysis

At a fixed point $\mathbf{x}^*$, local stability depends on the Jacobian eigenvalues $\{\mu_i\}$.

**Bifurcation types:**

| Eigenvalue condition | Bifurcation type |
|---------------------|------------------|
| Real $\mu$ crosses $+1$ | Saddle-node |
| Real $\mu$ crosses $-1$ | Period-doubling |
| Complex $|\mu|$ crosses $1$ | Neimark-Sacker (torus) |

### S6.2 Parameter Ranges for Rich Dynamics

Based on default parameters, typical regimes:

| $\mathbf{c}$ region | Dynamics |
|---------------------|----------|
| $\|\mathbf{c}\| < 0.5$ | Stable fixed point |
| $0.5 < \|\mathbf{c}\| < 1.0$ | Periodic orbits |
| $\|\mathbf{c}\| > 1.0$ | Chaotic or unbounded |

### S6.3 Role of Matrix $A$

The linear feedback matrix $A$ controls:

- **Diagonal elements** $A_{ii}$: Self-feedback strength
  - $|A_{ii}| < 1$: Damping
  - $|A_{ii}| > 1$: Amplification (usually unstable)

- **Off-diagonal elements** $A_{ij}$: Cross-coupling
  - Positive: Excitatory
  - Negative: Inhibitory

### S6.4 Role of Matrix $B$

The nonlinear coupling matrix $B$ determines:

- **Magnitude**: Strength of nonlinear contribution
  - Small $\|B\|$: Near-linear dynamics
  - Large $\|B\|$: Strong nonlinearity, potential chaos

- **Structure**: Distribution of nonlinear influence
  - Symmetric $B$: Conservative-like tendencies
  - Asymmetric $B$: Rotational/spiral dynamics

### S6.5 Role of Matrix $W$

The weight matrix $W$ affects:

- **Sensitivity**: How strongly state affects activation
  - Large $\|W\|$: Steep transitions (switch-like)
  - Small $\|W\|$: Gradual transitions

- **Near-identity** $W \approx I$: Each component primarily affects its own activation

### S6.6 Codimension-2 Bifurcations

At isolated points in $(c_1, c_2)$ space, codimension-2 bifurcations occur where two bifurcation curves intersect:

- **Cusp**: Two saddle-node curves meet
- **Bogdanov-Takens**: Saddle-node meets Neimark-Sacker
- **1:2 Resonance**: Period-doubling meets Neimark-Sacker

These organize the global bifurcation structure.

---

## S7. Attractor Classification Algorithm

### S7.1 Multi-Criterion Classification

```
ALGORITHM: Attractor Classification
INPUT: Model, initial condition x₀, steps N, transient T
OUTPUT: Attractor type

1. Simulate trajectory for N steps
2. Check divergence:
   IF max(||x_t||) > 100 THEN RETURN "unbounded"
3. Check fixed point:
   x_final ← trajectory[-1]
   IF ||f(x_final) - x_final|| < ε THEN RETURN "fixed_point"
4. Compute Lyapunov exponent λ
5. IF λ > 0.01 THEN RETURN "chaotic"
6. IF λ < -0.01 THEN RETURN "fixed_point"
7. RETURN "limit_cycle"
```

### S7.2 Period Detection

To detect period-$p$ cycles:

1. After transient, record trajectory $\{\mathbf{x}_T, \mathbf{x}_{T+1}, \ldots\}$
2. For $p = 1, 2, 3, \ldots, P_{\max}$:
   - Check if $\|\mathbf{x}_{T+p} - \mathbf{x}_T\| < \epsilon$
   - If yes, verify $\|\mathbf{x}_{T+kp} - \mathbf{x}_T\| < \epsilon$ for several $k$
3. Return smallest $p$ satisfying criterion

---

## S8. Trait-to-Parameter Mapping Derivation

### S8.1 Design Rationale

The mapping $\{\text{traits}\} \to \mathbf{c}$ is designed such that:

1. Traits in $[0, 1]$ map to $\mathbf{c}$ spanning the dynamically interesting region
2. Central trait values ($0.5$) map to moderate $\mathbf{c}$
3. Extreme traits explore bifurcation boundaries

### S8.2 Derivation

For openness $O \in [0, 1]$ and volatility $V \in [0, 1]$:

$$
c_1 = c_1^{\min} + (c_1^{\max} - c_1^{\min}) \cdot O + \alpha_V (V - 0.5)
$$

With $c_1^{\min} = -1$, $c_1^{\max} = 1$, $\alpha_V = 0.5$:

$$
c_1 = -1 + 2O + 0.5(V - 0.5)
$$

Similarly for $c_2$ using integration $I$ and focus $F$.

### S8.3 Inverse Mapping

Given $\mathbf{c}$, approximate trait values:

$$
O \approx \frac{c_1 + 1}{2}, \quad V \approx 0.5 + 2(c_1 + 1 - 2O)
$$

$$
I \approx \frac{c_2 + 1}{2}, \quad F \approx 0.5 + 2(c_2 + 1 - 2I)
$$

---

## S9. Computational Complexity

### S9.1 Single Trajectory

- One step: $O(d^2)$ (matrix-vector multiplication)
- $N$ steps: $O(N d^2)$
- Jacobian computation: $O(d^2)$
- Lyapunov exponent: $O(N d^2)$

### S9.2 Basin of Attraction

- Grid size: $R \times R$
- Per grid point: $O(N d^2)$
- Total: $O(R^2 N d^2)$

For $R = 200$, $N = 1000$, $d = 2$: $\sim 10^8$ operations

### S9.3 Parameter-Space Map

- Grid size: $R_c \times R_c$
- Per grid point: Lyapunov computation $O(N d^2)$
- Total: $O(R_c^2 N d^2)$

---

## References

[S1] Press, W. H., et al. (2007). *Numerical Recipes* (3rd ed.). Cambridge University Press.

[S2] Wolf, A., et al. (1985). Determining Lyapunov exponents from a time series. *Physica D*, 16(3), 285-317.

[S3] Grassberger, P., & Procaccia, I. (1983). Characterization of strange attractors. *Physical Review Letters*, 50(5), 346.

[S4] Kuznetsov, Y. A. (2004). *Elements of Applied Bifurcation Theory* (3rd ed.). Springer.

---

**Version:** 1.0.0
**Date:** 2025-11-26
