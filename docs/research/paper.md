# Scientific Paper

The theoretical foundation of MindFractal Lab.

## Abstract

We present a fractal dynamical systems framework for modeling consciousness states, metastability, and personality traits. The model implements a nonlinear discrete-time map exhibiting fixed points, limit cycles, and strange attractors with fractal basin boundaries.

## Citation

```bibtex
@software{mindfractal2025,
  title = {MindFractal Lab: Fractal Dynamical Consciousness Model},
  author = {MindFractal Lab Contributors},
  year = {2025},
  url = {https://github.com/Dezirae-Stark/mindfractal-lab}
}
```

## Full Paper

See the complete scientific paper: [paper.md](../paper.md)

## Key Results

### 1. Rich Dynamical Behavior

The model exhibits:

- **Fixed points** — stable equilibria representing settled states
- **Limit cycles** — periodic oscillations
- **Strange attractors** — chaotic dynamics with fractal structure

### 2. Fractal Basin Boundaries

Basin boundaries have fractal dimension $D_B > 1$, creating:

- Metastable regions
- Sensitive dependence on initial conditions near boundaries
- Complex transition dynamics

### 3. Lyapunov Analysis

Maximal Lyapunov exponent characterizes dynamics:

$$
\lambda = \lim_{n \to \infty} \frac{1}{n} \sum_{k=0}^{n-1} \ln \|\mathbf{J}(\mathbf{x}_k)\|
$$

| λ | Behavior |
|:--|:---------|
| < 0 | Stable (converges to attractor) |
| = 0 | Neutral (periodic/quasi-periodic) |
| > 0 | Chaotic (sensitive dependence) |

### 4. Parameter-Behavior Mapping

Systematic relationship between parameters and dynamics:

- **A matrix** — damping/amplification
- **B matrix** — nonlinear coupling strength
- **c vector** — external drive ("personality")

## Mathematical Details

See [Mathematical Supplement](../supplement.md) for complete derivations.

## Related Work

- Dynamical systems theory (Strogatz, 2015)
- Metastability in neural systems (Kelso, 2012)
- Fractal geometry (Mandelbrot, 1982)
- Consciousness models (Tononi, 2008)
