# Embeddings Explorer

Visualize trajectory embeddings in latent space using dimensionality reduction.

<script src="https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js"></script>
<link rel="stylesheet" href="../site/interactive/css/interactive.css">
<link rel="stylesheet" href="../site/interactive/css/interactive-mobile.css">
<script src="../site/interactive/js/ui_common.js"></script>
<script src="../site/interactive/js/pyodide_bootstrap.js"></script>
<script src="../site/interactive/js/embeddings_viewer.js"></script>

<div id="embeddings-viewer-container" class="interactive-demo"></div>

<script>
document.addEventListener('DOMContentLoaded', function() {
    new EmbeddingsViewer('embeddings-viewer-container');
});
</script>

## About This Tool

The Embeddings Explorer uses machine learning techniques to visualize high-dimensional trajectory data in 2D, revealing hidden structure in the space of possible dynamics.

### Trajectory Features

Each trajectory is characterized by statistical features:

| Feature | Description |
|---------|-------------|
| **Mean** | Average position per dimension |
| **Std** | Spread/variance per dimension |
| **Range** | Max - min per dimension |
| **Path length** | Total distance traveled |
| **Curvature** | Average direction change |

These features form a high-dimensional representation of the trajectory's character.

### PCA (Principal Component Analysis)

**PCA** finds the directions of maximum variance:

$$
\mathbf{y} = W^T (\mathbf{x} - \bar{\mathbf{x}})
$$

Where $W$ contains the principal eigenvectors of the covariance matrix.

**Strengths**: Fast, preserves global structure, interpretable axes

### t-SNE (t-distributed Stochastic Neighbor Embedding)

**t-SNE** preserves local neighborhood structure:

$$
p_{ij} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma^2)}{\sum_{k \neq l} \exp(-\|x_k - x_l\|^2 / 2\sigma^2)}
$$

**Strengths**: Better cluster separation, reveals local structure

### Classification by Dynamics

Trajectories are automatically classified:

- **Stable** (blue): Low variance, converging dynamics
- **Periodic** (green): Moderate variance, repeating patterns
- **Chaotic** (red): High variance, irregular patterns
- **Divergent** (gray): Escaping trajectories

### Parameter Space View

The **Parameter Space** view shows how dynamical classification maps onto the parameter plane $(c_1, c_2)$.

## Applications

Embedding analysis enables:

1. **Clustering**: Find groups of similar dynamics
2. **Outlier detection**: Identify unusual trajectories
3. **Interpolation**: Predict behavior for new parameters
4. **Classification**: Automatic labeling of dynamics

## Mathematical Background

The embedding process:

1. **Sample** $N$ trajectories from parameter space
2. **Extract** $d$ features per trajectory
3. **Construct** feature matrix $X \in \mathbb{R}^{N \times d}$
4. **Reduce** to 2D using PCA or t-SNE
5. **Visualize** with classification colors

## Usage Tips

1. **Start with PCA** for an overview of the structure
2. **Try t-SNE** to reveal clusters
3. **Parameter Space** shows geographic organization
4. **Increase samples** for smoother distributions
5. **Adjust range** to focus on specific regions
