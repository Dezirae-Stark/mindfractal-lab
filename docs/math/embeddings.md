# Embeddings & Latent Spaces

Machine learning approaches to representing fractal dynamics in low-dimensional latent spaces.

## Motivation

High-dimensional state/parameter spaces are difficult to:

- Visualize
- Navigate
- Classify

ML embeddings provide compact, meaningful representations.

## Autoencoder Approach

### Architecture

$$
\mathbf{z} = E(\mathbf{x}) \quad \text{(encoder)}
$$
$$
\hat{\mathbf{x}} = D(\mathbf{z}) \quad \text{(decoder)}
$$

Where $\mathbf{z} \in \mathbb{R}^k$ is the latent representation with $k \ll d$.

### Training

Minimize reconstruction loss:

$$
\mathcal{L} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2 + \beta \cdot \text{KL}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))
$$

The KL term (variational autoencoder) encourages structured latent space.

## Trajectory Embeddings

Embed entire trajectories, not just states:

$$
\mathbf{z}_\text{traj} = E_\text{seq}(\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_T)
$$

Using:

- **RNNs/LSTMs** — sequential processing
- **Transformers** — attention over time steps
- **Reservoir computing** — echo state networks

## Applications

### Attractor Classification

Latent space clusters by attractor type:

- Fixed points
- Limit cycles
- Chaotic attractors

### Anomaly Detection

Identify unusual states via latent space distance.

### Interpolation

Generate smooth transitions between states.

## Implementation

```python
from extensions.ml_embeddings.autoencoder import TrajectoryAutoencoder

model = TrajectoryAutoencoder(latent_dim=8)
model.train(trajectories)
z = model.encode(trajectory)
```
