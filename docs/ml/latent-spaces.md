# Latent Space Representations

Low-dimensional embeddings of fractal dynamics.

## Autoencoder Architecture

### Standard Autoencoder

```
Input x ──[Encoder]──→ Latent z ──[Decoder]──→ Reconstruction x̂
```

Loss: $\mathcal{L} = \|x - \hat{x}\|^2$

### Variational Autoencoder (VAE)

Adds probabilistic structure:

$$
\mathcal{L}_\text{VAE} = \mathbb{E}[\|x - \hat{x}\|^2] + \beta \cdot \text{KL}(q(z|x) \| p(z))
$$

Benefits:

- Smooth latent space
- Meaningful interpolation
- Generation capability

## Trajectory Embeddings

### Single State

Embed individual states:

$$
z = E(x) \in \mathbb{R}^k
$$

### Full Trajectory

Embed entire time series:

$$
z_\text{traj} = E_\text{seq}(x_0, x_1, \ldots, x_T)
$$

Methods:

- **LSTM/GRU** — recurrent encoding
- **Transformer** — attention mechanism
- **Time-delay embedding** — classical approach

## Latent Space Properties

### Clustering

Similar dynamics cluster together:

- Fixed points form tight clusters
- Limit cycles form rings
- Chaotic attractors spread diffusely

### Interpolation

Moving through latent space:

$$
z(\alpha) = (1-\alpha) z_A + \alpha z_B
$$

Generates intermediate dynamics.

## Implementation

```python
from extensions.ml_embeddings.vae import TrajectoryVAE

# Train
vae = TrajectoryVAE(input_dim=2, latent_dim=8)
vae.fit(trajectories, epochs=100)

# Encode
z = vae.encode(trajectory)

# Decode
x_recon = vae.decode(z)

# Interpolate
z_interp = vae.interpolate(z1, z2, steps=10)
```
