# Attractor Classification

ML models for classifying dynamical behavior.

## Classification Tasks

### Attractor Type

Predict the type of long-term behavior:

| Class | Description | Features |
|:------|:------------|:---------|
| 0 | Fixed point | Low variance, convergent |
| 1 | Limit cycle | Periodic, bounded variance |
| 2 | Torus | Quasi-periodic, two frequencies |
| 3 | Strange attractor | High variance, positive Lyapunov |

### Basin Membership

Given initial condition, predict which attractor:

$$
\hat{y} = f(x_0; \theta)
$$

### Stability

Binary classification: stable vs unstable.

## Features

### Hand-crafted Features

- Lyapunov exponent estimate
- Trajectory variance
- Autocorrelation structure
- Recurrence statistics

### Learned Features

- CNN on trajectory images
- RNN on raw time series
- Autoencoder latent vectors

## Models

### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
```

### Neural Network

```python
import torch.nn as nn

class AttractorClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)
```

## Training Data

Generate labeled examples via simulation:

```python
from mindfractal import FractalDynamicsModel

def generate_dataset(n_samples):
    X, y = [], []
    for _ in range(n_samples):
        # Random parameters
        model = FractalDynamicsModel(c=np.random.randn(2))
        traj = simulate_orbit(model, x0, n_steps=1000)

        # Compute label
        lyap = model.lyapunov_exponent_estimate(x0)
        label = classify_attractor(traj, lyap)

        X.append(extract_features(traj))
        y.append(label)

    return np.array(X), np.array(y)
```

## Evaluation

- Accuracy
- Confusion matrix
- ROC curves per class
- Cross-validation
