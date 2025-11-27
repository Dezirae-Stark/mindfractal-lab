# Installation

## Requirements

- Python 3.7+
- NumPy
- Matplotlib

## Installation Methods

### PyPI (Recommended)

```bash
pip install mindfractal
```

### From Source

```bash
git clone https://github.com/Dezirae-Stark/mindfractal-lab.git
cd mindfractal-lab
pip install -e .
```

### Android (Termux)

```bash
pkg install python numpy matplotlib git
git clone https://github.com/Dezirae-Stark/mindfractal-lab.git
cd mindfractal-lab
pip install -e .
```

### Android (PyDroid 3)

```python
import os
os.system('pip install numpy matplotlib')
os.system('pip install git+https://github.com/Dezirae-Stark/mindfractal-lab.git')
```

## Verify Installation

```python
from mindfractal import FractalDynamicsModel
model = FractalDynamicsModel()
print("Installation successful!")
```
