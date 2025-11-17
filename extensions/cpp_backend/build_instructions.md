# C++ Accelerated Backend Build Instructions

This extension provides a C++ implementation of the orbit simulation for 10-100x speedup.

## Requirements

- g++ or clang with C++17 support
- pybind11 (for Python bindings)

## Building on Termux (Android)

```bash
# Install compiler and tools
pkg install clang python numpy

# Install pybind11
pip install pybind11

# Compile the extension
cd extensions/cpp_backend
clang++ -O3 -Wall -shared -std=c++17 -fPIC \
  $(python3 -m pybind11 --includes) \
  pybind_wrapper.cpp fast_orbit.cpp \
  -o fast_orbit$(python3-config --extension-suffix)

# Test
python3 -c "import fast_orbit; print('C++ backend loaded successfully!')"
```

## Building on Linux/MacOS

```bash
# Install pybind11
pip install pybind11

# Compile
cd extensions/cpp_backend
g++ -O3 -Wall -shared -std=c++17 -fPIC \
  $(python3 -m pybind11 --includes) \
  pybind_wrapper.cpp fast_orbit.cpp \
  -o fast_orbit$(python3-config --extension-suffix)
```

## Usage

```python
import numpy as np
from extensions.cpp_backend import fast_orbit

# Define model matrices
A = np.eye(2) * 0.9
B = np.array([[0.2, 0.3], [0.3, 0.2]])
W = np.eye(2)
c = np.array([0.1, 0.1])

# Initial condition
x0 = np.array([0.5, 0.5])

# Simulate (C++ backend)
trajectory = fast_orbit.simulate_orbit(A, B, W, c, x0, n_steps=10000)

print(f"Simulated {len(trajectory)} steps (C++ backend)")
```

## Fallback

If C++ backend fails to build, the pure Python implementation will be used automatically.
