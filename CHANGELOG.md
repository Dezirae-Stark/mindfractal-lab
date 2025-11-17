# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-11-17

### Added
- Core 2D fractal dynamical consciousness model
  - `FractalDynamicsModel` class with step(), jacobian(), lyapunov_exponent_estimate()
  - Simulation engine with orbit generation and attractor classification
  - Visualization module with phase portraits, basin diagrams, bifurcation plots
  - Fractal map generator for parameter-space exploration
  - Command-line interface (CLI)

- Extension 1: 3D state space
  - `FractalDynamicsModel3D` for 3-dimensional dynamics
  - Full 3-exponent Lyapunov spectrum computation
  - 3D trajectory visualization

- Extension 2: Psychomapping
  - Trait-to-parameter mapping (openness, volatility, integration, focus → c)
  - Pre-defined personality profiles (balanced, creative_explorer, stable_focused, chaotic_fragmented, meditative)

- Extension 3: Kivy GUI
  - Interactive Android/desktop GUI with trait sliders
  - Real-time simulation and visualization

- Extension 4: FastAPI webapp
  - Browser-based interface
  - REST API endpoints for simulation and visualization
  - AJAX-based parameter adjustment

- Extension 5: C++ backend
  - High-performance orbit simulation (10-100× speedup)
  - pybind11 Python bindings
  - Build instructions for Termux/Linux

- Documentation
  - Scientific paper with mathematical framework
  - User guide with installation and usage instructions
  - Developer guide with API reference
  - Architecture document with system design diagrams

- Testing
  - Unit tests for core modules (model, simulate, visualize, fractal_map)
  - Extension tests (3D, psychomapping)
  - 90%+ coverage target

- Examples
  - Demo Jupyter notebook with 13 comprehensive examples
  - CLI usage examples

- Repository infrastructure
  - MIT License
  - README with badges and quick start
  - setup.py with extras (dev, gui, web, cpp)
  - .gitignore
  - CONTRIBUTING.md
  - CI/CD workflows (planned)

### Notes
- Android compatible (PyDroid 3, Termux)
- Pure CPU implementation (no GPU dependencies)
- Matplotlib Agg backend for headless rendering
- Production-ready research software

[0.1.0]: https://github.com/YOUR_USERNAME/mindfractal-lab/releases/tag/v0.1.0
