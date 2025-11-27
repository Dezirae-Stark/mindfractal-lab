MindFractal Lab API Reference
=============================

Welcome to the MindFractal Lab API documentation. This reference provides
detailed information about all modules, classes, and functions in the package.

.. toctree::
   :maxdepth: 2
   :caption: Core Modules

   api/mindfractal

.. toctree::
   :maxdepth: 2
   :caption: Extensions

   api/extensions

.. toctree::
   :maxdepth: 1
   :caption: Indices

   genindex
   modindex


Core Package
------------

The ``mindfractal`` package provides the core functionality:

.. autosummary::
   :toctree: api
   :recursive:

   mindfractal.model
   mindfractal.simulate
   mindfractal.visualize
   mindfractal.fractal_map


Quick Reference
---------------

Main Classes
~~~~~~~~~~~~

.. currentmodule:: mindfractal.model

.. autosummary::

   FractalDynamicsModel

Main Functions
~~~~~~~~~~~~~~

.. currentmodule:: mindfractal.simulate

.. autosummary::

   simulate_orbit
   find_fixed_points
   compute_attractor_type

.. currentmodule:: mindfractal.visualize

.. autosummary::

   plot_orbit
   plot_phase_space
   plot_basin_of_attraction
   plot_bifurcation

.. currentmodule:: mindfractal.fractal_map

.. autosummary::

   generate_fractal_map


Extensions
----------

Extensions provide additional functionality:

- **state3d**: 3D dynamics models
- **psychomapping**: Trait-to-parameter mapping
- **tenth_dimension_possibility**: Possibility Manifold
- **gui_kivy**: Android/desktop GUI
- **webapp**: FastAPI web interface
- **cpp_backend**: C++ accelerated backend


Installation
------------

.. code-block:: bash

   pip install mindfractal

Or for development:

.. code-block:: bash

   git clone https://github.com/Dezirae-Stark/mindfractal-lab.git
   cd mindfractal-lab
   pip install -e ".[dev]"


Example Usage
-------------

.. code-block:: python

   import numpy as np
   from mindfractal import FractalDynamicsModel, simulate_orbit, plot_orbit

   # Create model
   model = FractalDynamicsModel()

   # Simulate
   x0 = np.array([0.5, 0.5])
   trajectory = simulate_orbit(model, x0, n_steps=1000)

   # Visualize
   plot_orbit(model, x0, save_path='orbit.png')


Links
-----

- **GitHub**: https://github.com/Dezirae-Stark/mindfractal-lab
- **Documentation**: https://dezirae-stark.github.io/mindfractal-lab/
- **Issues**: https://github.com/Dezirae-Stark/mindfractal-lab/issues
