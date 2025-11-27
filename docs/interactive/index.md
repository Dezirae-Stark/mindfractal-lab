# Interactive Lab

Welcome to the **MindFractal Interactive Lab** — a browser-based playground for exploring fractal dynamical systems, consciousness modeling, and complex dynamics.

!!! info "Browser Requirements"
    The interactive modules use **Pyodide** (Python in WebAssembly) and require a modern browser with WebAssembly support. Best experienced on desktop Chrome, Firefox, or Safari.

## Available Interactive Tools

<div class="grid cards" markdown>

-   :material-cube-outline: **2D Fractal Explorer**

    ---

    Explore basin of attraction, Lyapunov maps, and orbit dynamics in the 2D fractal parameter space.

    [:octicons-arrow-right-24: Open Explorer](fractal_explorer.md)

-   :material-rotate-3d: **3D Attractor Explorer**

    ---

    Visualize strange attractors, Poincare sections, and the full 3D dynamics of the fractal consciousness model.

    [:octicons-arrow-right-24: Open Explorer](attractor_explorer.md)

-   :material-chart-bubble: **CY Slice Viewer**

    ---

    Explore Calabi-Yau inspired complex dynamics including Mandelbrot sets, Julia sets, and CY slices.

    [:octicons-arrow-right-24: Open Viewer](cy_slice_viewer.md)

-   :material-compass: **Possibility Navigator**

    ---

    Navigate the Possibility Manifold — sample trajectories, scan stability regions, and interpolate timelines.

    [:octicons-arrow-right-24: Open Navigator](possibility_navigator.md)

-   :material-brain: **Embeddings Explorer**

    ---

    Visualize trajectory embeddings in latent space using PCA and t-SNE dimensionality reduction.

    [:octicons-arrow-right-24: Open Explorer](embeddings_explorer.md)

</div>

## Quick Start

1. **Choose a tool** from the cards above
2. **Wait for Pyodide** to load (first load may take 10-20 seconds)
3. **Adjust parameters** using the sliders and controls
4. **Click Compute** to generate visualizations
5. **Toggle mode** between Explorer (simplified) and Researcher (code visible)

## Mode Toggle

Use the floating button in the bottom-right corner to switch between:

- **Explorer Mode**: Clean interface focused on visual exploration
- **Researcher Mode**: Shows Python code for each computation

## Mathematical Foundation

The interactive tools visualize the core dynamical system:

$$
\mathbf{x}_{n+1} = A\mathbf{x}_n + B \tanh(W\mathbf{x}_n) + \mathbf{c}
$$

Where:

- $\mathbf{x} \in \mathbb{R}^d$ is the state vector (consciousness state)
- $A, B, W$ are system matrices controlling dynamics
- $\mathbf{c}$ is the parameter vector (personality traits)

Different regions of parameter space exhibit:

- **Stable fixed points**: Converging to equilibrium states
- **Limit cycles**: Periodic oscillations
- **Strange attractors**: Chaotic dynamics with sensitive dependence
- **Divergence**: Unbounded trajectories

## Technical Details

The interactive modules are powered by:

- **Pyodide**: Python running in WebAssembly
- **NumPy**: Numerical computations
- **Matplotlib**: Server-side rendering to base64 images
- **Custom Python cores**: Optimized for browser execution

All computation happens locally in your browser — no data is sent to any server.
