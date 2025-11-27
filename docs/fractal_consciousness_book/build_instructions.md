# Building the Fractal Consciousness Book

This document explains how to compile the LaTeX book into a PDF.

## Prerequisites

You need a LaTeX distribution installed. Options include:

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install texlive-full
```

### Linux (Arch)
```bash
sudo pacman -S texlive-most
```

### macOS
Install [MacTeX](https://www.tug.org/mactex/):
```bash
brew install --cask mactex
```

### Windows
Install [MiKTeX](https://miktex.org/) or [TeX Live](https://www.tug.org/texlive/).

### Minimal Installation
If you want a smaller installation:
```bash
# Debian/Ubuntu
sudo apt install texlive-latex-base texlive-latex-extra texlive-fonts-recommended texlive-science

# Or use tlmgr to install specific packages
tlmgr install amsmath amssymb amsthm mathtools physics hyperref cleveref booktabs algorithm algorithmic tikz pgfplots tcolorbox fancyhdr geometry enumitem caption subcaption
```

## Building

### Using Make
```bash
cd docs/fractal_consciousness_book
make pdf
```

### Manual Build
```bash
cd docs/fractal_consciousness_book
pdflatex fractal_consciousness_book.tex
pdflatex fractal_consciousness_book.tex  # Run twice for references
```

### With Bibliography (if added)
```bash
pdflatex fractal_consciousness_book.tex
bibtex fractal_consciousness_book
pdflatex fractal_consciousness_book.tex
pdflatex fractal_consciousness_book.tex
```

### Alternative Engines
```bash
# For better Unicode support
xelatex fractal_consciousness_book.tex

# Alternative
lualatex fractal_consciousness_book.tex
```

## Output

The compiled PDF will be at:
```
docs/fractal_consciousness_book/fractal_consciousness_book.pdf
```

## Troubleshooting

### Missing Packages
If you get errors about missing packages, install them:
```bash
# Using tlmgr
tlmgr install <package-name>

# Or install full texlive
sudo apt install texlive-full
```

### Common Issues

1. **"File not found" for chapters**: Make sure you're running from the `fractal_consciousness_book` directory.

2. **Math errors**: Ensure `amsmath`, `amssymb`, `amsthm` are installed.

3. **Font issues**: Try `xelatex` or `lualatex` instead of `pdflatex`.

4. **References not updating**: Run `pdflatex` twice.

## Cleaning Up

Remove auxiliary files:
```bash
make clean
```

Remove everything including PDF:
```bash
make distclean
```

## Viewing

```bash
# Linux
xdg-open fractal_consciousness_book.pdf

# macOS
open fractal_consciousness_book.pdf

# Windows
start fractal_consciousness_book.pdf
```

## Continuous Preview

For live preview while editing, consider:

### Using latexmk
```bash
latexmk -pdf -pvc fractal_consciousness_book.tex
```

### Using VS Code
Install the "LaTeX Workshop" extension for VS Code.

### Using Overleaf
Upload the entire `fractal_consciousness_book` directory to [Overleaf](https://www.overleaf.com) for online editing and compilation.

## Book Structure

```
fractal_consciousness_book/
├── fractal_consciousness_book.tex  # Main document
├── chapters/
│   ├── 01_intro.tex               # Introduction
│   ├── 02_base_models.tex         # Base dynamical models
│   ├── 03_cy_dynamics.tex         # Calabi-Yau extension
│   ├── 04_possibility_manifold.tex# Possibility Manifold
│   ├── 05_tenth_dimension_metaphor.tex  # Tenth dimension
│   ├── 06_ml_embeddings.tex       # Machine learning
│   ├── 07_visualization_and_interfaces.tex  # Visualization
│   └── 08_future_work.tex         # Future directions
├── images/                         # Figures (placeholder)
├── Makefile                        # Build automation
└── build_instructions.md           # This file
```

## Adding Figures

Place figures in the `images/` directory and include them:
```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{images/phase_portrait.png}
    \caption{Phase portrait of the 2D dynamics}
    \label{fig:phase_portrait}
\end{figure}
```

## Customization

### Colors
Edit the color definitions in the main `.tex` file:
```latex
\definecolor{fractalblue}{RGB}{30, 90, 150}
\definecolor{fractalred}{RGB}{180, 60, 60}
```

### Fonts
For different fonts, use `xelatex` with `fontspec`:
```latex
\usepackage{fontspec}
\setmainfont{Linux Libertine}
```

### Page Layout
Modify geometry settings:
```latex
\geometry{
    a4paper,
    left=30mm,
    right=25mm,
    top=30mm,
    bottom=30mm
}
```
