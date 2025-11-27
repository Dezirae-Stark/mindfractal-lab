# LaTeX Book

A comprehensive textbook on the Fractal Consciousness Framework.

## Overview

The book provides complete documentation of the mathematical framework, from foundational dynamics through advanced extensions.

## Chapters

1. **Introduction** — Motivation, goals, overview
2. **Base Models** — 2D/3D real-valued dynamics
3. **CY Dynamics** — Complex Calabi-Yau extension
4. **Possibility Manifold** — The space of all possible states
5. **Tenth Dimension Metaphor** — Conceptual framework
6. **ML Embeddings** — Machine learning approaches
7. **Visualization & Interfaces** — Tools and GUIs
8. **Future Work** — Research directions

## Building the Book

### Requirements

- TeX Live (full installation recommended)
- latexmk

### Build Commands

```bash
cd docs/fractal_consciousness_book

# Build PDF
make pdf
# or
latexmk -pdf fractal_consciousness_book.tex

# Clean auxiliary files
make clean

# Clean everything including PDF
make cleanall
```

### Output

The compiled PDF is generated at:
```
docs/fractal_consciousness_book/fractal_consciousness_book.pdf
```

## Source Structure

```
docs/fractal_consciousness_book/
├── fractal_consciousness_book.tex  # Main document
├── chapters/
│   ├── 01_intro.tex
│   ├── 02_base_models.tex
│   ├── 03_cy_dynamics.tex
│   ├── 04_possibility_manifold.tex
│   ├── 05_tenth_dimension_metaphor.tex
│   ├── 06_ml_embeddings.tex
│   ├── 07_visualization_and_interfaces.tex
│   └── 08_future_work.tex
├── Makefile
└── build_instructions.md
```

## LaTeX Packages Used

- `amsmath`, `amssymb`, `amsthm` — Mathematics
- `hyperref` — Cross-references
- `graphicx` — Figures
- `listings` — Code blocks
- `tikz` — Diagrams
- `algorithm2e` — Pseudocode

## Contributing

To contribute to the book:

1. Edit relevant chapter in `chapters/`
2. Build locally to verify
3. Submit PR with changes

## Download

Pre-built PDF available in GitHub releases (when published).
