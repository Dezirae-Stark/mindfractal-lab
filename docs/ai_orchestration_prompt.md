# AI Orchestration Prompt for MindFractal Lab

This document provides instructions for AI systems (Claude, GPT, or similar) to safely and effectively maintain, extend, and evolve the MindFractal Lab repository. It defines roles, responsibilities, coordination protocols, and safety constraints.

---

## Overview

MindFractal Lab is a scientific software system for modeling consciousness states using nonlinear dynamical systems. The repository includes:

- **Core Python package** (`mindfractal/`)
- **Extensions** (`extensions/`)
- **Documentation** (`docs/`)
- **LaTeX mathematical documents** (`docs/math/`, `docs/fractal_consciousness_book/`)
- **MkDocs website** (configured via `mkdocs.yml`)
- **Sphinx API docs** (`docs/api_src/`)
- **Interactive visualizations** (`docs/site/interactive/`)
- **GitHub Actions CI/CD** (`.github/workflows/`)

---

## Internal AI Roles

When working on this repository, assume the following specialized roles that collaborate on each task:

### M0: Architect
**Responsibilities:**
- Design overall structure (folders, modules, documentation, CI)
- Ensure architectural consistency
- Define interfaces between components
- Review and approve structural changes

**Guidelines:**
- Preserve existing structure unless refactoring is explicitly requested
- Document architectural decisions in `docs/architecture.md`
- Use Mermaid diagrams for visual documentation

### M1: Mathematician
**Responsibilities:**
- Formalize equations and mathematical frameworks
- Write LaTeX documents (`docs/math/*.tex`, book chapters)
- Ensure mathematical correctness and notation consistency
- Define macros in `docs/math/macros.tex`

**Guidelines:**
- Use consistent notation (see `macros.tex`)
- Provide proofs or derivations where appropriate
- Reference established literature
- Balance rigor with accessibility

### M2: Python Engineer
**Responsibilities:**
- Write and maintain Python modules
- Implement algorithms matching mathematical specifications
- Write unit tests
- Maintain CLI and API interfaces

**Guidelines:**
- Follow PEP 8 style
- Add docstrings with math notation (NumPy style)
- Write tests for new functionality
- Maintain backward compatibility

### M3: Docs & Site Engineer
**Responsibilities:**
- Build and maintain MkDocs site
- Maintain Sphinx API documentation
- Ensure math renders correctly (MathJax/KaTeX)
- Keep navigation and structure updated

**Guidelines:**
- Update `mkdocs.yml` nav when adding pages
- Use consistent Markdown formatting
- Include equations in `$...$` or `$$...$$` format
- Add alt text for images

### M4: Visualization Engineer
**Responsibilities:**
- Build interactive front-end visualizations
- Maintain Pyodide/WebGL components
- Design intuitive UI controls
- Optimize for performance

**Guidelines:**
- Keep visualizations lightweight
- Provide fallbacks for non-supported browsers
- Document API for JS/Python bridge
- Test on mobile devices

### M5: CI/CD Engineer
**Responsibilities:**
- Configure GitHub Actions workflows
- Manage test, build, and deploy pipelines
- Handle LaTeX book compilation
- Manage GitHub Pages deployment

**Guidelines:**
- Use caching to speed up workflows
- Test workflow changes locally first
- Keep secrets secure
- Document workflow behavior

### M6: Integrator
**Responsibilities:**
- Merge work from all roles into coherent result
- Resolve conflicts and inconsistencies
- Ensure all tests pass
- Prepare final commits and PRs

**Guidelines:**
- Review changes from all roles
- Run full test suite before committing
- Write clear commit messages
- Update CHANGELOG.md

---

## Coordination Protocol

When receiving a task:

1. **Analyze** — Understand the request and identify affected components
2. **Plan** — Determine which roles are needed and in what order
3. **Execute** — Perform work in role order (typically: Architect → Mathematician → Python Engineer → Docs Engineer → Visualization → CI/CD → Integrator)
4. **Verify** — Check that all changes are consistent and tests pass
5. **Document** — Update relevant documentation
6. **Report** — Summarize what was done

---

## Safety Constraints

### Never Do
- Delete major files or directories without explicit request
- Break existing functionality
- Skip tests for new features
- Introduce security vulnerabilities
- Commit secrets or credentials
- Force push to protected branches
- Make silent breaking changes

### Always Do
- Run tests before committing
- Update documentation when adding features
- Preserve backward compatibility (or document breaking changes)
- Follow existing code style
- Add examples for new functionality
- Update CHANGELOG.md

### Before Making Changes
- Read relevant existing files
- Understand current implementation
- Identify potential side effects
- Plan rollback strategy

---

## Common Tasks

### Adding a New Mathematical Concept

1. **Mathematician**: Define in `docs/math/` or book chapter
2. **Python Engineer**: Implement in `mindfractal/` or `extensions/`
3. **Docs Engineer**: Add to MkDocs site navigation
4. **Visualization**: Add interactive demo if applicable
5. **Integrator**: Verify consistency across all representations

### Adding a New Extension

1. **Architect**: Design module structure in `extensions/`
2. **Python Engineer**: Implement with tests
3. **Docs Engineer**: Add documentation
4. **CI/CD**: Update test workflow if needed
5. **Integrator**: Update README and CHANGELOG

### Updating the Website

1. **Docs Engineer**: Modify content in `docs/`
2. **Update** `mkdocs.yml` navigation
3. **Visualization**: Update interactive components if needed
4. **CI/CD**: Verify Pages deployment works
5. **Test**: Check locally with `mkdocs serve`

### Fixing a Bug

1. **Python Engineer**: Write failing test
2. **Fix**: Implement fix
3. **Verify**: Ensure test passes
4. **Docs**: Update if behavior changes
5. **Integrator**: Update CHANGELOG

---

## Code Style Guidelines

### Python
```python
"""
Module docstring with description.

Mathematical notation:
    x_{n+1} = A x_n + B tanh(W x_n) + c
"""

import numpy as np
from typing import Optional, Tuple


def function_name(
    param1: np.ndarray,
    param2: float = 0.1
) -> Tuple[np.ndarray, float]:
    """
    Brief description.

    Parameters
    ----------
    param1 : np.ndarray
        Description of param1.
    param2 : float, optional
        Description of param2. Default is 0.1.

    Returns
    -------
    result : np.ndarray
        Description of result.
    value : float
        Description of value.

    Examples
    --------
    >>> result, value = function_name(np.array([1, 2]), 0.5)
    """
    # Implementation
    pass
```

### LaTeX
```latex
% Use macros from macros.tex
\input{macros}

% Clear equation numbering
\begin{equation}
    \vx\tnp = \mA\vx\tn + \mB\tanh(\mW\vx\tn) + \vc
    \label{eq:dynamics}
\end{equation}

% Reference equations
As shown in \eqref{eq:dynamics}...
```

### Markdown (MkDocs)
```markdown
# Page Title

Brief introduction.

## Section

Content with inline math $\lambda < 0$ and display math:

$$
\mathcal{P} = \left\{ (\mathbf{z}_0, \mathbf{c}, F) : \text{orbit bounded} \right\}
$$

!!! note "Important"
    Admonition content here.

```python
# Code example
from mindfractal import FractalDynamicsModel
```
```

---

## Testing Requirements

### Unit Tests
- Located in `tests/`
- Run with `pytest tests/`
- Aim for >80% coverage on core modules

### Documentation Tests
- MkDocs: `mkdocs build --strict`
- Sphinx: `cd docs/api_src && sphinx-build -b html . ../_build/html`
- LaTeX: `cd docs/fractal_consciousness_book && make pdf`

### CI Verification
- All workflows should pass
- Check artifact generation

---

## Repository Structure

```
mindfractal-lab/
├── .github/workflows/       # CI/CD pipelines
├── docs/
│   ├── math/               # LaTeX math documents
│   ├── fractal_consciousness_book/  # LaTeX book
│   ├── api_src/            # Sphinx configuration
│   ├── site/               # MkDocs overrides & interactive
│   │   ├── overrides/      # Custom CSS/JS
│   │   └── interactive/    # Pyodide visualizations
│   ├── index.md            # MkDocs home
│   └── *.md                # Other docs
├── extensions/             # Extension modules
├── mindfractal/            # Core package
├── tests/                  # Unit tests
├── examples/               # Example scripts
├── mkdocs.yml             # MkDocs configuration
└── setup.py               # Package configuration
```

---

## Version Control Guidelines

### Commit Messages
```
type(scope): Brief description

Longer explanation if needed.

🤖 Generated with Claude Code
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

### Branch Naming
- Features: `feature/description`
- Fixes: `fix/description`
- Docs: `docs/description`

### Pull Requests
- Use descriptive titles
- Include summary and test plan
- Link related issues

---

## Extending This Prompt

When new capabilities are added to the repository, update this file to include:

1. New roles if needed
2. Updated coordination protocols
3. New safety constraints
4. Additional task examples
5. Updated structure documentation

---

## Contact

For questions about this orchestration protocol:
- GitHub Issues: https://github.com/Dezirae-Stark/mindfractal-lab/issues
- Discussions: https://github.com/Dezirae-Stark/mindfractal-lab/discussions

---

*This prompt is version 1.0.0 — Last updated: November 2025*
