# Contributing to MindFractal Lab

Thank you for your interest in contributing to MindFractal Lab! This document provides guidelines and instructions for contributing.

## How to Contribute

### Reporting Issues

- Search existing issues before creating a new one
- Use the issue template (if provided)
- Include:
  - Clear description of the problem
  - Steps to reproduce
  - Expected vs actual behavior
  - Environment (OS, Python version, package versions)
  - Error messages and stack traces

### Suggesting Features

- Open an issue with the "enhancement" label
- Describe the feature and its use case
- Explain how it fits with the project's goals
- Consider providing a proof-of-concept implementation

### Pull Requests

1. **Fork the repository**
   ```bash
   gh repo fork YOUR_USERNAME/mindfractal-lab --clone
   cd mindfractal-lab
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/my-new-feature
   ```

3. **Set up development environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

4. **Make your changes**
   - Follow code style guidelines (see below)
   - Add tests for new functionality
   - Update documentation as needed
   - Keep commits focused and atomic

5. **Run tests**
   ```bash
   pytest tests/ -v
   pytest --cov=mindfractal tests/
   ```

6. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```

7. **Push to your fork**
   ```bash
   git push origin feature/my-new-feature
   ```

8. **Open a pull request**
   - Use a clear, descriptive title
   - Reference any related issues
   - Describe what changed and why
   - Include screenshots/plots if relevant

## Code Style Guidelines

### Python Code

- **PEP 8**: Follow Python style guide
- **Line length**: Max 100 characters (flexibility for readability)
- **Imports**: Group stdlib, third-party, local (separated by blank lines)
- **Naming**:
  - Functions/variables: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`

### Type Hints

Add type hints to all public functions:

```python
def simulate_orbit(
    model: FractalDynamicsModel,
    x0: np.ndarray,
    n_steps: int = 1000,
    return_all: bool = True
) -> np.ndarray:
    """Docstring here."""
    pass
```

### Docstrings

Use NumPy-style docstrings:

```python
def my_function(param1, param2):
    """
    Brief description.

    Detailed description (if needed).

    Parameters
    ----------
    param1 : type
        Description
    param2 : type
        Description

    Returns
    -------
    type
        Description

    Examples
    --------
    >>> my_function(1, 2)
    3
    """
    pass
```

### Comments

- Explain **why**, not **what**
- Keep comments up-to-date with code changes
- Use inline comments sparingly

## Testing Guidelines

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names: `test_step_returns_correct_shape`
- One assertion per test (when possible)
- Test edge cases and error conditions

### Test Structure

```python
class TestMyFeature:
    """Test suite for my feature"""

    def test_basic_case(self):
        """Test basic functionality"""
        result = my_function(input_data)
        assert result == expected

    def test_edge_case(self):
        """Test edge case"""
        with pytest.raises(ValueError):
            my_function(invalid_input)
```

### Coverage

- Aim for 90%+ coverage on core modules
- 70%+ on extensions
- Don't sacrifice readability for coverage

## Documentation

### Code Documentation

- All public functions/classes must have docstrings
- Include examples in docstrings when helpful
- Update API documentation when adding features

### User Documentation

- Update relevant guides (user_guide.md, developer.md)
- Add examples to notebooks if appropriate
- Update README if public API changes

## Development Workflow

### Branch Strategy

- `main`: Stable, production-ready code
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `research/*`: Experimental work
- `docs/*`: Documentation improvements

### Commit Messages

Use conventional commit format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `chore`: Maintenance

Example:
```
feat(fractal_map): Add adaptive refinement algorithm

Implements boundary-adaptive sampling to improve resolution
near fractal boundaries while maintaining performance.

Closes #42
```

### Code Review

All pull requests require review before merging. Reviewers check:
- Code quality and style
- Test coverage
- Documentation
- Performance implications
- Backward compatibility

## Performance Considerations

- Profile before optimizing
- Document performance-critical sections
- Consider C++ backend for CPU-intensive tasks
- Avoid unnecessary memory allocations in loops
- Use NumPy vectorization when possible

## Android Compatibility

When adding features, ensure compatibility with:
- PyDroid 3
- Termux
- Matplotlib Agg backend

Avoid:
- TkInter or other GUI backends (except Kivy)
- GPU-only dependencies
- Platform-specific code without fallbacks

## Licensing

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

- Open an issue for questions
- Use GitHub Discussions for broader topics
- Check existing documentation first

## Code of Conduct

Be respectful, inclusive, and constructive. We're all here to advance science and build useful tools.

---

**Thank you for contributing to MindFractal Lab!**
