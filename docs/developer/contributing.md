# Contributing

Guidelines for contributing to MindFractal Lab.

## Getting Started

### 1. Fork the Repository

```bash
# Fork on GitHub, then clone
git clone https://github.com/YOUR_USERNAME/mindfractal-lab.git
cd mindfractal-lab
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install in development mode
pip install -e ".[dev]"
```

### 3. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

## Development Workflow

### Code Style

- Follow PEP 8
- Use type hints
- Maximum line length: 100 characters
- Use Black for formatting

```bash
black mindfractal/ tests/
isort mindfractal/ tests/
```

### Documentation

- Docstrings for all public functions
- Update relevant docs/ pages
- Add examples where helpful

### Testing

- Write tests for new features
- Ensure all tests pass

```bash
pytest tests/ -v
```

### Commit Messages

Follow conventional commits:

```
feat: Add new visualization mode
fix: Correct Lyapunov calculation edge case
docs: Update API reference
test: Add basin boundary tests
refactor: Simplify simulation loop
```

## Pull Request Process

### 1. Update Documentation

- API docs if adding functions
- User guide if changing behavior
- README if significant feature

### 2. Run Checks

```bash
# Format
black --check mindfractal/ tests/
isort --check-only mindfractal/ tests/

# Lint
flake8 mindfractal/ tests/

# Type check
mypy mindfractal/

# Tests
pytest tests/ -v --cov=mindfractal
```

### 3. Submit PR

- Clear title describing change
- Reference any related issues
- Describe what/why/how

### 4. Review

- Address review comments
- Keep PR focused (one feature per PR)

## Areas for Contribution

### Good First Issues

- Documentation improvements
- Test coverage
- Example scripts
- Bug fixes

### Intermediate

- New visualization modes
- Performance optimization
- CLI enhancements

### Advanced

- New dynamical models
- ML integration
- C++ backend extensions

## Code of Conduct

- Be respectful
- Welcome newcomers
- Focus on constructive feedback
- Assume good intentions

## Questions?

- Open an issue for discussion
- Tag maintainers for help
- Check existing issues first
