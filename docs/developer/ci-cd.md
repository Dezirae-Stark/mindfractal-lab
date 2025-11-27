# CI/CD

Continuous Integration and Deployment pipelines.

## GitHub Actions Workflows

### Tests (`tests.yml`)

Runs on push/PR to main:

- Python 3.8-3.12
- Ubuntu, macOS, Windows
- pytest with coverage
- Codecov upload

```yaml
jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']
```

### Documentation Build (`docs_build.yml`)

Builds documentation on docs changes:

- MkDocs Material site
- Sphinx API documentation
- Artifact upload

### Pages Deploy (`pages_deploy.yml`)

Deploys to GitHub Pages on main push:

- Builds MkDocs site
- Deploys to gh-pages branch
- URL: https://dezirae-stark.github.io/mindfractal-lab/

### Book Build (`book_build.yml`)

Compiles LaTeX book:

- Full TeX Live installation
- PDF generation
- Artifact upload

## Workflow Triggers

| Workflow | Push | PR | Manual |
|:---------|:-----|:---|:-------|
| Tests | main, develop | main | Yes |
| Docs Build | main | main | Yes |
| Pages Deploy | main | - | Yes |
| Book Build | main | main | Yes |

## Path Filters

Workflows only run when relevant files change:

```yaml
paths:
  - 'mindfractal/**'
  - 'tests/**'
  - 'setup.py'
```

## Secrets

Required repository secrets:

| Secret | Purpose |
|:-------|:--------|
| `GITHUB_TOKEN` | Auto-provided for actions |
| `CODECOV_TOKEN` | Coverage upload (optional) |

## Local CI Simulation

Run checks locally before pushing:

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

# Build docs
mkdocs build --strict
```

## Deployment

### PyPI Release

Manual process:

```bash
# Build
python -m build

# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*
```

### GitHub Pages

Automatic on main push via `pages_deploy.yml`.

### Docker (Future)

Planned container deployment for webapp.

## Troubleshooting

### Workflow Failures

1. Check Actions tab for logs
2. Look for red X on specific job
3. Expand failed step
4. Fix locally and push

### Common Issues

- **pip cache error**: Remove `cache: 'pip'` if no requirements.txt
- **MkDocs strict mode**: Fix all broken links
- **Python version**: Ensure code works on 3.8+
