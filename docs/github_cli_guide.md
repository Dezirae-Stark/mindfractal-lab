# GitHub CLI Guide for MindFractal Lab (Termux)

Complete guide to using GitHub CLI (`gh`) with MindFractal Lab on Android Termux.

## Table of Contents

1. [Setup](#setup)
2. [Authentication](#authentication)
3. [Repository Operations](#repository-operations)
4. [Branch Management](#branch-management)
5. [Issues](#issues)
6. [Pull Requests](#pull-requests)
7. [Releases](#releases)
8. [Projects](#projects)
9. [Milestones](#milestones)
10. [Common Workflows](#common-workflows)

---

## Setup

### Install GitHub CLI in Termux

```bash
pkg update
pkg install gh git
```

### Verify Installation

```bash
gh --version
git --version
```

---

## Authentication

### Login to GitHub

```bash
gh auth login
```

Follow the prompts:
1. Choose "GitHub.com"
2. Choose "HTTPS" protocol
3. Authenticate via browser or paste token

### Check Authentication Status

```bash
gh auth status
```

### Logout

```bash
gh auth logout
```

---

## Repository Operations

### Create Repository

```bash
# Create public repository
gh repo create mindfractal-lab --public --source=. --remote=origin

# Create private repository
gh repo create mindfractal-lab --private --source=. --remote=origin --description="Fractal Dynamical Consciousness Model"
```

### Clone Repository

```bash
gh repo clone Dezirae-Stark/mindfractal-lab
cd mindfractal-lab
```

### View Repository

```bash
# View in terminal
gh repo view

# Open in browser
gh repo view --web
```

### Fork Repository

```bash
gh repo fork Dezirae-Stark/mindfractal-lab --clone
```

### List Repositories

```bash
gh repo list Dezirae-Stark
```

### Delete Repository (Use with Caution!)

```bash
gh repo delete Dezirae-Stark/mindfractal-lab
```

---

## Branch Management

### List Branches

```bash
# Local branches
git branch -a

# Remote branches on GitHub
gh api repos/:owner/:repo/branches | grep '"name"'
```

### Create and Push Branches

```bash
# Create branch locally
git branch feature/my-feature

# Create and switch to branch
git checkout -b feature/my-feature

# Push branch to GitHub
git push -u origin feature/my-feature

# Create multiple branches
git branch feature/3d-model
git branch feature/gui-kivy
git branch feature/webapp
git push origin feature/3d-model feature/gui-kivy feature/webapp
```

### Delete Branches

```bash
# Delete local branch
git branch -d feature/my-feature

# Delete remote branch
git push origin --delete feature/my-feature
```

### Rename Default Branch to Main

```bash
git branch -m master main
git push -u origin main
```

---

## Issues

### Create Issue

```bash
# Interactive creation
gh issue create

# With title and body
gh issue create --title "Add new feature" --body "Description of the feature"

# With label
gh issue create --title "Fix bug" --body "Bug description" --label "bug"

# From file
gh issue create --title "Feature request" --body-file issue_description.md
```

### List Issues

```bash
# All open issues
gh issue list

# All issues (open and closed)
gh issue list --state all

# Filter by label
gh issue list --label "enhancement"

# Filter by assignee
gh issue list --assignee Dezirae-Stark
```

### View Issue

```bash
# View in terminal
gh issue view 1

# Open in browser
gh issue view 1 --web
```

### Close Issue

```bash
gh issue close 1
```

### Reopen Issue

```bash
gh issue reopen 1
```

### Comment on Issue

```bash
gh issue comment 1 --body "This is a comment"
```

### Assign Issue

```bash
gh issue edit 1 --add-assignee Dezirae-Stark
```

---

## Pull Requests

### Create Pull Request

```bash
# Interactive creation
gh pr create

# With title and body
gh pr create --title "Add feature" --body "Feature description" --base main --head feature/my-feature

# From current branch
gh pr create --fill  # Auto-fills title and body from commits
```

### List Pull Requests

```bash
# Open PRs
gh pr list

# All PRs
gh pr list --state all

# Filter by author
gh pr list --author Dezirae-Stark
```

### View Pull Request

```bash
# View in terminal
gh pr view 1

# Open in browser
gh pr view 1 --web
```

### Check Out PR Locally

```bash
gh pr checkout 1
```

### Merge Pull Request

```bash
# Merge (create merge commit)
gh pr merge 1 --merge

# Squash and merge
gh pr merge 1 --squash

# Rebase and merge
gh pr merge 1 --rebase

# Auto-merge when checks pass
gh pr merge 1 --auto --squash
```

### Close PR Without Merging

```bash
gh pr close 1
```

### Review Pull Request

```bash
# Approve
gh pr review 1 --approve

# Request changes
gh pr review 1 --request-changes --body "Please address these issues"

# Comment
gh pr review 1 --comment --body "Looks good!"
```

---

## Releases

### Create Release

```bash
# Create release from tag
gh release create v0.1.0 --title "Version 0.1.0" --notes "Release notes"

# Create release with notes from file
gh release create v0.1.0 --title "Version 0.1.0" --notes-file RELEASE_NOTES.md

# Create draft release
gh release create v0.1.0 --draft --title "Version 0.1.0" --notes "Draft release"

# Create pre-release
gh release create v0.1.0-beta --prerelease --title "Beta Release" --notes "Beta version"
```

### List Releases

```bash
gh release list
```

### View Release

```bash
# View in terminal
gh release view v0.1.0

# Open in browser
gh release view v0.1.0 --web
```

### Download Release Assets

```bash
gh release download v0.1.0
```

### Delete Release

```bash
gh release delete v0.1.0
```

---

## Projects

### Create Project

```bash
# Create project for user
gh project create --owner Dezirae-Stark --title "MindFractal Lab Development"

# Create project for organization
gh project create --owner my-org --title "Project Name"
```

### List Projects

```bash
gh project list --owner Dezirae-Stark
```

### View Project

```bash
gh project view 1 --owner Dezirae-Stark
```

### Add Item to Project

```bash
gh project item-add 1 --owner Dezirae-Stark --url https://github.com/Dezirae-Stark/mindfractal-lab/issues/1
```

---

## Milestones

### Create Milestone

```bash
# Using API
gh api repos/:owner/:repo/milestones -X POST \
  -f title="M1: Core Engine" \
  -f description="Core 2D fractal dynamics model" \
  -f state="open"

# With due date
gh api repos/:owner/:repo/milestones -X POST \
  -f title="M1: Core Engine" \
  -f description="Core model implementation" \
  -f due_on="2025-12-31T23:59:59Z" \
  -f state="open"
```

### List Milestones

```bash
gh api repos/:owner/:repo/milestones | grep -E '"title"|"number"'
```

### View Milestone

```bash
gh api repos/:owner/:repo/milestones/1
```

### Assign Issue to Milestone

```bash
gh issue edit 1 --milestone "M1: Core Engine"
```

### Close Milestone

```bash
gh api repos/:owner/:repo/milestones/1 -X PATCH -f state="closed"
```

---

## Common Workflows

### Complete Repository Setup (MindFractal Lab Example)

```bash
# 1. Navigate to project directory
cd /data/data/com.termux/files/home/mindfractal-lab

# 2. Initialize git (if not already done)
git init
git branch -m main

# 3. Add files and commit
git add .
git commit -m "feat: Initial commit"

# 4. Create GitHub repository
gh repo create mindfractal-lab --private --source=. --remote=origin --description="Fractal Dynamical Consciousness Model"

# 5. Push to GitHub
git push -u origin main

# 6. Create feature branches
git branch feature/3d-model feature/gui-kivy feature/webapp feature/cpp-backend feature/psychomapping
git branch research/metastability-analysis research/parameter-fractal-experiments
git branch docs/enhancement

# 7. Push all branches
git push origin feature/3d-model feature/gui-kivy feature/webapp feature/cpp-backend feature/psychomapping research/metastability-analysis research/parameter-fractal-experiments docs/enhancement

# 8. Create issues
gh issue create --title "Add bifurcation diagram generator" --label "enhancement"
gh issue create --title "Investigate parameter-boundary instability"
gh issue create --title "Implement Lyapunov exponent calculator"
gh issue create --title "Speed up fractal map calculation"
gh issue create --title "GPU offload feasibility study"
gh issue create --title "WebGL visualization prototype"
gh issue create --title "Prepare package for PyPI publishing"

# 9. Create milestones
gh api repos/:owner/:repo/milestones -X POST -f title="M1: Core Engine (2D)" -f description="Core 2D model"
gh api repos/:owner/:repo/milestones -X POST -f title="M2: Visualizations" -f description="Visualization suite"
gh api repos/:owner/:repo/milestones -X POST -f title="M3: Extensions" -f description="3D, GUI, webapp"
gh api repos/:owner/:repo/milestones -X POST -f title="M4: C++ Backend" -f description="Performance backend"
gh api repos/:owner/:repo/milestones -X POST -f title="M5: 1.0 Release Preparation" -f description="Production readiness"

# 10. Create project board
gh project create --owner Dezirae-Stark --title "MindFractal Lab Development"

# 11. Create and push release tag
git tag -a v0.1.0 -m "Initial release v0.1.0"
git push origin v0.1.0

# 12. Create GitHub release
gh release create v0.1.0 --title "MindFractal Lab v0.1.0" --notes "Initial release with complete feature set"
```

### Fork and Contribute Workflow

```bash
# 1. Fork repository
gh repo fork Dezirae-Stark/mindfractal-lab --clone
cd mindfractal-lab

# 2. Create feature branch
git checkout -b feature/my-contribution

# 3. Make changes and commit
git add .
git commit -m "feat: Add my contribution"

# 4. Push to your fork
git push -u origin feature/my-contribution

# 5. Create pull request
gh pr create --base main --title "Add my contribution" --body "Description of changes"
```

### Issue Tracking Workflow

```bash
# 1. Create issue for bug
gh issue create --title "Fix memory leak" --body "Memory leak in fractal_map.py" --label "bug"

# 2. Create branch for fix
git checkout -b bugfix/memory-leak

# 3. Fix and commit
git add .
git commit -m "fix: Resolve memory leak in fractal_map

Closes #5"

# 4. Push and create PR
git push -u origin bugfix/memory-leak
gh pr create --fill

# 5. Merge when approved
gh pr merge --squash

# 6. Issue automatically closes due to "Closes #5" in commit message
```

### Release Workflow

```bash
# 1. Update version in setup.py and CHANGELOG.md
# (manual edit)

# 2. Commit version bump
git add setup.py CHANGELOG.md
git commit -m "chore: Bump version to 0.2.0"

# 3. Create and push tag
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0

# 4. Create GitHub release
gh release create v0.2.0 --title "MindFractal Lab v0.2.0" --notes-file RELEASE_NOTES.md

# 5. Verify release
gh release view v0.2.0
```

---

## Tips and Best Practices

### Authentication

- Store token securely
- Use `gh auth refresh` if token expires
- Use SSH for git operations: `gh auth setup-git`

### Repository Management

- Always create private repos for sensitive work
- Use descriptive names and clear descriptions
- Keep README.md updated

### Branch Strategy

- `main`: Stable, production-ready code
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `research/*`: Experimental work
- `docs/*`: Documentation updates

### Commit Messages

Use conventional commit format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Tests
- `refactor:` Code refactoring
- `chore:` Maintenance

### Issue Management

- Use labels for categorization
- Assign milestones for planning
- Reference issues in commits: `Closes #123`
- Use templates for consistency

### Pull Requests

- Keep PRs focused and small
- Write clear descriptions
- Link related issues
- Request reviews from teammates
- Use draft PRs for work-in-progress

### Releases

- Follow semantic versioning: MAJOR.MINOR.PATCH
- Write detailed release notes
- Tag releases consistently
- Include changelog

---

## Troubleshooting

### "Permission denied" Error

```bash
# Refresh authentication
gh auth refresh
gh auth status
```

### "Not found" Error

```bash
# Verify repository exists
gh repo view Dezirae-Stark/mindfractal-lab

# Check you have access
gh repo list Dezirae-Stark
```

### Rate Limit Exceeded

```bash
# Check rate limit status
gh api rate_limit

# Wait for reset or authenticate to increase limit
gh auth login
```

### Merge Conflicts

```bash
# Update branch from main
git checkout feature/my-feature
git fetch origin
git merge origin/main

# Resolve conflicts manually, then:
git add .
git commit -m "Merge main into feature branch"
git push
```

---

## Advanced: GitHub API with `gh api`

### Custom API Calls

```bash
# Get repository info
gh api repos/:owner/:repo

# List collaborators
gh api repos/:owner/:repo/collaborators

# Get commit history
gh api repos/:owner/:repo/commits

# Create webhook
gh api repos/:owner/:repo/hooks -X POST -f url="https://example.com/webhook" -f events[]="push"
```

### Pagination

```bash
# Get all issues (handles pagination)
gh api --paginate repos/:owner/:repo/issues
```

### GraphQL Queries

```bash
# GraphQL query
gh api graphql -f query='
  query {
    repository(owner: "Dezirae-Stark", name: "mindfractal-lab") {
      issues(first: 10) {
        nodes {
          title
          number
        }
      }
    }
  }
'
```

---

## References

- [GitHub CLI Documentation](https://cli.github.com/manual/)
- [GitHub API Documentation](https://docs.github.com/en/rest)
- [Termux Wiki](https://wiki.termux.com/)
- [Git Documentation](https://git-scm.com/doc)

---

**Last Updated**: 2025-11-17
**Version**: 1.0
**Repository**: https://github.com/Dezirae-Stark/mindfractal-lab
