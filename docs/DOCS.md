# Documentation

## Overview

This project uses MkDocs for documentation with the following stack:

- **Engine**: MkDocs
- **Docstring**: mkdocstrings (Python)
- **Jupyter**: mkdocs-jupyter
- **Theme**: mkdocs-shadcn
- **Minify**: mkdocs-minify-plugin

## Quick Commands

```bash
# Install documentation dependencies
uv sync --group docs

# Serve documentation locally
uv run mkdocs serve

# Build documentation
uv run mkdocs build

# Deploy to GitHub Pages
uv run mkdocs gh-deploy
```

## VS Code Tasks

The following tasks are available in VS Code:

- `docs-serve` - Start development server
- `docs-build` - Build static documentation
- `docs-deploy` - Deploy to GitHub Pages

## File Structure

```text
docs/
├── index.md                 # Home page
├── getting-started.md       # Installation and setup
├── usage.md                 # Usage guide
├── reference/               # API reference docs
│   ├── api.md
│   ├── config.md
│   ├── crawler.md
│   ├── db.md
│   ├── embedding.md
│   ├── ingest.md
│   ├── model.md
│   └── search.md
└── notebooks/               # Jupyter notebooks (symlinked)
    ├── api_usage.ipynb
    ├── benchmark_inference_servers.ipynb
    └── embedder.ipynb
```

## Configuration

Documentation configuration is in `mkdocs.yml`. Key features:

- Shadcn theme with modern design
- Code highlighting and copy buttons
- Search functionality
- Automatic API documentation generation
- Jupyter notebook integration
- HTML minification for production

## Deployment

The documentation can be deployed to GitHub Pages using:

```bash
uv run mkdocs gh-deploy
```

This will build the documentation and push it to the `gh-pages` branch.
