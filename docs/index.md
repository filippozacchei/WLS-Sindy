# MF-SINDy Documentation

Welcome to the documentation hub for MF-SINDy. The content is authored in Markdown under `docs/` and rendered via [MkDocs Material](https://squidfunk.github.io/mkdocs-material/). Start here for installation, tutorials, and automation notes.

## Quickstart

1. **Clone & install**
   ```bash
   git clone https://github.com/filippozacchei/2025_visiting.git
   cd 2025_visiting
   python -m venv .venv && source .venv/bin/activate
   pip install -e .[dev]
   ```
2. **Launch notebooks** – run `jupyter lab examples/<case>/part1.ipynb` (multi-trajectory GLS) or `part2.ipynb` (heteroscedastic GLS). The repository also includes a standalone forecasting notebook at `examples/double_pendulum/forecasting.ipynb`.
3. **Cache outputs** – figures/video exports land in `examples/<case>/results/` (git-ignored) so experiments remain reproducible without polluting the repo.

For manuscript writing, the repository root also includes `paper_results_section.tex`, a standalone draft of the paper's results section.

## Base Tutorials

Use the base notebooks for guided walkthroughs that skip the heavier case-study machinery:

- `examples/base/part1.ipynb` – multi-trajectory GLS with trajectory-wise fidelity weights.
- `examples/base/part2.ipynb` – heteroscedastic GLS along a single trajectory.

Together they cover the scalar ODE setup, trajectory-wise weighting across ensembles, weak regression assembly, and the weighted weak-form workflow in a lighter setting than the full case studies.

## Automation via GitHub Actions

- `.github/workflows/docs.yml` builds the MkDocs site (strict mode) and runs sanity checks on pull requests.
- `.github/workflows/mkdocs-deploy.yml` publishes the static site to GitHub Pages on pushes to `main`.

To preview locally:

```bash
pip install -e .[dev]
mkdocs serve
```

MkDocs will watch files and provide a hot-reload server at `http://127.0.0.1:8000/`.
