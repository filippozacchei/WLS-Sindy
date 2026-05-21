# MF-SINDy

Multi Fidelity Sparse Identification of Nonlinear Dynamics (WLS-SINDy) extends the weak-form SINDy framework with heteroscedastic noise models, multi-fidelity training data, and GLS-style whitening. This repository houses the research code, documentation, and a standalone paper-results draft.

The project follows the US-RSE recommendations for research software: a `src/` package installable with `pip`, reproducible experiments in `examples/`, versioned documentation, and automated quality gates via pre-commit + nox.

## Repository Layout

```
.
├── src/mfsindy               # installable Python package (pip/pyproject)
├── examples                  # GLS / WLS experiments & notebooks
├── docs                      # documentation entry point + assets
├── scripts                   # utility scripts for project maintenance
├── paper_results_section.tex  # standalone LaTeX draft for the results section
├── pyproject.toml            # packaging metadata
└── noxfile.py, .pre-commit-config.yaml (added below)
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]   # installs mfsindy plus dev tooling
```

The package exposes reusable case modules (`mfsindy.cases.*`), plotting helpers, and the custom `WeightedWeakPDELibrary` implementation.

## Documentation & Research Assets

- [Hosted documentation](https://filippozacchei.github.io/MFSindy/) (MkDocs Material) mirrors the Markdown under `docs/`. Use `mkdocs serve` for local previews; `.github/workflows/docs.yml` guards the build, while `.github/workflows/mkdocs-deploy.yml` publishes to GitHub Pages.
- `scripts/` contains lightweight project utilities.
- `paper_results_section.tex` is a standalone LaTeX draft of the results section that can be dropped into the manuscript source tree.

## Examples

Each standard case lives under `examples/<case>/` with `part1.ipynb` and `part2.ipynb` entry points. Typical workflow:

```bash
cd examples/lorenz
jupyter lab part1.ipynb   # multi-trajectory weighting scenario
jupyter lab part2.ipynb   # heteroskedastic GLS scenario
```

Use `part1.ipynb` for the multi-trajectory weighting scenario and `part2.ipynb` for the heteroskedastic run (Burgers, Hopf, Lorenz, pendulum, isothermal flow, and the base diffusion tutorial).

The repository can also host standalone workflows when the usual part-1/part-2 split is not the right fit. For example, `examples/double_pendulum/forecasting.ipynb` studies a forecasting benchmark with one HF and several LF observations of the same double-pendulum trajectory, comparing `HF`, `LF`, and `MF` weighted weak-SINDy models under either a polynomial or physics-informed library.

## Development Workflow

1. **Automation**: `nox` sessions (`lint`, `tests`, `docs`, etc.) encapsulate repeatable checks. Run `nox -s lint` before pushing.
2. **Pre-commit**: Install hooks via `pre-commit install` to lint staged files (ruff, black, end-of-file fixes, YAML formatting).
3. **Coding style**: follow PEP 8/pyproject formatting, keep notebooks in `docs/notebooks` or `examples/*/*.ipynb`, and store figures under `docs/assets/` when they are part of the documentation site.
4. **Data**: large simulation outputs belong in `examples/**/results/` (git-ignored) or external storage. Only commit configuration + lightweight references.

For additional context on the scientific motivation, see the Markdown documentation under `docs/` and the standalone manuscript draft in `paper_results_section.tex`.
