# Publication Readiness TODO

Checklist for bringing this repository to a paper-companion release standard from a research software engineering point of view.

## P0: Release Blockers

- [ ] Add a `LICENSE` file.
- [ ] Add a `CITATION.cff` file with the preferred citation for the software.
- [ ] Add archival metadata such as `.zenodo.json` and connect the repo to Zenodo for DOI minting.
- [ ] Cut the paper release from `main`, not from a feature branch.
- [ ] Create a tagged release that matches the paper submission / camera-ready version.

## P0: Reproducibility Blockers

- [ ] Pin the `pysindy` fork to an immutable commit or release in `pyproject.toml`.
- [ ] Pin the same `pysindy` dependency in `binder/requirements.txt`.
- [ ] Add one reproducible environment definition for external users.
Suggested options: `environment.yml`, `requirements-lock.txt`, or similar.
- [ ] Verify the project can be installed from scratch in a clean environment using only repository instructions.
- [ ] Verify Binder still builds after dependency pinning.

## P0: Validation Blockers

- [ ] Add a real `tests/` suite.
- [ ] Add at least one package-level smoke test for `import mfsindy`.
- [ ] Add at least one test for `WeightedWeakPDELibrary`.
- [ ] Add at least one lightweight end-to-end test covering a representative experiment path.
- [ ] Update `noxfile.py` so the `tests` session runs actual tests rather than falling back to `compileall` plus import-only checks.
- [ ] Add CI that runs lint, tests, and docs checks on pull requests.

## P1: Documentation and Repository Identity

- [ ] Pick one canonical repository/project name and use it consistently.
Current mismatch to resolve: `2025_visiting` vs `WLS-Sindy`.
- [ ] Align repository URLs across `README.md`, `mkdocs.yml`, GitHub Pages config, and Binder links.
- [ ] Review README claims about RSE readiness so they match what is actually enforced.
- [ ] Review README references to `nox` sessions and make sure the documented sessions exist.
- [ ] Expand the reproduction instructions for paper results.
Include which notebook/script produces which figures/tables.
- [ ] Document expected runtime, compute requirements, and any long-running steps.
- [ ] Decide whether the double-pendulum forecasting workflow should also appear in the MkDocs navigation.

## P1: Example and Notebook Hygiene

- [ ] Clean notebook outputs that expose local machine paths or environment-specific warnings.
- [ ] Re-run notebooks intended for release so committed outputs are intentional and reviewer-facing.
- [ ] Check that example outputs are either intentionally committed or clearly excluded from version control.
- [ ] Verify all Binder links open the intended notebooks successfully.

## P1: Docs and Build Assurance

- [ ] Add a local/docs environment step so `mkdocs build --strict` can be run and verified before release.
- [ ] Strengthen `scripts/check_docs.py` beyond heading presence checks if it is intended as a meaningful docs quality gate.
- [ ] Confirm the documentation site builds from a clean checkout.

## P2: Final Release Hygiene

- [ ] Check that no generated junk or OS-specific files are included in the release snapshot.
- [ ] Review `.gitignore` and release contents once before tagging.
- [ ] Confirm the published release contains only the files needed to reproduce the paper and understand the software.
- [ ] Add release notes summarizing:
software version, dependency baseline, reproduction entry points, and DOI/citation information.

## Suggested Release Sequence

- [ ] Add metadata files: `LICENSE`, `CITATION.cff`, `.zenodo.json`.
- [ ] Pin dependencies and add a reproducible environment file.
- [ ] Add tests and CI.
- [ ] Clean notebooks and docs links.
- [ ] Verify install, tests, docs build, and Binder.
- [ ] Tag the release and archive it for the paper.
