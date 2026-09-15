# Implementation plan for msrelapse

Source specification: docs/plan1.md (local, not tracked). This file tracks progress
and records the decisions taken where the specification left a choice open.

## Decisions and assumptions

- Data: the de-identified weekly series cannot be released without a decision from
  the co-authors, so the package ships a synthetic twin generated from the fitted
  parameters, with loud provenance, plus instructions for requesting the original.
- Python: 3.10 through 3.14 supported, local development on 3.14.
- Markdown lives in docs/ except README.md at the repository root, so CHANGELOG,
  CONTRIBUTING, the data provenance note and the JOSS draft live under docs/.
- CLI uses argparse to keep the dependency set minimal.
- Package name msrelapse, subject to the PyPI availability check in Stage 1.
- PyPI and Zenodo publishing need account credentials, so Stage 5 prepares the
  release workflow and instructions and leaves the actual release to the maintainer.
- ORCID and DOI fields in CITATION.cff and codemeta.json stay as placeholders
  until the maintainer fills them.
- The paper's printed mean durations are naive means over its own follow up
  windows (all runs including the censored final remission). A cohort generated
  at a true mean of 100 weeks shows a naive mean near 73 weeks over the Figure 3
  windows, so the paper twin is generated from generative means chosen so that
  the naive means reproduce the printed 4.3 and 100 weeks (cohort.naive_mean_targets,
  bordi2013_spec with match_naive_means True).
- The weekly rounding rule marks every week a relapse touches, so continuous
  time engines are aimed at tau_relapse minus 1 and tau_health plus 1 weeks
  (cohort.continuous_targets). Sub week remissions merge neighbouring relapses;
  that residual is documented, not hidden.
- Crossings read off the integration grid overestimate episode lengths by about
  C sqrt(dt); exit_times and the state machine both apply the Brownian bridge
  shift of 0.5826 sigma sqrt(dt) to the thresholds.
- Data files ship inside the package (src/msrelapse/data) with a loader in
  msrelapse.datasets, so they are available after pip install; the plan's data/
  directory is not used.

## Stage 1: Pin down the paper and the ecosystem
**Goal**: Authoritative facts document from the PDF (equations, sign conventions,
every reported number), numerical regression values for both potential
conventions, calibrated beta and sigma, dependency versions and CI action versions.
**Success Criteria**: Facts document written to docs/paper_facts.md; convention
decision recorded; regression values with tolerances listed; PyPI name confirmed.
**Tests**: None yet; produces the values the Stage 3 tests pin.
**Status**: Complete. Outcome: the paper's potential is V(x) = -x^2/2 + alpha x^4/4
+ beta x (equations 2 and 3), drift x - alpha x^3 - beta, noise variance epsilon,
barrier 1/(4 alpha) growing as alpha falls, health well at negative x. The plan's
provisional form was the other parameterisation and is dropped. Facts are in
docs/paper_facts.md. The name msrelapse is free on PyPI. Calibration at alpha = 1
to 100 and 4.3 weeks (mean first passage to the saddle) gives beta 0.210435 and
sigma 0.507768; the Kramers form cannot reach 4.3 weeks because its prefactor
floor is 4.44 weeks.

## Stage 2: Scaffold
**Goal**: Installable src layout package with pyproject extras and dev group,
ruff, mypy strict, pytest with coverage, pre-commit, CI matrix, _params.py holding
every number from the paper with provenance, empty test passing.
**Success Criteria**: uv sync, ruff check, ruff format --check, mypy, pytest all
green locally; CI workflow file present.
**Tests**: tests/test_params.py checks provenance strings are non-empty.
**Status**: Complete. Notes: mypy targets 3.12 because the numpy stubs use the
type statement; ruff excludes docs/ because it formats code blocks in prose;
the MIT classifier is omitted because PyPI rejects it next to a license
expression.

## Stage 3: Modules
**Goal**: model, renewal, io (wave 1); simulate, fit, stats (wave 2); cohort,
plots, cli and the public API with cite() (wave 3). Each module built test first
with the tests listed in plan sections 4.1 to 4.10, each reviewed for spec
compliance and then for code quality by separate agents before the wave is committed.
**Success Criteria**: All plan section 4 tests pass with fixed seeds; coverage at
least 90 percent; mypy strict clean; ruff clean.
**Tests**: tests/test_model.py, test_simulate.py, test_renewal.py, test_fit.py,
test_stats.py, test_io.py, test_cohort.py, test_plots.py, test_cli.py.
**Status**: In Progress

## Stage 4: Notebooks, documentation, metadata, data
**Goal**: Four notebooks executed headlessly, synthetic data files with provenance,
mkdocs site (index, theory, reproducing, citing, api), CITATION.cff, codemeta.json,
CHANGELOG, CONTRIBUTING, issue templates, release workflow, JOSS draft.
**Success Criteria**: Notebook 01 closing table within tolerance when executed in
CI; mkdocs build passes strict mode; CITATION.cff validates with cffconvert;
test_reproduce_paper.py green.
**Tests**: tests/test_reproduce_paper.py, notebook execution job, cffconvert.
**Status**: Not Started

## Stage 5: Reviews and release preparation
**Goal**: Three independent review rounds over the whole implementation, each
followed by fixes and re-verification, then release checklist for v1.0.0.
**Success Criteria**: Round three finds no critical or important issues; all
quality gates green; definition of done checklist in plan section 11 addressed
or explicitly deferred to the maintainer.
**Tests**: Full suite on three operating systems in CI.
**Status**: Not Started
