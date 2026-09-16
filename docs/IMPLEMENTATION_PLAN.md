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
  at a true mean of 100 weeks shows a naive mean near 80 weeks over the Figure 3
  windows (79.7 measured over 20000 windows), so the paper twin is generated from
  generative means chosen so that the naive means reproduce the printed 4.3 and
  100 weeks (cohort.naive_mean_targets, bordi2013_spec with match_naive_means True).
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
- The hysteresis band fraction has two defaults on purpose: model.DEFAULT_BAND_FRACTION
  of 0.3, which every band taking function of model and simulate uses, and
  cohort.SDE_ENGINE_BAND_FRACTION of 0.4, which the sde engine of CohortSpec uses.
  The reason is measured rather than stylistic. A band decides how many complete
  remissions of a path are shorter than a week, and a sub week remission merges the
  two relapses around it once the path is written as weeks. On the rounding corrected
  101 and 3.3 weeks the measured table of CohortSpec was built on, about one complete
  remission in five and a half is shorter than a week at 0.3 and about one in ten at
  0.4, which puts the naive means of an sde cohort about a fifth to a third above the
  renewal ones at 0.3 and about a tenth to a fifth above them at 0.4. It is not
  widened further: at 0.5 the cohort of the paper has no calibration at all, because
  the asymmetry a wider band asks for passes the saddle-node fold at
  fold_beta(1) = 0.3849. The table of measurements is in the Notes of CohortSpec, and
  the cost of the wider default, a pair of durations that calibrates at 0.3 and raises
  at 0.4, is the worked example beside it.
- The closing table of datasets.reproduction_table carries one barrier ratio row, the
  cohort one. The per patient ratios of patients 23, 32 and 53 are deliberately not
  rows of it, because they are read off the durations the article prints rather than
  off any record, and the table measures a weekly frame. They live in
  cohort.paper_patients, one row per patient, with each printed value beside the
  value recomputed from its neighbours.
- Time varying parameters are out of scope for version 0.1 and are recorded as future
  work. The article's beta and sigma are constants of a patient, and a slowly varying
  beta(t) or sigma(t), which would let a barrier drift over a follow up, would need a
  calibration procedure and a validation the article gives no material for. Nothing in
  the equations forbids it, so it is a later version rather than a closed question.
- date-released in CITATION.cff is the planned first release date rather than a date
  anything was released on, and it is updated with the release if that date moves. The
  "comment" key of codemeta.json is kept as the maintainer note it is, saying which
  identifier fields Zenodo fills in at that release and where the ORCID belongs.

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
**Status**: Complete. All modules, the CLI, the public API with cite(), a
doctest harness over every module, and the shipped synthetic twin are in place.
test_periodicity tests relapse onsets (flat spectrum under the memoryless null)
rather than the state series (red spectrum). The closing table's barrier ratio
tolerance is 0.2 absolute for the synthetic twin because seventy records carry
sampling noise in the two naive means; 0.05 applies only to a deterministic
recomputation from the printed means.

## Stage 4: Notebooks, documentation, metadata, data
**Goal**: Four notebooks executed headlessly, synthetic data files with provenance,
mkdocs site (index, theory, reproducing, citing, api), CITATION.cff, codemeta.json,
CHANGELOG, CONTRIBUTING, issue templates, release workflow, JOSS draft.
**Success Criteria**: Notebook 01 closing table within tolerance when executed in
CI; mkdocs build passes strict mode; CITATION.cff validates with cffconvert;
test_reproduce_paper.py green.
**Tests**: tests/test_reproduce_paper.py, notebook execution job, cffconvert.
**Status**: Complete. Four notebooks built by notebooks/build_notebooks.py and
executed in tests; mkdocs site with theory, reproducing, data, citing, API and
paper facts pages; CITATION.cff, codemeta.json, issue forms, release workflow,
docs deploy workflow, JOSS draft under docs/paper.

## Stage 5: Reviews and release preparation
**Goal**: Three independent review rounds over the whole implementation, each
followed by fixes and re-verification, then release checklist for v1.0.0.
**Success Criteria**: Round three finds no critical or important issues; all
quality gates green; definition of done checklist in plan section 11 addressed
or explicitly deferred to the maintainer.
**Tests**: Full suite on three operating systems in CI.
**Status**: In Progress
