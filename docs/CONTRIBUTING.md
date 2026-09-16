# Contributing

Bug reports, reproductions on other data and pull requests are all welcome. The
package is a reference implementation of one article, so the first question
about any change is whether it keeps the implementation faithful to the source.

## Setting up

The project is managed with [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/renato-umeton/multiple-sclerosis-modeling-bordi
cd multiple-sclerosis-modeling-bordi
uv sync --all-extras --all-groups
uv run pre-commit install
```

`uv sync` downloads the Python it needs, creates `.venv` and installs from
`uv.lock`, so everyone runs the same versions. Prefix commands with `uv run`
rather than activating the environment, and name the extras or the group a
command needs so that `uv run` installs them if they are missing.

## Quality gates

Everything below has to pass before a change is merged. The test suite runs in
continuous integration on Linux, macOS and Windows across Python 3.10 to 3.14;
the lint, documentation, notebook and distribution jobs run once, on Linux and
Python 3.13.

```bash
uv run ruff check .                            # lint
uv run ruff format --check --diff .            # formatting
uv run mypy                                    # strict: src, tests, notebooks
uv run --all-extras pytest --cov=msrelapse     # the suite, with coverage
uv run --group docs mkdocs build --strict      # the documentation site
uvx cffconvert --validate -i CITATION.cff      # the citation metadata
uv build && uvx twine check --strict dist/*    # the distribution
```

- **ruff** lints and formats, with the numpy docstring convention. Every public
  function and class carries a numpy style docstring: a one line summary,
  Parameters, Returns, Raises where it raises, and Examples where an example
  helps. The gate above only checks the formatting; `uv run ruff format .`
  applies it.
- **mypy** runs in strict mode over `src`, `tests` and `notebooks`. Type hints
  everywhere, including in tests and in `notebooks/build_notebooks.py`, which
  is held to the same bar because the test suite imports it.
- **pytest** is configured with `--strict-markers` and `--strict-config`.
  Coverage is off unless it is asked for, so pass `--cov=msrelapse` as the gate
  above does; a measured run fails below 90 percent. Two markers exist: `slow`
  for long running tests and `notebook` for the notebook execution tests.
- **cffconvert and twine** cover the metadata. The lint job validates
  `CITATION.cff`, which Zenodo and the reference managers read, and the build
  job builds the wheel and the sdist, checks them with `twine check --strict`,
  runs the packaged suite from the unpacked sdist and installs the wheel on its
  own to run the command line. A change to the version, the classifiers, the
  sdist include list or the README rendering fails there.
- **pre-commit** runs ruff, mypy and `nbstripout`, the last of which keeps
  notebook outputs out of the repository. Never bypass the hooks.
- **House style** is a test rather than a convention left to review. The sweep
  in `tests/test_docs.py` reads every text file in your working tree that
  `.gitignore` does not name outright, source, prose, notebooks and workflow
  files alike, and fails on an em dash, on a line of three hyphens, or on the
  name of a generative model or its tooling. A change can therefore fail the
  suite on a sentence rather than on code, and the fix is to rewrite the
  sentence: a comma or a colon where an em dash is tempting, a heading or a
  blank line where a horizontal rule is. The sweep asks git nothing, so a stray
  untracked file left in the tree, a scratch note or a pasted transcript, is
  read like any source file; move it out of the tree rather than rewrite it.

## Working on the code

Write the failing test first, make it pass with the smallest change that does,
then clean up with the tests green. Small commits that each compile and pass
are preferred over one large one.

### Test conventions

- **Every random operation takes a generator, an integer seed or None.** Tests
  pass a fixed seed and no test depends on a global random state.
- **No flaky tolerances.** A stochastic assertion states its tolerance and the
  tolerance is chosen from repeated runs, not from the first run that passed.
  Where a bound is looser than a reader would expect, the source says in a
  comment what was measured and why the bound is where it is.
- **Test behaviour, not implementation.** Assert on what a function returns and
  on the errors it raises, not on how it computes them.
- Tests that execute a notebook carry the `notebook` marker, so the rest of the
  suite can be run with `-m 'not notebook'`.

### Numbers from the article

Every number the article reports lives in `src/msrelapse/_params.py`, in the
`PAPER` object, each with its unit and the sentence it came from. **No other
module computes with one: import `PAPER` instead.** This is what lets a reader
check the package against the article one field at a time, and it is what keeps
the same value from drifting between two places. `msrelapse params` prints the
whole set.

Two ways of writing a printed value down are allowed, and both are in the
repository already. A docstring or a doctest may quote one as an illustration,
which several do, because an example reads better with the article's own
numbers than with an attribute lookup. A test may quote one where quoting it is
the point: `tests/test_params.py` exists to pin each field against the article,
and a test that passes a printed value as an ordinary fixture, rather than as a
claim about the article, is fine as well.

A number derived from the article, rather than reported by it, is also welcome
in `PAPER`, as long as its source string says it is derived. A number that is
this package's own choice, a time step or a tolerance, belongs in the module
that uses it, as a named constant with a comment saying what was measured to
pick it.

If you disagree with a value, the place to argue is `docs/paper_facts.md`,
which records how each was verified against the article, and any change to a
value has to update the provenance string with it.

## Notebooks

Notebooks are stored without outputs, which `nbstripout` enforces. Execute them
headlessly the way continuous integration does:

```bash
uv run --group notebooks pytest -m notebook
```

That runs each notebook end to end and fails on the first error. Notebook 01
also checks that its closing table stays inside the tolerances documented in
[Reproducing the paper](reproducing.md), so a change that moves a reproduced
number will be caught there.

## Documentation

The site is built with mkdocs-material and mkdocstrings.

```bash
uv run --group docs mkdocs serve           # live preview on localhost
uv run --group docs mkdocs build --strict  # what continuous integration runs
```

`--strict` turns a broken cross reference or a page missing from the navigation
into a failure, which is the point of building the docs in continuous
integration at all. API pages are one file per module under `docs/api/`, each
holding a short introduction and an mkdocstrings directive, so a new module
needs a new page and a new navigation entry in `mkdocs.yml`. Prose pages avoid
em dashes and horizontal rules, which the house style sweep above enforces over
every file it reaches, and the mathematics is written in the article's own
convention, $V(x) = -x^2/2 + \alpha x^4/4 + \beta x$.

The suite builds the site once more, in `tests/test_docs.py`, and reads the
built API pages back. That is the only check on what a visitor is served rather
than on what the sources say, so it is the one that catches a Sphinx role the
mkdocstrings handler passes through. It carries the `slow` marker and is
skipped where the `docs` group is not installed. Set `MSRELAPSE_REQUIRE_DOCS`
in an environment that is meant to have the group, and that skip becomes a
failure naming what is missing:

```bash
MSRELAPSE_REQUIRE_DOCS=1 uv run --group docs pytest -m slow tests/test_docs.py
```

## Releasing

Releases are the maintainer's, and the steps are:

1. Update `docs/CHANGELOG.md`, moving the Unreleased entries under the new
   version and its date.
2. Bump the version in `pyproject.toml`, `src/msrelapse/__init__.py`,
   `CITATION.cff` and `codemeta.json`. The suite compares those four against
   each other, so leaving one behind is a failing test rather than a quiet
   mismatch. Two further copies sit in prose: the `version` field of the
   software entry on `docs/citing.md`, which the suite checks against the
   BibTeX the package prints, and the citation sentence near the end of
   `README.md`, which nothing checks.
3. Set the release date in the four places that carry one: `date-released` in
   `CITATION.cff`, `datePublished` and `dateModified` in `codemeta.json`, and
   the `year` of the software entry in `src/msrelapse/_citation.py`. The suite
   pins `date-released`, so moving it is a failing test until the assertion
   moves with it; the other three are checked by nothing. `dateCreated` is the
   date the work started and stays where it is.
4. Check that the whole gate list above is green on a clean checkout.
5. Tag the commit as `vX.Y.Z` and push the tag. The release workflow builds the
   distributions, checks them with `twine check --strict`, and publishes to
   PyPI through trusted publishing, so no API token exists anywhere in the
   repository. The publishing environment requires the maintainer's approval,
   so an accidental tag cannot publish unattended.
6. Zenodo archives the GitHub release and mints the archive DOI. Put that DOI
   into `CITATION.cff`, `codemeta.json` and the software BibTeX entry in
   `src/msrelapse/_citation.py`, replacing the placeholder. The same
   placeholder sits in the BibTeX block of `docs/citing.md`, which the suite
   checks against the entry the package prints, and in the badge at the top of
   `README.md`, which nothing checks. Then release the patch that carries them.
7. Before submitting `docs/paper/paper.md` to the Journal of Open Source
   Software, fill in the affiliation and the ORCID the header leaves as
   placeholders, and turn the fenced `yaml` block at the top of that file into
   the front matter JOSS asks for. It is fenced in the repository because the
   house style keeps a line of three hyphens out of every file here.

## Conduct

Be accurate and be kind. Claims about the article belong in
`docs/paper_facts.md` with the evidence attached, and claims about patients
belong nowhere in this repository: the records it ships are synthetic.
